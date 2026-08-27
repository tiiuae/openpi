"""Terminal UI for the Trossen client: a pinned input line plus rich logging.

The control loop logs while the operator types instructions into the same
terminal. With ordinary line-buffered input the two fight each other — a log
record lands mid-word and the half-typed instruction scrolls away. Here the
input line is pinned to the bottom of the terminal with rich's ``Live`` display:
log records scroll above it while the line being typed stays put, along with a
status line showing what the policy is doing.

Doing that means reading stdin a character at a time (the terminal's own line
editor can't be redrawn by us), so the terminal is put in cbreak mode. ``ISIG``
stays enabled, so Ctrl+C still interrupts the client as usual.

Whenever that isn't possible — no TTY, no termios, output piped to a file — the
plain line-buffered listener is used instead and everything still works, just
without the pinned line.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import atexit
from collections.abc import Iterable
import logging
import os
import select
import sys
import threading

try:
    import termios
    import tty
except ImportError:  # not a POSIX terminal (Windows) — the plain listener covers it
    termios = None
    tty = None

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text

logger = logging.getLogger(__name__)

console = Console()

_ENTER = ("\r", "\n")
_BACKSPACE = ("\x7f", "\b")
_CTRL_D = "\x04"
_ESCAPE = "\x1b"
_REFRESH_PER_SECOND = 8

# Pinned above the status line so the keyword commands stay visible while the
# log scrolls; "help" prints the full table with explanations.
_COMMAND_HINT = Text(
    "record · hold/stop = end take · pass · fail · reject · open/close · home · sleep · twist · wave · help · quit",
    style="dim",
)


def setup_logging(*, debug: bool = False) -> None:
    """Route logging through rich, on the same console as the pinned input line.

    Sharing the console is what lets ``Live`` redraw the input line below each
    log record instead of being overwritten by it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True, log_time_format="[%H:%M:%S]")],
        force=True,  # drop the basicConfig handler installed at import time
    )
    if debug:
        # Only this client's loggers — the root logger would also unleash
        # lerobot's per-frame camera debug output.
        logging.getLogger("__main__").setLevel(logging.DEBUG)
        logging.getLogger("scripted_motions").setLevel(logging.DEBUG)


def print_help(rows: Iterable[tuple[str, str]]) -> None:
    """Render the keyword-command list as a table."""
    table = Table(title="Keyword commands", title_style="bold", header_style="bold", box=None, padding=(0, 2))
    table.add_column("type", style="cyan", no_wrap=True)
    table.add_column("effect")
    for keys, effect in rows:
        table.add_row(keys, effect)
    console.print(table)
    console.print(
        "Anything else you type becomes the task instruction; Enter alone restores the default. "
        "A keyword command pauses the policy until you type an instruction.\n",
        style="dim",
    )


class BasePromptListener(ABC):
    """Shared state for the two listeners: typed lines in, status out.

    Args:
        default_prompt: Instruction restored when the operator hits Enter on an
            empty line.
    """

    pinned = False

    def __init__(self, default_prompt: str) -> None:
        self._default_prompt = default_prompt
        self._pending: str | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._task = default_prompt
        self._paused = False
        self._rate_hz: float | None = None
        self._rec_state: str | None = None  # None | "recording" | "pending"
        self._rec_steps = 0
        self._saving: str | None = None

    # -- control loop API --------------------------------------------------

    def poll(self) -> str | None:
        """Return a line typed since the last call, or None."""
        with self._lock:
            pending = self._pending
            self._pending = None
        return pending

    def set_task(self, task: str) -> None:
        with self._lock:
            self._task = task

    def set_paused(self, paused: bool) -> None:  # noqa: FBT001 - mirrors the loop's own flag
        with self._lock:
            self._paused = paused

    def set_rate(self, rate_hz: float) -> None:
        with self._lock:
            self._rate_hz = rate_hz

    def set_recording(self, state: str | None, steps: int) -> None:
        """Mirror the EpisodeRecorder state ("recording"/"pending"/idle) on the status line."""
        with self._lock:
            self._rec_state = None if state == "idle" else state
            self._rec_steps = steps

    def set_saving(self, text: str | None) -> None:
        """Show background episode-save progress (called from the saver thread)."""
        with self._lock:
            self._saving = text

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    # -- internals ---------------------------------------------------------

    def _submit(self, line: str) -> None:
        with self._lock:
            self._pending = line or self._default_prompt


class PlainPromptListener(BasePromptListener):
    """Line-buffered fallback: a daemon thread parked on ``readline()``.

    No pinned input line — log records and typing interleave — but it works
    anywhere, including when stdout is redirected to a file.
    """

    def start(self) -> None:
        if sys.stdin is None or not sys.stdin.isatty():
            logger.info("stdin is not a TTY — live instruction updates disabled")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="prompt-listener")
        self._thread.start()
        logger.info("Type an instruction + Enter to switch task (Enter alone = %r)", self._default_prompt)

    def stop(self) -> None:
        # The thread is blocked in readline() and is a daemon, so flag it rather
        # than joining on a keypress that may never come.
        self._running = False

    def _loop(self) -> None:
        while self._running:
            try:
                line = sys.stdin.readline()
            except Exception:
                logger.exception("PromptListener: stdin read failed — live updates stopped")
                return
            if not line:  # EOF
                return
            self._submit(line.strip())


class PinnedPromptListener(BasePromptListener):
    """Rich listener: status + input line pinned below the scrolling log."""

    pinned = True

    def __init__(self, default_prompt: str) -> None:
        super().__init__(default_prompt)
        self._buffer = ""
        self._live: Live | None = None
        self._fd = sys.stdin.fileno()
        self._saved_mode = None

    def start(self) -> None:
        self._running = True
        self._saved_mode = termios.tcgetattr(self._fd)
        # The reader is a daemon thread, and daemon threads are killed without
        # running their finally blocks when the main thread dies. Without this
        # net, a crash would leave the operator with an echo-less terminal.
        atexit.register(self._restore_terminal)
        self._live = Live(
            get_renderable=self._render,
            console=console,
            refresh_per_second=_REFRESH_PER_SECOND,
            transient=True,  # leave a clean terminal behind on exit
        )
        self._live.start()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="prompt-listener")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            # The reader wakes every 100 ms, so this returns promptly — and only
            # once the thread has restored the terminal mode in its finally.
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _restore_terminal(self) -> None:
        """Put the terminal back into line mode. Safe to call more than once."""
        if self._saved_mode is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_mode)
            self._saved_mode = None

    def _loop(self) -> None:
        fd = self._fd
        try:
            tty.setcbreak(fd)  # unbuffered, no echo; leaves ISIG so Ctrl+C works
            while self._running:
                if not select.select([fd], [], [], 0.1)[0]:
                    continue
                # Read the fd directly: sys.stdin.read(1) would pull the whole
                # line into its own buffer, after which select() reports the fd
                # as empty and the rest of the keystrokes are stranded there.
                data = os.read(fd, 1024).decode(errors="ignore")
                if not data:  # EOF
                    return
                for line in self._consume(data):
                    self._submit(line)
                if _CTRL_D in data:
                    return
        except Exception:
            logger.exception("PromptListener: stdin read failed — live updates stopped")
        finally:
            self._restore_terminal()

    def _consume(self, data: str) -> list[str]:
        """Fold keystrokes into the buffer, returning any lines Enter completed."""
        committed: list[str] = []
        index = 0
        with self._lock:
            while index < len(data):
                char = data[index]
                index += 1
                if char == _ESCAPE:
                    # Arrow keys, Home, ... — skip the sequence so it can't land
                    # in the buffer as stray characters.
                    while index < len(data) and not data[index].isalpha() and data[index] != "~":
                        index += 1
                    index += 1
                elif char in _ENTER:
                    committed.append(self._buffer.strip())
                    self._buffer = ""
                elif char in _BACKSPACE:
                    self._buffer = self._buffer[:-1]
                elif char.isprintable():
                    self._buffer += char
        return committed

    def _render(self) -> Group:
        with self._lock:
            buffer, paused, task, rate_hz = self._buffer, self._paused, self._task, self._rate_hz
            rec_state, rec_steps, saving = self._rec_state, self._rec_steps, self._saving

        if paused:
            status = Text.assemble(
                ("⏸ paused", "bold yellow"),
                (" · arm holding position · type an instruction to resume", "dim"),
            )
        else:
            rate = f" · {rate_hz:.0f} Hz" if rate_hz else ""
            status = Text.assemble(
                ("▶ running", "bold green"),
                (rate, "dim"),
                (" · task: ", "dim"),
                (f"{task!r}", "italic"),
            )
        if rec_state == "recording":
            status.append(" · ", style="dim")
            status.append(f"● REC {rec_steps} steps", style="bold red")
        elif rec_state == "pending":
            status.append(" · ", style="dim")
            status.append(f"■ take pending ({rec_steps} steps): pass / fail / reject", style="bold yellow")
        if saving is not None:
            status.append(" · ", style="dim")
            status.append(f"💾 saving {saving}", style="bold magenta")
        # Trailing reversed space renders as a block cursor.
        entry = Text.assemble(("❯ ", "bold cyan"), (buffer, ""), (" ", "reverse"))  # noqa: RUF001 - prompt glyph
        return Group(_COMMAND_HINT, status, entry)


def make_prompt_listener(default_prompt: str) -> BasePromptListener:
    """Return the pinned listener when the terminal supports it, else the plain one."""
    if not (sys.stdin and sys.stdin.isatty() and console.is_terminal and termios is not None):
        return PlainPromptListener(default_prompt)
    return PinnedPromptListener(default_prompt)
