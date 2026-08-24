from __future__ import annotations

import threading

from terminal_ui import BasePromptListener


class _TestPromptListener(BasePromptListener):
    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False


def test_prompt_listener_preserves_submissions_in_order() -> None:
    submitted: list[str] = []

    def on_submit(line: str) -> bool:
        submitted.append(line)
        return line != "help"

    listener = _TestPromptListener("default task", on_submit=on_submit)
    listener._submit("home")  # noqa: SLF001
    listener._submit("pick up the green cup")  # noqa: SLF001

    assert submitted == ["home", "pick up the green cup"]
    assert listener.has_pending()
    assert listener.has_interrupting_input()
    assert listener.poll() == "home"
    assert listener.has_pending()
    assert listener.poll() == "pick up the green cup"
    assert not listener.has_pending()
    assert not listener.has_interrupting_input()
    assert listener.poll() is None


def test_non_interrupting_line_does_not_stop_policy_motion() -> None:
    listener = _TestPromptListener("default task", on_submit=lambda line: line != "help")
    listener._submit("help")  # noqa: SLF001

    assert listener.has_pending()
    assert not listener.has_interrupting_input()
    assert listener.poll() == "help"


def test_empty_line_queues_default_prompt() -> None:
    callback_lines: list[str] = []

    def on_submit(line: str) -> bool:
        callback_lines.append(line)
        return True

    listener = _TestPromptListener("default task", on_submit=on_submit)
    listener._submit("")  # noqa: SLF001

    assert callback_lines == ["default task"]
    assert listener.poll() == "default task"


def test_interrupt_is_visible_before_submit_callback_finishes() -> None:
    callback_started = threading.Event()
    release_callback = threading.Event()

    def blocking_callback(line: str) -> bool:
        callback_started.set()
        assert release_callback.wait(timeout=2.0)
        return True

    listener = _TestPromptListener("default task", on_submit=blocking_callback)
    submitter = threading.Thread(target=listener._submit, args=("new task",))  # noqa: SLF001
    dispatched = threading.Event()
    dispatch_result: list[bool] = []
    dispatcher: threading.Thread | None = None
    submitter.start()
    try:
        assert callback_started.wait(timeout=2.0)
        assert listener.has_pending()
        assert listener.has_interrupting_input()
        # The line cannot be handled until its invalidation callback completes.
        assert listener.poll() is None
        dispatcher = threading.Thread(
            target=lambda: dispatch_result.append(
                listener.dispatch_if_current(dispatched.set)
            )
        )
        dispatcher.start()
        assert not dispatched.wait(timeout=0.02)
    finally:
        release_callback.set()
        submitter.join(timeout=2.0)
        if dispatcher is not None:
            dispatcher.join(timeout=2.0)

    assert not submitter.is_alive()
    assert dispatcher is not None and not dispatcher.is_alive()
    assert dispatch_result == [False]
    assert not dispatched.is_set()
    assert listener.poll() == "new task"


def test_fifo_is_preserved_with_rtc_submit_callback() -> None:
    listener = _TestPromptListener("default task", on_submit=lambda line: True)
    listener._submit("home")  # noqa: SLF001
    listener._submit("new task")  # noqa: SLF001

    assert listener.poll() == "home"
    assert listener.poll() == "new task"


def test_default_non_rtc_listener_keeps_original_latest_wins_behavior() -> None:
    listener = _TestPromptListener("default task")
    listener._submit("first")  # noqa: SLF001
    listener._submit("second")  # noqa: SLF001

    assert listener.poll() == "second"
    assert listener.poll() is None


def test_rtc_dispatch_is_rejected_after_invalidating_input_arrives() -> None:
    listener = _TestPromptListener("default task", on_submit=lambda line: True)
    listener._submit("new task")  # noqa: SLF001
    dispatched = False

    def dispatch() -> None:
        nonlocal dispatched
        dispatched = True

    assert listener.dispatch_if_current(dispatch) is False
    assert dispatched is False
