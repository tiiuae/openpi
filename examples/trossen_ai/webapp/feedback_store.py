"""Persist user feedback as dated markdown files."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path


def save_feedback(feedback_dir: str | Path, *, name: str, email: str, feedback: str) -> str:
    """Write one feedback submission to feedback_dir/YYYY-MM-DD-HHMMSS[-N].md.

    Returns the path written. Creates the directory if needed.
    """
    d = Path(feedback_dir)
    d.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now()
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    path = d / f"{stamp}.md"
    suffix = 1
    while path.exists():
        path = d / f"{stamp}-{suffix}.md"
        suffix += 1
    body = (
        f"# Feedback — {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"- **Name:** {name}\n"
        f"- **Email:** {email}\n\n"
        f"{feedback}\n"
    )
    path.write_text(body)
    return str(path)
