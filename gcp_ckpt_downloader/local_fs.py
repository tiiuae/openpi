"""Local directory listing for the destination picker.

`list_directory` is adapted near-verbatim from the LeRobot dataset wizard's
`examples/trossen_ai/webapp/files_api.py`, minus the `is_dataset` flag (this
app has no notion of LeRobot datasets).
"""

from __future__ import annotations

import contextlib
from pathlib import Path


def list_directory(path_str: str) -> dict:
    """List the immediate subdirectories of `path_str`.

    Returns {"path", "parent", "entries": [{"name", "type": "dir", "path"}]}
    on success, or {"error": ...} if `path_str` doesn't exist or isn't a
    directory. `path_str` is expanded ("~") and resolved to an absolute path.
    """
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return {"error": f"not a directory: {path_str}"}
    entries = []
    with contextlib.suppress(PermissionError):
        entries.extend(
            {"name": child.name, "type": "dir", "path": str(child)} for child in sorted(p.iterdir()) if child.is_dir()
        )
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "entries": entries}


def existing_names(dest: str, names: list[str]) -> list[str]:
    """Return the subset of `names` that already exist as children of `dest`.

    Used by the UI to badge GCS folders that would overwrite something
    already present at the chosen download destination.
    """
    base = Path(dest).expanduser()
    return [name for name in names if (base / name).exists()]
