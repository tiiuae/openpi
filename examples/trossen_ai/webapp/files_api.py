"""Directory listing for the folder browser (ported from the dataset wizard).

Traverses outside the repo via expanduser().resolve(); returns directories only,
flagging LeRobot datasets (meta/info.json present).
"""
from __future__ import annotations

from pathlib import Path


def list_directory(path_str: str) -> dict:
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        return {"error": f"not a directory: {path_str}"}
    entries = []
    try:
        for child in sorted(p.iterdir()):
            if child.is_dir():
                entries.append({
                    "name": child.name,
                    "type": "dir",
                    "is_dataset": (child / "meta" / "info.json").exists(),
                    "path": str(child),
                })
    except PermissionError:
        pass
    parent = str(p.parent) if p.parent != p else None
    return {"path": str(p), "parent": parent, "entries": entries}
