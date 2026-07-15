"""Named config presets persisted as JSON files in a directory."""
from __future__ import annotations

import json
import re
from pathlib import Path

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigStore:
    def __init__(self, presets_dir: str | Path) -> None:
        self._dir = Path(presets_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        if not _SAFE.match(name):
            raise ValueError(f"Invalid preset name {name!r} (use letters, digits, . _ -)")
        return self._dir / f"{name}.json"

    def save(self, name: str, config: dict) -> None:
        self._path(name).write_text(json.dumps(config, indent=2))

    def load(self, name: str) -> dict:
        path = self._path(name)
        if not path.is_file():
            raise KeyError(name)
        return json.loads(path.read_text())

    def list_names(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)
