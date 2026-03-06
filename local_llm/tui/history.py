"""Persistent composer history for the TUI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from local_llm.constants import HISTORY_DIR


class HistoryStore:
    """Persist input history scoped by working directory."""

    def __init__(self, cwd: Path, limit: int = 100, root: Path | None = None) -> None:
        self.cwd = cwd.resolve()
        self.limit = max(limit, 1)
        self.root = root or HISTORY_DIR
        self.root.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(self.cwd).encode()).hexdigest()[:16]
        self.path = self.root / f"{digest}.json"
        self.entries = self._load()
        self.index = len(self.entries)
        self.pending = ""

    def push(self, value: str) -> None:
        """Append a history entry and persist it."""
        value = value.rstrip()
        if not value:
            return
        if not self.entries or self.entries[-1] != value:
            self.entries.append(value)
        self.entries = self.entries[-self.limit :]
        self.index = len(self.entries)
        self.pending = ""
        self._save()

    def older(self, current_value: str) -> str | None:
        """Return the previous history entry."""
        if not self.entries:
            return None
        if self.index == len(self.entries):
            self.pending = current_value
        if self.index > 0:
            self.index -= 1
        return self.entries[self.index]

    def newer(self) -> str | None:
        """Return the next history entry, if any."""
        if not self.entries:
            return None
        if self.index < len(self.entries) - 1:
            self.index += 1
            return self.entries[self.index]
        self.index = len(self.entries)
        return self.pending

    def _load(self) -> list[str]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw if isinstance(item, str)]

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.entries, indent=2) + "\n", encoding="utf-8")
