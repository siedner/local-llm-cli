"""macOS memory pressure helpers."""

from __future__ import annotations

import re
import subprocess
import time

_CACHE_TTL_SECONDS = 2.0
_cached_snapshot: dict | None = None
_cached_at = 0.0


def get_memory_pressure() -> dict:
    """Return a coarse memory-pressure snapshot for macOS."""
    global _cached_at, _cached_snapshot

    now = time.time()
    if _cached_snapshot is not None and (now - _cached_at) < _CACHE_TTL_SECONDS:
        return _cached_snapshot

    snapshot = {
        "state": "unknown",
        "free_percent": None,
        "raw": "",
    }
    try:
        result = subprocess.run(
            ["memory_pressure", "-Q"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        raw = result.stdout.strip()
        snapshot["raw"] = raw
        match = re.search(r"System-wide memory free percentage:\s+(\d+)%", raw)
        if match:
            free_percent = int(match.group(1))
            snapshot["free_percent"] = free_percent
            if free_percent <= 8:
                snapshot["state"] = "red"
            elif free_percent <= 18:
                snapshot["state"] = "yellow"
            else:
                snapshot["state"] = "green"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    _cached_snapshot = snapshot
    _cached_at = now
    return snapshot
