"""Configuration management for local-llm."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from .constants import (
    CONFIG_DIR, CONFIG_FILE, DEFAULT_PRESET, DEFAULT_SYSTEM_PROMPT,
    GENERATION_PRESETS, PROFILES,
)


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load config from disk, returning defaults if missing."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return _default_config()
    return _default_config()


def save_config(config: dict) -> None:
    """Write config to disk."""
    _ensure_config_dir()
    CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")


def _default_config() -> dict:
    return {
        "profile": None,
        "favorite_models": [],
        "generation": _default_generation_settings(),
    }


def _default_generation_settings() -> dict:
    preset = GENERATION_PRESETS[DEFAULT_PRESET]
    return {
        "preset": DEFAULT_PRESET,
        "temp": preset["temp"],
        "top_p": preset["top_p"],
        "max_tokens": preset["max_tokens"],
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }


def get_generation_settings(config: dict | None = None) -> dict:
    """Return generation settings from config, with defaults for missing keys."""
    if config is None:
        config = load_config()
    defaults = _default_generation_settings()
    saved = config.get("generation", {})
    # Merge: saved values override defaults
    return {**defaults, **saved}


def save_generation_settings(settings: dict) -> None:
    """Persist generation settings to config."""
    config = load_config()
    config["generation"] = settings
    save_config(config)


def get_profile_name(config: dict) -> Optional[str]:
    """Return the active profile name, or None."""
    return config.get("profile")


def get_profile(config: dict) -> Optional[dict]:
    """Return the active profile dict, or None."""
    name = get_profile_name(config)
    if name and name in PROFILES:
        return PROFILES[name]
    return None


def detect_profile() -> Optional[str]:
    """Auto-detect profile by reading macOS sysctl for chip and memory."""
    chip = _detect_chip()
    memory_gb = _detect_memory_gb()
    if chip is None or memory_gb is None:
        return None
    return match_profile(chip, memory_gb)


def match_profile(chip: str, memory_gb: int) -> Optional[str]:
    """Match a chip string and memory amount to a profile name."""
    for name, profile in PROFILES.items():
        pattern = profile["chip_pattern"]
        mem = profile["memory_gb"]
        if pattern.lower() in chip.lower() and memory_gb >= mem:
            return name
    # Fallback: match by memory alone if chip partially matches
    for name, profile in PROFILES.items():
        if memory_gb >= profile["memory_gb"]:
            return name
    return None


def _detect_chip() -> Optional[str]:
    """Detect Apple Silicon chip name via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_memory_gb() -> Optional[int]:
    """Detect total system memory in GB via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            bytes_val = int(result.stdout.strip())
            return bytes_val // (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None
