"""Configuration management for local-llm."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .constants import (
    CONFIG_DIR,
    CONFIG_FILE,
    CONFIG_SCHEMA_VERSION,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PRESET,
    DEFAULT_QUEUE_LIMIT,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SYSTEM_PROMPT,
    GENERATION_PRESETS,
    OUTPUT_SIZE_PROFILES,
    PROFILES,
)


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load config from disk, returning defaults if missing."""
    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                _preserve_invalid_config()
                return _default_config()
            return _normalize_config(raw)
        except (json.JSONDecodeError, OSError):
            _preserve_invalid_config()
            return _default_config()
    return _default_config()


def save_config(config: dict) -> None:
    """Write config to disk."""
    _ensure_config_dir()
    normalized = _normalize_config(config)
    CONFIG_FILE.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def _default_config() -> dict:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "profile": None,
        "favorite_models": [],
        "generation": _default_generation_settings(),
        "runtime": _default_runtime_settings(),
        "session_defaults": _default_session_defaults(),
        "tui": _default_tui_settings(),
        "calibration": {},
        "benchmarks": [],
        "models": {},
    }


def _default_runtime_settings() -> dict:
    return {
        "backend": "mlx",
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "queue_limit": DEFAULT_QUEUE_LIMIT,
        "request_timeout_seconds": DEFAULT_REQUEST_TIMEOUT_SECONDS,
        "keep_alive_seconds": None,
        "safe_mode": True,
    }


def _default_session_defaults() -> dict:
    return {
        "keep_alive_seconds": None,
        "max_context": None,
        "max_output": None,
    }


def _default_generation_settings() -> dict:
    preset = GENERATION_PRESETS[DEFAULT_PRESET]
    return {
        "preset": DEFAULT_PRESET,
        "temp": preset["temp"],
        "top_p": preset["top_p"],
        "top_k": preset["top_k"],
        "max_tokens": preset["max_tokens"],
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    }


def _default_tui_settings() -> dict:
    return {
        "show_statusline": True,
        "statusline_fields": [
            "state",
            "model",
            "profile",
            "session",
            "safe",
            "memory",
            "queue",
            "warm",
        ],
        "multiline_hint_mode": "compact",
        "vim_mode": False,
        "history_size": 100,
        "show_examples_in_help": True,
    }


def _normalize_config(config: dict) -> dict:
    """Merge loaded config with defaults and preserve unknown keys."""
    defaults = _default_config()

    favorite_models = config.get("favorite_models", defaults["favorite_models"])
    if not isinstance(favorite_models, list):
        favorite_models = defaults["favorite_models"]

    generation = config.get("generation", {})
    if not isinstance(generation, dict):
        generation = {}

    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}

    session_defaults = config.get("session_defaults", {})
    if not isinstance(session_defaults, dict):
        session_defaults = {}

    tui = config.get("tui", {})
    if not isinstance(tui, dict):
        tui = {}

    calibration = config.get("calibration", defaults["calibration"])
    if not isinstance(calibration, dict):
        calibration = defaults["calibration"]

    benchmarks = config.get("benchmarks", defaults["benchmarks"])
    if not isinstance(benchmarks, list):
        benchmarks = defaults["benchmarks"]

    models = config.get("models", defaults["models"])
    if not isinstance(models, dict):
        models = defaults["models"]

    normalized = {**defaults, **config}
    normalized["schema_version"] = CONFIG_SCHEMA_VERSION
    normalized["favorite_models"] = favorite_models
    normalized["generation"] = {**defaults["generation"], **generation}
    normalized["runtime"] = {**defaults["runtime"], **runtime}
    normalized["session_defaults"] = {**defaults["session_defaults"], **session_defaults}
    normalized["tui"] = {**defaults["tui"], **tui}
    normalized["calibration"] = calibration
    normalized["benchmarks"] = benchmarks
    normalized["models"] = models
    return normalized


def _preserve_invalid_config() -> None:
    """Move a broken config aside so the next save does not overwrite it."""
    if not CONFIG_FILE.exists():
        return

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup = CONFIG_FILE.with_name(
        f"{CONFIG_FILE.stem}.invalid-{timestamp}{CONFIG_FILE.suffix}"
    )
    try:
        shutil.move(str(CONFIG_FILE), str(backup))
    except OSError:
        pass


def get_generation_settings(config: dict | None = None) -> dict:
    """Return generation settings from config, with defaults for missing keys."""
    if config is None:
        config = load_config()
    return {**_default_generation_settings(), **config.get("generation", {})}


def save_generation_settings(settings: dict) -> None:
    """Persist generation settings to config."""
    config = load_config()
    config["generation"] = {
        **_default_generation_settings(),
        **config.get("generation", {}),
        **settings,
    }
    save_config(config)


def get_runtime_settings(config: dict | None = None) -> dict:
    """Return normalized runtime settings."""
    if config is None:
        config = load_config()
    return {**_default_runtime_settings(), **config.get("runtime", {})}


def save_runtime_settings(settings: dict) -> None:
    """Persist runtime settings to config."""
    config = load_config()
    config["runtime"] = {
        **_default_runtime_settings(),
        **config.get("runtime", {}),
        **settings,
    }
    save_config(config)


def get_session_defaults(config: dict | None = None) -> dict:
    """Return normalized session defaults."""
    if config is None:
        config = load_config()
    return {**_default_session_defaults(), **config.get("session_defaults", {})}


def normalize_output_size_profile(name: str | None) -> str | None:
    """Normalize a user-provided output size profile label."""
    if not name:
        return None
    normalized = str(name).strip().upper()
    return normalized if normalized in OUTPUT_SIZE_PROFILES else None


def get_output_size_profile(name: str | None) -> dict | None:
    """Return the output size profile payload for a label."""
    normalized = normalize_output_size_profile(name)
    if normalized is None:
        return None
    return {"name": normalized, **OUTPUT_SIZE_PROFILES[normalized]}


def detect_output_size_profile(max_tokens: int, request_timeout_seconds: int) -> str | None:
    """Return the matching output size profile label, if any."""
    for name, profile in OUTPUT_SIZE_PROFILES.items():
        if (
            int(profile["max_tokens"]) == int(max_tokens)
            and int(profile["request_timeout_seconds"]) == int(request_timeout_seconds)
        ):
            return name
    return None


def record_benchmark(result: dict) -> None:
    """Append a benchmark record to config."""
    config = load_config()
    benchmarks = list(config.get("benchmarks", []))
    benchmarks.append(result)
    config["benchmarks"] = benchmarks[-50:]
    save_config(config)


def get_tui_settings(config: dict | None = None) -> dict:
    """Return normalized TUI settings."""
    if config is None:
        config = load_config()
    return {**_default_tui_settings(), **config.get("tui", {})}


def save_tui_settings(settings: dict) -> None:
    """Persist TUI settings to config."""
    config = load_config()
    config["tui"] = {
        **_default_tui_settings(),
        **settings,
    }
    save_config(config)


def save_calibration(profile_name: str, calibration: dict) -> None:
    """Persist profile calibration metadata."""
    config = load_config()
    saved = dict(config.get("calibration", {}))
    saved[profile_name] = calibration
    config["calibration"] = saved
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


def get_effective_profile(config: dict | None = None, name: str | None = None) -> tuple[str, dict]:
    """Return the active profile name and merged settings."""
    if config is None:
        config = load_config()

    profile_name = name or config.get("profile") or detect_profile() or "m1pro32"
    if profile_name not in PROFILES:
        profile_name = "m1pro32"

    profile = dict(PROFILES[profile_name])
    calibration = config.get("calibration", {}).get(profile_name, {})
    calibrated_runtime = calibration.get("runtime", {}) if isinstance(calibration, dict) else {}
    session_defaults = config.get("session_defaults", {})

    # Resolution order is built-in defaults -> calibration -> explicit user/session overrides.
    profile.update({k: v for k, v in calibrated_runtime.items() if k in profile})

    if session_defaults.get("max_context") is not None:
        profile["default_context"] = min(
            profile["hard_context"],
            int(session_defaults["max_context"]),
        )
    if session_defaults.get("max_output") is not None:
        profile["max_tokens"] = min(
            int(session_defaults["max_output"]),
            profile["max_tokens"],
        )
    if session_defaults.get("keep_alive_seconds") is not None:
        profile["keep_alive_seconds"] = int(session_defaults["keep_alive_seconds"])

    profile["default_context"] = min(int(profile["default_context"]), int(profile["hard_context"]))
    profile["max_tokens"] = min(int(profile["max_tokens"]), int(PROFILES[profile_name]["max_tokens"]))
    profile["keep_alive_seconds"] = max(60, int(profile["keep_alive_seconds"]))
    return profile_name, profile


def detect_profile() -> Optional[str]:
    """Auto-detect profile by reading macOS sysctl for chip and memory."""
    chip = _detect_chip()
    memory_gb = _detect_memory_gb()
    if chip is None or memory_gb is None:
        return None
    return match_profile(chip, memory_gb)


def match_profile(chip: str, memory_gb: int) -> Optional[str]:
    """Match a chip string and memory amount to a profile name."""
    ordered = sorted(
        PROFILES.items(),
        key=lambda item: (item[1]["memory_gb"], item[0]),
        reverse=True,
    )
    chip_lower = chip.lower()
    for name, profile in ordered:
        if profile["chip_pattern"].lower() in chip_lower and memory_gb >= profile["memory_gb"]:
            return name
    for name, profile in ordered:
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
