"""Tests for config management."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from local_llm.config import (
    _default_config,
    detect_output_size_profile,
    detect_profile,
    get_effective_profile,
    get_output_size_profile,
    get_runtime_settings,
    load_config,
    match_profile,
    normalize_output_size_profile,
    save_generation_settings,
    save_runtime_settings,
    get_tui_settings,
    save_config,
    save_tui_settings,
)
from local_llm.constants import CONFIG_SCHEMA_VERSION


def test_default_config():
    config = _default_config()
    assert config["schema_version"] == CONFIG_SCHEMA_VERSION
    assert config["profile"] is None
    assert config["favorite_models"] == []


def test_save_and_load_config(tmp_path):
    config_file = tmp_path / "config.json"
    with patch("local_llm.config.CONFIG_FILE", config_file), \
         patch("local_llm.config.CONFIG_DIR", tmp_path):
        config = {"profile": "m1pro32", "favorite_models": ["test/model"]}
        save_config(config)

        assert config_file.exists()
        loaded = load_config()
        assert loaded["profile"] == "m1pro32"
        assert loaded["favorite_models"] == ["test/model"]


def test_load_config_missing(tmp_path):
    config_file = tmp_path / "nonexistent" / "config.json"
    with patch("local_llm.config.CONFIG_FILE", config_file):
        config = load_config()
        assert config == _default_config()


def test_load_config_invalid_json(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text("not json")
    with patch("local_llm.config.CONFIG_FILE", config_file), \
         patch("local_llm.config.CONFIG_DIR", tmp_path):
        config = load_config()
        assert config == _default_config()
        backups = list(tmp_path.glob("config.invalid-*.json"))
        assert len(backups) == 1
        assert not config_file.exists()


def test_load_config_normalizes_generation(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"generation": {"temp": 0.2}}))
    with patch("local_llm.config.CONFIG_FILE", config_file), \
         patch("local_llm.config.CONFIG_DIR", tmp_path):
        config = load_config()
        assert config["generation"]["temp"] == 0.2
        assert "top_p" in config["generation"]
        assert "max_tokens" in config["generation"]


def test_load_config_normalizes_tui_settings(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"tui": {"history_size": 42}}))
    with patch("local_llm.config.CONFIG_FILE", config_file), \
         patch("local_llm.config.CONFIG_DIR", tmp_path):
        config = load_config()
        assert config["tui"]["history_size"] == 42
        assert config["tui"]["show_statusline"] is True
        assert "statusline_fields" in config["tui"]


def test_save_tui_settings(tmp_path):
    config_file = tmp_path / "config.json"
    with patch("local_llm.config.CONFIG_FILE", config_file), \
         patch("local_llm.config.CONFIG_DIR", tmp_path):
        save_tui_settings({"history_size": 12, "show_statusline": False})
        settings = get_tui_settings()
        assert settings["history_size"] == 12
        assert settings["show_statusline"] is False


def test_save_generation_settings_preserves_unspecified_values(tmp_path):
    config_file = tmp_path / "config.json"
    with patch("local_llm.config.CONFIG_FILE", config_file), \
         patch("local_llm.config.CONFIG_DIR", tmp_path):
        save_config({"generation": {"system_prompt": "Keep this", "temp": 0.2}})
        save_generation_settings({"max_tokens": 4096})
        config = load_config()
        assert config["generation"]["max_tokens"] == 4096
        assert config["generation"]["system_prompt"] == "Keep this"
        assert config["generation"]["temp"] == 0.2


def test_save_runtime_settings_preserves_unspecified_values(tmp_path):
    config_file = tmp_path / "config.json"
    with patch("local_llm.config.CONFIG_FILE", config_file), \
         patch("local_llm.config.CONFIG_DIR", tmp_path):
        save_config({"runtime": {"host": "0.0.0.0", "safe_mode": False}})
        save_runtime_settings({"request_timeout_seconds": 900})
        runtime = get_runtime_settings()
        assert runtime["request_timeout_seconds"] == 900
        assert runtime["host"] == "0.0.0.0"
        assert runtime["safe_mode"] is False


def test_effective_profile_prefers_user_session_defaults_over_calibration():
    config = _default_config()
    config["profile"] = "m1pro32"
    config["calibration"] = {
        "m1pro32": {
            "runtime": {
                "default_context": 12000,
                "keep_alive_seconds": 900,
            }
        }
    }
    config["session_defaults"] = {
        "max_context": 4096,
        "keep_alive_seconds": 300,
    }

    name, profile = get_effective_profile(config)
    assert name == "m1pro32"
    assert profile["default_context"] == 4096
    assert profile["keep_alive_seconds"] == 300


def test_output_size_profile_helpers():
    assert normalize_output_size_profile("xl") == "XL"
    profile = get_output_size_profile("XL")
    assert profile is not None
    assert profile["max_tokens"] == 3072
    assert profile["request_timeout_seconds"] == 900
    assert detect_output_size_profile(3072, 900) == "XL"
    assert detect_output_size_profile(1234, 900) is None


def test_match_profile_m1pro32():
    assert match_profile("Apple M1 Pro", 32) == "m1pro32"


def test_match_profile_m4mini16():
    assert match_profile("Apple M4", 16) == "m4mini16"


def test_match_profile_unknown_chip():
    # Falls back to memory-based match
    result = match_profile("Apple M3 Ultra", 64)
    assert result is not None  # Should match something


def test_match_profile_no_match():
    # Very low memory, won't match anything
    result = match_profile("Unknown Chip", 4)
    assert result is None
