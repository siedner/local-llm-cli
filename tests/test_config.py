"""Tests for config management."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from local_llm.config import (
    _default_config,
    detect_profile,
    load_config,
    match_profile,
    save_config,
)


def test_default_config():
    config = _default_config()
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
    with patch("local_llm.config.CONFIG_FILE", config_file):
        config = load_config()
        assert config == _default_config()


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
