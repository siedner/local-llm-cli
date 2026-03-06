"""Tests for HF cache scanning."""

from pathlib import Path

import pytest

from local_llm.hf_cache import find_model_path, list_installed_models


def test_list_installed_models_empty(tmp_path):
    models = list_installed_models(cache_dir=tmp_path)
    assert models == []


def test_list_installed_models(tmp_path):
    # Create fake model dirs
    (tmp_path / "models--org1--model-a").mkdir()
    (tmp_path / "models--org2--model-b").mkdir()
    (tmp_path / "version.txt").touch()  # should be ignored
    (tmp_path / "random-dir").mkdir()  # should be ignored

    models = list_installed_models(cache_dir=tmp_path)
    assert len(models) == 2
    repos = [m["repo"] for m in models]
    assert "org1/model-a" in repos
    assert "org2/model-b" in repos


def test_list_installed_models_structure(tmp_path):
    (tmp_path / "models--MyOrg--MyModel").mkdir()
    models = list_installed_models(cache_dir=tmp_path)
    assert len(models) == 1
    m = models[0]
    assert m["repo"] == "MyOrg/MyModel"
    assert m["org"] == "MyOrg"
    assert m["name"] == "MyModel"
    assert m["path"] == tmp_path / "models--MyOrg--MyModel"


def test_find_model_path_exists(tmp_path):
    (tmp_path / "models--org--name").mkdir()
    result = find_model_path("org/name", cache_dir=tmp_path)
    assert result == tmp_path / "models--org--name"


def test_find_model_path_missing(tmp_path):
    result = find_model_path("org/name", cache_dir=tmp_path)
    assert result is None


def test_find_model_path_invalid_repo(tmp_path):
    result = find_model_path("noslash", cache_dir=tmp_path)
    assert result is None


def test_list_installed_models_nonexistent_dir():
    models = list_installed_models(cache_dir=Path("/nonexistent/path"))
    assert models == []
