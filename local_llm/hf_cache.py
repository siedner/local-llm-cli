"""Hugging Face cache scanning utilities."""

from __future__ import annotations

import os
import subprocess
import json
from pathlib import Path
from typing import Optional


def get_hf_cache_dir() -> Path:
    """Return the HF hub cache directory."""
    if "HF_HUB_CACHE" in os.environ:
        return Path(os.environ["HF_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def list_installed_models(cache_dir: Optional[Path] = None, filter_relevant: bool = True) -> list[dict]:
    """Scan HF cache for installed models.

    Returns list of dicts with keys: repo, org, name, path.
    If filter_relevant is True, attempts to filter for text generation models
    by checking config.json architectures.
    """
    cache = cache_dir or get_hf_cache_dir()
    if not cache.exists():
        return []

    models = []
    for entry in sorted(cache.iterdir()):
        if entry.is_dir() and entry.name.startswith("models--"):
            parts = entry.name.split("--", 2)
            if len(parts) == 3:
                org, name = parts[1], parts[2]
                
                is_relevant = True
                if filter_relevant:
                    # Look for config.json in snapshots
                    snapshots_dir = entry / "snapshots"
                    if snapshots_dir.exists():
                        is_relevant = False
                        for snapshot in snapshots_dir.iterdir():
                            config_path = snapshot / "config.json"
                            if config_path.exists():
                                try:
                                    with open(config_path, "r") as f:
                                        config = json.load(f)
                                        archs = config.get("architectures", [])
                                        if any("CausalLM" in a or ("ConditionalGeneration" in a and "Whisper" not in a) for a in archs):
                                            is_relevant = True
                                            break
                                except Exception:
                                    pass
                                    
                if is_relevant:
                    models.append({
                        "repo": f"{org}/{name}",
                        "org": org,
                        "name": name,
                        "path": entry,
                    })
    return models


def get_model_disk_usage(model_path: Path) -> str:
    """Get disk usage of a model directory using du."""
    try:
        result = subprocess.run(
            ["du", "-sh", str(model_path)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\t")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def find_model_path(repo: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the cache path for a given HF repo (org/name)."""
    cache = cache_dir or get_hf_cache_dir()
    parts = repo.split("/", 1)
    if len(parts) != 2:
        return None
    org, name = parts
    candidate = cache / f"models--{org}--{name}"
    if candidate.exists():
        return candidate
    return None
