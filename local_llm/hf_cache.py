"""Hugging Face cache scanning utilities."""

from __future__ import annotations

import os
import subprocess
import json
from pathlib import Path
from typing import Optional


_TEXT_LLM_ARCH_HINTS = (
    "CausalLM",
    "ForConditionalGeneration",
    "ForQuestionAnswering",
)
_EXCLUDED_ARCH_HINTS = (
    "Whisper",
    "Diffusion",
    "Flux",
    "Codec",
    "Vocoder",
    "Speech",
    "Audio",
    "Vision",
)
_EXCLUDED_MODEL_TYPES = {
    "whisper",
    "flux",
    "audioldm2",
    "musicgen",
    "encodec",
    "dac",
}


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
                    is_relevant = _entry_is_relevant_llm(entry)
                                    
                if is_relevant:
                    models.append({
                        "repo": f"{org}/{name}",
                        "org": org,
                        "name": name,
                        "path": entry,
                    })
    return models


def _entry_is_relevant_llm(entry: Path) -> bool:
    """Return True when a cache entry looks like a text-generation model."""
    snapshots_dir = entry / "snapshots"
    if not snapshots_dir.exists():
        return False

    for snapshot in snapshots_dir.iterdir():
        config_path = snapshot / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if _config_is_relevant_llm(config, snapshot):
            return True
    return False


def _config_is_relevant_llm(config: dict, snapshot_dir: Path) -> bool:
    """Classify a config as a relevant text/chat LLM."""
    architectures = config.get("architectures") or []
    model_type = str(config.get("model_type") or "").lower()

    if model_type in _EXCLUDED_MODEL_TYPES:
        return False

    if any(any(excluded in arch for excluded in _EXCLUDED_ARCH_HINTS) for arch in architectures):
        return False

    has_text_config = isinstance(config.get("text_config"), dict)
    has_audio_config = isinstance(config.get("audio_config"), dict)
    has_vision_config = isinstance(config.get("vision_config"), dict)

    if has_audio_config and not has_text_config:
        return False
    if has_vision_config and not has_text_config:
        return False

    # local-llm currently supports text-generation models only.
    # Multimodal repos with a vision tower routinely fail under mlx_lm.
    if has_vision_config:
        return False

    tokenizer_present = any(
        (snapshot_dir / name).exists()
        for name in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json")
    )

    if has_text_config and tokenizer_present:
        return True

    if any(any(hint in arch for hint in _TEXT_LLM_ARCH_HINTS) for arch in architectures):
        return tokenizer_present

    return False


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
