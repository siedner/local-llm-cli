"""Constants and paths for local-llm."""

from pathlib import Path
import os

CONFIG_DIR = Path(os.environ.get("LOCAL_LLM_CONFIG_DIR", Path.home() / ".config" / "local-llm"))
CONFIG_FILE = CONFIG_DIR / "config.json"

DATA_DIR = Path(os.environ.get("LOCAL_LLM_DATA_DIR", Path.home() / ".local" / "share" / "local-llm"))
PIDS_DIR = DATA_DIR / "pids"
LOGS_DIR = DATA_DIR / "logs"
VENV_DIR = DATA_DIR / "venv"

DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"

PROFILES = {
    "m1pro32": {
        "chip_pattern": "Apple M1 Pro",
        "memory_gb": 32,
        "max_tokens": 2048,
        "decode_concurrency": 1,
        "prompt_concurrency": 1,
        "recommended_models": [
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4",
        ],
        "risk_hints": [
            "M1 Pro 32GB: models up to ~8B params are safe",
            "Large OpenCode contexts (>10k tokens) may cause Metal OOM with bigger models",
        ],
    },
    "m4mini16": {
        "chip_pattern": "Apple M4",
        "memory_gb": 16,
        "max_tokens": 1024,
        "decode_concurrency": 1,
        "prompt_concurrency": 1,
        "recommended_models": [
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4",
        ],
        "risk_hints": [
            "M4 16GB: prefer <= 4B models; avoid big contexts",
            "OpenCode sends large prefills — start with small repos to validate",
        ],
    },
}

RECOMMENDED_MODELS = [
    {
        "repo": "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
        "description": "Qwen3.5 4B, MXFP4 quantization, text-only, MLX-converted",
        "estimated_size": "2.1G",
    },
    {
        "repo": "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4",
        "description": "Qwen3.5 4B, NVFP4 quantization, text-only, MLX-converted",
        "estimated_size": "2.2G",
    },
]

# ---------------------------------------------------------------------------
# Generation presets
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

GENERATION_PRESETS = {
    "precise": {
        "description": "Factual, deterministic responses",
        "temp": 0.3,
        "top_p": 0.85,
        "max_tokens": 2048,
    },
    "balanced": {
        "description": "Good all-around default",
        "temp": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
    },
    "creative": {
        "description": "More varied, expressive output",
        "temp": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
    },
    "coding": {
        "description": "Code generation — low temp, high token limit",
        "temp": 0.2,
        "top_p": 0.9,
        "max_tokens": 4096,
    },
}

DEFAULT_PRESET = "balanced"
