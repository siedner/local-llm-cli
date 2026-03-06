"""Constants and paths for local-llm."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = Path(os.environ.get("LOCAL_LLM_CONFIG_DIR", Path.home() / ".config" / "local-llm"))
CONFIG_FILE = CONFIG_DIR / "config.json"

DATA_DIR = Path(os.environ.get("LOCAL_LLM_DATA_DIR", Path.home() / ".local" / "share" / "local-llm"))
PIDS_DIR = DATA_DIR / "pids"
LOGS_DIR = DATA_DIR / "logs"
STATE_DIR = DATA_DIR / "state"
VENV_DIR = DATA_DIR / "venv"
HISTORY_DIR = DATA_DIR / "history"

DEFAULT_PORT = 8080
DEFAULT_HOST = "127.0.0.1"
DEFAULT_QUEUE_LIMIT = 0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300

DAEMON_PID_FILE = PIDS_DIR / "daemon.pid"
DAEMON_LOG_FILE = LOGS_DIR / "daemon.log"
DAEMON_JSONL_LOG_FILE = LOGS_DIR / "daemon.jsonl"
DAEMON_STATE_FILE = STATE_DIR / "daemon-state.json"

LAUNCHD_LABEL = "dev.local-llm.daemon"
LAUNCHD_PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
PROJECT_COMMANDS_DIR = Path(".local-llm") / "commands"
USER_COMMANDS_DIR = CONFIG_DIR / "commands"

CONFIG_SCHEMA_VERSION = 2

PROFILES = {
    "m1pro32": {
        "chip_pattern": "Apple M1 Pro",
        "memory_gb": 32,
        "default_context": 8192,
        "hard_context": 16384,
        "max_tokens": 1024,
        "decode_concurrency": 1,
        "prompt_concurrency": 1,
        "keep_alive_seconds": 20 * 60,
        "queue_limit": 0,
        "request_timeout_seconds": 300,
        "recommended_models": [
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4",
        ],
        "risk_hints": [
            "M1 Pro 32GB: keep one warm model and prefer <=8B quantized models.",
            "Large repo prompts should be preflighted before generation.",
        ],
    },
    "m432": {
        "chip_pattern": "Apple M4",
        "memory_gb": 32,
        "default_context": 8192,
        "hard_context": 16384,
        "max_tokens": 1024,
        "decode_concurrency": 1,
        "prompt_concurrency": 1,
        "keep_alive_seconds": 20 * 60,
        "queue_limit": 0,
        "request_timeout_seconds": 300,
        "recommended_models": [
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4",
        ],
        "risk_hints": [
            "M4 32GB: one warm 4B/8B model is the default safe operating mode.",
            "Prefer calibrated settings over aggressive concurrency.",
        ],
    },
    "m4mini16": {
        "chip_pattern": "Apple M4",
        "memory_gb": 16,
        "default_context": 4096,
        "hard_context": 8192,
        "max_tokens": 512,
        "decode_concurrency": 1,
        "prompt_concurrency": 1,
        "keep_alive_seconds": 10 * 60,
        "queue_limit": 0,
        "request_timeout_seconds": 180,
        "recommended_models": [
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
            "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4",
        ],
        "risk_hints": [
            "M4 16GB: keep prompts compact and prefer <=4B quantized models.",
            "Reject oversize prompts instead of pushing unified memory into yellow/red.",
        ],
    },
}

RECOMMENDED_MODELS = [
    {
        "repo": "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
        "description": "Fast default for Apple Silicon, MXFP4 quantization",
        "estimated_size": "2.1G",
    },
    {
        "repo": "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4",
        "description": "Alternative Qwen3.5 4B quantization for Apple Silicon",
        "estimated_size": "2.2G",
    },
    {
        "repo": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
        "description": "Recommended 9B upgrade for 32GB Macs using standard mlx-lm",
        "estimated_size": "5.6G",
    },
    {
        "repo": "NexVeridian/Qwen3.5-9B-4bit",
        "description": "Conservative mlx-lm 9B fallback if the OptiQ build gives trouble",
        "estimated_size": "5.6G",
    },
    {
        "repo": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "description": "Strong local coding model for 32GB Apple Silicon machines",
        "estimated_size": "16G",
    },
]

MODEL_METADATA = {
    "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4": {
        "family": "Qwen3.5 4B",
        "quantization": "MXFP4",
        "summary": "Best default for Apple Silicon chat. Prioritizes speed and low memory use.",
        "when_to_use": "Use first on 16GB/32GB Macs when you want the safest fast chat profile.",
        "tradeoffs": "Lowest memory pressure of the bundled options; slightly more aggressive quantization than NVFP4.",
    },
    "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4": {
        "family": "Qwen3.5 4B",
        "quantization": "NVFP4",
        "summary": "Alternative 4-bit variant that may preserve slightly more quality on some prompts.",
        "when_to_use": "Try when MXFP4 feels too lossy or you want to compare quality vs speed on the same 4B base model.",
        "tradeoffs": "Usually a touch heavier than MXFP4; naming is similar, so choose it intentionally for A/B testing.",
    },
    "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit": {
        "family": "Qwen3 Coder 30B-A3B",
        "quantization": "4-bit",
        "summary": "Large coding model. Significantly heavier than the 4B chat models.",
        "when_to_use": "Use on 32GB machines for coding-heavy tasks when latency is less important than stronger reasoning.",
        "tradeoffs": "Much slower to warm and decode; not the default recommendation for everyday chat.",
    },
    "mlx-community/Qwen3.5-9B-OptiQ-4bit": {
        "family": "Qwen3.5 9B",
        "quantization": "4-bit",
        "summary": "Best current 9B upgrade for this app on Apple Silicon using standard mlx-lm.",
        "when_to_use": "Install first on 32GB Macs when you want a clear quality step up from 4B without jumping to a huge model.",
        "tradeoffs": "Heavier than 4B; still may need prompt/template tuning to avoid verbose reasoning.",
    },
    "NexVeridian/Qwen3.5-9B-4bit": {
        "family": "Qwen3.5 9B",
        "quantization": "4-bit",
        "summary": "Fallback 9B mlx-lm conversion when you want a simpler non-OptiQ option.",
        "when_to_use": "Use if the OptiQ build causes issues or you want a more conservative baseline for A/B testing.",
        "tradeoffs": "Similar memory footprint to OptiQ; usually not the first install unless you are troubleshooting.",
    },
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

GENERATION_PRESETS = {
    "precise": {
        "description": "Factual, deterministic responses",
        "temp": 0.2,
        "top_p": 0.85,
        "top_k": 40,
        "max_tokens": 512,
    },
    "balanced": {
        "description": "Fast, general purpose chat",
        "temp": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_tokens": 1024,
    },
    "creative": {
        "description": "More varied, expressive output",
        "temp": 1.0,
        "top_p": 0.95,
        "top_k": 80,
        "max_tokens": 1024,
    },
    "coding": {
        "description": "Code generation with tighter sampling and longer output",
        "temp": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 2048,
    },
}

DEFAULT_PRESET = "balanced"
