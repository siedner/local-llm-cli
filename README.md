# local-llm

Mac-first local inference manager using the **`mlx-lm` Python API** on Apple Silicon.

Designed for:
- **MacBook Pro M1 Pro, 32GB RAM**
- **Mac mini M4, 16GB RAM**
- **Mac mini / MacBook Pro M4, 32GB RAM**

## Install

```bash
# Option 1: pipx (recommended)
pipx install .

# Option 2: pip
pip install .
```

Requires Python 3.10 or newer.

## Quick start

```bash
# Check your environment
local-llm doctor

# Auto-detect your hardware profile
local-llm profile auto

# Download a model
local-llm models install RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4

# Start a chat
local-llm chat RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4

# Start the background daemon and warm a model
local-llm serve start RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4

# Inspect daemon/runtime state
local-llm daemon status
local-llm ps
```

## Commands

### Doctor

Check environment and install missing prerequisites:

```bash
local-llm doctor
```

Example output:
```
  [OK] python3: Python 3.12.0 at /opt/homebrew/bin/python3
  [OK] mlx_lm in PATH: found mlx_lm.generate at /opt/homebrew/bin/mlx_lm.generate
  [OK] mlx_lm importable: /opt/homebrew/lib/python3.12/site-packages/mlx_lm/__init__.py
  [OK] virtual env: none detected (system Python will be used)
  [OK] HF cache: /Users/you/.cache/huggingface/hub
  [OK] ssh: found at /usr/bin/ssh
  [OK] lsof: found at /usr/sbin/lsof

All checks passed.
```

Auto-fix missing dependencies:
```bash
local-llm doctor --fix        # interactive
local-llm doctor --fix --yes  # auto-confirm
```

### Profiles

```bash
local-llm profile list      # show available profiles
local-llm profile current   # show active profile
local-llm profile auto      # auto-detect from hardware
local-llm profile set m4mini16  # manual override
```

### Models

```bash
local-llm models list              # list installed models
local-llm models list --disk       # with disk usage
local-llm models list --no-disk    # skip disk usage
local-llm models list --json       # JSON output
local-llm models recommended       # show recommended models

local-llm models install RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4
local-llm models remove RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4
```

### Chat

```bash
local-llm chat RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4
local-llm chat RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4 --temp 0.5 --top-p 0.95
local-llm chat RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4 --session repo-chat --max-context 4096
```

Tune output length and timeout without editing config files:

```bash
local-llm config show
local-llm config size L
local-llm config max-tokens 4096
local-llm config request-timeout 900
```

Size profiles:

```text
S   = 512 tokens / 180s
M   = 1024 tokens / 300s
L   = 2048 tokens / 600s
XL  = 3072 tokens / 900s
XXL = 4096 tokens / 1200s
```

### Serve

Start the local daemon and warm a model:

```bash
# Conservative profile-aware limits
local-llm serve start RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4 --safe

# Override keep-alive/profile
local-llm serve start RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4 --profile m432 --keep-alive 1200

# Custom port
local-llm serve start RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4 --port 9090
```

Example output:
```
Profile:  m1pro32
Model:    RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4
Host:     127.0.0.1
Port:     8080
Max tkns: 2048
Decode/Prompt concurrency: 1/1

  WARNING: M1 Pro 32GB: models up to ~8B params are safe
  WARNING: Large OpenCode contexts (>10k tokens) may cause Metal OOM with bigger models

Base URL: http://127.0.0.1:8080/v1

OpenCode provider snippet:
{
  "provider": {
    "name": "MLX Local",
    "type": "openai",
    "url": "http://127.0.0.1:8080/v1",
    "model": "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4"
  }
}

Warm path: reused
```

Manage the daemon:
```bash
local-llm serve status              # inspect loaded-model runtime state
local-llm serve stop                # offload the current model, keep the daemon alive
local-llm daemon status             # inspect the daemon process itself
local-llm daemon stop               # stop the daemon process
local-llm daemon start              # start daemon explicitly
local-llm daemon install-launchd    # start at login via launchd
local-llm serve options             # show runtime defaults and safe operating model
local-llm logs --follow             # tail daemon logs
```

Terminology:
```text
daemon  = the background Python process listening on the port
runtime = the loaded model state managed inside that daemon
```

### OpenCode Integration

```bash
# Print OpenCode provider snippet
local-llm opencode snippet RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4

# Custom port and provider name
local-llm opencode snippet RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4 --port 9090 --provider-name "My MLX"
```

### Guide

```bash
local-llm guide opencode   # best practices for MLX-LM + OpenCode
```

### SSH Tunnels (opt-in)

SSH tunneling is off by default. Use it to forward a remote MLX-LM server to your local machine.

```bash
# Create a tunnel (foreground)
local-llm ssh tunnel --to user@mac-mini.local

# Create a tunnel (background)
local-llm ssh tunnel --to user@mac-mini.local --detach

# Custom ports and key
local-llm ssh tunnel --to user@mac-mini.local \
  --remote-port 8080 --local-port 9090 \
  --key ~/.ssh/id_ed25519

# Check tunnel status
local-llm ssh status

# Stop tunnel
local-llm ssh stop

# Print command without running
local-llm ssh snippet --to user@mac-mini.local
```

## Configuration

Config is stored at `~/.config/local-llm/config.json`.

Runtime data, logs, and daemon state are stored at `~/.local/share/local-llm/`.

## Recommended Models

Text-only, MLX-converted models that work well on Apple Silicon:

| Model | Quantization | Notes |
|-------|-------------|-------|
| `RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4` | MXFP4 | Recommended for 16GB machines |
| `RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4` | NVFP4 | Alternative quantization |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT
