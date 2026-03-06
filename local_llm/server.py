"""Server management: serve, stop, status."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import ui
from .config import get_profile, load_config
from .constants import DEFAULT_HOST, DEFAULT_PORT, LOGS_DIR, PIDS_DIR, PROFILES
from .doctor import get_mlx_python


def _ensure_dirs():
    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _pid_file(port: int) -> Path:
    return PIDS_DIR / f"server-{port}.pid"


def _log_file(port: int) -> Path:
    return LOGS_DIR / f"server-{port}.log"


def serve(
    model: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_tokens: Optional[int] = None,
    detach: bool = False,
    log: Optional[str] = None,
    safe: bool = False,
) -> None:
    """Start the MLX-LM OpenAI-compatible server."""
    _ensure_dirs()
    config = load_config()
    profile = get_profile(config)

    # Determine server args
    decode_concurrency = 1
    prompt_concurrency = 1

    if max_tokens is None:
        if profile:
            max_tokens = profile["max_tokens"]
        else:
            max_tokens = 1024  # conservative default

    if safe:
        decode_concurrency = 1
        prompt_concurrency = 1
        if profile and max_tokens > profile["max_tokens"]:
            max_tokens = profile["max_tokens"]
        ui.warning("SAFE MODE  Concurrency=1, reduced max-tokens")

    # Print summary panel
    config_table = ui.styled_table(box=None, padding=(0, 1, 0, 0), show_header=False)
    config_table.add_column("Key", style="white", no_wrap=True)
    config_table.add_column("Value", style="dim")

    profile_name = config.get("profile", "none")
    config_table.add_row("Profile", profile_name)
    config_table.add_row("Model", model)
    config_table.add_row("Host", host)
    config_table.add_row("Port", str(port))
    config_table.add_row("Max tokens", str(max_tokens))
    config_table.add_row("Concurrency", f"decode={decode_concurrency} prompt={prompt_concurrency}")

    ui.rich_panel(config_table, title="Server Config")

    # Risk hints
    if profile:
        for hint in profile.get("risk_hints", []):
            ui.warning(hint)
        ui.console.print()

    base_url = f"http://{host}:{port}/v1"
    ui.kv("Base URL", base_url)
    ui.console.print()
    _print_opencode_snippet(model, port)
    ui.console.print()

    python = get_mlx_python()
    cmd = [
        python, "-m", "mlx_lm.server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--max-tokens", str(max_tokens),
        "--decode-concurrency", str(decode_concurrency),
        "--prompt-concurrency", str(prompt_concurrency),
    ]

    if detach:
        log_path = Path(log) if log else _log_file(port)
        ui.info(f"Starting in background...")
        ui.kv("Log", str(log_path))
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_file = _pid_file(port)
        pid_file.write_text(str(proc.pid))
        ui.kv("PID", str(proc.pid))
        ui.kv("Stop", f"local-llm serve stop --port {port}")
    else:
        ui.info("Starting server (Ctrl+C to stop)...")
        ui.divider()
        try:
            proc = subprocess.run(cmd)
        except KeyboardInterrupt:
            ui.console.print()
            ui.info("Server stopped.")


def serve_stop(port: int = DEFAULT_PORT) -> None:
    """Stop a detached server by port."""
    pid_file = _pid_file(port)
    if not pid_file.exists():
        ui.warning(f"No PID file for port {port}. Server may not be running.")
        _try_lsof_kill(port)
        return

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        ui.success(f"Sent SIGTERM to PID {pid} (port {port}).")
    except ProcessLookupError:
        ui.info(f"Process {pid} not found (already stopped?).")
    pid_file.unlink(missing_ok=True)


def _fingerprint_port_pid(port: int) -> dict | None:
    """Find the PID listening on port and fingerprint its commandline.

    Returns a dict with keys: pid, cmdline, is_mlx_server, model
    or None if nothing is listening.
    """
    try:
        lsof = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        raw_pids = [p.strip() for p in lsof.stdout.strip().split("\n") if p.strip()]
        if not raw_pids:
            return None

        # Pick the first PID (usually only one listens on the port)
        pid = int(raw_pids[0])

        ps = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        )
        cmdline = ps.stdout.strip()

        is_mlx = "mlx_lm.server" in cmdline or "mlx_lm" in cmdline

        # Try to extract --model argument from command line
        model = None
        if "--model" in cmdline:
            parts = cmdline.split()
            for i, part in enumerate(parts):
                if part == "--model" and i + 1 < len(parts):
                    model = parts[i + 1]
                    break

        return {"pid": pid, "cmdline": cmdline, "is_mlx_server": is_mlx, "model": model}
    except Exception:
        return None


def serve_status(port: int = DEFAULT_PORT) -> None:
    """Show status of a server on a given port."""
    pid_file = _pid_file(port)
    log_file = _log_file(port)

    # Case 1: We have a PID file from a managed server
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)  # 0 = just check if alive
            info = _fingerprint_port_pid(port)
            ui.success(f"Server running on port {port}")
            ui.kv("PID", str(pid))
            if info and info["model"]:
                ui.kv("Model", info["model"])
            if log_file.exists():
                ui.kv("Log", str(log_file))
            ui.kv("Stop", f"local-llm serve stop --port {port}")
        except ProcessLookupError:
            ui.warning(f"Stale PID file: process {pid} is no longer running.")
            pid_file.unlink(missing_ok=True)
            # Fall through to port scan below
            _check_port_unmanaged(port)
        return

    # Case 2: No PID file – check if anything is on the port
    _check_port_unmanaged(port)


def _check_port_unmanaged(port: int) -> None:
    """Check whether an unmanaged process is listening on port and fingerprint it."""
    info = _fingerprint_port_pid(port)
    if not info:
        ui.info(f"No server detected on port {port}.")
        return

    pid = info["pid"]
    if info["is_mlx_server"]:
        model_str = f" ({info['model']}" + ")" if info["model"] else ""
        ui.warning(f"mlx_lm.server detected on port {port} (PID {pid}){model_str}")
        ui.info("This looks like an orphaned local-llm server from a previous session.")
        ui.kv("Cmdline", info["cmdline"][:80] + ("..." if len(info["cmdline"]) > 80 else ""))
        if ui.confirm("Kill it and reclaim the port?"):
            try:
                os.kill(pid, signal.SIGTERM)
                ui.success(f"Sent SIGTERM to PID {pid}.")
            except Exception as e:
                ui.error(f"Failed to kill PID {pid}: {e}")
    else:
        ui.warning(f"Port {port} is in use by a non-mlx process (PID {pid}).")
        ui.kv("Cmdline", info["cmdline"][:80] + ("..." if len(info["cmdline"]) > 80 else ""))
        ui.info("This is not a local-llm server — not touching it.")


def _try_lsof_kill(port: int) -> None:
    """Try to find and kill process on port via lsof."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            ui.info(f"Found process(es) on port {port}: {', '.join(pids)}")
            if ui.confirm("Kill them?"):
                for pid in pids:
                    os.kill(int(pid), signal.SIGTERM)
                ui.success("Sent SIGTERM.")
    except Exception:
        pass


def _print_opencode_snippet(model: str, port: int) -> None:
    """Print an OpenCode provider snippet."""
    snippet = {
        "provider": {
            "name": "MLX Local",
            "type": "openai",
            "url": f"http://127.0.0.1:{port}/v1",
            "model": model,
        }
    }
    ui.info("OpenCode provider snippet:")
    ui.code_block(json.dumps(snippet, indent=2), lang="json")


def opencode_snippet(model: str, port: int = DEFAULT_PORT, provider_name: str = "MLX Local") -> None:
    """Print an OpenCode provider JSON block."""
    snippet = {
        "provider": {
            "name": provider_name,
            "type": "openai",
            "url": f"http://127.0.0.1:{port}/v1",
            "model": model,
        }
    }
    ui.code_block(json.dumps(snippet, indent=2), lang="json")


def serve_options() -> None:
    """Print known server flags and safe defaults."""
    ui.header("MLX-LM Server Options")
    ui.console.print()

    flags = (
        "  --model <hf_repo>        Model to serve\n"
        "  --host <addr>            Bind address (default: 127.0.0.1)\n"
        "  --port <port>            Port (default: 8080)\n"
        "  --max-tokens <n>         Max tokens per response\n"
        "  --decode-concurrency <n> Concurrent decode requests\n"
        "  --prompt-concurrency <n> Concurrent prompt processing"
    )
    ui.panel(flags, title="Key flags for `python -m mlx_lm.server`")
    ui.console.print()

    defaults = (
        "  --decode-concurrency 1\n"
        "  --prompt-concurrency 1\n"
        "  --max-tokens 1024  (M4 16GB)\n"
        "  --max-tokens 2048  (M1 Pro 32GB)"
    )
    ui.panel(defaults, title="Safe defaults (recommended for OpenCode)")
    ui.console.print()

    ui.warning("OpenCode sends large prefills (tool schema + repo context).")
    ui.warning("This can exceed 10k tokens and cause Metal OOM crashes.")
    ui.warning("Use `--safe` flag with `local-llm serve` to enforce safe limits.")


def guide_opencode() -> None:
    """Print best practices for using MLX-LM with OpenCode."""
    ui.header("Guide: Using MLX-LM with OpenCode")

    sections = [
        ("1. Use text-only models (not multimodal)",
         "Recommended: Qwen3.5-4B with MLX quantization"),
        ("2. Keep context small",
         "OpenCode sends tool schemas + repo context in every request.\n"
         "This can easily exceed 10k tokens on prefill.\n"
         "Large prefills cause Metal OOM on 16GB machines."),
        ("3. Start with a small repo to validate",
         "Test with a tiny project first.\n"
         "If it works, gradually increase project size."),
        ("4. Use --safe mode",
         "local-llm serve <model> --safe"),
        ("5. Monitor memory usage",
         "Watch Activity Monitor for 'Memory Pressure'.\n"
         "If yellow/red, stop the server and use a smaller model."),
        ("6. Recommended models",
         "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4\n"
         "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-nvfp4"),
    ]

    for title, body in sections:
        ui.console.print()
        ui.panel(body, title=title)
