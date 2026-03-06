"""SSH tunnel support (opt-in, off by default)."""

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Optional

from . import ui
from .constants import DEFAULT_PORT, LOGS_DIR, PIDS_DIR


def _ensure_dirs():
    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _ssh_pid_file(local_port: int) -> Path:
    return PIDS_DIR / f"ssh-tunnel-{local_port}.pid"


def tunnel(
    to: str,
    remote_port: int = DEFAULT_PORT,
    local_port: int = DEFAULT_PORT,
    key: str = "~/.ssh/id_ed25519",
    bind: str = "127.0.0.1",
    detach: bool = False,
) -> None:
    """Create an SSH local forward tunnel."""
    _ensure_dirs()
    key_path = os.path.expanduser(key)

    if not os.path.exists(key_path):
        ui.error(f"SSH key not found: {key_path}")
        ui.info("Specify a different key with --key")
        return

    cmd = [
        "ssh",
        "-i", key_path,
        "-N",
        "-L", f"{bind}:{local_port}:127.0.0.1:{remote_port}",
        to,
    ]

    cmd_str = " ".join(cmd)
    base_url = f"http://{bind}:{local_port}/v1"

    from rich.text import Text

    summary = Text()
    pairs = [
        ("Command", cmd_str),
        ("Forward", f"{bind}:{local_port} -> {to}:127.0.0.1:{remote_port}"),
        ("Base URL", base_url),
    ]
    for i, (k, v) in enumerate(pairs):
        summary.append(f"  {k}", style="white")
        summary.append(f"  {v}", style="dim")
        if i < len(pairs) - 1:
            summary.append("\n")

    ui.rich_panel(summary, title="SSH Tunnel")

    if detach:
        log_path = LOGS_DIR / f"ssh-tunnel-{local_port}.log"
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_file = _ssh_pid_file(local_port)
        pid_file.write_text(str(proc.pid))
        ui.console.print()
        ui.kv("PID", str(proc.pid))
        ui.kv("Log", str(log_path))
        ui.kv("Stop", f"local-llm ssh stop --local-port {local_port}")
    else:
        ui.console.print()
        ui.info("Starting tunnel (Ctrl+C to stop)...")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            ui.console.print()
            ui.info("Tunnel closed.")


def ssh_status(local_port: int = DEFAULT_PORT) -> None:
    """Check status of an SSH tunnel."""
    pid_file = _ssh_pid_file(local_port)
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            ui.kv("SSH tunnel", f"running on local port {local_port} (PID {pid})")
        except ProcessLookupError:
            ui.warning(f"PID file exists but process {pid} is not running.")
            pid_file.unlink(missing_ok=True)
    else:
        ui.info(f"No SSH tunnel tracked for local port {local_port}.")


def ssh_stop(local_port: int = DEFAULT_PORT) -> None:
    """Stop an SSH tunnel."""
    pid_file = _ssh_pid_file(local_port)
    if not pid_file.exists():
        ui.warning(f"No PID file for SSH tunnel on local port {local_port}.")
        return

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        ui.success(f"Sent SIGTERM to SSH tunnel (PID {pid}).")
    except ProcessLookupError:
        ui.info(f"Process {pid} not found (already stopped?).")
    pid_file.unlink(missing_ok=True)


def ssh_snippet(
    to: str,
    remote_port: int = DEFAULT_PORT,
    local_port: int = DEFAULT_PORT,
    key: str = "~/.ssh/id_ed25519",
    bind: str = "127.0.0.1",
) -> None:
    """Print the SSH command and OpenCode baseURL for a tunnel."""
    key_path = os.path.expanduser(key)
    cmd = f"ssh -i {key_path} -N -L {bind}:{local_port}:127.0.0.1:{remote_port} {to}"

    ui.info("SSH tunnel command:")
    ui.code_block(cmd, lang="bash")
    ui.console.print()
    ui.kv("Base URL", f"http://{bind}:{local_port}/v1")
    ui.console.print()
    ui.info("Or run:")
    ui.code_block(
        f"local-llm ssh tunnel --to {to} --remote-port {remote_port} --local-port {local_port}",
        lang="bash",
    )
