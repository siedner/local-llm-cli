"""Daemon management and CLI compatibility wrappers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import ui
from .config import detect_profile, get_effective_profile, load_config, save_calibration
from .constants import (
    DAEMON_LOG_FILE,
    DAEMON_PID_FILE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LAUNCHD_LABEL,
    LAUNCHD_PLIST,
    PACKAGE_ROOT,
)
from .daemon_client import DaemonClient, DaemonError
from .doctor import get_mlx_python
from .launchd import write_launchd_plist
from .runtime import get_listening_process_info, get_process_info


def _daemon_env() -> dict:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    roots = [str(PACKAGE_ROOT)]
    if pythonpath:
        roots.append(pythonpath)
    env["PYTHONPATH"] = ":".join(roots)
    return env


def _daemon_cmd(host: str, port: int) -> list[str]:
    python = get_mlx_python()
    return [python, "-m", "local_llm.daemon", "--host", host, "--port", str(port)]


def _wait_for_daemon(client: DaemonClient, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.health()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def daemon_start(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    detach: bool = True,
    log: Optional[str] = None,
) -> None:
    """Start the local-llm daemon."""
    client = DaemonClient(host, port)
    try:
        health = client.health()
        ui.success(f"Daemon already running on {host}:{port}")
        if health.get("loaded_model"):
            ui.kv("Loaded model", health["loaded_model"])
        return
    except Exception:
        pass

    cmd = _daemon_cmd(host, port)
    env = _daemon_env()
    log_path = Path(log) if log else DAEMON_LOG_FILE
    if detach:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
                cwd=str(PACKAGE_ROOT),
            )
        DAEMON_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        DAEMON_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        ui.info("Starting daemon in background...")
        ui.kv("PID", str(proc.pid))
        ui.kv("Log", str(log_path))
        if _wait_for_daemon(client):
            ui.success(f"Daemon ready at http://{host}:{port}/v1")
        else:
            exited = proc.poll()
            DAEMON_PID_FILE.unlink(missing_ok=True)
            if exited is not None:
                ui.error(f"Daemon exited during startup with code {exited}.")
            else:
                ui.warning(f"Daemon did not report healthy within timeout. Check {log_path}")
    else:
        ui.info("Starting daemon in foreground (Ctrl+C to stop)...")
        subprocess.run(cmd, env=env, cwd=str(PACKAGE_ROOT))


def daemon_stop(port: int = DEFAULT_PORT) -> None:
    """Stop the daemon process."""
    pid = None
    info = None
    if DAEMON_PID_FILE.exists():
        pid = int(DAEMON_PID_FILE.read_text().strip())
        info = get_process_info(pid)
    else:
        info = get_listening_process_info(port)
        pid = info.pid if info else None
        if pid is None:
            ui.warning("No daemon PID file found.")
            return

    if not info or info.module_name != "local_llm.daemon":
        ui.error(f"PID {pid} is not a local_llm.daemon process; refusing to kill it.")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + 10
        while time.time() < deadline:
            if get_process_info(pid) is None:
                DAEMON_PID_FILE.unlink(missing_ok=True)
                ui.success(f"Stopped daemon PID {pid}.")
                return
            time.sleep(0.2)
        ui.warning(f"Daemon PID {pid} did not exit after SIGTERM; sending SIGKILL.")
        os.kill(pid, signal.SIGKILL)
        ui.success(f"Killed daemon PID {pid}.")
    except OSError as exc:
        ui.error(str(exc))
    finally:
        DAEMON_PID_FILE.unlink(missing_ok=True)


def daemon_status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Show daemon status."""
    client = DaemonClient(host, port)
    try:
        snapshot = client.health()
    except Exception:
        ui.info(f"No daemon detected on {host}:{port}.")
        return

    ui.header("Daemon Status")
    ui.kv("State", snapshot["status"])
    ui.kv("Profile", snapshot["profile"])
    ui.kv("Base URL", f"http://{host}:{port}/v1")
    ui.kv("Loaded model", snapshot.get("loaded_model") or "none")
    ui.kv("Memory pressure", snapshot["memory_pressure"]["state"])
    ui.kv("Keep alive", str(snapshot["keep_alive_seconds"]))
    if snapshot.get("active_request_id"):
        ui.kv("Active request", snapshot["active_request_id"])


def serve(
    model: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_tokens: Optional[int] = None,
    profile: Optional[str] = None,
    keep_alive_seconds: Optional[int] = None,
    detach: bool = False,
    log: Optional[str] = None,
    safe: bool = False,
) -> None:
    """Start or warm the daemon-backed local runtime."""
    if not detach:
        ui.info("`serve start` warms models through the background daemon; starting it detached.")
    daemon_start(host=host, port=port, detach=True, log=log)
    client = DaemonClient(host, port)
    config = load_config()
    profile_name, profile = get_effective_profile(config, profile)
    effective_max_tokens = min(max_tokens or profile["max_tokens"], profile["max_tokens"])
    try:
        result = client.warm(
            model,
            keep_alive_seconds=keep_alive_seconds or profile["keep_alive_seconds"],
            profile=profile_name,
            safe_mode=safe,
        )
    except DaemonError as exc:
        ui.error(str(exc))
        return

    ui.header("Runtime")
    ui.kv("Profile", profile_name)
    ui.kv("Model", model)
    ui.kv("Base URL", f"http://{host}:{port}/v1")
    ui.kv("Max output", str(effective_max_tokens))
    ui.kv("Context", f"{profile['default_context']} / hard {profile['hard_context']}")
    ui.kv("Keep alive", f"{result['keep_alive_seconds']}s")
    ui.kv("Warm path", "reused" if result["reused"] else "fresh load")
    if result.get("load_duration_seconds") is not None:
        ui.kv("Load time", f"{result['load_duration_seconds']:.2f}s")
    ui.console.print()
    _print_opencode_snippet(model, port)


def serve_stop(port: int = DEFAULT_PORT) -> None:
    """Stop the daemon."""
    daemon_stop(port=port)


def serve_status(port: int = DEFAULT_PORT) -> None:
    """Show runtime status."""
    daemon_status(port=port)


def show_ps(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Show runtime snapshot."""
    client = DaemonClient(host, port)
    try:
        snapshot = client.ps()
    except Exception as exc:
        ui.error(str(exc))
        return
    ui.header("Runtime")
    ui.kv("State", snapshot["status"])
    ui.kv("Model", snapshot.get("loaded_model") or "none")
    ui.kv("Profile", snapshot["profile"])
    ui.kv("Sessions", str(snapshot["session_count"]))
    ui.kv("Memory pressure", snapshot["memory_pressure"]["state"])


def inspect_identifier(identifier: str, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Inspect a session or model."""
    client = DaemonClient(host, port)
    try:
        result = client.inspect(identifier)
    except Exception as exc:
        ui.error(str(exc))
        return
    ui.code_block(json.dumps(result, indent=2), lang="json")


def tail_logs(follow: bool = False) -> None:
    """Print daemon logs."""
    if not DAEMON_LOG_FILE.exists():
        ui.warning("No daemon log file found.")
        return
    if follow:
        subprocess.run(["tail", "-f", str(DAEMON_LOG_FILE)])
    else:
        subprocess.run(["tail", "-n", "100", str(DAEMON_LOG_FILE)])


def benchmark_runtime(
    model: str,
    *,
    runs: int = 5,
    prompt: Optional[str] = None,
    profile: Optional[str] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> dict | None:
    """Run a benchmark through the daemon."""
    daemon_start(host=host, port=port, detach=True)
    client = DaemonClient(host, port)
    try:
        result = client.benchmark(
            {
                "model": model,
                "runs": runs,
                "prompt": prompt,
                "profile": profile,
            }
        )
    except Exception as exc:
        ui.error(str(exc))
        return None

    ui.header("Benchmark")
    ui.kv("Model", result["model"])
    ui.kv("Profile", result["profile"])
    ui.kv("Runs", str(result["runs"]))
    ui.kv("Avg total", f"{result['avg_total_seconds']:.2f}s")
    ui.kv("Avg TTFT", f"{result['avg_ttft_seconds']:.2f}s")
    ui.kv("Avg tok/s", f"{result['avg_generation_tps']:.2f}")
    return result


def calibrate_profile(
    model: str,
    *,
    profile: Optional[str] = None,
    runs: int = 5,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    """Benchmark and persist conservative calibration metadata."""
    benchmark = benchmark_runtime(model, runs=runs, profile=profile, host=host, port=port)
    if benchmark is None:
        return
    config = load_config()
    profile_name, effective = get_effective_profile(config, profile or detect_profile())
    calibration = {
        "updated_at": time.time(),
        "runtime": {
            "default_context": effective["default_context"],
            "hard_context": effective["hard_context"],
            "max_tokens": effective["max_tokens"],
            "keep_alive_seconds": effective["keep_alive_seconds"],
        },
        "benchmark": benchmark,
    }
    save_calibration(profile_name, calibration)
    ui.success(f"Saved calibration for profile {profile_name}.")


def install_launchd(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Install the daemon as a launchd agent."""
    plist = write_launchd_plist(host=host, port=port)
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"], check=False)
    subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)], check=False)
    ui.success(f"Installed launchd agent {LAUNCHD_LABEL}.")
    ui.kv("Plist", str(plist))


def uninstall_launchd() -> None:
    """Remove the launchd agent."""
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"], check=False)
    LAUNCHD_PLIST.unlink(missing_ok=True)
    ui.success(f"Removed launchd agent {LAUNCHD_LABEL}.")


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


def _print_opencode_snippet(model: str, port: int) -> None:
    ui.info("OpenCode provider snippet:")
    opencode_snippet(model, port=port)


def serve_options() -> None:
    """Print runtime defaults and controls."""
    ui.header("Daemon Runtime Defaults")
    ui.panel(
        "  one warm model per machine\n"
        "  one active request at a time\n"
        "  profile-aware context caps\n"
        "  keep-alive based idle eviction\n"
        "  macOS memory-pressure admission control",
        title="Operating model",
    )


def guide_opencode() -> None:
    """Print best practices for using MLX-LM with OpenCode."""
    ui.header("Guide: Using local-llm with OpenCode")
    ui.panel(
        "Keep one small model warm, clamp max output aggressively, and prefer shorter tool/repo context windows.\n"
        "Use `local-llm profile calibrate` on each Mac before trusting bigger contexts.",
        title="Best practices",
    )
