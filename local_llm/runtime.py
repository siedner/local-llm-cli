"""Helpers for inspecting and safely managing runtime processes."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProcessInfo:
    """Normalized process metadata for a server-like process."""

    pid: int
    cmdline: str
    module_name: Optional[str]
    is_mlx_server: bool
    model: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None


def pid_exists(pid: int) -> bool:
    """Return True if a PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_process_info(pid: int) -> Optional[ProcessInfo]:
    """Return normalized metadata for a PID, if available."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    cmdline = result.stdout.strip()
    if result.returncode != 0 or not cmdline:
        return None
    return parse_process_info(pid, cmdline)


def get_listening_pid(port: int) -> Optional[int]:
    """Return the first PID listening on a TCP port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    for raw_pid in result.stdout.splitlines():
        raw_pid = raw_pid.strip()
        if raw_pid:
            try:
                return int(raw_pid)
            except ValueError:
                continue
    return None


def get_listening_process_info(port: int) -> Optional[ProcessInfo]:
    """Return process metadata for the listener on a TCP port."""
    pid = get_listening_pid(port)
    if pid is None:
        return None
    return get_process_info(pid)


def parse_process_info(pid: int, cmdline: str) -> ProcessInfo:
    """Parse a command line into normalized process metadata."""
    tokens = _split_cmdline(cmdline)

    module_name = _extract_module_name(tokens)
    is_mlx_server = module_name == "mlx_lm.server" or "mlx_lm.server" in cmdline

    port = _extract_int_flag(tokens, "--port")
    return ProcessInfo(
        pid=pid,
        cmdline=cmdline,
        module_name=module_name,
        is_mlx_server=is_mlx_server,
        model=_extract_flag(tokens, "--model"),
        host=_extract_flag(tokens, "--host"),
        port=port,
    )


def _split_cmdline(cmdline: str) -> list[str]:
    try:
        return shlex.split(cmdline)
    except ValueError:
        return cmdline.split()


def _extract_module_name(tokens: list[str]) -> Optional[str]:
    for idx, token in enumerate(tokens):
        if token == "-m" and idx + 1 < len(tokens):
            return tokens[idx + 1]
    return None


def _extract_flag(tokens: list[str], flag: str) -> Optional[str]:
    for idx, token in enumerate(tokens):
        if token == flag and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def _extract_int_flag(tokens: list[str], flag: str) -> Optional[int]:
    value = _extract_flag(tokens, flag)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
