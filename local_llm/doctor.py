"""Doctor command: detect and install prerequisites."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .constants import VENV_DIR
from .hf_cache import get_hf_cache_dir


class Check:
    """Result of a single doctor check."""

    def __init__(self, name: str, ok: bool, detail: str, fix_hint: Optional[str] = None):
        self.name = name
        self.ok = ok
        self.detail = detail
        self.fix_hint = fix_hint


def run_checks() -> list[Check]:
    """Run all environment checks and return results."""
    checks = []
    checks.append(_check_python())
    checks.append(_check_mlx_lm_path())
    checks.append(_check_mlx_lm_import())
    checks.append(_check_venv_or_conda())
    checks.append(_check_hf_cache())
    checks.append(_check_ssh())
    checks.append(_check_lsof())
    return checks


def _find_best_python() -> tuple:
    """Find the best available Python 3.10+. Returns (path, version_str, major, minor) or Nones."""
    # Check versioned pythons first (e.g. python3.12, python3.11, python3.10), then python3
    candidates = ["python3.13", "python3.12", "python3.11", "python3.10", "python3"]
    for cmd in candidates:
        path = shutil.which(cmd)
        if not path:
            continue
        try:
            result = subprocess.run(
                [path, "--version"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                continue
            version_str = result.stdout.strip()
            parts = version_str.replace("Python ", "").split(".")
            major, minor = int(parts[0]), int(parts[1])
            if major >= 3 and minor >= 10:
                return (path, version_str, major, minor)
        except Exception:
            continue
    return (None, None, None, None)


def _check_python() -> Check:
    path, version_str, major, minor = _find_best_python()
    if path:
        return Check("python3", True, f"{version_str} at {path}")

    # Fall back to whatever python3 is for error reporting
    python = shutil.which("python3")
    if not python:
        return Check("python3", False, "not found in PATH",
                      fix_hint="Install Python 3.10+: https://www.python.org/downloads/ or `brew install python`")
    try:
        result = subprocess.run(
            ["python3", "--version"], capture_output=True, text=True, timeout=5,
        )
        version_str = result.stdout.strip()
        return Check("python3", False, f"{version_str} (need 3.10+)",
                      fix_hint="Upgrade to Python 3.10+ (you may have python3.12 — check `python3.12 --version`)")
    except Exception as e:
        return Check("python3", False, f"error: {e}")


def _check_mlx_lm_path() -> Check:
    # Check if mlx_lm generate/chat/server scripts are callable
    for cmd in ["mlx_lm.generate", "mlx_lm.chat"]:
        path = shutil.which(cmd)
        if path:
            return Check("mlx_lm in PATH", True, f"found {cmd} at {path}")
    # Check managed venv
    venv_bin = VENV_DIR / "bin"
    for cmd in ["mlx_lm.generate", "mlx_lm.chat"]:
        candidate = venv_bin / cmd
        if candidate.exists():
            return Check("mlx_lm in PATH", True, f"found in managed venv: {candidate}")
    # Check conda envs
    for conda_base in [Path.home() / "miniforge3", Path.home() / "anaconda3",
                       Path.home() / "miniconda3"]:
        envs_dir = conda_base / "envs"
        if not envs_dir.exists():
            continue
        for env_dir in sorted(envs_dir.iterdir()):
            for cmd in ["mlx_lm.generate", "mlx_lm.chat"]:
                candidate = env_dir / "bin" / cmd
                if candidate.exists():
                    return Check("mlx_lm in PATH", True,
                                  f"found in conda env '{env_dir.name}': {candidate}")
    return Check("mlx_lm in PATH", False, "mlx_lm commands not found",
                  fix_hint="Run `local-llm doctor --fix` to install mlx-lm")


def _check_mlx_lm_import() -> Check:
    # Try finding any python that can import mlx_lm
    mlx_py = _find_mlx_python()
    if mlx_py:
        try:
            result = subprocess.run(
                [mlx_py, "-c", "import mlx_lm; print(mlx_lm.__file__)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return Check("mlx_lm importable", True,
                              f"{result.stdout.strip()} (via {mlx_py})")
        except Exception:
            pass
    # Fallback: try the default python
    python = _get_python()
    try:
        result = subprocess.run(
            [python, "-c", "import mlx_lm; print(mlx_lm.__file__)"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return Check("mlx_lm importable", True, result.stdout.strip())
        return Check("mlx_lm importable", False, "cannot import mlx_lm",
                      fix_hint="Run `local-llm doctor --fix` to install mlx-lm")
    except Exception as e:
        return Check("mlx_lm importable", False, f"error: {e}")


def _check_venv_or_conda() -> Check:
    in_conda = "CONDA_DEFAULT_ENV" in os.environ
    in_venv = sys.prefix != sys.base_prefix
    managed_venv = VENV_DIR.exists()

    parts = []
    if in_conda:
        parts.append(f"conda env: {os.environ['CONDA_DEFAULT_ENV']}")
    if in_venv:
        parts.append(f"venv: {sys.prefix}")
    if managed_venv:
        parts.append(f"managed venv: {VENV_DIR}")
    if not parts:
        return Check("virtual env", True, "none detected (system Python will be used)")
    return Check("virtual env", True, "; ".join(parts))


def _check_hf_cache() -> Check:
    cache = get_hf_cache_dir()
    if cache.exists():
        return Check("HF cache", True, str(cache))
    return Check("HF cache", True, f"{cache} (will be created on first download)")


def _check_ssh() -> Check:
    ssh = shutil.which("ssh")
    if ssh:
        return Check("ssh", True, f"found at {ssh}")
    return Check("ssh", False, "ssh not found")


def _check_lsof() -> Check:
    lsof = shutil.which("lsof")
    if lsof:
        return Check("lsof", True, f"found at {lsof}")
    return Check("lsof", False, "lsof not found")


def _find_mlx_python() -> Optional[str]:
    """Search for a python that can import mlx_lm across all known environments."""
    candidates = []

    # 1. Managed venv
    venv_python = VENV_DIR / "bin" / "python3"
    if venv_python.exists():
        candidates.append(str(venv_python))

    # 2. Conda envs (check miniforge3 and anaconda3)
    for conda_base in [Path.home() / "miniforge3", Path.home() / "anaconda3",
                       Path.home() / "miniconda3"]:
        envs_dir = conda_base / "envs"
        if envs_dir.exists():
            for env_dir in sorted(envs_dir.iterdir()):
                py = env_dir / "bin" / "python3"
                if py.exists():
                    candidates.append(str(py))
        # Also check base env
        base_py = conda_base / "bin" / "python3"
        if base_py.exists():
            candidates.append(str(base_py))

    # 3. System pythons
    for cmd in ["python3.13", "python3.12", "python3.11", "python3.10", "python3"]:
        path = shutil.which(cmd)
        if path:
            candidates.append(path)

    # Test each candidate
    for py in candidates:
        try:
            result = subprocess.run(
                [py, "-c", "import mlx_lm"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return py
        except Exception:
            continue
    return None


def _get_python() -> str:
    """Return the python executable to use for mlx_lm operations."""
    # First try to find one that already has mlx_lm
    mlx_py = _find_mlx_python()
    if mlx_py:
        return mlx_py
    # Otherwise, prefer managed venv python
    venv_python = VENV_DIR / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    # Find best system python 3.10+
    path, _, _, _ = _find_best_python()
    if path:
        return path
    return "python3"


def get_mlx_python() -> str:
    """Return the python executable that can import mlx_lm."""
    return _get_python()


def fix_missing(yes: bool = False) -> list[str]:
    """Attempt to install missing mlx-lm. Returns list of actions taken."""
    actions = []

    # Check if mlx_lm is already importable
    python = _get_python()
    result = subprocess.run(
        [python, "-c", "import mlx_lm"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        actions.append("mlx-lm is already installed")
        return actions

    # Strategy 1: pipx
    pipx = shutil.which("pipx")
    if pipx:
        if not yes:
            from . import ui
            ui.info("mlx-lm is not installed.")
            if not ui.confirm("Install via pipx?"):
                actions.append("Skipped pipx install (user declined)")
                _print_manual_instructions()
                return actions
        from . import ui
        ui.info("Installing mlx-lm via pipx...")
        result = subprocess.run(
            [pipx, "install", "mlx-lm"],
            capture_output=False, timeout=300,
        )
        if result.returncode == 0:
            actions.append("Installed mlx-lm via pipx")
        else:
            actions.append("pipx install failed, falling back to managed venv")
            actions.extend(_install_managed_venv(yes))
        return actions

    # Strategy 2: managed venv
    actions.extend(_install_managed_venv(yes))
    return actions


def _install_managed_venv(yes: bool) -> list[str]:
    """Install mlx-lm into a managed venv."""
    actions = []
    if not yes:
        from . import ui
        ui.info(f"mlx-lm is not installed.")
        if not ui.confirm(f"Create managed venv at {VENV_DIR}?"):
            actions.append("Skipped managed venv install (user declined)")
            _print_manual_instructions()
            return actions

    python3 = _get_python()
    if not shutil.which(python3):
        actions.append("ERROR: python3 not found. Cannot create venv.")
        return actions

    from . import ui
    ui.info(f"Creating managed venv with {python3} at {VENV_DIR}...")
    VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [python3, "-m", "venv", str(VENV_DIR)],
        capture_output=False, timeout=60,
    )
    if result.returncode != 0:
        actions.append("ERROR: Failed to create venv")
        return actions
    actions.append(f"Created venv at {VENV_DIR}")

    pip = str(VENV_DIR / "bin" / "pip")
    ui.info("Installing mlx-lm (this may take a few minutes)...")
    result = subprocess.run(
        [pip, "install", "mlx-lm"],
        capture_output=False, timeout=600,
    )
    if result.returncode == 0:
        actions.append("Installed mlx-lm into managed venv")
    else:
        actions.append("ERROR: pip install mlx-lm failed")
    return actions


def _print_manual_instructions():
    """Print manual install instructions."""
    from . import ui

    ui.console.print()
    code = (
        "# Option 1: pipx (recommended)\n"
        "brew install pipx  # if needed\n"
        "pipx install mlx-lm\n"
        "\n"
        "# Option 2: managed venv\n"
        f"python3 -m venv {VENV_DIR}\n"
        f"{VENV_DIR}/bin/pip install mlx-lm\n"
        "\n"
        "# Option 3: pip (if you know what you're doing)\n"
        "pip install mlx-lm"
    )
    ui.panel(code, title="Manual install options")
