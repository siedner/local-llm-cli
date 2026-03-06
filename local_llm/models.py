"""Model management: list, install, remove."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import ui
from .doctor import get_mlx_python
from .hf_cache import find_model_path, get_hf_cache_dir, get_model_disk_usage, list_installed_models


def list_models(disk: bool = False, as_json: bool = False, filter_relevant: bool = True) -> list[dict]:
    """List installed models with optional disk usage."""
    models = list_installed_models(filter_relevant=filter_relevant)
    if disk:
        for m in models:
            m["disk_usage"] = get_model_disk_usage(m["path"])
    return models


def install_model(repo: str, yes: bool = False) -> bool:
    """Install/download a model by triggering a warm-download.

    Uses mlx_lm.generate --max-tokens 1 to download model weights.
    Returns True on success.
    """
    if not yes:
        ui.info(f"This will download model '{repo}' to your HF cache.")
        if not ui.confirm("Continue?"):
            ui.info("Cancelled.")
            return False

    python = get_mlx_python()
    ui.info(f"Downloading {repo} (this may take a while)...")
    result = subprocess.run(
        [python, "-m", "mlx_lm.generate",
         "--model", repo,
         "--max-tokens", "1",
         "--prompt", "hi"],
        capture_output=False,
        timeout=1800,  # 30 minutes for large downloads
    )
    if result.returncode == 0:
        ui.success(f"Model '{repo}' is now installed.")
        return True
    else:
        ui.error(f"Failed to download model '{repo}'.")
        return False


def remove_model(repo: str, yes: bool = False) -> bool:
    """Remove a model from the HF cache."""
    model_path = find_model_path(repo)
    if model_path is None:
        ui.error(f"Model '{repo}' not found in cache.")
        return False

    disk = get_model_disk_usage(model_path)
    if not yes:
        ui.warning(f"Remove '{repo}' ({disk}) from {model_path}?")
        if not ui.confirm("This cannot be undone. Continue?"):
            ui.info("Cancelled.")
            return False

    shutil.rmtree(model_path)
    ui.success(f"Removed '{repo}' ({disk}).")
    return True


def scan_models() -> None:
    """Scan the system for large LLM weights and print a report."""
    from rich.table import Table
    
    ui.header("LLM & MODEL HOUSEKEEPING SCAN")
    
    scan_dir = Path.home()
    
    ui.info(f"Scanning {scan_dir} for large model weights (>200MB)...")
    ui.info("Looking for: .gguf, .safetensors, .pth, .pt, .ckpt, .bin, .onnx")
    ui.console.print("This may take a minute or two depending on your disk speed...\n", style="dim")
    
    cmd = [
        "find", str(scan_dir), "-type", "f",
        "(",
        "-name", "*.gguf", "-o",
        "-name", "*.safetensors", "-o",
        "-name", "*.pth", "-o",
        "-name", "*.pt", "-o",
        "-name", "*.ckpt", "-o",
        "-name", "*.bin", "-o",
        "-name", "*.onnx",
        ")",
        "-size", "+200M",
        "-print0"
    ]
    
    try:
        find_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        xargs_proc = subprocess.Popen(["xargs", "-0", "ls", "-lh"], stdin=find_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        find_proc.stdout.close()
        out, _ = xargs_proc.communicate()
        
        lines = out.decode("utf-8").strip().split('\n')
        total_bytes = 0
        
        table = Table(show_header=True)
        table.add_column("Size", justify="right", style="cyan", no_wrap=True)
        table.add_column("Path", style="dim")
        
        # Helper to parse ls -lh size output back to bytes
        def parse_size(size_str: str) -> float:
            try:
                if size_str.endswith('G'): return float(size_str[:-1]) * 1024**3
                if size_str.endswith('M'): return float(size_str[:-1]) * 1024**2
                if size_str.endswith('K'): return float(size_str[:-1]) * 1024
                if size_str.endswith('B'): return float(size_str[:-1])
                return float(size_str)
            except ValueError:
                return 0.0

        if lines and lines[0]:
            for line in lines:
                parts = line.split()
                if len(parts) >= 9:
                    size_str = parts[4]
                    path_str = " ".join(parts[8:])
                    table.add_row(size_str, path_str)
                    total_bytes += parse_size(size_str)
        
        # Check Ollama blobs
        ollama_blobs = scan_dir / ".ollama" / "models" / "blobs"
        if ollama_blobs.exists() and ollama_blobs.is_dir():
            cmd_ollama = ["find", str(ollama_blobs), "-type", "f", "-size", "+200M", "-print0"]
            find_o = subprocess.Popen(cmd_ollama, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            xargs_o = subprocess.Popen(["xargs", "-0", "ls", "-lh"], stdin=find_o.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            find_o.stdout.close()
            out_o, _ = xargs_o.communicate()
            
            lines_o = out_o.decode("utf-8").strip().split('\n')
            if lines_o and lines_o[0]:
                for line in lines_o:
                    parts = line.split()
                    if len(parts) >= 9:
                        size_str = parts[4]
                        path_str = " ".join(parts[8:])
                        table.add_row(size_str, path_str + " (Ollama)")
                        total_bytes += parse_size(size_str)
        
        if table.row_count > 0:
            ui.console.print(table)
        else:
            ui.info("No large LLM weights found.")
            
        total_gb = total_bytes / (1024**3)
        ui.console.print()
        ui.success(f"TOTAL ESTIMATED STORAGE: {total_gb:.2f} GB")
        
    except Exception as e:
        ui.error(f"Scan failed: {e}")
