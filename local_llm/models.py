"""Model management: list, install, remove."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from . import ui
from .constants import MODEL_METADATA, RECOMMENDED_MODELS
from .doctor import get_mlx_python
from .hf_cache import (
    find_model_path,
    get_hf_cache_dir,
    get_model_disk_usage,
    list_installed_models,
)

MIN_RELEVANT_MODEL_BYTES = 100 * 1024 * 1024


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += (Path(root) / filename).stat().st_size
            except OSError:
                continue
    return total


def _infer_quantization(repo: str) -> str | None:
    lower = repo.lower()
    if "mxfp4" in lower:
        return "MXFP4"
    if "nvfp4" in lower:
        return "NVFP4"
    if "4bit" in lower or "4-bit" in lower:
        return "4-bit"
    return None


def _default_summary(repo: str) -> str:
    quant = _infer_quantization(repo)
    if quant == "MXFP4":
        return "Apple-Silicon-oriented fast quantization for lower memory pressure."
    if quant == "NVFP4":
        return "Alternative 4-bit quantization aimed at similar size with slightly different quality/speed tradeoffs."
    if quant == "4-bit":
        return "Quantized model variant tuned for lower memory use than full precision weights."
    return "Installed model found in the Hugging Face cache."


def enrich_model_info(model: dict) -> dict:
    enriched = dict(model)
    metadata = MODEL_METADATA.get(enriched["repo"], {})
    if not metadata:
        for candidate in RECOMMENDED_MODELS:
            if candidate["repo"] == enriched["repo"]:
                metadata = {
                    "summary": candidate.get("description"),
                }
                break
    quantization = metadata.get("quantization") or _infer_quantization(enriched["repo"])
    enriched["family"] = metadata.get("family")
    enriched["quantization"] = quantization
    enriched["summary"] = metadata.get("summary") or _default_summary(enriched["repo"])
    enriched["when_to_use"] = metadata.get("when_to_use", "")
    enriched["tradeoffs"] = metadata.get("tradeoffs", "")
    return enriched


def _include_model(model: dict, *, filter_relevant: bool) -> bool:
    if not filter_relevant:
        return True
    size_bytes = _directory_size_bytes(model["path"])
    model["size_bytes"] = size_bytes
    return size_bytes >= MIN_RELEVANT_MODEL_BYTES


def list_models(disk: bool = False, as_json: bool = False, filter_relevant: bool = True) -> list[dict]:
    """List installed models with optional disk usage."""
    models = list_installed_models(filter_relevant=filter_relevant)
    filtered: list[dict] = []
    for model in models:
        if not _include_model(model, filter_relevant=filter_relevant):
            continue
        if disk:
            model["disk_usage"] = get_model_disk_usage(model["path"])
        filtered.append(enrich_model_info(model))
    return filtered


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
    existing_path = find_model_path(repo)
    had_existing_cache = existing_path is not None
    ui.info(f"Downloading {repo} (this may take a while)...")
    result = subprocess.run(
        [python, "-m", "mlx_lm", "generate",
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
        if not had_existing_cache:
            failed_path = find_model_path(repo)
            if failed_path is not None:
                shutil.rmtree(failed_path, ignore_errors=True)
                ui.info(f"Removed partial cache entry at {failed_path}.")
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


def verify_model(repo: str) -> bool:
    """Verify that a cached model has the minimum expected files."""
    model_path = find_model_path(repo)
    if model_path is None:
        ui.error(f"Model '{repo}' not found in cache.")
        return False

    snapshots = model_path / "snapshots"
    if not snapshots.exists():
        ui.error(f"Model '{repo}' is missing snapshots/")
        return False

    snapshot_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        ui.error(f"Model '{repo}' has no snapshot directories.")
        return False

    latest = snapshot_dirs[0]
    required = ["config.json", "tokenizer.json"]
    missing = [name for name in required if not (latest / name).exists()]
    if missing:
        ui.error(f"Model '{repo}' is missing files: {', '.join(missing)}")
        return False

    ui.success(f"Model '{repo}' passed integrity checks.")
    return True


def repair_model(repo: str, yes: bool = False) -> bool:
    """Repair a model by re-running install if verification fails."""
    if verify_model(repo):
        ui.info("Nothing to repair.")
        return True
    ui.warning(f"Repairing '{repo}' by re-downloading the model.")
    return install_model(repo, yes=yes)


def prune_models(yes: bool = False) -> int:
    """Prune invalid cache entries."""
    cache_dir = get_hf_cache_dir()
    if not cache_dir.exists():
        ui.info("HF cache does not exist.")
        return 0

    pruned = 0
    for entry in sorted(cache_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        snapshots = entry / "snapshots"
        refs = entry / "refs"
        if snapshots.exists() and any(snapshots.iterdir()):
            continue
        if refs.exists() and any(refs.iterdir()):
            continue
        if not yes:
            ui.warning(f"Prune invalid cache entry {entry}?")
            if not ui.confirm("Continue?"):
                continue
        shutil.rmtree(entry, ignore_errors=True)
        pruned += 1

    if pruned:
        ui.success(f"Pruned {pruned} invalid cache entr{'y' if pruned == 1 else 'ies'}.")
    else:
        ui.info("No invalid cache entries found.")
    return pruned


def warm_model(repo: str, host: str = "127.0.0.1", port: int = 8080) -> bool:
    """Warm a model through the daemon."""
    from .server import serve

    serve(repo, host=host, port=port, detach=True, safe=True)
    return True


def _format_size(size_bytes: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    value = float(size_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)}B"
    return f"{value:.1f}{unit}"


def _iter_large_files(root: Path, *, threshold_bytes: int, extensions: set[str], skip_roots: set[Path] | None = None) -> Iterable[tuple[Path, int]]:
    skip_roots = skip_roots or set()
    for current_root, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current_root)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if (current_path / dirname) not in skip_roots
        ]

        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() not in extensions:
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError:
                continue
            if size_bytes >= threshold_bytes:
                yield path, size_bytes


def _write_scan_report(report_file: Path, rows: list[tuple[str, Path, int]], total_bytes: int, *, root: Path, threshold_mb: int) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        handle.write("LLM SCAN REPORT\n")
        handle.write(f"Root: {root}\n")
        handle.write(f"Minimum Size: {threshold_mb}MB\n\n")
        handle.write("CATEGORY\tSIZE\tPATH\n")
        for category, path, size_bytes in rows:
            handle.write(f"{category}\t{_format_size(size_bytes)}\t{path}\n")
        handle.write("\n")
        handle.write(f"TOTAL ESTIMATED SIZE: {total_bytes / (1024 ** 3):.2f} GB\n")


def scan_models(root: Path | None = None, *, save_report: Path | None = None, min_size_mb: int = 200) -> None:
    """Scan the system for large LLM weights and print a report."""
    from rich.table import Table

    ui.header("LLM & MODEL HOUSEKEEPING SCAN")

    scan_dir = root or Path.home()
    report_file = save_report or (Path.home() / "llm_scan_report.txt")

    ui.info(f"Scanning {scan_dir} for large model weights (>{min_size_mb}MB)...")
    ui.info("Looking for: .gguf, .safetensors, .pth, .pt, .ckpt, .bin, .onnx")
    ui.info(f"Report will be saved to: {report_file}")
    ui.console.print("This may take a minute or two depending on your disk speed...\n", style="dim")

    threshold_bytes = min_size_mb * 1024 * 1024
    extensions = {".gguf", ".safetensors", ".pth", ".pt", ".ckpt", ".bin", ".onnx"}

    try:
        total_bytes = 0
        rows: list[tuple[str, Path, int]] = []
        table = Table(show_header=True)
        table.add_column("Category", style="white", no_wrap=True)
        table.add_column("Size", justify="right", style="cyan", no_wrap=True)
        table.add_column("Path", style="dim")

        files_found = 0
        with ui.console.status("[bold white]Scanning filesystem for model weights...", spinner="dots") as status:
            for path, size_bytes in _iter_large_files(scan_dir, threshold_bytes=threshold_bytes, extensions=extensions, skip_roots={scan_dir / ".ollama"}):
                rows.append(("File", path, size_bytes))
                files_found += 1
                total_bytes += size_bytes
                status.update(f"[bold white]Scanning filesystem...[/] found {files_found} large files ({total_bytes / (1024 ** 3):.2f} GB)")

            ollama_blobs = scan_dir / ".ollama" / "models" / "blobs"
            if ollama_blobs.exists() and ollama_blobs.is_dir():
                status.update(f"[bold white]Scanning Ollama blobs...[/] found {files_found} large files ({total_bytes / (1024 ** 3):.2f} GB)")
                for path, size_bytes in _iter_large_files(ollama_blobs, threshold_bytes=threshold_bytes, extensions=extensions):
                    rows.append(("Ollama", path, size_bytes))
                    files_found += 1
                    total_bytes += size_bytes
                    status.update(f"[bold white]Scanning Ollama blobs...[/] found {files_found} large files ({total_bytes / (1024 ** 3):.2f} GB)")

        for category, path, size_bytes in rows:
            table.add_row(category, _format_size(size_bytes), str(path))

        if table.row_count > 0:
            ui.console.print(table)
        else:
            ui.info("No large LLM weights found.")

        _write_scan_report(report_file, rows, total_bytes, root=scan_dir, threshold_mb=min_size_mb)
        total_gb = total_bytes / (1024 ** 3)
        ui.console.print()
        ui.success(f"TOTAL ESTIMATED STORAGE: {total_gb:.2f} GB")
        ui.info(f"Saved report: {report_file}")

    except Exception as e:
        ui.error(f"Scan failed: {e}")
