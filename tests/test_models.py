"""Tests for model listing and enrichment."""

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from local_llm.models import _write_scan_report, enrich_model_info, install_model, list_models


def test_enrich_model_info_adds_quantization_metadata():
    model = {"repo": "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4", "path": Path("/tmp/model")}
    enriched = enrich_model_info(model)
    assert enriched["quantization"] == "MXFP4"
    assert "Apple Silicon" in enriched["summary"]
    assert "16GB/32GB" in enriched["when_to_use"]


def test_list_models_filters_tiny_entries(tmp_path):
    tiny = tmp_path / "tiny"
    tiny.mkdir()
    (tiny / "readme.txt").write_text("x")

    large = tmp_path / "large"
    large.mkdir()
    payload = large / "weights.bin"
    payload.write_bytes(b"0" * (101 * 1024 * 1024))

    installed = [
        {"repo": "HeartMuLa/HeartCodec-oss", "org": "HeartMuLa", "name": "HeartCodec-oss", "path": tiny},
        {
            "repo": "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",
            "org": "RepublicOfKorokke",
            "name": "Qwen3.5-4B-mlx-lm-mxfp4",
            "path": large,
        },
    ]

    with patch("local_llm.models.list_installed_models", return_value=installed), \
         patch("local_llm.models.get_model_disk_usage", return_value="2.1G"):
        models = list_models(disk=True, filter_relevant=True)

    assert [model["repo"] for model in models] == ["RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4"]


def test_install_model_cleans_partial_cache_on_failure(tmp_path):
    repo = "mlx-community/Qwen3.5-9B-4bit"
    failed_path = tmp_path / "models--mlx-community--Qwen3.5-9B-4bit"

    def fake_run(*_args, **_kwargs):
        failed_path.mkdir(parents=True, exist_ok=True)
        (failed_path / "snapshots").mkdir(exist_ok=True)
        return CompletedProcess(args=[], returncode=1)

    with patch("local_llm.models.get_mlx_python", return_value="python3"), \
         patch("local_llm.models.find_model_path", side_effect=[None, failed_path]), \
         patch("local_llm.models.subprocess.run", side_effect=fake_run), \
         patch("local_llm.models.ui.info"), \
         patch("local_llm.models.ui.error"):
        assert install_model(repo, yes=True) is False

    assert failed_path.exists() is False


def test_write_scan_report_saves_summary(tmp_path):
    report = tmp_path / "llm_scan_report.txt"
    rows = [
        ("File", Path("/Users/mac/models/foo.gguf"), 300 * 1024 * 1024),
        ("Ollama", Path("/Users/mac/.ollama/models/blobs/bar"), 500 * 1024 * 1024),
    ]

    _write_scan_report(report, rows, 800 * 1024 * 1024, root=Path("/Users/mac"), threshold_mb=200)

    contents = report.read_text()
    assert "LLM SCAN REPORT" in contents
    assert "CATEGORY\tSIZE\tPATH" in contents
    assert "File\t300.0M\t/Users/mac/models/foo.gguf" in contents
    assert "Ollama\t500.0M\t/Users/mac/.ollama/models/blobs/bar" in contents
    assert "TOTAL ESTIMATED SIZE:" in contents
