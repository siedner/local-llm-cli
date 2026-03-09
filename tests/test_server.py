"""Tests for daemon/runtime control commands."""

from unittest.mock import Mock, patch

from local_llm.server import daemon_status, serve_status, serve_stop


def test_serve_stop_evicts_loaded_model():
    client = Mock()
    client.health.return_value = {"loaded_model": "test/model"}
    client.evict.return_value = {"ok": True, "evicted_model": "test/model", "reason": "manual"}

    with patch("local_llm.server.DaemonClient", return_value=client), \
         patch("local_llm.server.ui.success") as success, \
         patch("local_llm.server.ui.info") as info:
        serve_stop(port=9090)

    client.health.assert_called_once_with()
    client.evict.assert_called_once_with("test/model")
    success.assert_called_once_with("Offloaded model test/model.")
    info.assert_called_once_with("Daemon is still running. Use `local-llm daemon stop` to stop the process.")


def test_serve_stop_reports_when_no_model_is_loaded():
    client = Mock()
    client.health.return_value = {"loaded_model": None}

    with patch("local_llm.server.DaemonClient", return_value=client), \
         patch("local_llm.server.ui.info") as info:
        serve_stop(port=9090)

    client.health.assert_called_once_with()
    client.evict.assert_not_called()
    info.assert_called_once_with("No warm model loaded on 127.0.0.1:9090.")


def test_daemon_status_explains_process_view():
    client = Mock()
    client.health.return_value = {
        "status": "warm",
        "profile": "m1pro32",
        "loaded_model": "test/model",
        "active_request_id": None,
        "request_timeout_seconds": 300,
    }

    with patch("local_llm.server.DaemonClient", return_value=client), \
         patch("local_llm.server.ui.header") as header, \
         patch("local_llm.server.ui.kv") as kv, \
         patch("local_llm.server.ui.info") as info:
        daemon_status(port=9090)

    header.assert_called_once_with("Daemon Process")
    kv.assert_any_call("Status", "running")
    kv.assert_any_call("Runtime state", "warm")
    kv.assert_any_call("Loaded model", "test/model")
    kv.assert_any_call("Request timeout", "300s")
    info.assert_called_once_with(
        "Use `local-llm serve status` for the loaded-model view, or `local-llm daemon stop` to stop the process."
    )


def test_serve_status_explains_runtime_view():
    client = Mock()
    client.health.return_value = {
        "status": "stopped",
        "profile": "m1pro32",
        "loaded_model": None,
        "session_count": 0,
        "memory_pressure": {"state": "green"},
        "keep_alive_seconds": 1200,
        "request_timeout_seconds": 300,
    }

    with patch("local_llm.server.DaemonClient", return_value=client), \
         patch("local_llm.server.ui.header") as header, \
         patch("local_llm.server.ui.kv") as kv, \
         patch("local_llm.server.ui.info") as info:
        serve_status(port=9090)

    header.assert_called_once_with("Runtime (Loaded Model State)")
    kv.assert_any_call("State", "stopped")
    kv.assert_any_call("Model", "none")
    kv.assert_any_call("Sessions", "0")
    kv.assert_any_call("Request timeout", "300s")
    info.assert_called_once_with(
        "The daemon process is running. Use `local-llm daemon status` for process-level status."
    )
