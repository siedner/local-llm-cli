"""Tests for process inspection helpers."""

from local_llm.runtime import parse_process_info


def test_parse_python_module_server_command():
    info = parse_process_info(
        1234,
        "/usr/bin/python3 -m mlx_lm.server --model org/model --host 127.0.0.1 --port 8080",
    )
    assert info.pid == 1234
    assert info.is_mlx_server is True
    assert info.model == "org/model"
    assert info.host == "127.0.0.1"
    assert info.port == 8080


def test_parse_direct_server_command():
    info = parse_process_info(
        4321,
        "/opt/homebrew/bin/mlx_lm.server --model org/model --port=9090",
    )
    assert info.is_mlx_server is True
    assert info.model == "org/model"
    assert info.port == 9090


def test_parse_non_server_command():
    info = parse_process_info(
        999,
        "/usr/bin/python3 -m http.server 8080",
    )
    assert info.is_mlx_server is False
    assert info.model is None
    assert info.port is None
