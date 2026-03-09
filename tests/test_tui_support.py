"""Tests for TUI helper modules."""

from pathlib import Path
from unittest.mock import patch

from rich.console import Group

from local_llm.tui.commands import canonical_name, get_command
from local_llm.tui.custom_commands import discover_custom_commands
from local_llm.tui.history import HistoryStore
from local_llm.tui.app import _clean_thinking_text, _render_assistant_text, _split_assistant_sections


def test_command_aliases_resolve_to_canonical():
    assert canonical_name("/set") == "/config"
    assert canonical_name("/select") == "/model"
    assert canonical_name("/ps") == "/status"
    assert get_command("/serve").canonical == "/runtime"


def test_runtime_and_daemon_command_descriptions_explain_the_split():
    runtime = get_command("/runtime")
    daemon = get_command("/daemon")
    status = get_command("/status")

    assert runtime is not None
    assert daemon is not None
    assert status is not None
    assert "loaded model inside the daemon process" in runtime.description
    assert "daemon process" in daemon.description
    assert "loaded-model snapshot" in status.description


def test_discover_custom_commands(tmp_path):
    project_commands = tmp_path / ".local-llm" / "commands"
    user_commands = tmp_path / "user-commands"
    project_commands.mkdir(parents=True)
    user_commands.mkdir(parents=True)

    (project_commands / "summarize.md").write_text(
        "---\n"
        "description: Summarize code\n"
        "argument-hint: <topic>\n"
        "model: test/model\n"
        "safe: true\n"
        "---\n"
        "Summarize this repository for $1.\n"
    )
    (user_commands / "brainstorm.md").write_text("Brainstorm ideas about $ARGUMENTS.\n")

    with patch("local_llm.tui.custom_commands.USER_COMMANDS_DIR", user_commands):
        commands = discover_custom_commands(tmp_path)

    summarize = commands["/summarize"]
    assert summarize.description == "Summarize code"
    assert summarize.model == "test/model"
    assert summarize.safe is True
    assert summarize.expand(["daemon"]) == "Summarize this repository for daemon."

    brainstorm = commands["/brainstorm"]
    assert brainstorm.expand(["mlx", "latency"]) == "Brainstorm ideas about mlx latency."


def test_history_store_persists_and_navigates(tmp_path):
    store = HistoryStore(tmp_path / "workspace", limit=3, root=tmp_path / "history")
    store.push("/help")
    store.push("/status")
    store.push("hello")

    assert store.older("") == "hello"
    assert store.older("") == "/status"
    assert store.newer() == "hello"

    reloaded = HistoryStore(tmp_path / "workspace", limit=3, root=tmp_path / "history")
    assert reloaded.entries == ["/help", "/status", "hello"]


def test_render_assistant_text_preserves_think_blocks_separately():
    renderable = _render_assistant_text("<think>drafting</think>\n\nFinal answer", final=True)
    assert isinstance(renderable, Group)


def test_split_assistant_sections_handles_reasoning_prefix_without_tags():
    thinking, answer = _split_assistant_sections(
        "Thinking Process:\n1. Draft reply\n\nHey there! How can I help?"
    )
    assert thinking.startswith("Thinking Process:")
    assert answer == "Hey there! How can I help?"


def test_clean_thinking_text_strips_reasoning_prefix_and_dangling_tag():
    cleaned = _clean_thinking_text("Thinking Process:\n1. Draft reply\n</think>")
    assert cleaned == "1. Draft reply"
