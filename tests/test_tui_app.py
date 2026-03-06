"""Pilot tests for the redesigned Textual TUI."""

import asyncio
from pathlib import Path
from unittest.mock import patch

from textual.widgets import OptionList

from local_llm.tui.app import CommandPalette


def test_drawer_filters_slash_commands():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._set_composer_text("/stat")
                await pilot.pause()
                drawer = app.query_one("#drawer-list", OptionList)
                prompts = [str(option.prompt) for option in drawer._options]
                assert any("/status" in prompt for prompt in prompts)

    asyncio.run(run())


def test_enter_submits_chat_prompt():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                called = {}

                def fake_handle_chat(prompt: str, **_kwargs) -> None:
                    called["prompt"] = prompt

                app._handle_chat = fake_handle_chat  # type: ignore[method-assign]
                app._set_composer_text("hello")
                await pilot.press("enter")
                await pilot.pause()
                assert called["prompt"] == "hello"

    asyncio.run(run())


def test_arrow_keys_navigate_drawer():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._set_composer_text("/h")
                await pilot.pause()
                drawer = app.query_one("#drawer-list", OptionList)
                assert drawer.highlighted == 0
                await pilot.press("down")
                await pilot.pause()
                assert drawer.highlighted == 1

    asyncio.run(run())


def test_slash_focuses_composer_from_other_widget():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                transcript = app.query_one("#transcript")
                transcript.focus()
                await pilot.pause()
                await pilot.press("/")
                await pilot.pause()
                assert app.focused is app.composer
                assert app.composer.text == "/"

    asyncio.run(run())


def test_models_install_drawer_shows_curated_install_targets():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._set_composer_text("/models install ")
                await pilot.pause()
                drawer = app.query_one("#drawer-list", OptionList)
                prompts = [str(option.prompt) for option in drawer._options]
                assert any("Qwen3.5-9B-OptiQ-4bit" in prompt for prompt in prompts)
                assert any("Paste a Hugging Face repo" in prompt for prompt in prompts)

    asyncio.run(run())


def test_models_install_hint_explains_next_step():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._set_composer_text("/models install ")
                await pilot.pause()
                assert "Type or paste a Hugging Face repo" in app._command_hint_message()

    asyncio.run(run())


def test_models_install_command_context_shows_waiting_for_repo():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._set_composer_text("/models install ")
                await pilot.pause()
                assert "waiting for <repo>" in app._command_context_markup()

    asyncio.run(run())


def test_models_install_requires_explicit_repo():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            app.selected_model = "RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4"
            async with app.run_test() as pilot:
                app._set_composer_text("/models install")
                await pilot.press("enter")
                await pilot.pause()
                assert app._validate_command("/models install") == "Install requires a Hugging Face repo. Paste one or choose a recommended target."
                assert app.composer.text.strip() == "/models install"

    asyncio.run(run())


def test_page_down_scrolls_transcript():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                for index in range(40):
                    app.log_msg(f"line {index}", style="system-msg")
                await pilot.pause()
                transcript = app.query_one("#transcript")
                before = transcript.scroll_y
                await pilot.press("pagedown")
                await pilot.pause()
                assert transcript.scroll_y >= before

    asyncio.run(run())


def test_config_alias_updates_runtime_flag(tmp_path):
    async def run() -> None:
        config_file = tmp_path / "config.json"
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}), \
             patch("local_llm.config.CONFIG_FILE", config_file), \
             patch("local_llm.config.CONFIG_DIR", tmp_path):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._handle_command("/set safe off")
                await pilot.pause()
                assert app.runtime_safe is False

    asyncio.run(run())


def test_history_navigation_uses_previous_entry(tmp_path):
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}), \
             patch("local_llm.tui.app.HistoryStore", wraps=__import__("local_llm.tui.history", fromlist=["HistoryStore"]).HistoryStore):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._set_composer_text("/help")
                app._submit_current_input()
                await pilot.pause()
                app._history_older()
                assert app.composer.text.strip() == "/help"

    asyncio.run(run())


def test_interrupt_cancels_without_quitting():
    async def run() -> None:
        with patch("local_llm.tui.app._cached_list_models", return_value=[]), \
             patch("local_llm.tui.app.discover_custom_commands", return_value={}):
            app = CommandPalette()
            async with app.run_test() as pilot:
                app._generation_inflight = True
                cancelled = {}

                def _cancel() -> None:
                    cancelled["called"] = True

                app._cancel_active_request = _cancel  # type: ignore[method-assign]
                app.action_interrupt()
                await pilot.pause()
                assert cancelled["called"] is True
                assert app.is_running

    asyncio.run(run())
