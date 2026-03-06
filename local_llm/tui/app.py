"""Main Textual app for local-llm TUI (Redesigned)."""

from __future__ import annotations

import subprocess
import re
import shlex
import time
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll, Horizontal
from textual.reactive import reactive
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option
from rich.markdown import Markdown

from local_llm import __version__
from local_llm.config import (
    get_generation_settings, load_config, save_generation_settings,
)
from local_llm.constants import (
    DEFAULT_SYSTEM_PROMPT, GENERATION_PRESETS,
)
from local_llm.doctor import get_mlx_python
from local_llm.engine import Engine
from local_llm.models import list_models


COMMANDS = {
    "/select": "Select active model for chat/serve",
    "/models": "List, install, or remove models",
    "/serve": "Start, stop, or check local MLX server",
    "/profile": "View or configure hardware profile",
    "/set": "Configure chat parameters (temp, top_p, max_tokens)",
    "/doctor": "Check system environment",
    "/ssh": "Manage SSH tunnels",
    "/guide": "View setup guides",
    "/chat": "Start chat directly from input (default behavior without /)",
    "/clear": "Clear conversation history (model forgets context)",
    "/copy": "Copy last model response to clipboard",
    "/quit": "Quit the application",
}

# Model list cache
_model_cache: list[dict] | None = None
_model_cache_ts: float = 0.0
_MODEL_CACHE_TTL = 30.0  # seconds


def _cached_list_models(**kwargs) -> list[dict]:
    """Return model list with a cache to avoid filesystem scans on every keystroke."""
    global _model_cache, _model_cache_ts
    now = time.time()
    if _model_cache is not None and (now - _model_cache_ts) < _MODEL_CACHE_TTL:
        return _model_cache
    _model_cache = list_models(**kwargs)
    _model_cache_ts = now
    return _model_cache


def _invalidate_model_cache() -> None:
    global _model_cache, _model_cache_ts
    _model_cache = None
    _model_cache_ts = 0.0


class CommandPalette(App):
    """Interactive Command-Palette style TUI for local-llm."""

    TITLE = "local-llm"
    SUB_TITLE = f"v{__version__}"

    CSS = """
    Screen {
        background: transparent;
        layout: vertical;
    }

    #log-area {
        height: 1fr;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        overflow-y: scroll;
    }

    #bottom-bar {
        dock: bottom;
        width: 100%;
        height: auto;
        background: transparent;
    }

    #input-container {
        width: 100%;
        height: 1;
        padding: 0;
        margin: 0;
    }

    #prompt-icon {
        color: #55ff55;
        text-style: bold;
        width: 3;
        content-align: center middle;
    }

    #command-input {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }

    #command-input:focus {
        border: none;
    }

    #dropdown-container {
        display: none;
        height: auto;
        max-height: 10;
        margin: 0 0 1 3;
        background: transparent;
        border: none;
    }
    
    #dropdown-container.-visible {
        display: block;
    }

    #command-dropdown {
        width: 100%;
        height: auto;
        max-height: 9;
        background: transparent;
        border: none;
        padding: 0;
        scrollbar-size: 0 0;
    }
    
    #command-dropdown:focus > .option-list--option-highlighted {
        background: #333333;
        text-style: bold;
    }
    
    #command-dropdown:focus {
        border: none;
    }

    #activity-indicator {
        dock: bottom;
        margin-bottom: 1;
        content-align: left middle;
        width: 100%;
        height: 1;
        padding: 0 1;
        color: #55ff55;
        text-style: bold;
        layer: overlay;
        display: none;
        background: transparent;
    }
    
    #activity-indicator.-active {
        display: block;
    }

    .system-msg {
        color: #888888;
        text-style: italic;
    }
    .user-msg {
        color: #ffffff;
        text-style: bold;
    }
    .error-msg {
        color: #ff5555;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("ctrl+y", "copy_last", "Copy"),
        Binding("escape", "dismiss", "Dismiss", show=False),
        Binding("tab", "cycle_focus", "Focus", show=False),
        Binding("down", "focus_dropdown", "Options", show=False),
        Binding("up", "focus_log", "Log", show=False),
    ]

    selected_model: reactive[str] = reactive("")
    _last_response: str = ""

    # Conversation history (OpenAI messages format)
    _chat_history: list[dict] = []

    # Generation parameters (loaded from config on mount)
    gen_temp: float = 0.7
    gen_top_p: float = 0.9
    gen_max_tokens: int = 2048
    gen_preset: str = "balanced"
    gen_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Engine instance (created on mount)
    _engine: Engine | None = None

    # Spinner state
    _spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _spinner_idx = 0
    _spinner_timer = None

    def start_thinking(self) -> None:
        indicator = self.query_one("#activity-indicator", Static)
        indicator.add_class("-active")
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.1, self._update_spinner)

    def stop_thinking(self) -> None:
        indicator = self.query_one("#activity-indicator", Static)
        indicator.remove_class("-active")
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
            self._spinner_timer = None

    def _update_spinner(self) -> None:
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self.query_one("#activity-indicator", Static).update(self._spinner_frames[self._spinner_idx])

    def compose(self) -> ComposeResult:
        yield Static("", id="activity-indicator")
        with VerticalScroll(id="log-area"):
             pass
        
        from textual.containers import Vertical
        with Vertical(id="bottom-bar"):
            with Horizontal(id="input-container"):
                yield Static("❯", id="prompt-icon")
                yield Input(placeholder="Ask a question or type '/' for commands...", id="command-input")
                
            with Container(id="dropdown-container"):
                yield OptionList(id="command-dropdown")

    def on_mount(self) -> None:
        self._engine = Engine()
        self._chat_history = []

        # Load persisted generation settings
        gs = get_generation_settings()
        self.gen_temp = gs["temp"]
        self.gen_top_p = gs["top_p"]
        self.gen_max_tokens = gs["max_tokens"]
        self.gen_preset = gs.get("preset", "balanced")
        self.gen_system_prompt = gs.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

        self.query_one("#command-input").focus()
        self.log_msg(f"[bold]local-llm v{__version__}[/] initialized. Type [bold]/[/] for commands.", style="system-msg")
        self.log_msg(
            "[dim]Ctrl+Y copy last response · Ctrl+L clear log · /clear reset history · Tab cycle focus[/]",
            style="system-msg",
        )

        # Recover an already-running server
        running = self._engine.get_running_model()
        if running:
            self.selected_model = running
            self.log_msg(f"Recovered running server: [bold]{running}[/]", style="system-msg")
            return

        # Auto-select a model
        config = load_config()
        favs = config.get("favorite_models", [])
        if favs:
            self.selected_model = favs[0]
            self.log_msg(f"Auto-selected favorite model: [bold]{self.selected_model}[/]", style="system-msg")
        else:
            models = _cached_list_models()
            if models:
                self.selected_model = models[0]['repo']
                self.log_msg(f"Auto-selected first available model: [bold]{self.selected_model}[/]", style="system-msg")

    def log_msg(self, msg: str, style: str = "") -> None:
        log = self.query_one("#log-area", VerticalScroll)
        # Use classes for coloring; only escape if we're not explicitly intending rich tags like [bold]
        # But for general log_msg, it may contain [bold], so we leave it as is.
        widget = Static(msg, classes=style)
        log.mount(widget)
        log.scroll_end(animate=False)

    def action_clear_log(self) -> None:
        self.query_one("#log-area", VerticalScroll).remove_children()

    def action_copy_last(self) -> None:
        """Copy the last model response to the system clipboard."""
        if not self._last_response:
            self.log_msg("Nothing to copy yet.", style="system-msg")
            return
        try:
            subprocess.run(["pbcopy"], input=self._last_response.encode(), check=True)
            self.log_msg("Copied last response to clipboard.", style="system-msg")
        except Exception as e:
            self.log_msg(f"Copy failed: {e}", style="error-msg")

    def action_dismiss(self) -> None:
        """Dismiss the dropdown and refocus input."""
        self.query_one("#dropdown-container").remove_class("-visible")
        self.query_one("#command-input", Input).focus()

    def action_cycle_focus(self) -> None:
        """Cycle focus: input → dropdown (if visible) → log → input."""
        focused = self.focused
        inp = self.query_one("#command-input", Input)
        dropdown = self.query_one("#command-dropdown", OptionList)
        log_area = self.query_one("#log-area", VerticalScroll)
        container = self.query_one("#dropdown-container")

        if focused == inp:
            if container.has_class("-visible"):
                dropdown.focus()
            else:
                log_area.focus()
        elif focused == dropdown:
            log_area.focus()
        else:
            inp.focus()

    def action_focus_dropdown(self) -> None:
        dropdown = self.query_one("#command-dropdown", OptionList)
        container = self.query_one("#dropdown-container")
        if container.has_class("-visible") and dropdown.option_count > 0:
            dropdown.focus()
            if dropdown.highlighted is None:
                dropdown.highlighted = 0
            
    def action_focus_log(self) -> None:
        self.query_one("#log-area").focus()

    # --- Dropdown Logic ---

    def _populate_dropdown(self, filter_text: str = "") -> None:
        dropdown = self.query_one("#command-dropdown", OptionList)
        dropdown.clear_options()
        
        search = filter_text.lower().lstrip()
        parts = search.split(" ", 1)
        base_cmd = parts[0]
        
        if base_cmd == "/models" and len(parts) > 1:
            sub = parts[1].lstrip()
            actions = {
                "list": "Show installed models",
                "install": "Download a model",
                "remove": "Delete a model",
                "scan": "Scan system for all LLM weights",
            }
            action_match = next((a for a in actions if sub.startswith(a + " ") or sub == a), None)
            
            if action_match in ["install", "remove"]:
                from local_llm.constants import RECOMMENDED_MODELS
                term = sub[len(action_match):].strip()
                options = list_models(disk=True, filter_relevant=False) if action_match != "install" else RECOMMENDED_MODELS
                added = 0
                for item in options:
                    repo = item["repo"]
                    if not term or term in repo.lower():
                        if action_match == "install":
                            display = f"[cyan]{repo}[/cyan] [dim italic]{item.get('estimated_size', 'unknown')}[/] [dim]- {item.get('description', '')}[/]"
                        else:
                            display = f"[cyan]{repo}[/cyan] [dim italic]{item.get('disk_usage', 'unknown')}[/]"
                        dropdown.add_option(Option(display, id=f"/models {action_match} {repo}"))
                        added += 1
                if added == 0:
                     dropdown.add_option(Option(f"Press enter to {action_match}{' ' + term if term else ''}", id=f"/models {action_match} {term}"))
            else:
                added = 0
                for a, desc in actions.items():
                    if not sub or a.startswith(sub):
                        dropdown.add_option(Option(f"[white]{a}[/white] [dim]- {desc}[/dim]", id=f"/models {a} "))
                        added += 1
                if added == 0:
                     dropdown.add_option(Option(f"Press enter to execute: [white]/models {sub}[/white]", id=f"/models {sub}"))
                     
        elif base_cmd == "/select" and len(parts) > 1:
            sub = parts[1].lstrip()
            options = _cached_list_models(disk=True, filter_relevant=False)
            added = 0
            for item in options:
                repo = item["repo"]
                if not sub or sub in repo.lower():
                    dropdown.add_option(Option(f"[cyan]{repo}[/cyan] [dim italic]{item.get('disk_usage', 'unknown')}[/]", id=f"/select {repo}"))
                    added += 1
            if added == 0:
                dropdown.add_option(Option(f"Press enter to select: [cyan]{sub}[/cyan]", id=f"/select {sub}"))
                     
        elif base_cmd == "/serve" and len(parts) > 1:
            sub = parts[1].lstrip()
            added = 0
            for action in ["start", "stop", "status"]:
                if not sub or action.startswith(sub):
                    dropdown.add_option(Option(action, id=f"/serve {action} "))
                    added += 1
            if added == 0:
                dropdown.add_option(Option(f"Press enter to execute: /serve {sub}", id=f"/serve {sub}"))
                    
        elif base_cmd == "/profile" and len(parts) > 1:
            sub = parts[1].lstrip()
            added = 0
            for action in ["auto", "current", "list", "set"]:
                if not sub or action.startswith(sub):
                    dropdown.add_option(Option(action, id=f"/profile {action} "))
                    added += 1
            if added == 0:
                dropdown.add_option(Option(f"Press enter to execute: /profile {sub}", id=f"/profile {sub}"))
                    
        elif base_cmd == "/set" and len(parts) > 1:
            sub = parts[1].lstrip()
            sub_parts = sub.split(" ", 1)
            sub_cmd = sub_parts[0]

            # Show presets if typing "/set preset"
            if sub_cmd == "preset" or "preset".startswith(sub_cmd):
                if len(sub_parts) > 1:
                    filter_text = sub_parts[1].strip()
                else:
                    filter_text = ""
                added = 0
                for name, info in GENERATION_PRESETS.items():
                    if not filter_text or filter_text in name:
                        marker = " ✓" if name == self.gen_preset else ""
                        dropdown.add_option(Option(
                            f"[cyan]{name}[/cyan] [dim]— {info['description']}[/dim]{marker}",
                            id=f"/set preset {name}"
                        ))
                        added += 1
                if added == 0:
                    dropdown.add_option(Option(f"Unknown preset: {filter_text}", id=f"/set preset {filter_text}"))
            else:
                added = 0
                settings_list = [
                    ("preset", self.gen_preset),
                    ("temp", self.gen_temp),
                    ("top_p", self.gen_top_p),
                    ("max_tokens", self.gen_max_tokens),
                    ("system_prompt", self.gen_system_prompt[:40] + ("…" if len(self.gen_system_prompt) > 40 else "")),
                    ("reset", "restore defaults"),
                ]
                for k, cur in settings_list:
                    if not sub or k.startswith(sub):
                        dropdown.add_option(Option(f"[white]{k}[/white] [dim](current: {cur})[/dim]", id=f"/set {k} "))
                        added += 1
                if added == 0:
                    dropdown.add_option(Option(f"Press enter to execute: [white]/set {sub}[/white]", id=f"/set {sub}"))
                
        elif base_cmd == "/ssh" and len(parts) > 1:
            sub = parts[1].lstrip()
            added = 0
            for action in ["status", "stop", "tunnel", "snippet"]:
                if not sub or action.startswith(sub):
                    dropdown.add_option(Option(action, id=f"/ssh {action} "))
                    added += 1
            if added == 0:
                dropdown.add_option(Option(f"Press enter to execute: /ssh {sub}", id=f"/ssh {sub}"))
                    
        elif base_cmd == "/guide" and len(parts) > 1:
            sub = parts[1].lstrip()
            if not sub or "opencode".startswith(sub):
                 dropdown.add_option(Option("opencode", id="/guide opencode "))
            else:
                 dropdown.add_option(Option(f"Press enter to execute: /guide {sub}", id=f"/guide {sub}"))
                
        else:
            # Base command autocomplete
            added = 0
            for cmd, desc in COMMANDS.items():
                if cmd.lower().startswith(search) or (search and search in desc.lower()):
                    dropdown.add_option(Option(f"[white]{cmd}[/white] [dim]- {desc}[/dim]", id=cmd + " "))
                    added += 1
            if added == 0 and search.startswith("/"):
                dropdown.add_option(Option(f"Press enter to execute: [white]{search}[/white]", id=search))

        dropdown.highlighted = None

    def on_input_changed(self, event: Input.Changed) -> None:
        val = event.value
        container = self.query_one("#dropdown-container")
        
        if val.startswith("/"):
            container.add_class("-visible")
            self._populate_dropdown(val)
        else:
            container.remove_class("-visible")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        val = event.value.strip()
        if not val:
            return
            
        container = self.query_one("#dropdown-container")
        dropdown = self.query_one("#command-dropdown", OptionList)
        
        # Quick submit if dropdown visible and input matches exactly, or nothing typed, etc.
        # It's cleaner to read the input value directly here.
        event.input.value = ""
        container.remove_class("-visible")
        
        if val.startswith("/"):
            self._handle_command(val)
        else:
            self.log_msg(f"You: {val}", style="user-msg")
            self._handle_chat(val)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        command = str(event.option.id)
        inp = self.query_one("#command-input", Input)
        container = self.query_one("#dropdown-container")
        
        if command.startswith("/models install ") or command.startswith("/models remove ") or command.startswith("/serve ") or command.startswith("/set ") or command.startswith("/select "):
            inp.value = command
            inp.cursor_position = len(inp.value)
            container.add_class("-visible")
            self._populate_dropdown(command)
            inp.focus()
        elif command.startswith("/models") and command.strip() == "/models":
            inp.value = "/models "
            inp.cursor_position = len(inp.value)
            self._populate_dropdown("/models ")
            inp.focus()
        else:
             inp.value = ""
             container.remove_class("-visible")
             self._handle_command(command)
             inp.focus()

    # --- Command Routing ---

    def _handle_command(self, cmd_string: str) -> None:
        try:
            parts = shlex.split(cmd_string)
        except ValueError:
            parts = [cmd_string]
            
        cmd = parts[0]
        args = parts[1:]
        
        if cmd in ["/quit", "/q", "exit", "quit" ]:
            self.exit()
            
        elif cmd == "/doctor":
            self.log_msg("Running Doctor checks...", style="system-msg")
            self._run_cli_command(["doctor"] + args)
            
        elif cmd == "/select":
            if not args:
                self.log_msg("Usage: /select <repo>", style="error-msg")
                return
            repo = args[0]
            self.selected_model = repo
            self._chat_history = []  # reset history on model change
            self.log_msg(f"Selected model: [bold]{repo}[/]", style="system-msg")
            self._ensure_server_for_model(repo)
            
        elif cmd == "/serve":
            action = args[0] if args else "start"
            flags = args[1:] if args else []
            
            if action == "stop":
                self._run_cli_command(["serve", "stop"] + flags)
            elif action == "status":
                self._run_cli_command(["serve", "status"] + flags)
            else:
                if not self.selected_model:
                    self.log_msg("Error: Select a model first using /models before starting the server", style="error-msg")
                    return
                # Default to detached unless they provide --no-detach
                detached_flags = []
                if "--no-detach" in flags:
                    flags.remove("--no-detach")
                elif "--detach" not in flags:
                    detached_flags.append("--detach")
                    
                self.log_msg(f"Starting server for {self.selected_model}...", style="system-msg")
                self._run_cli_command(["serve", "start", self.selected_model] + detached_flags + flags)
            
        elif cmd == "/models":
            action = args[0] if args else "list"
            flags = args[1:] if args else []
            
            if action == "list":
                _invalidate_model_cache()  # force fresh scan
                self.log_msg("Installed models:", style="system-msg")
                self._run_cli_command(["models", "list"] + flags)
            elif action == "scan":
                self._run_cli_command(["models", "scan"])
            elif action == "install":
                if not flags:
                    self.log_msg("Usage: /models install <repo>", style="error-msg")
                    return
                repo = flags[0]
                _invalidate_model_cache()  # new model coming
                self.log_msg(f"Installing {repo}...", style="system-msg")
                self._run_cli_command(["models", "install", repo, "-y"])
            elif action == "remove":
                if not flags:
                    self.log_msg("Usage: /models remove <repo>", style="error-msg")
                    return
                repo = flags[0]
                _invalidate_model_cache()  # model leaving
                self._run_cli_command(["models", "remove", repo, "-y"])
            else:
                 self.log_msg(f"Unknown /models subcommand: {action}", style="error-msg")

        elif cmd == "/profile":
            action = args[0] if args else "current"
            flags = args[1:] if args else []
            self._run_cli_command(["profile", action] + flags)
            
        elif cmd == "/set":
            if not args:
                # Show settings summary
                self._show_settings_summary()
                return
            param = args[0].lower()
            val = " ".join(args[1:]) if len(args) > 1 else ""

            if param == "preset":
                if val not in GENERATION_PRESETS:
                    names = ", ".join(GENERATION_PRESETS.keys())
                    self.log_msg(f"Unknown preset '{val}'. Available: {names}", style="error-msg")
                    return
                p = GENERATION_PRESETS[val]
                self.gen_preset = val
                self.gen_temp = p["temp"]
                self.gen_top_p = p["top_p"]
                self.gen_max_tokens = p["max_tokens"]
                self._persist_gen_settings()
                self.log_msg(
                    f"Applied preset [bold]{val}[/]: temp={p['temp']}  top_p={p['top_p']}  max_tokens={p['max_tokens']}",
                    style="system-msg",
                )

            elif param == "reset":
                from local_llm.config import _default_generation_settings
                defaults = _default_generation_settings()
                self.gen_preset = defaults["preset"]
                self.gen_temp = defaults["temp"]
                self.gen_top_p = defaults["top_p"]
                self.gen_max_tokens = defaults["max_tokens"]
                self.gen_system_prompt = defaults["system_prompt"]
                self._persist_gen_settings()
                self.log_msg("Settings reset to defaults.", style="system-msg")
                self._show_settings_summary()

            elif param == "system_prompt":
                if not val:
                    self.log_msg(f"Current system prompt: {self.gen_system_prompt}", style="system-msg")
                    return
                self.gen_system_prompt = val
                self._persist_gen_settings()
                self.log_msg(f"System prompt updated.", style="system-msg")

            else:
                if not val:
                    self.log_msg(f"Usage: /set {param} <value>", style="error-msg")
                    return
                try:
                    if param == "temp":
                        self.gen_temp = float(val)
                    elif param == "top_p":
                        self.gen_top_p = float(val)
                    elif param == "max_tokens":
                        self.gen_max_tokens = int(val)
                    else:
                        self.log_msg(f"Unknown parameter '{param}'. Try: /set", style="error-msg")
                        return
                    self.gen_preset = "custom"
                    self._persist_gen_settings()
                    self.log_msg(f"Set [bold]{param}[/] to {val}  (preset → custom)", style="system-msg")
                except ValueError:
                    self.log_msg(f"Invalid value for {param}: {val}", style="error-msg")
            
        elif cmd == "/chat":
            if not self.selected_model:
                self.log_msg("Error: Select a model first using /models", style="error-msg")
                return
            self.log_msg(f"Launching interactive chat with {self.selected_model}...", style="system-msg")
            # Defer execution to let UI paint first
            self.set_timer(0.1, self._launch_interactive_chat)
            
        elif cmd == "/guide":
            action = args[0] if args else "opencode"
            self._run_cli_command(["guide", action])
            
        elif cmd == "/ssh":
            action = args[0] if args else "status"
            flags = args[1:] if args else []
            self._run_cli_command(["ssh", action] + flags)

        elif cmd == "/copy":
            self.action_copy_last()

        elif cmd == "/clear":
            self._chat_history = []
            self.log_msg("Conversation history cleared.", style="system-msg")

        else:
            self.log_msg(f"Unknown command: {cmd}", style="error-msg")


    def _show_settings_summary(self) -> None:
        """Display a formatted table of all current generation settings."""
        lines = [
            f"  [bold]preset[/]         {self.gen_preset}",
            f"  [bold]temp[/]           {self.gen_temp}",
            f"  [bold]top_p[/]          {self.gen_top_p}",
            f"  [bold]max_tokens[/]     {self.gen_max_tokens}",
            f"  [bold]system_prompt[/]  {self.gen_system_prompt}",
            "",
            "  [dim]Presets:[/dim]",
        ]
        for name, info in GENERATION_PRESETS.items():
            marker = " [bold green]✓[/]" if name == self.gen_preset else ""
            lines.append(
                f"    [cyan]{name:10}[/cyan] temp={info['temp']}  top_p={info['top_p']}  "
                f"max_tokens={info['max_tokens']}  [dim]{info['description']}[/dim]{marker}"
            )
        lines.append("")
        lines.append("  [dim]Usage: /set preset <name> | /set <param> <value> | /set reset[/dim]")
        self.log_msg("\n".join(lines), style="system-msg")

    def _persist_gen_settings(self) -> None:
        """Save current generation settings to config."""
        save_generation_settings({
            "preset": self.gen_preset,
            "temp": self.gen_temp,
            "top_p": self.gen_top_p,
            "max_tokens": self.gen_max_tokens,
            "system_prompt": self.gen_system_prompt,
        })

    @work(thread=True)
    def _run_cli_command(self, args: list[str]) -> None:
        import sys
        from rich.markup import escape
        
        self.app.call_from_thread(self.start_thinking)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "local_llm.cli"] + args,
                capture_output=True,
                text=True,
            )
            
            if result.stdout:
                # Basic cleanup of rich CLI formatting markers that break RichLog occasionally
                out = result.stdout
                self.app.call_from_thread(self._append_to_log, escape(out))
            if result.stderr:
                self.app.call_from_thread(self._append_to_log, escape(result.stderr), "error-msg")
                
        except Exception as e:
            self.app.call_from_thread(self._append_to_log, str(e), "error-msg")
        finally:
            self.app.call_from_thread(self.stop_thinking)

    def _append_to_log(self, text: str, style: str = "") -> None:
        self.log_msg(text, style)

    def _launch_interactive_chat(self) -> None:
        python = get_mlx_python()
        with self.suspend():
            subprocess.run([python, "-m", "mlx_lm", "chat", "--model", self.selected_model])
        self.log_msg("Chat session ended.", style="system-msg")

    def _create_stream_widget(self, widget_id: str, style: str = "") -> None:
        log = self.query_one("#log-area", VerticalScroll)
        # Disable markup so Model tokens like [ or ] don't crash Textual rendering
        widget = Static("", id=widget_id, classes=style, markup=False)
        log.mount(widget)
        log.scroll_end(animate=False)

    def _update_stream_widget(self, widget_id: str, full_text: str, is_final: bool = False) -> None:
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            if is_final:
                widget.update(Markdown(full_text))
            else:
                widget.update(full_text)
            self.query_one("#log-area", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def _restyle_stream_widget(self, widget_id: str, style: str) -> None:
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.set_classes(style)
        except Exception:
            pass

    @work(thread=True)
    def _ensure_server_for_model(self, model: str) -> None:
        """Start or swap the MLX server for the given model (background worker)."""
        if not self._engine:
            return
        self.app.call_from_thread(self.start_thinking)
        try:
            def on_status(msg: str) -> None:
                self.app.call_from_thread(self._append_to_log, msg, "system-msg")

            self._engine.ensure_server(model, on_status=on_status)
        except Exception as e:
            self.app.call_from_thread(self._append_to_log, f"Server error: {e}", "error-msg")
        finally:
            self.app.call_from_thread(self.stop_thinking)

    @work(thread=True)
    def _handle_chat(self, prompt: str) -> None:
        import uuid
        if not self.selected_model:
            self.app.call_from_thread(self._append_to_log, "Error: Select a model first.", "error-msg")
            return

        if not self._engine:
            self.app.call_from_thread(self._append_to_log, "Engine not initialized.", "error-msg")
            return

        self.app.call_from_thread(self.start_thinking)

        try:
            widget_id = f"stream-{uuid.uuid4().hex}"
            self.app.call_from_thread(self._create_stream_widget, widget_id, "system-msg")

            # Ensure server is running for the selected model
            if self._engine.get_running_model() != self.selected_model:
                self.app.call_from_thread(
                    self._update_stream_widget, widget_id,
                    f"Starting server for {self.selected_model}… (first message takes longer)"
                )
                self._engine.ensure_server(
                    self.selected_model,
                    on_status=lambda msg: self.app.call_from_thread(
                        self._update_stream_widget, widget_id, msg
                    ),
                )

            self.app.call_from_thread(
                self._update_stream_widget, widget_id, "Generating…"
            )

            # Build messages: system prompt + history + new user message
            self._chat_history.append({"role": "user", "content": prompt})
            messages = [
                {"role": "system", "content": self.gen_system_prompt},
                *self._chat_history,
            ]

            # Stream tokens via HTTP
            full_text = ""
            token_count = 0
            for chunk in self._engine.chat_stream(
                messages,
                temperature=self.gen_temp,
                top_p=self.gen_top_p,
                max_tokens=self.gen_max_tokens,
            ):
                full_text += chunk
                token_count += 1

                # Update widget periodically to avoid overwhelming UI
                if token_count % 3 == 0 or chunk in ("\n", ". ", "? ", "! "):
                    display_text = re.sub(
                        r'<think>.*?(</think>|$)', '', full_text, flags=re.DOTALL
                    ).strip()
                    self.app.call_from_thread(
                        self._update_stream_widget, widget_id, display_text
                    )

            # Final render as markdown
            display_text = re.sub(
                r'<think>.*?(</think>|$)', '', full_text, flags=re.DOTALL
            ).strip()
            self._last_response = display_text
            self._chat_history.append({"role": "assistant", "content": full_text})
            self.app.call_from_thread(
                self._update_stream_widget, widget_id, display_text, True
            )

        except Exception as e:
            self.app.call_from_thread(
                self._append_to_log, f"Chat error: {e}", "error-msg"
            )
        finally:
            self.app.call_from_thread(self.stop_thinking)

    def on_unmount(self) -> None:
        """Clean up: stop the MLX server when the TUI exits."""
        if self._engine:
            try:
                self._engine.stop_server()
            except Exception:
                pass

if __name__ == "__main__":
    app = CommandPalette()
    app.run(mouse=False)
