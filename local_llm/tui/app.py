"""Claude-Code-style Textual app for local-llm."""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from pathlib import Path
import re

from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from local_llm import __version__
from local_llm.config import (
    _default_generation_settings,
    detect_output_size_profile,
    get_effective_profile,
    get_generation_settings,
    get_output_size_profile,
    get_runtime_settings,
    get_session_defaults,
    get_tui_settings,
    load_config,
    normalize_output_size_profile,
    save_config,
    save_generation_settings,
    save_runtime_settings,
    save_tui_settings,
)
from local_llm.constants import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_SYSTEM_PROMPT,
    GENERATION_PRESETS,
    MODEL_METADATA,
    RECOMMENDED_MODELS,
)
from local_llm.engine import Engine
from local_llm.models import list_models
from local_llm.tui.commands import CommandSpec, canonical_name, get_command, iter_commands
from local_llm.tui.custom_commands import CustomCommand, discover_custom_commands
from local_llm.tui.history import HistoryStore


_model_cache: dict[tuple[tuple[str, object], ...], tuple[float, list[dict]]] = {}
_MODEL_CACHE_TTL = 30.0
STATUSLINE_FIELDS = ("state", "model", "profile", "session", "safe", "memory", "queue", "warm")
THINK_RE = re.compile(r"<think>(.*?)(</think>|$)", re.DOTALL)
REASONING_PREFIXES = (
    "Thinking Process:",
    "Thought Process:",
    "Reasoning:",
    "Analysis:",
)


def _cached_list_models(**kwargs) -> list[dict]:
    global _model_cache
    key = tuple(sorted(kwargs.items()))
    now = time.time()
    cached = _model_cache.get(key)
    if cached and (now - cached[0]) < _MODEL_CACHE_TTL:
        return cached[1]
    models = list_models(**kwargs)
    _model_cache[key] = (now, models)
    return models


def _invalidate_model_cache() -> None:
    global _model_cache
    _model_cache = {}


def _split_assistant_sections(text: str) -> tuple[str, str]:
    """Split assistant text into reasoning and final answer sections."""
    text = text.strip()
    if not text:
        return "", ""

    explicit = THINK_RE.search(text)
    if explicit:
        thinking = explicit.group(1).strip()
        answer = (text[:explicit.start()] + text[explicit.end():]).strip()
        return thinking, answer

    stripped = text.lstrip()
    if stripped.startswith(REASONING_PREFIXES):
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", stripped) if chunk.strip()]
        if len(chunks) >= 2:
            return "\n\n".join(chunks[:-1]), chunks[-1]
        return stripped, ""

    return "", text


def _clean_thinking_text(text: str) -> str:
    """Normalize reasoning text before rendering."""
    text = text.replace("<think>", "").replace("</think>", "").strip()
    for prefix in REASONING_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
            break
    return text.strip()


def _render_assistant_text(text: str, *, final: bool) -> Group | Markdown | Text:
    """Render assistant output with a distinct muted block for reasoning sections."""
    thinking, answer = _split_assistant_sections(text)
    parts: list[object] = []
    if thinking:
        thinking = _clean_thinking_text(thinking)
        parts.append(
            Panel(
                Markdown(thinking) if final else Text(thinking, style="italic #7d8590"),
                title="Thinking",
                title_align="left",
                style="dim italic",
                border_style="#30363d",
                padding=(0, 1),
            )
        )
    if answer:
        parts.append(Markdown(answer) if final else Text(answer))

    if not parts:
        return Text("")
    if len(parts) == 1:
        return parts[0]
    return Group(*parts)


class Composer(TextArea):
    """Multiline composer with explicit submit/navigation actions."""

    BINDINGS = [
        Binding("enter", "submit", "Submit", show=False),
        Binding("ctrl+j", "newline", "Newline", show=False),
        Binding("alt+enter", "newline", "Newline", show=False),
        Binding("up", "history_up", "History up", show=False),
        Binding("down", "history_down", "History down", show=False),
        Binding("escape", "escape_key", "Escape", show=False),
    ]

    def action_submit(self) -> None:
        self.app.composer_submit()

    def action_newline(self) -> None:
        self.app.composer_newline()

    def action_history_up(self) -> None:
        if not self.app.composer_history_up():
            self.action_cursor_up()

    def action_history_down(self) -> None:
        if not self.app.composer_history_down():
            self.action_cursor_down()

    def action_escape_key(self) -> None:
        self.app.composer_escape()

    def on_key(self, event: Key) -> None:
        if event.key == "enter":
            self.action_submit()
            event.stop()
        elif event.key in {"ctrl+j", "alt+enter"}:
            self.action_newline()
            event.stop()
        elif event.key == "up":
            self.action_history_up()
            event.stop()
        elif event.key == "down":
            self.action_history_down()
            event.stop()
        elif event.key == "escape":
            self.action_escape_key()
            event.stop()


class CommandPalette(App):
    """Keyboard-first TUI for managing and chatting with local-llm."""

    TITLE = "local-llm"
    SUB_TITLE = f"v{__version__}"

    CSS = """
    Screen {
        layout: vertical;
        background: #081123;
        color: #f5f7fa;
    }

    #transcript {
        height: 1fr;
        padding: 1 2;
        background: transparent;
    }

    #drawer {
        height: 12;
        display: none;
        border-top: solid #22314d;
        background: #0d1830;
    }

    #drawer.-visible {
        display: block;
    }

    #drawer-list {
        width: 45%;
        height: 100%;
        border-right: solid #22314d;
        background: transparent;
    }

    #drawer-details {
        width: 55%;
        padding: 1 2;
        color: #c4d0e3;
    }

    #composer-shell {
        height: auto;
        border-top: solid #22314d;
        padding: 0 1 0 1;
        background: #0b1528;
    }

    #composer-header {
        height: auto;
        padding: 0 1;
        color: #8ba0c2;
    }

    #command-context {
        display: none;
        height: auto;
        padding: 0 1;
        color: #c4d0e3;
    }

    #command-context.-active {
        display: block;
    }

    #composer-mode {
        color: #58a6ff;
        text-style: bold;
    }

    #composer {
        height: auto;
        min-height: 1;
        max-height: 8;
        border: round #22314d;
        background: #09101f;
    }

    #composer.-disabled {
        border: round #161b22;
        color: #484f58;
    }

    #composer-hints {
        padding: 0 1;
        color: #8ba0c2;
        height: auto;
    }

    #activity {
        padding: 0 1;
        color: #58a6ff;
        text-style: bold;
        display: none;
    }

    #activity.-active {
        display: block;
    }

    #statusline {
        height: 1;
        padding: 0 1;
        background: #060d19;
        color: #9eb0cc;
    }

    .system-msg {
        color: #98a8c4;
    }

    .user-msg {
        color: #f5f7fa;
        text-style: bold;
    }

    .error-msg {
        color: #ff7b72;
        text-style: bold;
    }

    .summary-msg {
        color: #8ba0c2;
    }

    .assistant-label {
        color: #8ba0c2;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("/", "focus_composer_command", "Command", show=False),
        Binding("ctrl+d", "quit", "Quit"),
        Binding("ctrl+c", "interrupt", "Cancel"),
        Binding("ctrl+l", "clear_transcript", "Clear"),
        Binding("ctrl+y", "copy_last", "Copy"),
        Binding("ctrl+r", "refresh_runtime", "Refresh"),
        Binding("pageup", "transcript_page_up", "Scroll Up", show=False),
        Binding("pagedown", "transcript_page_down", "Scroll Down", show=False),
        Binding("home", "transcript_home", "Top", show=False),
        Binding("end", "transcript_end", "Bottom", show=False),
        Binding("ctrl+up", "transcript_line_up", "Line Up", show=False),
        Binding("ctrl+down", "transcript_line_down", "Line Down", show=False),
        Binding("tab", "focus_next", "Focus", show=False),
        Binding("shift+tab", "focus_previous", "Focus", show=False),
    ]

    selected_model: reactive[str] = reactive("")
    _last_response: str = ""

    def __init__(self) -> None:
        super().__init__()
        self._engine: Engine | None = None
        self._chat_history: list[dict] = []
        self._session_id = ""
        self._generation_inflight = False
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        self._spinner_timer = None
        self._thinking_label = "Working"
        self._drawer_entries: dict[str, dict] = {}
        self._runtime_snapshot: dict = {}
        self._snapshot_refresh_inflight = False
        self._last_escape_at = 0.0
        self._last_user_prompt = ""
        self._history: HistoryStore | None = None
        self._custom_commands: dict[str, CustomCommand] = {}
        self._tui_settings: dict = {}

        self.gen_temp = 0.7
        self.gen_top_p = 0.9
        self.gen_top_k = 50
        self.gen_max_tokens = 1024
        self.gen_preset = "balanced"
        self.gen_system_prompt = DEFAULT_SYSTEM_PROMPT
        self.runtime_max_context: int | None = None
        self.runtime_keep_alive: int | None = None
        self.runtime_request_timeout = DEFAULT_REQUEST_TIMEOUT_SECONDS
        self.runtime_safe = True
        self._profile_name = "auto"

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="transcript"):
            pass
        with Horizontal(id="drawer"):
            yield OptionList(id="drawer-list")
            yield Static("", id="drawer-details")
        with Vertical(id="composer-shell"):
            yield Static("", id="composer-header")
            yield Static("", id="command-context")
            yield Composer("", id="composer")
            yield Static("", id="composer-hints")
            yield Static("", id="activity")
        yield Static("", id="statusline")

    def on_mount(self) -> None:
        self._engine = Engine()
        self._session_id = f"tui-{uuid.uuid4().hex[:8]}"
        self._chat_history = []

        config = load_config()
        generation = get_generation_settings(config)
        runtime = get_runtime_settings(config)
        session_defaults = get_session_defaults(config)
        self._tui_settings = get_tui_settings(config)
        self._profile_name, _profile = get_effective_profile(config)

        self.gen_temp = generation["temp"]
        self.gen_top_p = generation["top_p"]
        self.gen_top_k = generation["top_k"]
        self.gen_max_tokens = generation["max_tokens"]
        self.gen_preset = generation["preset"]
        self.gen_system_prompt = generation["system_prompt"]
        self.runtime_max_context = session_defaults.get("max_context")
        self.runtime_keep_alive = session_defaults.get("keep_alive_seconds")
        self.runtime_request_timeout = int(runtime.get("request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS))
        self.runtime_safe = bool(runtime.get("safe_mode", True))

        self._history = HistoryStore(Path.cwd(), limit=int(self._tui_settings.get("history_size", 100)))
        self._custom_commands = discover_custom_commands()

        self.query_one("#composer", Composer).focus()
        self.log_msg(
            f"[bold]local-llm v{__version__}[/] ready — type [bold]/[/] for commands · [dim]Ctrl+C cancel · Ctrl+D quit · /help keys[/dim]",
            style="system-msg",
        )
        self.log_msg("[dim]─────────────────────────────────────────[/dim]", style="system-msg")

        running = self._engine.get_running_model() if self._engine else None
        if running:
            self.selected_model = running
            self.log_msg(f"Recovered warm runtime: [bold]{running}[/]", style="system-msg")
        else:
            favorites = config.get("favorite_models", [])
            if favorites:
                self.selected_model = favorites[0]
                self.log_msg(f"Auto-selected favorite model: [bold]{self.selected_model}[/]", style="system-msg")
            else:
                models = _cached_list_models()
                if models:
                    self.selected_model = models[0]["repo"]
                    self.log_msg(f"Auto-selected first available model: [bold]{self.selected_model}[/]", style="system-msg")

        self._render_composer_header()
        self._render_command_context()
        self._render_composer_hints()
        self._render_statusline()
        self.set_interval(3.0, self._queue_runtime_refresh)
        self._queue_runtime_refresh()

    def log_msg(self, message: str, style: str = "") -> None:
        log = self.query_one("#transcript", VerticalScroll)
        widget = Static(message, classes=style)
        log.mount(widget)
        log.scroll_end(animate=False)

    def _log_user_msg(self, text: str) -> None:
        log = self.query_one("#transcript", VerticalScroll)
        panel = Static(
            Panel(
                Text(text),
                title="[bold]You[/bold]",
                title_align="left",
                border_style="#58a6ff",
                padding=(0, 1),
            )
        )
        log.mount(panel)
        log.scroll_end(animate=False)

    def action_clear_transcript(self) -> None:
        self.query_one("#transcript", VerticalScroll).remove_children()
        self.log_msg("Transcript cleared.", style="system-msg")

    def action_copy_last(self) -> None:
        if not self._last_response:
            self.log_msg("Nothing to copy yet.", style="system-msg")
            return
        try:
            subprocess.run(["pbcopy"], input=self._last_response.encode(), check=True)
            self.notify("Copied to clipboard", severity="information", timeout=2)
        except Exception as exc:
            self.notify(f"Copy failed: {exc}", severity="error", timeout=3)

    def action_interrupt(self) -> None:
        if not self._generation_inflight:
            self.log_msg("No active generation to cancel.", style="system-msg")
            return
        self.log_msg("Cancelling active generation…", style="system-msg")
        self._cancel_active_request()

    def action_refresh_runtime(self) -> None:
        self._queue_runtime_refresh()

    def action_transcript_page_up(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_page_up(animate=False)

    def action_transcript_page_down(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_page_down(animate=False)

    def action_transcript_home(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_home(animate=False)

    def action_transcript_end(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def action_transcript_line_up(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_up(animate=False)

    def action_transcript_line_down(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_down(animate=False)

    def start_thinking(self, label: str = "Working") -> None:
        self._thinking_label = label
        activity = self.query_one("#activity", Static)
        activity.add_class("-active")
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.1, self._update_spinner)

    def stop_thinking(self) -> None:
        activity = self.query_one("#activity", Static)
        activity.remove_class("-active")
        if self._spinner_timer is not None:
            self._spinner_timer.pause()
            self._spinner_timer = None
        self._thinking_label = "Working"

    def _update_spinner(self) -> None:
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
        self.query_one("#activity", Static).update(f"{self._spinner_frames[self._spinner_idx]} {self._thinking_label}…")

    @property
    def composer(self) -> Composer:
        return self.query_one("#composer", Composer)

    def _composer_text(self) -> str:
        return self.composer.text

    def _set_composer_text(self, value: str) -> None:
        self.composer.load_text(value)
        lines = value.splitlines() or [""]
        self.composer.move_cursor((len(lines) - 1, len(lines[-1])))
        self._render_composer_header()
        self._render_command_context()
        self._render_composer_hints()
        self._refresh_drawer()

    def _clear_composer(self) -> None:
        self._set_composer_text("")

    def _render_composer_header(self) -> None:
        mode = "command" if self._composer_text().lstrip().startswith("/") else "chat"
        selected = self.selected_model or "none"
        self.query_one("#composer-header", Static).update(
            f"[bold]mode[/]: {mode}   [bold]model[/]: {selected}   [bold]profile[/]: {self._profile_name}   [bold]session[/]: {self._session_id}"
        )

    def _render_command_context(self) -> None:
        widget = self.query_one("#command-context", Static)
        markup = self._command_context_markup()
        if markup:
            widget.update(markup)
            widget.add_class("-active")
        else:
            widget.update("")
            widget.remove_class("-active")

    def _command_context_markup(self) -> str:
        text = self._composer_text().strip()
        if not text.startswith("/"):
            return ""

        parts = text.split()
        command = canonical_name(parts[0])
        args = parts[1:]

        if command == "/models":
            if not args:
                return "[bold]/models[/]  [dim]→ choose an action: list, install, warm, remove, verify, repair, prune[/dim]"
            action = args[0]
            if action == "install":
                if len(args) == 1:
                    return (
                        "[bold]/models[/] [dim]›[/] [bold]install[/]  [cyan]waiting for <repo>[/cyan]\n"
                        "[dim]Paste a Hugging Face repo or choose a curated install target from the drawer.[/dim]"
                    )
                return (
                    f"[bold]/models[/] [dim]›[/] [bold]install[/] [cyan]{escape(args[1])}[/cyan]\n"
                    "[dim]Press Enter to install this repo.[/dim]"
                )
            if action in {"remove", "verify", "repair", "warm"}:
                target = args[1] if len(args) > 1 else (self.selected_model or "<repo>")
                return (
                    f"[bold]/models[/] [dim]›[/] [bold]{escape(action)}[/] [cyan]{escape(target)}[/cyan]\n"
                    "[dim]Choose an installed model from the drawer or type its full repo id.[/dim]"
                )

        if command == "/model":
            if len(args) == 0:
                return "[bold]/model[/]  [cyan]waiting for <installed-repo>[/cyan]\n[dim]Pick one installed model to make it the active chat target.[/dim]"
            return f"[bold]/model[/] [cyan]{escape(args[0])}[/cyan]\n[dim]Press Enter to select this model and reset the chat session.[/dim]"

        if command == "/runtime":
            action = args[0] if args else "start"
            target = args[1] if len(args) > 1 and not args[1].startswith("-") else (self.selected_model or "selected-model")
            if action == "start":
                return (
                    f"[bold]/runtime[/] [dim]›[/] [bold]start[/] [cyan]{escape(target)}[/cyan]\n"
                    "[dim]Press Enter to warm this model into the background daemon process.[/dim]"
                )
            if action == "stop":
                return "[bold]/runtime[/] [dim]›[/] [bold]stop[/]\n[dim]Press Enter to offload the current model and keep the daemon process running.[/dim]"
            if action == "status":
                return "[bold]/runtime[/] [dim]›[/] [bold]status[/]\n[dim]Press Enter to inspect the loaded-model state inside the daemon process.[/dim]"
            if action == "options":
                return "[bold]/runtime[/] [dim]›[/] [bold]options[/]\n[dim]Press Enter to show runtime defaults for the daemon-managed model state.[/dim]"
            return f"[bold]/runtime[/] [dim]›[/] [bold]{escape(action)}[/]\n[dim]Press Enter to run this runtime action.[/dim]"

        if command == "/daemon":
            action = args[0] if args else "status"
            if action == "start":
                return "[bold]/daemon[/] [dim]›[/] [bold]start[/]\n[dim]Press Enter to start the background daemon process.[/dim]"
            if action == "stop":
                return "[bold]/daemon[/] [dim]›[/] [bold]stop[/]\n[dim]Press Enter to stop the daemon process itself.[/dim]"
            if action == "status":
                return "[bold]/daemon[/] [dim]›[/] [bold]status[/]\n[dim]Press Enter to inspect the daemon process and any loaded runtime state.[/dim]"
            return f"[bold]/daemon[/] [dim]›[/] [bold]{escape(action)}[/]\n[dim]Press Enter to run this daemon action.[/dim]"

        if command == "/config":
            if not args:
                return "[bold]/config[/]  [dim]→ choose a setting to inspect or update[/dim]"
            key = args[0]
            value = " ".join(args[1:]) if len(args) > 1 else "<value>"
            return f"[bold]/config[/] [dim]›[/] [bold]{escape(key)}[/] [cyan]{escape(value)}[/cyan]\n[dim]Press Enter to apply this setting.[/dim]"

        return ""

    def _render_composer_hints(self, error: str | None = None) -> None:
        mode = "command" if self._composer_text().lstrip().startswith("/") else "chat"
        multiline_hint = "Ctrl+J / Option+Enter newline"
        submit_hint = "Enter submit"
        cancel_hint = "Ctrl+C cancel"
        if error:
            message = f"[red]{escape(error)}[/red]"
        elif mode == "command":
            message = self._command_hint_message()
        else:
            message = "[dim]Chat mode · Enter sends · type / for commands · Esc Esc edits previous prompt[/dim]"
        self.query_one("#composer-hints", Static).update(
            f"{message}\n[dim]{submit_hint} · {multiline_hint} · {cancel_hint} · PgUp/PgDn scroll transcript[/dim]"
        )

    def _command_hint_message(self) -> str:
        text = self._composer_text().strip()
        if not text.startswith("/"):
            return "[dim]Slash mode · Enter executes when syntax is complete · Tab focuses suggestions[/dim]"
        parts = text.split()
        command = canonical_name(parts[0])
        if command == "/models":
            if len(parts) == 1:
                return "[dim]Models · Choose an action like list, install, warm, remove, or verify[/dim]"
            action = parts[1]
            if action == "install":
                return "[dim]Install · Type or paste a Hugging Face repo, or pick a recommended target from the drawer[/dim]"
            if action in {"remove", "verify", "repair", "warm"}:
                return f"[dim]{escape(action.title())} · Pick an installed model from the drawer or type its full repo id[/dim]"
        if command == "/model":
            return "[dim]Model · Pick one installed model to make it the active chat target[/dim]"
        if command == "/runtime":
            return "[dim]Runtime = loaded model state inside the daemon. Start warms a model; stop offloads it; `daemon stop` kills the process.[/dim]"
        if command == "/daemon":
            return "[dim]Daemon = background process. Start/stop control the process itself; runtime commands only warm or offload models inside it.[/dim]"
        if command == "/config":
            return "[dim]Config · Use `size` for S/M/L/XL/XXL, or set `max_tokens` and `request_timeout` directly[/dim]"
        return "[dim]Slash mode · Enter executes when syntax is complete · Tab focuses suggestions[/dim]"

    def _statusline_value(self, field: str) -> str:
        snapshot = self._runtime_snapshot or {}
        if field == "state":
            return f"state:{snapshot.get('status', 'offline')}"
        if field == "model":
            return f"model:{snapshot.get('loaded_model') or self.selected_model or 'none'}"
        if field == "profile":
            return f"profile:{snapshot.get('profile') or self._profile_name}"
        if field == "session":
            return f"session:{self._session_id}"
        if field == "safe":
            return f"safe:{'on' if self.runtime_safe else 'off'}"
        if field == "memory":
            memory = snapshot.get("memory_pressure", {}).get("state", "unknown")
            return f"memory:{memory}"
        if field == "queue":
            return f"queue:{snapshot.get('queue_depth', 0)}"
        if field == "warm":
            loaded = snapshot.get("loaded_model")
            status = "warm" if loaded and loaded == self.selected_model else "cold"
            return f"path:{status}"
        return field

    def _render_statusline(self) -> None:
        if not self._tui_settings.get("show_statusline", True):
            self.query_one("#statusline", Static).update("")
            return
        fields = self._tui_settings.get("statusline_fields", list(STATUSLINE_FIELDS))
        text = "  |  ".join(self._statusline_value(field) for field in fields if field in STATUSLINE_FIELDS)
        self.query_one("#statusline", Static).update(text)

    def _set_generation_inflight(self, active: bool) -> None:
        self._generation_inflight = active
        self.composer.disabled = active
        if active:
            self.composer.add_class("-disabled")
        else:
            self.composer.remove_class("-disabled")
        self._render_composer_hints()
        self._render_statusline()

    def _queue_runtime_refresh(self) -> None:
        if not self._snapshot_refresh_inflight:
            self._refresh_runtime_snapshot()

    @work(thread=True)
    def _refresh_runtime_snapshot(self) -> None:
        if not self._engine:
            return
        self._snapshot_refresh_inflight = True
        try:
            snapshot = self._engine.client.ps()
        except Exception:
            snapshot = {
                "status": "offline",
                "loaded_model": None,
                "profile": self._profile_name,
                "queue_depth": 0,
                "memory_pressure": {"state": "unknown"},
            }
        self.call_from_thread(self._set_runtime_snapshot, snapshot)
        self._snapshot_refresh_inflight = False

    def _set_runtime_snapshot(self, snapshot: dict) -> None:
        self._runtime_snapshot = snapshot
        loaded = snapshot.get("loaded_model")
        if loaded:
            self.selected_model = loaded
        self._render_statusline()
        self._render_composer_header()
        self._render_command_context()

    @on(TextArea.Changed, "#composer")
    def _on_composer_changed(self, _event: TextArea.Changed) -> None:
        self._render_composer_header()
        self._render_composer_hints()
        self._refresh_drawer()

    def _refresh_drawer(self) -> None:
        text = self._composer_text().lstrip()
        drawer = self.query_one("#drawer", Horizontal)
        if not text.startswith("/"):
            drawer.remove_class("-visible")
            self.query_one("#drawer-list", OptionList).clear_options()
            self.query_one("#drawer-details", Static).update("")
            return
        drawer.add_class("-visible")
        self._populate_drawer(text)

    def _drawer_visible(self) -> bool:
        return self.query_one("#drawer", Horizontal).has_class("-visible")

    def _move_drawer_selection(self, delta: int) -> bool:
        if not self._drawer_visible():
            return False
        option_list = self.query_one("#drawer-list", OptionList)
        if option_list.option_count == 0:
            return False
        current = option_list.highlighted if option_list.highlighted is not None else 0
        new_index = max(0, min(option_list.option_count - 1, current + delta))
        option_list.highlighted = new_index
        entry = self._drawer_entries.get(f"drawer-{new_index}")
        if entry:
            self._show_drawer_detail(entry.get("detail", ""))
        return True

    def _apply_highlighted_drawer_option(self) -> bool:
        if not self._drawer_visible():
            return False
        option_list = self.query_one("#drawer-list", OptionList)
        index = option_list.highlighted
        if index is None:
            return False
        entry = self._drawer_entries.get(f"drawer-{index}")
        if not entry:
            return False
        value = entry["value"]
        execute_now = not value.endswith(" ")
        if execute_now:
            self._set_composer_text(value)
            self._submit_current_input()
        else:
            self._set_composer_text(value)
        return True

    def _populate_drawer(self, text: str) -> None:
        option_list = self.query_one("#drawer-list", OptionList)
        option_list.clear_options()
        self._drawer_entries = {}

        raw = text.strip()
        body = raw[1:]
        name_fragment, _, remainder = body.partition(" ")
        slash_name = f"/{name_fragment}" if name_fragment else ""
        normalized = canonical_name(slash_name) if slash_name else ""
        suggestions: list[tuple[str, str, str]] = []

        if not name_fragment:
            for spec in iter_commands():
                suggestions.append(self._spec_option(spec))
            for custom in self._custom_commands.values():
                suggestions.append(self._custom_option(custom))
        elif slash_name in self._custom_commands:
            custom = self._custom_commands[slash_name]
            suggestions.append(self._custom_option(custom))
        else:
            spec = get_command(slash_name)
            if spec is None and not raw.endswith(" "):
                for item in iter_commands():
                    if any(name.startswith(slash_name) for name in item.names) or name_fragment in item.description.lower():
                        suggestions.append(self._spec_option(item))
                for custom in self._custom_commands.values():
                    if custom.slash_name.startswith(slash_name):
                        suggestions.append(self._custom_option(custom))
            else:
                target = spec or get_command(normalized)
                if target is not None:
                    suggestions.extend(self._argument_suggestions(target, remainder))

        if not suggestions:
            suggestions.append((raw, raw, "Press Enter to execute the current command"))

        for index, (label, value, detail) in enumerate(suggestions):
            option_id = f"drawer-{index}"
            self._drawer_entries[option_id] = {"value": value, "detail": detail}
            option_list.add_option(Option(label, id=option_id))

        if option_list.option_count:
            option_list.highlighted = 0
            self._show_drawer_detail(self._drawer_entries["drawer-0"]["detail"])

    def _spec_option(self, spec: CommandSpec) -> tuple[str, str, str]:
        aliases = f" aliases: {', '.join(spec.aliases)}" if spec.aliases else ""
        label = f"[cyan]{spec.canonical}[/cyan] [dim]- {spec.description}[/dim]"
        detail = f"[bold]{spec.canonical}[/]\n{escape(spec.description)}{aliases}\n[dim]{escape(spec.argument_hint or '(no arguments)')}[/dim]"
        if self._tui_settings.get("show_examples_in_help", True) and spec.examples:
            examples = "\n".join(f"  {escape(example)}" for example in spec.examples)
            detail += f"\n[bold]Examples[/]\n{examples}"
        return label, spec.canonical + (" " if spec.argument_hint else ""), detail

    def _custom_option(self, custom: CustomCommand) -> tuple[str, str, str]:
        hint = f" {custom.argument_hint}" if custom.argument_hint else ""
        label = f"[magenta]{custom.slash_name}[/magenta] [dim]- {custom.description} ({custom.source})[/dim]"
        detail = f"[bold]{custom.slash_name}[/] ({custom.source})\n{escape(custom.description)}\n[dim]{escape(hint or '(no arguments)')}[/dim]"
        return label, custom.slash_name + (" " if custom.argument_hint else ""), detail

    def _argument_suggestions(self, spec: CommandSpec, remainder: str) -> list[tuple[str, str, str]]:
        remainder = remainder.lstrip()
        if spec.canonical == "/model":
            return self._model_repo_suggestions("/model", remainder)
        if spec.canonical == "/models":
            return self._models_suggestions(remainder)
        if spec.canonical == "/runtime":
            return self._simple_subcommand_suggestions(spec, remainder, ["start", "stop", "status", "options"])
        if spec.canonical == "/daemon":
            return self._simple_subcommand_suggestions(spec, remainder, ["start", "stop", "status", "install-launchd", "uninstall-launchd"])
        if spec.canonical == "/profile":
            return self._simple_subcommand_suggestions(spec, remainder, ["auto", "current", "list", "set", "calibrate"])
        if spec.canonical == "/ssh":
            return self._simple_subcommand_suggestions(spec, remainder, ["status", "stop", "tunnel", "snippet"])
        if spec.canonical == "/statusline":
            return self._simple_subcommand_suggestions(spec, remainder, ["show", "hide", "fields", "reset"])
        if spec.canonical == "/help":
            suggestions = [("commands", "/help commands", "Show all command categories"), ("keys", "/help keys", "Show keyboard shortcuts")]
            for item in iter_commands():
                suggestions.append((item.canonical, f"/help {item.canonical[1:]}", item.description))
            return [(
                f"[cyan]{label}[/cyan]",
                value,
                detail,
            ) for label, value, detail in suggestions if not remainder or label.startswith(remainder)]
        if spec.canonical == "/config":
            settings = [
                "preset",
                "size",
                "temp",
                "top_p",
                "top_k",
                "max_tokens",
                "request_timeout",
                "max_context",
                "keep_alive",
                "safe",
                "system_prompt",
                "reset",
            ]
            return [(
                f"[cyan]{setting}[/cyan] [dim]- configure {setting}[/dim]",
                f"/config {setting} ",
                f"Update {setting}",
            ) for setting in settings if not remainder or setting.startswith(remainder)]
        if spec.canonical == "/inspect":
            values = [self._session_id]
            if self.selected_model:
                values.append(self.selected_model)
            return [(
                f"[cyan]{value}[/cyan]",
                f"/inspect {value}",
                f"Inspect {value}",
            ) for value in values if value and (not remainder or remainder in value)]
        if spec.canonical == "/benchmark":
            if self.selected_model:
                return [(
                    f"[cyan]{self.selected_model}[/cyan]",
                    f"/benchmark {self.selected_model}",
                    "Benchmark the currently selected model",
                )]
        return [self._spec_option(spec)]

    def _simple_subcommand_suggestions(self, spec: CommandSpec, remainder: str, subcommands: list[str]) -> list[tuple[str, str, str]]:
        return [(
            f"[cyan]{item}[/cyan] [dim]- {spec.description}[/dim]",
            f"{spec.canonical} {item}",
            f"{spec.canonical} {item}",
        ) for item in subcommands if not remainder or item.startswith(remainder)]

    def _models_suggestions(self, remainder: str) -> list[tuple[str, str, str]]:
        parts = remainder.split(" ", 1)
        action = parts[0] if parts[0] else ""
        tail = parts[1].strip() if len(parts) > 1 else ""
        actions = ["list", "install", "remove", "verify", "repair", "warm", "prune", "recommended", "scan"]
        if action in {"install", "remove", "verify", "repair", "warm"}:
            prefix = f"/models {action}"
            return self._model_repo_suggestions(prefix, tail, install=(action == "install"))
        return [(
            f"[cyan]{item}[/cyan] [dim]- /models {item}[/dim]",
            f"/models {item}" + (" " if item in {"install", "remove", "verify", "repair", "warm"} else ""),
            self._models_action_detail(item),
        ) for item in actions if not remainder or item.startswith(remainder)]

    def _model_repo_suggestions(self, prefix: str, remainder: str, *, install: bool = False) -> list[tuple[str, str, str]]:
        if install:
            return self._install_repo_suggestions(prefix, remainder)
        repos = _cached_list_models(disk=True, filter_relevant=not install)
        suggestions = []
        for item in repos:
            repo = item["repo"]
            if remainder and remainder.lower() not in repo.lower():
                continue
            suffix = item.get("disk_usage") or item.get("estimated_size", "unknown")
            quant = item.get("quantization") or "unknown"
            summary = item.get("summary") or "Installed model"
            suggestions.append((
                f"[cyan]{repo}[/cyan] [dim]{quant} · {suffix}[/dim]",
                f"{prefix} {repo}",
                f"[bold]{repo}[/]\n{escape(summary)}\n[dim]format={quant} · size={suffix}[/dim]"
                + (f"\n[dim]{escape(item.get('when_to_use', ''))}[/dim]" if item.get("when_to_use") else ""),
            ))
        return suggestions or [(f"{prefix} {remainder}".strip(), f"{prefix} {remainder}".strip(), "Execute the current command")]

    def _install_repo_suggestions(self, prefix: str, remainder: str) -> list[tuple[str, str, str]]:
        suggestions: list[tuple[str, str, str]] = []
        typed_repo = remainder.strip()
        if typed_repo:
            suggestions.append((
                f"[cyan]{typed_repo}[/cyan] [dim]- install typed repo[/dim]",
                f"{prefix} {typed_repo}",
                f"[bold]Install typed repo[/]\nInstall [cyan]{escape(typed_repo)}[/cyan] from Hugging Face.\n[dim]Press Enter to run exactly what you typed.[/dim]",
            ))
        else:
            suggestions.append((
                "[cyan]Paste a Hugging Face repo[/cyan] [dim]- then press Enter[/dim]",
                f"{prefix} ",
                "[bold]Install a model[/]\nType a repo like [cyan]mlx-community/Qwen3.5-9B-OptiQ-4bit[/cyan], or pick one of the curated suggestions below.",
            ))

        for item in RECOMMENDED_MODELS:
            repo = item["repo"]
            if typed_repo and typed_repo.lower() not in repo.lower():
                continue
            metadata = MODEL_METADATA.get(repo, {})
            quant = metadata.get("quantization") or "unknown"
            size = item.get("estimated_size", "unknown")
            summary = metadata.get("summary") or item.get("description") or "Recommended install target"
            when_to_use = metadata.get("when_to_use", "")
            suggestions.append((
                f"[cyan]{repo}[/cyan] [dim]{quant} · {size}[/dim]",
                f"{prefix} {repo}",
                f"[bold]{repo}[/]\n{escape(summary)}\n[dim]format={quant} · size={size}[/dim]"
                + (f"\n[dim]{escape(when_to_use)}[/dim]" if when_to_use else ""),
            ))

        return suggestions

    @staticmethod
    def _models_action_detail(action: str) -> str:
        details = {
            "list": "Show installed, relevant LLMs with quantization and usage guidance.",
            "install": "Download a model into the Hugging Face cache. After selecting install, paste a Hugging Face repo or pick a curated suggestion.",
            "remove": "Delete a cached model from disk.",
            "verify": "Check required cache files and snapshot integrity.",
            "repair": "Re-download a broken model cache entry.",
            "warm": "Load a model into the daemon without changing chat history.",
            "prune": "Remove invalid or incomplete cache directories.",
            "recommended": "Show curated models for Apple Silicon profiles.",
            "scan": "Scan the machine for large model-weight files.",
        }
        return details.get(action, f"/models {action}")

    @on(OptionList.OptionHighlighted, "#drawer-list")
    def _on_drawer_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        option_id = str(event.option.id)
        detail = self._drawer_entries.get(option_id, {}).get("detail", "")
        self._show_drawer_detail(detail)

    @on(OptionList.OptionSelected, "#drawer-list")
    def _on_drawer_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = str(event.option.id)
        value = self._drawer_entries.get(option_id, {}).get("value", "")
        execute_now = not value.endswith(" ")
        if execute_now:
            self._set_composer_text(value)
            self._submit_current_input()
        else:
            self._set_composer_text(value)
            self.composer.focus()

    def _show_drawer_detail(self, detail: str) -> None:
        hint = "\n[dim]Enter select · Esc dismiss · Up/Down navigate[/dim]"
        self.query_one("#drawer-details", Static).update(detail + hint)

    def action_focus_composer_command(self) -> None:
        self._focus_composer_command()

    def _focus_composer_command(self) -> None:
        if self.focused is self.composer:
            self.composer.insert("/")
            return
        existing = self._composer_text()
        self.composer.focus()
        if existing == "/":
            return
        if existing.startswith("/"):
            return
        self.composer.insert("/")

    def on_key(self, event: Key) -> None:
        if event.character == "/" and self.focused is not self.composer:
            self._focus_composer_command()
            event.stop()
            return
        if self.focused is not self.composer:
            return
        if event.key == "escape":
            self._handle_escape()
            event.stop()

    def _should_use_history(self, *, direction: str) -> bool:
        if self._history is None:
            return False
        row, _col = self.composer.cursor_location
        last_row = len(self._composer_text().splitlines() or [""]) - 1
        if direction == "up":
            return row == 0
        return row == last_row

    def _history_older(self) -> None:
        if not self._history:
            return
        value = self._history.older(self._composer_text())
        if value is not None:
            self._set_composer_text(value)

    def _history_newer(self) -> None:
        if not self._history:
            return
        value = self._history.newer()
        if value is not None:
            self._set_composer_text(value)

    def _handle_escape(self) -> None:
        now = time.time()
        if now - self._last_escape_at < 0.5:
            self._last_escape_at = 0.0
            self._load_previous_prompt()
            return
        self._last_escape_at = now
        drawer = self.query_one("#drawer", Horizontal)
        if drawer.has_class("-visible"):
            drawer.remove_class("-visible")
            self._render_composer_hints()
        elif self._last_user_prompt:
            self._render_composer_hints(error="Press Esc again to load previous prompt")

    def _load_previous_prompt(self) -> None:
        if not self._last_user_prompt:
            self.log_msg("No previous chat prompt to edit.", style="system-msg")
            return
        self._set_composer_text(self._last_user_prompt)
        self.log_msg("Loaded previous prompt into the composer.", style="system-msg")

    def _insert_newline(self) -> None:
        self.composer.insert("\n")

    def _handle_enter_key(self) -> None:
        text = self._composer_text()
        if not text.strip():
            return
        if text.endswith("\\"):
            self._set_composer_text(text[:-1] + "\n")
            return
        if text.lstrip().startswith("/"):
            validation = self._validate_command(text)
            if validation is not None:
                if self._apply_highlighted_drawer_option():
                    return
                self._render_composer_hints(validation)
                return
        self._submit_current_input()

    def composer_submit(self) -> None:
        self._handle_enter_key()

    def composer_newline(self) -> None:
        self._insert_newline()

    def composer_history_up(self) -> bool:
        text = self._composer_text()
        if text.lstrip().startswith("/") and self._drawer_visible():
            return self._move_drawer_selection(-1)
        if self._should_use_history(direction="up"):
            self._history_older()
            return True
        return False

    def composer_history_down(self) -> bool:
        text = self._composer_text()
        if text.lstrip().startswith("/") and self._drawer_visible():
            return self._move_drawer_selection(1)
        if self._should_use_history(direction="down"):
            self._history_newer()
            return True
        return False

    def composer_escape(self) -> None:
        self._handle_escape()

    def _submit_current_input(self) -> None:
        raw = self._composer_text().strip()
        if not raw:
            return
        if self._history:
            self._history.push(raw)
        self._clear_composer()
        if raw.startswith("/"):
            self._handle_command(raw)
        else:
            if self._generation_inflight:
                self.log_msg("Generation already in progress. Cancel or wait for the current reply first.", style="system-msg")
                return
            self._last_user_prompt = raw
            self._set_generation_inflight(True)
            self._log_user_msg(raw)
            self._handle_chat(raw)

    def _validate_command(self, command_line: str) -> str | None:
        try:
            parts = shlex.split(command_line)
        except ValueError:
            return "Incomplete command syntax. Close any open quote before executing."
        if not parts:
            return None
        command = parts[0]
        if command in self._custom_commands:
            return None
        spec = get_command(command)
        if spec is None:
            return f"Unknown command: {command}"
        canonical = spec.canonical
        args = parts[1:]
        if canonical == "/model" and not args:
            return "Model selection requires a repo name."
        if canonical == "/models" and args:
            action = args[0]
            if action == "install" and len(args) < 2:
                return "Install requires a Hugging Face repo. Paste one or choose a recommended target."
        if canonical == "/chat" and not self.selected_model:
            return "Select a model before launching CLI chat."
        if canonical == "/inspect" and not args and not (self.selected_model or self._session_id):
            return "Nothing to inspect yet."
        if canonical == "/runtime" and (not args or args[0] == "start") and not self.selected_model and len(args) < 2:
            return "Select a model or pass one explicitly before starting the runtime."
        if canonical == "/config" and args:
            key = args[0]
            if key == "size" and len(args) > 1:
                if normalize_output_size_profile(args[1]) is None:
                    return "size accepts S, M, L, XL, or XXL."
            if key in {"temp", "top_p"} and len(args) > 1:
                try:
                    float(args[1])
                except ValueError:
                    return f"{key} must be numeric."
            if key == "request_timeout" and len(args) > 1:
                try:
                    int(args[1])
                except ValueError:
                    return "request_timeout must be an integer."
            if key in {"top_k", "max_tokens", "max_context", "keep_alive"} and len(args) > 1:
                if args[1].lower() not in {"auto", "none", "reset", "default"}:
                    try:
                        int(args[1])
                    except ValueError:
                        return f"{key} must be an integer or auto."
            if key == "safe" and len(args) > 1 and args[1].lower() not in {"1", "0", "true", "false", "yes", "no", "on", "off", "safe", "unsafe"}:
                return "safe accepts on/off, true/false, or safe/unsafe."
        if canonical == "/statusline" and args:
            action = args[0]
            if action not in {"show", "hide", "fields", "reset"}:
                return "statusline accepts show, hide, fields, or reset."
            if action == "fields":
                invalid = [field for field in args[1:] if field not in STATUSLINE_FIELDS]
                if invalid:
                    return f"Unknown statusline field(s): {', '.join(invalid)}"
        return None

    def _handle_command(self, command_line: str) -> None:
        try:
            parts = shlex.split(command_line)
        except ValueError as exc:
            self.log_msg(f"Command parse error: {exc}", style="error-msg")
            return
        if not parts:
            return

        command = parts[0]
        if command in self._custom_commands:
            self._run_custom_command(self._custom_commands[command], parts[1:], command_line)
            return

        spec = get_command(command)
        if spec is None:
            self.log_msg(f"Unknown command: {command}", style="error-msg")
            return

        command = spec.canonical
        args = parts[1:]

        if command == "/quit":
            self.exit()
        elif command == "/help":
            self._show_help(args[0] if args else None)
        elif command == "/doctor":
            self._run_cli_command(["doctor"] + args)
        elif command == "/model":
            self._handle_model_select(args)
        elif command == "/runtime":
            self._handle_runtime_command(args)
        elif command == "/daemon":
            self._run_cli_command((["daemon"] + args) if args else ["daemon", "status"])
        elif command == "/status":
            self._show_status_panel()
            self._run_cli_command(["ps"] + args)
        elif command == "/inspect":
            identifier = args[0] if args else (self.selected_model or self._session_id)
            self._run_cli_command(["inspect", identifier] + args[1:])
        elif command == "/logs":
            if "--follow" in args:
                self.set_timer(0.1, self._follow_logs)
            else:
                self._run_cli_command(["logs"] + args)
        elif command == "/benchmark":
            cli_args = ["benchmark"]
            if not args or args[0].startswith("-"):
                if not self.selected_model:
                    self.log_msg("Select a model first or pass one explicitly.", style="error-msg")
                    return
                cli_args.append(self.selected_model)
                cli_args.extend(args)
            else:
                cli_args.extend(args)
            self._run_cli_command(cli_args)
        elif command == "/models":
            self._handle_models_command(args)
        elif command == "/profile":
            self._handle_profile_command(args)
        elif command == "/config":
            self._handle_config_command(args)
        elif command == "/statusline":
            self._handle_statusline_command(args)
        elif command == "/chat":
            if not self.selected_model:
                self.log_msg("Select a model first.", style="error-msg")
                return
            self.log_msg(f"Launching CLI chat for {self.selected_model}…", style="system-msg")
            self.set_timer(0.1, self._launch_interactive_chat)
        elif command == "/guide":
            self._run_cli_command(["guide"] + (args or ["opencode"]))
        elif command == "/ssh":
            self._run_cli_command(["ssh"] + (args or ["status"]))
        elif command == "/copy":
            self.action_copy_last()
        elif command == "/clear":
            self._chat_history = []
            self._session_id = f"tui-{uuid.uuid4().hex[:8]}"
            self.log_msg("[dim]─────────────────────────────────────────[/dim]", style="system-msg")
            self.log_msg("New session started.", style="system-msg")
            self._render_composer_header()
            self._render_statusline()

    def _run_custom_command(self, custom: CustomCommand, args: list[str], raw: str) -> None:
        prompt = custom.expand(args)
        if not prompt:
            self.log_msg(f"{custom.slash_name} expanded to an empty prompt.", style="error-msg")
            return
        if custom.model and custom.model != self.selected_model:
            self.selected_model = custom.model
            self._chat_history = []
            self._session_id = f"tui-{uuid.uuid4().hex[:8]}"
            self.log_msg(f"Custom command selected model: [bold]{custom.model}[/]", style="system-msg")
        safe_override = self.runtime_safe if custom.safe is None else custom.safe
        profile_override = custom.profile
        self._last_user_prompt = prompt
        self._set_generation_inflight(True)
        self._log_user_msg(raw)
        self.log_msg(f"[dim]expanded → {escape(prompt)}[/dim]", style="system-msg")
        self._handle_chat(prompt, profile_override=profile_override, safe_override=safe_override)

    def _handle_model_select(self, args: list[str]) -> None:
        if not args:
            self.log_msg("Usage: /model <repo>", style="error-msg")
            return
        repo = args[0]
        self.selected_model = repo
        self._chat_history = []
        self._session_id = f"tui-{uuid.uuid4().hex[:8]}"
        self.log_msg(f"Selected model: [bold]{repo}[/]", style="system-msg")
        self._ensure_server_for_model(repo)
        self._render_composer_header()
        self._render_statusline()

    def _handle_runtime_command(self, args: list[str]) -> None:
        action = args[0] if args else "start"
        flags = args[1:] if args else []
        if action == "start":
            explicit_model = None
            if flags and not flags[0].startswith("-"):
                explicit_model = flags[0]
                flags = flags[1:]
            model = explicit_model or self.selected_model
            if not model:
                self.log_msg("Select a model before starting the runtime.", style="error-msg")
                return
            self.log_msg(f"Starting runtime for [bold]{model}[/]…", style="system-msg")
            self._run_cli_command(["serve", "start", model] + flags)
            self.selected_model = model
        elif action in {"stop", "status", "options"}:
            self._run_cli_command(["serve", action] + flags)
        else:
            self.log_msg(f"Unknown runtime action: {action}", style="error-msg")

    def _handle_models_command(self, args: list[str]) -> None:
        action = args[0] if args else "list"
        tail = args[1:] if args else []
        if action == "list":
            _invalidate_model_cache()
            self._run_cli_command(["models", "list"] + tail)
        elif action == "scan":
            self._run_cli_command(["models", "scan"])
        elif action == "recommended":
            self._run_cli_command(["models", "recommended"])
        elif action == "prune":
            self._run_cli_command(["models", "prune", "-y"] + tail)
        elif action in {"install", "remove", "verify", "repair", "warm"}:
            repo = tail[0] if tail else (self.selected_model if action != "install" else None)
            if not repo:
                self.log_msg(f"Usage: /models {action} <repo>", style="error-msg")
                return
            _invalidate_model_cache()
            command = ["models", action, repo]
            if action in {"install", "remove", "repair"}:
                command.append("-y")
            command.extend(tail[1:] if tail and repo == tail[0] else tail)
            self._run_cli_command(command)
        else:
            self.log_msg(f"Unknown /models action: {action}", style="error-msg")

    def _handle_profile_command(self, args: list[str]) -> None:
        action = args[0] if args else "current"
        tail = args[1:] if args else []
        if action == "calibrate" and (not tail or tail[0].startswith("-")) and self.selected_model:
            self._run_cli_command(["profile", "calibrate", self.selected_model] + tail)
        else:
            self._run_cli_command(["profile", action] + tail)
        config = load_config()
        self._profile_name, _profile = get_effective_profile(config)
        self._render_composer_header()
        self._render_statusline()

    def _handle_config_command(self, args: list[str]) -> None:
        if not args:
            self._show_config_panel()
            return
        key = args[0].lower()
        value = " ".join(args[1:]) if len(args) > 1 else ""
        if key == "preset":
            if value not in GENERATION_PRESETS:
                self.log_msg(f"Unknown preset '{value}'.", style="error-msg")
                return
            preset = GENERATION_PRESETS[value]
            self.gen_preset = value
            self.gen_temp = preset["temp"]
            self.gen_top_p = preset["top_p"]
            self.gen_top_k = preset["top_k"]
            self.gen_max_tokens = preset["max_tokens"]
            self._persist_gen_settings()
            self.log_msg(f"Applied preset [bold]{value}[/].", style="system-msg")
            return
        if key == "size":
            profile_name = normalize_output_size_profile(value)
            if profile_name is None:
                self.log_msg("size accepts S, M, L, XL, or XXL.", style="error-msg")
                return
            profile = get_output_size_profile(profile_name)
            assert profile is not None
            self.gen_max_tokens = int(profile["max_tokens"])
            self.runtime_request_timeout = int(profile["request_timeout_seconds"])
            self._persist_gen_settings()
            self._persist_runtime_settings()
            self.log_msg(
                f"Applied size [bold]{profile_name}[/] → max_tokens={self.gen_max_tokens} request_timeout={self.runtime_request_timeout}s",
                style="system-msg",
            )
            return
        if key == "reset":
            defaults = _default_generation_settings()
            self.gen_preset = defaults["preset"]
            self.gen_temp = defaults["temp"]
            self.gen_top_p = defaults["top_p"]
            self.gen_top_k = defaults["top_k"]
            self.gen_max_tokens = defaults["max_tokens"]
            self.gen_system_prompt = defaults["system_prompt"]
            self.runtime_max_context = None
            self.runtime_keep_alive = None
            self.runtime_request_timeout = DEFAULT_REQUEST_TIMEOUT_SECONDS
            self.runtime_safe = True
            self._persist_gen_settings()
            self._persist_runtime_settings()
            self._persist_session_defaults()
            self.log_msg("Config reset to defaults.", style="system-msg")
            self._show_config_panel()
            return
        if key == "system_prompt":
            if not value:
                self.log_msg(f"Current system prompt: {escape(self.gen_system_prompt)}", style="system-msg")
                return
            self.gen_system_prompt = value
            self._persist_gen_settings()
            self.log_msg("System prompt updated.", style="system-msg")
            return
        if not value:
            self.log_msg(f"Usage: /config {key} <value>", style="error-msg")
            return
        try:
            if key == "temp":
                self.gen_temp = float(value)
            elif key == "top_p":
                self.gen_top_p = float(value)
            elif key == "top_k":
                self.gen_top_k = int(value)
            elif key == "max_tokens":
                self.gen_max_tokens = int(value)
            elif key == "request_timeout":
                self.runtime_request_timeout = int(value)
                self._persist_runtime_settings()
                self.log_msg(f"request_timeout → {self.runtime_request_timeout}s", style="system-msg")
                self._render_statusline()
                return
            elif key == "max_context":
                self.runtime_max_context = None if value.lower() in {"auto", "none", "reset", "default"} else int(value)
                self._persist_session_defaults()
                self.log_msg(f"max_context → {self.runtime_max_context or 'auto'}", style="system-msg")
                return
            elif key == "keep_alive":
                self.runtime_keep_alive = None if value.lower() in {"auto", "none", "reset", "default"} else int(value)
                self._persist_session_defaults()
                self.log_msg(f"keep_alive → {self.runtime_keep_alive or 'auto'}", style="system-msg")
                return
            elif key == "safe":
                lowered = value.lower()
                self.runtime_safe = lowered in {"1", "true", "yes", "on", "safe"}
                self._persist_runtime_settings()
                self.log_msg(f"safe → {'on' if self.runtime_safe else 'off'}", style="system-msg")
                self._render_statusline()
                return
            else:
                self.log_msg(f"Unknown config key: {key}", style="error-msg")
                return
            self.gen_preset = "custom"
            self._persist_gen_settings()
            self.log_msg(f"{key} → {value} (preset → custom)", style="system-msg")
        except ValueError:
            self.log_msg(f"Invalid value for {key}: {value}", style="error-msg")
        self._render_statusline()

    def _show_config_panel(self) -> None:
        lines = [
            "[bold]Current Config[/]",
            f"preset={self.gen_preset}  temp={self.gen_temp}  top_p={self.gen_top_p}  top_k={self.gen_top_k}",
            f"size={detect_output_size_profile(self.gen_max_tokens, self.runtime_request_timeout) or 'custom'}  max_tokens={self.gen_max_tokens}  request_timeout={self.runtime_request_timeout}s",
            f"max_context={self.runtime_max_context or 'auto'}  keep_alive={self.runtime_keep_alive or 'auto'}  safe={'on' if self.runtime_safe else 'off'}",
            f"system_prompt={escape(self.gen_system_prompt)}",
            "",
            "[dim]Examples[/dim]",
            "  /config preset balanced",
            "  /config size XL",
            "  /config request_timeout 900",
            "  /config max_context 8192",
            "  /config safe off",
            "  /config system_prompt You are a concise coding assistant.",
        ]
        self.log_msg("\n".join(lines), style="system-msg")

    def _handle_statusline_command(self, args: list[str]) -> None:
        action = args[0] if args else "fields"
        if action == "show":
            self._tui_settings["show_statusline"] = True
        elif action == "hide":
            self._tui_settings["show_statusline"] = False
        elif action == "reset":
            self._tui_settings["show_statusline"] = True
            self._tui_settings["statusline_fields"] = list(STATUSLINE_FIELDS)
        elif action == "fields":
            if args[1:]:
                self._tui_settings["statusline_fields"] = [field for field in args[1:] if field in STATUSLINE_FIELDS]
        else:
            self.log_msg("Usage: /statusline show|hide|fields|reset [field ...]", style="error-msg")
            return
        save_tui_settings(self._tui_settings)
        self._render_statusline()
        self.log_msg(
            f"Status line → {'shown' if self._tui_settings.get('show_statusline', True) else 'hidden'}; fields={', '.join(self._tui_settings.get('statusline_fields', []))}",
            style="system-msg",
        )

    def _show_status_panel(self) -> None:
        snapshot = self._runtime_snapshot or {}
        lines = [
            "[bold]Runtime Snapshot (loaded model state)[/]",
            f"state={snapshot.get('status', 'offline')}",
            f"model={snapshot.get('loaded_model') or 'none'}",
            f"profile={snapshot.get('profile') or self._profile_name}",
            f"sessions={snapshot.get('session_count', 0)}",
            f"queue={snapshot.get('queue_depth', 0)}",
            f"memory={snapshot.get('memory_pressure', {}).get('state', 'unknown')}",
            f"keep_alive={snapshot.get('keep_alive_seconds', self.runtime_keep_alive)}",
            f"request_timeout={snapshot.get('request_timeout_seconds', self.runtime_request_timeout)}",
        ]
        self.log_msg("\n".join(lines), style="system-msg")

    def _show_help(self, subject: str | None = None) -> None:
        if subject in {None, "", "commands"}:
            lines = [
                "[bold]Commands[/]",
                "[dim]Daemon = background process. Runtime = loaded model state managed inside that process.[/dim]",
            ]
            categories: dict[str, list[CommandSpec]] = {}
            for spec in iter_commands():
                categories.setdefault(spec.category, []).append(spec)
            for category, items in categories.items():
                lines.append(f"\n[bold]{category}[/]")
                for spec in items:
                    alias_text = f" [dim](aliases: {', '.join(spec.aliases)})[/dim]" if spec.aliases else ""
                    lines.append(f"  [cyan]{spec.canonical}[/cyan] {spec.description}{alias_text}")
                    lines.append(f"  [dim]{spec.argument_hint or '(no arguments)'}[/dim]")
            if self._custom_commands:
                lines.append("\n[bold]Custom Commands[/]")
                for custom in self._custom_commands.values():
                    lines.append(f"  [magenta]{custom.slash_name}[/magenta] {custom.description} [dim]({custom.source})[/dim]")
            self.log_msg("\n".join(lines), style="system-msg")
            return
        if subject == "keys":
            self.log_msg(
                "\n".join(
                    [
                        "[bold]Keyboard Shortcuts[/]",
                        "  Enter              submit current command or chat prompt",
                        "  Ctrl+J / Option+Enter  insert newline",
                        "  Up / Down          history navigation at top/bottom of composer",
                        "  Esc Esc            load previous chat prompt",
                        "  Ctrl+C             cancel active generation",
                        "  Ctrl+D             quit",
                        "  Ctrl+L             clear transcript",
                        "  Ctrl+Y             copy last assistant response",
                        "  Ctrl+R             refresh runtime snapshot",
                    ]
                ),
                style="system-msg",
            )
            return
        spec = get_command(subject)
        if spec is None:
            normalized = subject if subject.startswith("/") else f"/{subject}"
            custom = self._custom_commands.get(normalized)
            if custom:
                detail = self._custom_option(custom)[2]
                self.log_msg(detail, style="system-msg")
            else:
                self.log_msg(f"Unknown help topic: {subject}", style="error-msg")
            return
        detail = self._spec_option(spec)[2]
        self.log_msg(detail, style="system-msg")

    def _persist_gen_settings(self) -> None:
        save_generation_settings(
            {
                "preset": self.gen_preset,
                "temp": self.gen_temp,
                "top_p": self.gen_top_p,
                "top_k": self.gen_top_k,
                "max_tokens": self.gen_max_tokens,
                "system_prompt": self.gen_system_prompt,
            }
        )

    def _persist_runtime_settings(self) -> None:
        save_runtime_settings(
            {
                "safe_mode": self.runtime_safe,
                "request_timeout_seconds": self.runtime_request_timeout,
            }
        )

    def _persist_session_defaults(self) -> None:
        config = load_config()
        config["session_defaults"] = {
            **config.get("session_defaults", {}),
            "max_context": self.runtime_max_context,
            "keep_alive_seconds": self.runtime_keep_alive,
        }
        save_config(config)

    @work(thread=True)
    def _run_cli_command(self, args: list[str]) -> None:
        import sys

        self.call_from_thread(self.start_thinking, f"Running /{args[0]}" if args else "Working")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "local_llm.cli"] + args,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                self.call_from_thread(self.log_msg, escape(result.stdout), "system-msg")
            if result.stderr:
                self.call_from_thread(self.log_msg, escape(result.stderr), "error-msg")
        except Exception as exc:
            self.call_from_thread(self.log_msg, str(exc), "error-msg")
        finally:
            self.call_from_thread(self.stop_thinking)
            self.call_from_thread(self._queue_runtime_refresh)

    def _follow_logs(self) -> None:
        import sys

        with self.suspend():
            subprocess.run([sys.executable, "-m", "local_llm.cli", "logs", "--follow"])
        self.log_msg("Stopped following logs.", style="system-msg")

    def _launch_interactive_chat(self) -> None:
        import sys

        cli_args = [
            sys.executable,
            "-m",
            "local_llm.cli",
            "chat",
            self.selected_model,
            "--session",
            self._session_id,
            "--temp",
            str(self.gen_temp),
            "--top-p",
            str(self.gen_top_p),
            "--top-k",
            str(self.gen_top_k),
            "--max-tokens",
            str(self.gen_max_tokens),
        ]
        if self.runtime_max_context is not None:
            cli_args.extend(["--max-context", str(self.runtime_max_context)])
        if self.runtime_keep_alive is not None:
            cli_args.extend(["--keep-alive", str(self.runtime_keep_alive)])
        cli_args.append("--safe" if self.runtime_safe else "--unsafe")
        with self.suspend():
            subprocess.run(cli_args)
        self.log_msg("CLI chat session ended.", style="system-msg")
        self._queue_runtime_refresh()

    def _create_stream_widget(self, widget_id: str) -> None:
        log = self.query_one("#transcript", VerticalScroll)
        label = Static("[dim bold]Assistant[/dim bold]", classes="assistant-label")
        widget = Static("", id=widget_id, markup=False)
        log.mount(label)
        log.mount(widget)
        log.scroll_end(animate=False)

    def _update_stream_widget(self, widget_id: str, text: str, final: bool = False) -> None:
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(_render_assistant_text(text, final=final))
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def _append_summary(self, summary: dict) -> None:
        metrics = summary.get("local_llm", {}).get("metrics", {})
        usage = summary.get("usage", {})
        if not metrics and not usage:
            return
        line = (
            f"[dim]summary "
            f"ttft={metrics.get('ttft_seconds', 0):.2f}s "
            f"tok/s={metrics.get('generation_tps', 0):.2f} "
            f"cache={'hit' if metrics.get('cache_hit') else 'miss'} "
            f"prompt={usage.get('prompt_tokens', 0)} "
            f"output={usage.get('completion_tokens', 0)} "
            f"finish={metrics.get('finish_reason', 'unknown')}[/dim]"
        )
        self.log_msg(line, style="summary-msg")

    @work(thread=True)
    def _ensure_server_for_model(self, model: str) -> None:
        if not self._engine:
            return
        self.call_from_thread(self.start_thinking, "Warming model")
        try:
            self._engine.ensure_server(
                model,
                keep_alive_seconds=self.runtime_keep_alive,
                safe_mode=self.runtime_safe,
                on_status=lambda message: self.call_from_thread(self.log_msg, message, "system-msg"),
            )
        except Exception as exc:
            self.call_from_thread(self.log_msg, f"Runtime error: {exc}", "error-msg")
        finally:
            self.call_from_thread(self.stop_thinking)
            self.call_from_thread(self._queue_runtime_refresh)

    @work(thread=True)
    def _cancel_active_request(self) -> None:
        if not self._engine:
            return
        try:
            result = self._engine.client.cancel()
            message = "Generation cancelled." if result.get("cancelled") else "No active request was cancelled."
            self.call_from_thread(self.log_msg, message, "system-msg")
        except Exception as exc:
            self.call_from_thread(self.log_msg, f"Cancel failed: {exc}", "error-msg")

    @work(thread=True)
    def _handle_chat(
        self,
        prompt: str,
        *,
        profile_override: str | None = None,
        safe_override: bool | None = None,
    ) -> None:
        if not self._engine:
            self.call_from_thread(self.log_msg, "Engine not initialized.", "error-msg")
            self.call_from_thread(self._set_generation_inflight, False)
            return
        if not self.selected_model:
            self.call_from_thread(self.log_msg, "Select a model first.", "error-msg")
            self.call_from_thread(self._set_generation_inflight, False)
            return

        safe_value = self.runtime_safe if safe_override is None else safe_override
        self.call_from_thread(self.start_thinking, "Generating")

        widget_id = f"stream-{uuid.uuid4().hex}"
        self.call_from_thread(self._create_stream_widget, widget_id)

        try:
            running_model = self._engine.get_running_model()
            if running_model != self.selected_model:
                self.call_from_thread(self._update_stream_widget, widget_id, f"Warming {self.selected_model}…")
                self._engine.ensure_server(
                    self.selected_model,
                    profile=profile_override,
                    keep_alive_seconds=self.runtime_keep_alive,
                    safe_mode=safe_value,
                    on_status=lambda message: self.call_from_thread(self._update_stream_widget, widget_id, message),
                )

            self.call_from_thread(self._update_stream_widget, widget_id, "Generating…")
            self._chat_history.append({"role": "user", "content": prompt})
            messages = [{"role": "system", "content": self.gen_system_prompt}, *self._chat_history]

            full_text = ""
            last_render = 0.0
            for chunk in self._engine.chat_stream(
                messages,
                temperature=self.gen_temp,
                top_p=self.gen_top_p,
                top_k=self.gen_top_k,
                max_tokens=self.gen_max_tokens,
                session=self._session_id,
                max_context=self.runtime_max_context,
                keep_alive_seconds=self.runtime_keep_alive,
                profile=profile_override,
                safe=safe_value,
            ):
                full_text += chunk
                now = time.monotonic()
                if now - last_render >= 0.05 or chunk.endswith(("\n", ".", "!", "?")):
                    self.call_from_thread(self._update_stream_widget, widget_id, full_text)
                    last_render = now

            cleaned = re.sub(r"<think>.*?(</think>|$)", "", full_text, flags=re.DOTALL).strip()
            self._last_response = cleaned
            self._chat_history.append({"role": "assistant", "content": cleaned})
            self.call_from_thread(self._update_stream_widget, widget_id, full_text, True)
            self.call_from_thread(self._append_summary, self._engine.last_summary)
        except Exception as exc:
            self.call_from_thread(self.log_msg, f"Chat error: {exc}", "error-msg")
        finally:
            self.call_from_thread(self._set_generation_inflight, False)
            self.call_from_thread(self.stop_thinking)
            self.call_from_thread(self._queue_runtime_refresh)


if __name__ == "__main__":
    CommandPalette().run(mouse=False)
