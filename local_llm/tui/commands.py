"""Command registry for the interactive TUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CommandSpec:
    """Metadata for a slash command shown in the TUI."""

    canonical: str
    aliases: tuple[str, ...]
    category: str
    description: str
    argument_hint: str = ""
    examples: tuple[str, ...] = ()
    requires_model: bool = False
    requires_runtime: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        canonical="/help",
        aliases=(),
        category="General",
        description="Show command help and keyboard shortcuts",
        argument_hint="[command|commands|keys]",
        examples=("/help", "/help model", "/help keys"),
    ),
    CommandSpec(
        canonical="/model",
        aliases=("/select",),
        category="Runtime",
        description="Select the active model for chat and for `/runtime start`",
        argument_hint="<repo>",
        examples=("/model RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4",),
    ),
    CommandSpec(
        canonical="/models",
        aliases=(),
        category="Runtime",
        description="List, install, verify, repair, warm, or prune models",
        argument_hint="list|install|remove|verify|repair|warm|prune|recommended|scan [args]",
        examples=("/models list", "/models warm RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4"),
    ),
    CommandSpec(
        canonical="/runtime",
        aliases=("/serve",),
        category="Runtime",
        description="Manage the loaded model inside the daemon process",
        argument_hint="start|stop|status|options [flags]",
        examples=("/runtime start", "/runtime stop", "/runtime status"),
        requires_runtime=False,
    ),
    CommandSpec(
        canonical="/daemon",
        aliases=(),
        category="Runtime",
        description="Manage the background daemon process and launchd agent",
        argument_hint="start|stop|status|install-launchd|uninstall-launchd [flags]",
        examples=("/daemon status", "/daemon start --port 8080"),
    ),
    CommandSpec(
        canonical="/status",
        aliases=("/ps",),
        category="Runtime",
        description="Show the loaded-model snapshot reported by the daemon",
        argument_hint="[--port N]",
        examples=("/status", "/status --port 8080"),
    ),
    CommandSpec(
        canonical="/inspect",
        aliases=(),
        category="Runtime",
        description="Inspect a session id or loaded model",
        argument_hint="<session-id|model> [--port N]",
        examples=("/inspect tui-1234abcd", "/inspect RepublicOfKorokke/Qwen3.5-4B-mlx-lm-mxfp4"),
        requires_runtime=False,
    ),
    CommandSpec(
        canonical="/logs",
        aliases=(),
        category="Runtime",
        description="Show daemon logs or follow them in a suspended terminal",
        argument_hint="[--follow]",
        examples=("/logs", "/logs --follow"),
    ),
    CommandSpec(
        canonical="/benchmark",
        aliases=(),
        category="Runtime",
        description="Benchmark the selected or specified model through the daemon",
        argument_hint="[model] [--runs N] [--prompt TEXT] [--profile NAME]",
        examples=("/benchmark", "/benchmark --runs 3"),
    ),
    CommandSpec(
        canonical="/profile",
        aliases=(),
        category="Runtime",
        description="View, set, or calibrate the Apple Silicon hardware profile",
        argument_hint="list|current|auto|set <name>|calibrate [model] [--runs N]",
        examples=("/profile current", "/profile calibrate"),
    ),
    CommandSpec(
        canonical="/config",
        aliases=("/set",),
        category="Composer",
        description="Show or update generation and runtime defaults",
        argument_hint="[preset|size|temp|top_p|top_k|max_tokens|request_timeout|max_context|keep_alive|safe|system_prompt|reset]",
        examples=("/config", "/config size XL", "/config request_timeout 900"),
    ),
    CommandSpec(
        canonical="/statusline",
        aliases=(),
        category="Composer",
        description="Configure the fields shown in the bottom status line",
        argument_hint="show|hide|fields|reset [field ...]",
        examples=("/statusline fields state model memory queue", "/statusline hide"),
    ),
    CommandSpec(
        canonical="/doctor",
        aliases=(),
        category="Tools",
        description="Check the MLX/macOS environment and dependencies",
        argument_hint="",
        examples=("/doctor",),
    ),
    CommandSpec(
        canonical="/ssh",
        aliases=(),
        category="Tools",
        description="Manage optional SSH tunneling for remote access",
        argument_hint="status|stop|tunnel|snippet [flags]",
        examples=("/ssh status", "/ssh snippet --to user@host"),
    ),
    CommandSpec(
        canonical="/guide",
        aliases=(),
        category="Tools",
        description="Show integration guides",
        argument_hint="opencode",
        examples=("/guide opencode",),
    ),
    CommandSpec(
        canonical="/chat",
        aliases=(),
        category="Composer",
        description="Suspend the TUI and launch the interactive CLI chat",
        argument_hint="",
        examples=("/chat",),
        requires_model=True,
    ),
    CommandSpec(
        canonical="/clear",
        aliases=(),
        category="Composer",
        description="Clear the current conversation history and start a new session",
        argument_hint="",
        examples=("/clear",),
    ),
    CommandSpec(
        canonical="/copy",
        aliases=(),
        category="Composer",
        description="Copy the last assistant response to the macOS clipboard",
        argument_hint="",
        examples=("/copy",),
    ),
    CommandSpec(
        canonical="/quit",
        aliases=("/q",),
        category="General",
        description="Quit the application",
        argument_hint="",
        examples=("/quit",),
    ),
)

_SPEC_BY_NAME = {name: spec for spec in COMMAND_SPECS for name in spec.names}


def get_command(name: str) -> CommandSpec | None:
    """Return the command spec for a canonical name or alias."""
    normalized = name if name.startswith("/") else f"/{name}"
    return _SPEC_BY_NAME.get(normalized)


def iter_commands() -> Iterable[CommandSpec]:
    """Yield canonical command specs."""
    return COMMAND_SPECS


def canonical_name(name: str) -> str:
    """Normalize a command or alias to its canonical slash form."""
    spec = get_command(name)
    return spec.canonical if spec else (name if name.startswith("/") else f"/{name}")


def command_names() -> tuple[str, ...]:
    """Return canonical command names."""
    return tuple(spec.canonical for spec in COMMAND_SPECS)
