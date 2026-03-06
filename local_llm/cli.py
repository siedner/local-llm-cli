"""CLI entrypoint for local-llm."""

import json
import sys
from typing import Optional

try:
    import typer
    from typer import Argument, Option
except ImportError:
    print("Error: typer is required. Install with: pip install typer")
    sys.exit(1)

from . import __version__
from .config import detect_profile, get_profile_name, load_config, save_config
from .constants import DEFAULT_HOST, DEFAULT_PORT, PROFILES, RECOMMENDED_MODELS
from . import ui

app = typer.Typer(
    name="local-llm",
    help="Manage and run local LLMs using MLX-LM on macOS.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    """Launch interactive TUI when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from .tui import CommandPalette

        app_tui = CommandPalette()
        app_tui.run(mouse=False)

# ── doctor ──────────────────────────────────────────────────────────

@app.command()
def doctor(
    fix: bool = Option(False, "--fix", help="Attempt to install missing dependencies"),
    yes: bool = Option(False, "--yes", "-y", help="Auto-confirm installs"),
):
    """Check environment and install missing prerequisites."""
    from .doctor import fix_missing, run_checks

    ui.header("Doctor")
    checks = run_checks()
    for c in checks:
        ui.check(c.name, c.ok, detail=c.detail, hint=c.fix_hint or "")
    ui.console.print()

    all_ok = all(c.ok for c in checks)
    if all_ok:
        ui.success("All checks passed.")
    elif fix:
        ui.info("Attempting to fix issues...")
        actions = fix_missing(yes=yes)
        for a in actions:
            ui.info(f"-> {a}")
    else:
        ui.warning("Some checks failed. Run `local-llm doctor --fix` to attempt auto-install.")


# ── profile ─────────────────────────────────────────────────────────

profile_app = typer.Typer(help="Manage hardware profiles.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")


@profile_app.command("list")
def profile_list():
    """List available profiles."""
    config = load_config()
    current = get_profile_name(config)
    ui.header("Profiles")
    for name, p in PROFILES.items():
        marker = "  <-- active" if name == current else ""
        ui.kv(name, f"{p['chip_pattern']} {p['memory_gb']}GB, max_tokens={p['max_tokens']}{marker}")


@profile_app.command("current")
def profile_current():
    """Show the current profile."""
    config = load_config()
    name = get_profile_name(config)
    if name:
        ui.kv("Current profile", name)
    else:
        ui.warning("No profile set. Run `local-llm profile auto` or `local-llm profile set <name>`.")


@profile_app.command("set")
def profile_set(name: str = Argument(help="Profile name")):
    """Set the active profile."""
    if name not in PROFILES:
        ui.error(f"Unknown profile '{name}'. Available: {', '.join(PROFILES.keys())}")
        raise typer.Exit(1)
    config = load_config()
    config["profile"] = name
    save_config(config)
    ui.success(f"Profile set to: {name}")


@profile_app.command("auto")
def profile_auto():
    """Auto-detect and set profile based on hardware."""
    detected = detect_profile()
    if detected:
        config = load_config()
        config["profile"] = detected
        save_config(config)
        ui.success(f"Auto-detected profile: {detected}")
    else:
        ui.warning("Could not auto-detect profile. Set manually with `local-llm profile set <name>`.")


# ── models ──────────────────────────────────────────────────────────

models_app = typer.Typer(help="Manage local models.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    disk: bool = Option(True, "--no-disk", help="Skip showing disk usage"),
    as_json: bool = Option(False, "--json", help="Output as JSON"),
):
    """List installed models from HF cache."""
    from .models import list_models

    models = list_models(disk=disk)
    if not models:
        ui.info("No models found in HF cache.")
        ui.info("Install one with: local-llm models install <hf_repo>")
        return

    if as_json:
        output = []
        for m in models:
            entry = {"repo": m["repo"], "path": str(m["path"])}
            if disk:
                entry["disk_usage"] = m.get("disk_usage", "unknown")
            output.append(entry)
        ui.console.print_json(json.dumps(output))
        return

    table = ui.styled_table(title="Installed Models")
    table.add_column("Model", style="white")
    if disk:
        table.add_column("Size", style="dim")
    for m in models:
        row = [m["repo"]]
        if disk:
            row.append(m.get("disk_usage", "unknown"))
        table.add_row(*row)
    ui.console.print(table)


@models_app.command("install")
def models_install(
    repo: str = Argument(help="HF repo (org/name)"),
    yes: bool = Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Download a model (warm-download via mlx_lm.generate)."""
    from .models import install_model
    install_model(repo, yes=yes)


@models_app.command("remove")
def models_remove(
    repo: str = Argument(help="HF repo (org/name)"),
    yes: bool = Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a model from HF cache."""
    from .models import remove_model
    remove_model(repo, yes=yes)


@models_app.command("recommended")
def models_recommended():
    """Show recommended models."""
    ui.header("Recommended Models")
    ui.console.print()
    for m in RECOMMENDED_MODELS:
        ui.kv(m["repo"], m["description"])
    ui.console.print()
    config = load_config()
    favs = config.get("favorite_models", [])
    if favs:
        ui.header("Your Favorites")
        for f in favs:
            ui.info(f)


@models_app.command("scan")
def models_scan():
    """Scan the system for all LLM weights and show a housekeeping report."""
    from .models import scan_models
    scan_models()


# ── chat ────────────────────────────────────────────────────────────

@app.command()
def chat(
    model: str = Argument(help="HF repo (org/name) of the model"),
    temp: float = Option(0.7, "--temp", help="Temperature"),
    top_p: float = Option(0.9, "--top-p", help="Top-p sampling"),
    top_k: int = Option(50, "--top-k", help="Top-k sampling"),
):
    """Start an interactive chat session."""
    from .chat import chat as do_chat
    do_chat(model, temp=temp, top_p=top_p, top_k=top_k)


# ── serve ───────────────────────────────────────────────────────────

serve_app = typer.Typer(help="Manage the MLX-LM server.", no_args_is_help=True)
app.add_typer(serve_app, name="serve")


@serve_app.command("start")
def serve_start(
    model: str = Argument(help="HF repo (org/name) of the model"),
    host: str = Option(DEFAULT_HOST, "--host", help="Bind address"),
    port: int = Option(DEFAULT_PORT, "--port", help="Port"),
    max_tokens: Optional[int] = Option(None, "--max-tokens", help="Max tokens per response"),
    detach: bool = Option(False, "--detach", help="Run in background"),
    log: Optional[str] = Option(None, "--log", help="Log file path (detach mode)"),
    safe: bool = Option(False, "--safe", help="Force safe concurrency/token limits"),
):
    """Start the MLX-LM OpenAI-compatible server."""
    from .server import serve
    serve(model, host=host, port=port, max_tokens=max_tokens, detach=detach, log=log, safe=safe)


@serve_app.command("stop")
def serve_stop_cmd(
    port: int = Option(DEFAULT_PORT, "--port", help="Port of the server to stop"),
):
    """Stop a detached server."""
    from .server import serve_stop
    serve_stop(port=port)


@serve_app.command("status")
def serve_status_cmd(
    port: int = Option(DEFAULT_PORT, "--port", help="Port to check"),
):
    """Show server status."""
    from .server import serve_status
    serve_status(port=port)


@serve_app.command("options")
def serve_options_cmd():
    """Show MLX-LM server options and safe defaults."""
    from .server import serve_options
    serve_options()


# ── opencode ────────────────────────────────────────────────────────

opencode_app = typer.Typer(help="OpenCode integration helpers.", no_args_is_help=True)
app.add_typer(opencode_app, name="opencode")


@opencode_app.command("snippet")
def opencode_snippet_cmd(
    model: str = Argument(help="HF repo (org/name)"),
    port: int = Option(DEFAULT_PORT, "--port", help="Server port"),
    provider_name: str = Option("MLX Local", "--provider-name", help="Provider name"),
):
    """Print an OpenCode provider JSON snippet."""
    from .server import opencode_snippet
    opencode_snippet(model, port=port, provider_name=provider_name)


# ── guide ───────────────────────────────────────────────────────────

guide_app = typer.Typer(help="Guides and best practices.", no_args_is_help=True)
app.add_typer(guide_app, name="guide")


@guide_app.command("opencode")
def guide_opencode_cmd():
    """Print best practices for using MLX-LM with OpenCode."""
    from .server import guide_opencode
    guide_opencode()


# ── ssh ─────────────────────────────────────────────────────────────

ssh_app = typer.Typer(help="SSH tunnel management (opt-in).", no_args_is_help=True)
app.add_typer(ssh_app, name="ssh")


@ssh_app.command("tunnel")
def ssh_tunnel_cmd(
    to: str = Option(..., "--to", help="SSH destination (user@host)"),
    remote_port: int = Option(DEFAULT_PORT, "--remote-port", help="Remote server port"),
    local_port: int = Option(DEFAULT_PORT, "--local-port", help="Local port to forward to"),
    key: str = Option("~/.ssh/id_ed25519", "--key", help="SSH key path"),
    bind: str = Option("127.0.0.1", "--bind", help="Local bind address"),
    detach: bool = Option(False, "--detach", help="Run in background"),
):
    """Create an SSH local forward tunnel."""
    from .ssh import tunnel
    tunnel(to=to, remote_port=remote_port, local_port=local_port, key=key, bind=bind, detach=detach)


@ssh_app.command("status")
def ssh_status_cmd(
    local_port: int = Option(DEFAULT_PORT, "--local-port", help="Local port to check"),
):
    """Check status of an SSH tunnel."""
    from .ssh import ssh_status
    ssh_status(local_port=local_port)


@ssh_app.command("stop")
def ssh_stop_cmd(
    local_port: int = Option(DEFAULT_PORT, "--local-port", help="Local port of tunnel to stop"),
):
    """Stop an SSH tunnel."""
    from .ssh import ssh_stop
    ssh_stop(local_port=local_port)


@ssh_app.command("snippet")
def ssh_snippet_cmd(
    to: str = Option(..., "--to", help="SSH destination (user@host)"),
    remote_port: int = Option(DEFAULT_PORT, "--remote-port", help="Remote server port"),
    local_port: int = Option(DEFAULT_PORT, "--local-port", help="Local port"),
    key: str = Option("~/.ssh/id_ed25519", "--key", help="SSH key path"),
    bind: str = Option("127.0.0.1", "--bind", help="Local bind address"),
):
    """Print SSH command and OpenCode base URL."""
    from .ssh import ssh_snippet
    ssh_snippet(to=to, remote_port=remote_port, local_port=local_port, key=key, bind=bind)


# ── version ─────────────────────────────────────────────────────────

@app.command()
def version():
    """Show version."""
    ui.banner(__version__)


# ── main ────────────────────────────────────────────────────────────

def main():
    app()


if __name__ == "__main__":
    main()
