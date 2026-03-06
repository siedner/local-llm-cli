"""CLI entrypoint for local-llm."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

try:
    import typer
    from typer import Argument, Option
except ImportError:
    print("Error: typer is required. Install with: pip install typer")
    sys.exit(1)

from . import __version__, ui
from .config import detect_profile, get_effective_profile, get_profile_name, load_config, save_config
from .constants import DEFAULT_HOST, DEFAULT_PORT, PROFILES, RECOMMENDED_MODELS
from .server import (
    benchmark_runtime,
    calibrate_profile,
    daemon_start,
    daemon_status,
    daemon_stop,
    guide_opencode,
    inspect_identifier,
    install_launchd,
    opencode_snippet,
    serve,
    serve_options,
    serve_status,
    serve_stop,
    show_ps,
    tail_logs,
    uninstall_launchd,
)

app = typer.Typer(
    name="local-llm",
    help="Mac-first local inference manager for Apple Silicon.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context):
    """Launch interactive TUI when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        from .tui import CommandPalette

        CommandPalette().run(mouse=False)


@app.command()
def doctor(
    fix: bool = Option(False, "--fix", help="Attempt to install missing dependencies"),
    yes: bool = Option(False, "--yes", "-y", help="Auto-confirm installs"),
):
    """Check environment and install missing prerequisites."""
    from .doctor import fix_missing, run_checks

    ui.header("Doctor")
    checks = run_checks()
    for check in checks:
        ui.check(check.name, check.ok, detail=check.detail, hint=check.fix_hint or "")
    ui.console.print()

    if all(check.ok for check in checks):
        ui.success("All checks passed.")
    elif fix:
        ui.info("Attempting to fix issues...")
        for action in fix_missing(yes=yes):
            ui.info(f"-> {action}")
    else:
        ui.warning("Some checks failed. Run `local-llm doctor --fix` to attempt auto-install.")


profile_app = typer.Typer(help="Manage hardware profiles.", no_args_is_help=True)
app.add_typer(profile_app, name="profile")


@profile_app.command("list")
def profile_list():
    """List available profiles."""
    config = load_config()
    current = get_profile_name(config)
    ui.header("Profiles")
    for name, profile in PROFILES.items():
        marker = "  <-- active" if name == current else ""
        ui.kv(
            name,
            f"{profile['chip_pattern']} {profile['memory_gb']}GB, "
            f"context={profile['default_context']}/{profile['hard_context']}, "
            f"max_output={profile['max_tokens']}{marker}",
        )


@profile_app.command("current")
def profile_current():
    """Show the current profile."""
    config = load_config()
    name, profile = get_effective_profile(config)
    ui.kv("Current profile", name)
    ui.kv("Context", f"{profile['default_context']} / hard {profile['hard_context']}")
    ui.kv("Keep alive", f"{profile['keep_alive_seconds']}s")


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


@profile_app.command("calibrate")
def profile_calibrate(
    model: Optional[str] = Argument(None, help="Model to benchmark"),
    profile: Optional[str] = Option(None, "--profile", help="Profile to calibrate"),
    runs: int = Option(5, "--runs", help="Benchmark runs"),
):
    """Run a conservative calibration benchmark and persist the results."""
    config = load_config()
    profile_name, effective = get_effective_profile(config, profile)
    benchmark_model = model or effective["recommended_models"][0]
    calibrate_profile(benchmark_model, profile=profile_name, runs=runs)


models_app = typer.Typer(help="Manage local models.", no_args_is_help=True)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list(
    disk: bool = Option(True, "--disk/--no-disk", help="Show disk usage"),
    as_json: bool = Option(False, "--json", help="Output as JSON"),
):
    """List installed models from HF cache."""
    from .models import list_models

    models = list_models(disk=disk)
    if as_json:
        ui.console.print_json(json.dumps(models, default=str))
        return

    if not models:
        ui.info("No models found in HF cache.")
        return

    table = ui.styled_table(title="Installed Models")
    table.add_column("Model")
    table.add_column("Format", style="cyan")
    if disk:
        table.add_column("Size", style="dim")
    table.add_column("Use", style="dim")
    for model in models:
        row = [
            model["repo"],
            model.get("quantization") or "unknown",
        ]
        if disk:
            row.append(model.get("disk_usage", "unknown"))
        row.append(model.get("summary", "Installed model"))
        table.add_row(*row)
    ui.console.print(table)


@models_app.command("install")
def models_install(
    repo: str = Argument(help="HF repo (org/name)"),
    yes: bool = Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Download a model."""
    from .models import install_model

    install_model(repo, yes=yes)


@models_app.command("remove")
def models_remove(
    repo: str = Argument(help="HF repo (org/name)"),
    yes: bool = Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a model from cache."""
    from .models import remove_model

    remove_model(repo, yes=yes)


@models_app.command("verify")
def models_verify(repo: str = Argument(help="HF repo (org/name)")):
    """Verify model cache integrity."""
    from .models import verify_model

    if not verify_model(repo):
        raise typer.Exit(1)


@models_app.command("repair")
def models_repair(
    repo: str = Argument(help="HF repo (org/name)"),
    yes: bool = Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Repair a broken model by re-downloading it."""
    from .models import repair_model

    if not repair_model(repo, yes=yes):
        raise typer.Exit(1)


@models_app.command("prune")
def models_prune(
    yes: bool = Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Prune invalid cache entries."""
    from .models import prune_models

    prune_models(yes=yes)


@models_app.command("warm")
def models_warm(
    repo: str = Argument(help="HF repo (org/name)"),
    host: str = Option(DEFAULT_HOST, "--host"),
    port: int = Option(DEFAULT_PORT, "--port"),
):
    """Warm a model in the daemon."""
    from .models import warm_model

    warm_model(repo, host=host, port=port)


@models_app.command("recommended")
def models_recommended():
    """Show recommended models."""
    ui.header("Recommended Models")
    for model in RECOMMENDED_MODELS:
        ui.kv(model["repo"], model["description"])


@models_app.command("scan")
def models_scan(
    root: Path = Option(Path.home(), "--root", help="Directory root to scan for model weights"),
    save: Path = Option(Path.home() / "llm_scan_report.txt", "--save", help="Path to save the scan report"),
    min_size_mb: int = Option(200, "--min-size-mb", help="Minimum file size in MB to include"),
):
    """Scan the system for large LLM weights."""
    from .models import scan_models

    scan_models(root=root, save_report=save, min_size_mb=min_size_mb)


@app.command()
def chat(
    model: str = Argument(help="HF repo (org/name) of the model"),
    temp: Optional[float] = Option(None, "--temp", help="Temperature"),
    top_p: Optional[float] = Option(None, "--top-p", help="Top-p sampling"),
    top_k: Optional[int] = Option(None, "--top-k", help="Top-k sampling"),
    max_tokens: Optional[int] = Option(None, "--max-tokens", help="Max tokens per response"),
    profile: Optional[str] = Option(None, "--profile", help="Hardware profile override"),
    session: Optional[str] = Option(None, "--session", help="Stable session id for warm-path reuse"),
    keep_alive: Optional[int] = Option(None, "--keep-alive", help="Idle keep-alive in seconds"),
    safe: bool = Option(True, "--safe/--unsafe", help="Use conservative profile limits"),
    max_context: Optional[int] = Option(None, "--max-context", help="Context budget for the request"),
):
    """Start an interactive chat session."""
    from .chat import chat as do_chat

    do_chat(
        model,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        profile=profile,
        session=session,
        keep_alive_seconds=keep_alive,
        safe=safe,
        max_context=max_context,
    )


serve_app = typer.Typer(help="Manage the daemon-backed runtime.", no_args_is_help=True)
app.add_typer(serve_app, name="serve")


@serve_app.command("start")
def serve_start(
    model: str = Argument(help="HF repo (org/name) of the model"),
    host: str = Option(DEFAULT_HOST, "--host"),
    port: int = Option(DEFAULT_PORT, "--port"),
    max_tokens: Optional[int] = Option(None, "--max-tokens"),
    profile: Optional[str] = Option(None, "--profile", help="Hardware profile override"),
    keep_alive: Optional[int] = Option(None, "--keep-alive", help="Idle keep-alive in seconds"),
    log: Optional[str] = Option(None, "--log", help="Daemon log file"),
    safe: bool = Option(True, "--safe/--unsafe", help="Use conservative profile limits"),
):
    """Start the daemon and warm a model."""
    serve(
        model,
        host=host,
        port=port,
        max_tokens=max_tokens,
        profile=profile,
        keep_alive_seconds=keep_alive,
        log=log,
        safe=safe,
    )


@serve_app.command("stop")
def serve_stop_cmd(port: int = Option(DEFAULT_PORT, "--port")):
    """Stop the daemon."""
    serve_stop(port=port)


@serve_app.command("status")
def serve_status_cmd(port: int = Option(DEFAULT_PORT, "--port")):
    """Show runtime status."""
    serve_status(port=port)


@serve_app.command("options")
def serve_options_cmd():
    """Show daemon runtime defaults."""
    serve_options()


daemon_app = typer.Typer(help="Manage the local daemon.", no_args_is_help=True)
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start_cmd(
    host: str = Option(DEFAULT_HOST, "--host"),
    port: int = Option(DEFAULT_PORT, "--port"),
    detach: bool = Option(True, "--detach/--foreground"),
    log: Optional[str] = Option(None, "--log"),
):
    """Start the daemon."""
    daemon_start(host=host, port=port, detach=detach, log=log)


@daemon_app.command("stop")
def daemon_stop_cmd(port: int = Option(DEFAULT_PORT, "--port")):
    """Stop the daemon."""
    daemon_stop(port=port)


@daemon_app.command("status")
def daemon_status_cmd(
    host: str = Option(DEFAULT_HOST, "--host"),
    port: int = Option(DEFAULT_PORT, "--port"),
):
    """Show daemon status."""
    daemon_status(host=host, port=port)


@daemon_app.command("install-launchd")
def daemon_install_launchd_cmd(
    host: str = Option(DEFAULT_HOST, "--host"),
    port: int = Option(DEFAULT_PORT, "--port"),
):
    """Install the daemon as a launchd agent."""
    install_launchd(host=host, port=port)


@daemon_app.command("uninstall-launchd")
def daemon_uninstall_launchd_cmd():
    """Remove the launchd agent."""
    uninstall_launchd()


@app.command("ps")
def ps_cmd(
    host: str = Option(DEFAULT_HOST, "--host"),
    port: int = Option(DEFAULT_PORT, "--port"),
):
    """Show the current daemon runtime state."""
    show_ps(host=host, port=port)


@app.command()
def inspect(
    identifier: str = Argument(help="Session id or model repo"),
    host: str = Option(DEFAULT_HOST, "--host"),
    port: int = Option(DEFAULT_PORT, "--port"),
):
    """Inspect a session or the loaded model."""
    inspect_identifier(identifier, host=host, port=port)


@app.command()
def logs(
    follow: bool = Option(False, "--follow", "-f", help="Follow daemon logs"),
):
    """Print daemon logs."""
    tail_logs(follow=follow)


@app.command()
def benchmark(
    model: str = Argument(help="HF repo (org/name)"),
    runs: int = Option(5, "--runs"),
    prompt: Optional[str] = Option(None, "--prompt"),
    profile: Optional[str] = Option(None, "--profile"),
):
    """Run a warm-path benchmark through the daemon."""
    benchmark_runtime(model, runs=runs, prompt=prompt, profile=profile)


opencode_app = typer.Typer(help="OpenCode integration helpers.", no_args_is_help=True)
app.add_typer(opencode_app, name="opencode")


@opencode_app.command("snippet")
def opencode_snippet_cmd(
    model: str = Argument(help="HF repo (org/name)"),
    port: int = Option(DEFAULT_PORT, "--port"),
    provider_name: str = Option("MLX Local", "--provider-name"),
):
    """Print an OpenCode provider JSON snippet."""
    opencode_snippet(model, port=port, provider_name=provider_name)


guide_app = typer.Typer(help="Guides and best practices.", no_args_is_help=True)
app.add_typer(guide_app, name="guide")


@guide_app.command("opencode")
def guide_opencode_cmd():
    """Print best practices for using local-llm with OpenCode."""
    guide_opencode()


ssh_app = typer.Typer(help="SSH tunnel management (opt-in).", no_args_is_help=True)
app.add_typer(ssh_app, name="ssh")


@ssh_app.command("tunnel")
def ssh_tunnel_cmd(
    to: str = Option(..., "--to", help="SSH destination (user@host)"),
    remote_port: int = Option(DEFAULT_PORT, "--remote-port"),
    local_port: int = Option(DEFAULT_PORT, "--local-port"),
    key: str = Option("~/.ssh/id_ed25519", "--key"),
    bind: str = Option("127.0.0.1", "--bind"),
    detach: bool = Option(False, "--detach"),
):
    """Create an SSH local forward tunnel."""
    from .ssh import tunnel

    tunnel(to=to, remote_port=remote_port, local_port=local_port, key=key, bind=bind, detach=detach)


@ssh_app.command("status")
def ssh_status_cmd(local_port: int = Option(DEFAULT_PORT, "--local-port")):
    """Check status of an SSH tunnel."""
    from .ssh import ssh_status

    ssh_status(local_port=local_port)


@ssh_app.command("stop")
def ssh_stop_cmd(local_port: int = Option(DEFAULT_PORT, "--local-port")):
    """Stop an SSH tunnel."""
    from .ssh import ssh_stop

    ssh_stop(local_port=local_port)


@ssh_app.command("snippet")
def ssh_snippet_cmd(
    to: str = Option(..., "--to", help="SSH destination (user@host)"),
    remote_port: int = Option(DEFAULT_PORT, "--remote-port"),
    local_port: int = Option(DEFAULT_PORT, "--local-port"),
    key: str = Option("~/.ssh/id_ed25519", "--key"),
    bind: str = Option("127.0.0.1", "--bind"),
):
    """Print SSH command and base URL."""
    from .ssh import ssh_snippet

    ssh_snippet(to=to, remote_port=remote_port, local_port=local_port, key=key, bind=bind)


@app.command()
def version():
    """Show version."""
    ui.banner(__version__)


def main():
    app()


if __name__ == "__main__":
    main()
