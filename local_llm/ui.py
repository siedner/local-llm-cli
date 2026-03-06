"""Minimal monochrome TUI theme and helpers (Claude Code-inspired)."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── Theme ────────────────────────────────────────────────────────────

theme = Theme(
    {
        "info": "dim",
        "success": "bold white",
        "warning": "yellow",
        "error": "bold red",
        "accent": "bold white",
        "muted": "dim",
        "key": "white",
        "value": "dim",
    }
)

console = Console(theme=theme)


# ── Helpers ──────────────────────────────────────────────────────────


def banner(version: str) -> None:
    """Styled startup header with version."""
    text = Text()
    text.append("local-llm", style="bold white")
    text.append(f" v{version}", style="dim")
    console.print(text)


def header(title: str) -> None:
    """Section header with subtle underline."""
    console.print()
    console.print(Text(title, style="bold white"))
    console.print(Text("─" * len(title), style="dim"))


def kv(key: str, value: str, indent: int = 2) -> None:
    """Key: value pair — key bold, value dim."""
    pad = " " * indent
    text = Text()
    text.append(f"{pad}{key}", style="white")
    text.append(f"  {value}", style="dim")
    console.print(text)


def success(msg: str) -> None:
    """Success message."""
    console.print(Text(f"  {msg}", style="bold white"))


def error(msg: str) -> None:
    """Error message."""
    console.print(Text(f"  {msg}", style="bold red"))


def warning(msg: str) -> None:
    """Warning message."""
    console.print(Text(f"  {msg}", style="yellow"))


def info(msg: str) -> None:
    """Informational (dim) message."""
    console.print(Text(f"  {msg}", style="dim"))


def check(name: str, ok: bool, detail: str = "", hint: str = "") -> None:
    """Status check line: ✓ or ✗ with name and detail."""
    mark = Text("✓", style="bold white") if ok else Text("✗", style="bold red")
    line = Text("  ")
    line.append(mark)
    line.append(f" {name}", style="white" if ok else "red")
    if detail:
        line.append(f"  {detail}", style="dim")
    console.print(line)
    if hint and not ok:
        console.print(Text(f"    hint: {hint}", style="dim"))


def panel(content: str, title: str = "", padding: tuple[int, int] = (0, 1), box_style=box.ROUNDED) -> None:
    """Bordered panel with rounded box, dim border."""
    console.print(
        Panel(
            content,
            title=f"[bold white]{title}[/]" if title else None,
            title_align="left",
            border_style="dim",
            padding=padding,
            box=box_style,
            expand=False,
        )
    )


def rich_panel(renderable, title: str = "", padding: tuple[int, int] = (0, 1), box_style=box.ROUNDED) -> None:
    """Panel that accepts any Rich renderable (Text, Table, etc.)."""
    console.print(
        Panel(
            renderable,
            title=f"[bold white]{title}[/]" if title else None,
            title_align="left",
            border_style="dim",
            padding=padding,
            box=box_style,
            expand=False,
        )
    )


def code_block(code: str, lang: str = "json") -> None:
    """Syntax-highlighted code block inside a panel."""
    syntax = Syntax(code, lang, theme="ansi_dark", padding=0)
    console.print(
        Panel(
            syntax,
            border_style="dim",
            padding=(0, 1),
            box=box.ROUNDED,
            expand=False,
        )
    )


def confirm(prompt: str) -> bool:
    """Styled y/N confirmation prompt. Returns True if user confirms."""
    text = Text()
    text.append(f"  {prompt} ", style="white")
    text.append("[y/N]", style="dim")
    console.print(text, end=" ")
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return False
    return answer in ("y", "yes")


def divider() -> None:
    """Subtle horizontal rule."""
    console.print(Text("─" * 40, style="dim"))


def styled_table(title: str = "", **kwargs) -> Table:
    """Create a table with the project's style conventions."""
    return Table(
        title=f"[bold white]{title}[/]" if title else None,
        border_style="dim",
        header_style="bold white",
        show_edge=True,
        **kwargs,
    )
