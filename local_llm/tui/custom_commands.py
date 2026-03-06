"""Project and user custom slash commands for the TUI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from local_llm.constants import PROJECT_COMMANDS_DIR, USER_COMMANDS_DIR


FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
VALID_COMMAND_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class CustomCommand:
    """A prompt-template slash command loaded from markdown."""

    name: str
    source: str
    path: Path
    description: str
    argument_hint: str
    template: str
    model: str | None = None
    profile: str | None = None
    safe: bool | None = None

    @property
    def slash_name(self) -> str:
        return f"/{self.name}"

    def expand(self, arguments: list[str]) -> str:
        """Expand placeholders into a final prompt."""
        expanded = self.template.replace("$ARGUMENTS", " ".join(arguments).strip())
        for index, value in enumerate(arguments, start=1):
            expanded = expanded.replace(f"${index}", value)
        return expanded.strip()


def discover_custom_commands(cwd: Path | None = None) -> dict[str, CustomCommand]:
    """Load project and user custom commands."""
    root = cwd or Path.cwd()
    commands: dict[str, CustomCommand] = {}
    for source, directory in (("project", root / PROJECT_COMMANDS_DIR), ("user", USER_COMMANDS_DIR)):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            command = _parse_command(path, source)
            if command is not None:
                commands[command.slash_name] = command
    return commands


def _parse_command(path: Path, source: str) -> CustomCommand | None:
    stem = path.stem.lower()
    if not VALID_COMMAND_RE.match(stem):
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None

    metadata: dict[str, str] = {}
    template = raw
    match = FRONTMATTER_RE.match(raw)
    if match:
        metadata = _parse_frontmatter(match.group(1))
        template = match.group(2).strip()

    safe_value = metadata.get("safe")
    safe = None
    if safe_value is not None:
        safe = safe_value.lower() in {"1", "true", "yes", "on", "safe"}

    return CustomCommand(
        name=stem,
        source=source,
        path=path,
        description=metadata.get("description", "Custom prompt command"),
        argument_hint=metadata.get("argument-hint", ""),
        template=template,
        model=metadata.get("model"),
        profile=metadata.get("profile"),
        safe=safe,
    )


def _parse_frontmatter(raw: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata
