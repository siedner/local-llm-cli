"""Chat command: interactive chat with a model via Engine."""

from __future__ import annotations

import re
import sys

from . import ui
from .config import get_generation_settings
from .engine import Engine


def chat(
    model: str,
    temp: float | None = None,
    top_p: float | None = None,
    top_k: int = 50,
    max_tokens: int | None = None,
) -> None:
    """Start an interactive chat session backed by the MLX server.

    Parameters default to the persisted generation settings from config.
    The server starts automatically on the first message and persists
    across the session so model weights stay in memory.
    """
    # Load persisted settings as defaults
    gs = get_generation_settings()
    if temp is None:
        temp = gs["temp"]
    if top_p is None:
        top_p = gs["top_p"]
    if max_tokens is None:
        max_tokens = gs["max_tokens"]
    system_prompt = gs.get("system_prompt", "You are a helpful assistant.")

    if top_k != 50:
        ui.warning("Note: --top-k is not directly used via the server and will be ignored.")

    engine = Engine()
    ui.info(f"Chat with {model}")
    ui.info(f"Settings: temp={temp}  top_p={top_p}  max_tokens={max_tokens}")
    ui.info("Type your messages.  Ctrl+C or Ctrl+D to exit.\n")

    # Start server eagerly so user doesn't wait on the first message
    try:
        engine.ensure_server(model, on_status=lambda m: ui.info(m))
    except RuntimeError as e:
        ui.error(str(e))
        return

    ui.divider()

    history: list[dict] = []

    while True:
        try:
            prompt = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            ui.console.print()
            ui.info("Chat ended.")
            return

        if not prompt:
            continue

        history.append({"role": "user", "content": prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
        ]

        try:
            full_text = ""
            sys.stdout.write("\n")
            for chunk in engine.chat_stream(
                messages,
                temperature=temp,
                top_p=top_p,
                max_tokens=max_tokens,
            ):
                full_text += chunk
                sys.stdout.write(chunk)
                sys.stdout.flush()

            sys.stdout.write("\n\n")

            # Strip thinking tags for history storage
            clean = re.sub(r'<think>.*?(</think>|$)', '', full_text, flags=re.DOTALL).strip()
            history.append({"role": "assistant", "content": clean})

        except RuntimeError as e:
            ui.error(f"Generation error: {e}")
        except KeyboardInterrupt:
            ui.console.print()
            ui.info("Generation interrupted.")
