"""Chat command: interactive chat with a model via Engine."""

from __future__ import annotations

import re
import sys
import uuid

from . import ui
from .config import get_generation_settings, get_runtime_settings
from .engine import Engine


def chat(
    model: str,
    temp: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    profile: str | None = None,
    session: str | None = None,
    keep_alive_seconds: int | None = None,
    safe: bool | None = None,
    max_context: int | None = None,
) -> None:
    """Start an interactive chat session backed by the local daemon.

    Parameters default to the persisted generation settings from config.
    The daemon starts automatically on the first message and persists
    across the session so model weights stay warm in memory.
    """
    # Load persisted settings as defaults
    gs = get_generation_settings()
    runtime = get_runtime_settings()
    if temp is None:
        temp = gs["temp"]
    if top_p is None:
        top_p = gs["top_p"]
    if max_tokens is None:
        max_tokens = gs["max_tokens"]
    if top_k is None:
        top_k = gs.get("top_k", 50)
    system_prompt = gs.get("system_prompt", "You are a helpful assistant.")

    engine = Engine()
    session_id = session or f"cli-{uuid.uuid4().hex[:8]}"
    ui.info(f"Chat with {model}")
    ui.info(
        "Settings: "
        f"temp={temp}  top_p={top_p}  top_k={top_k}  "
        f"max_tokens={max_tokens}  request_timeout={runtime['request_timeout_seconds']}s"
    )
    ui.info("Type your messages.  Ctrl+C or Ctrl+D to exit.\n")

    # Start server eagerly so user doesn't wait on the first message
    try:
        engine.ensure_server(
            model,
            profile=profile,
            keep_alive_seconds=keep_alive_seconds,
            safe_mode=True if safe is None else safe,
            on_status=lambda m: ui.info(m),
        )
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
                top_k=top_k,
                max_tokens=max_tokens,
                session=session_id,
                max_context=max_context,
                keep_alive_seconds=keep_alive_seconds,
                profile=profile,
                safe=safe,
            ):
                full_text += chunk
                sys.stdout.write(chunk)
                sys.stdout.flush()

            sys.stdout.write("\n\n")
            summary = engine.last_summary.get("local_llm", {}).get("metrics", {})
            usage = engine.last_summary.get("usage", {})
            if summary or usage:
                ui.info(
                    "Summary: "
                    f"finish={summary.get('finish_reason', 'unknown')}  "
                    f"output={usage.get('completion_tokens', 0)}  "
                    f"ttft={summary.get('ttft_seconds', 0):.2f}s  "
                    f"tok/s={summary.get('generation_tps', 0):.2f}"
                )
                ui.console.print()

            # Strip thinking tags for history storage
            clean = re.sub(r'<think>.*?(</think>|$)', '', full_text, flags=re.DOTALL).strip()
            history.append({"role": "assistant", "content": clean})

        except RuntimeError as e:
            ui.error(f"Generation error: {e}")
        except KeyboardInterrupt:
            ui.console.print()
            ui.info("Generation interrupted.")
