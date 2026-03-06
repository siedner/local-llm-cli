"""Daemon-backed chat engine."""

from __future__ import annotations

import time
from typing import Generator, Optional

from .config import get_runtime_settings
from .constants import DEFAULT_HOST, DEFAULT_PORT
from .daemon_client import DaemonClient, DaemonError
from .server import daemon_start


class Engine:
    """Client-side engine that speaks to the local daemon."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        runtime = get_runtime_settings()
        self.host = host or runtime.get("host") or DEFAULT_HOST
        self.port = port or runtime.get("port") or DEFAULT_PORT
        self.client = DaemonClient(self.host, self.port)
        self._model: Optional[str] = None
        self.last_summary: dict = {}

    def get_running_model(self) -> Optional[str]:
        """Return the currently loaded model, if any."""
        try:
            snapshot = self.client.ps()
        except Exception:
            return None
        self._model = snapshot.get("loaded_model")
        return self._model

    @property
    def owns_process(self) -> bool:
        """Compatibility shim for the old TUI lifecycle."""
        return False

    def ensure_server(
        self,
        model: str,
        *,
        max_tokens: Optional[int] = None,
        profile: Optional[str] = None,
        keep_alive_seconds: Optional[int] = None,
        safe_mode: bool = True,
        on_status: Optional[callable] = None,
    ) -> None:
        """Ensure the daemon is running and the target model is warm."""

        def _status(message: str) -> None:
            if on_status:
                on_status(message)

        try:
            self.client.health()
        except Exception:
            _status("Starting daemon…")
            daemon_start(host=self.host, port=self.port, detach=True)
            deadline = time.time() + 30
            while time.time() < deadline:
                try:
                    self.client.health()
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise RuntimeError("Daemon did not become healthy within 30 seconds.")

        try:
            _status(f"Warming model {model}…")
            self.client.warm(
                model,
                keep_alive_seconds=keep_alive_seconds,
                profile=profile,
                safe_mode=safe_mode,
            )
            self._model = model
            _status(f"Runtime ready — {model}")
        except DaemonError as exc:
            raise RuntimeError(str(exc)) from exc

    def stop_server(self) -> None:
        """Do not auto-stop the daemon from interactive clients."""

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 512,
        session: Optional[str] = None,
        max_context: Optional[int] = None,
        keep_alive_seconds: Optional[int] = None,
        profile: Optional[str] = None,
        safe: Optional[bool] = None,
    ) -> Generator[str, None, None]:
        """Stream content from the daemon."""
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "stream": True,
            "session": session,
        }
        if max_context is not None:
            payload["max_context"] = max_context
        if keep_alive_seconds is not None:
            payload["keep_alive_seconds"] = keep_alive_seconds
        if profile is not None:
            payload["profile"] = profile
        if safe is not None:
            payload["safe"] = safe

        try:
            stream = self.client.chat_stream(payload)
            while True:
                try:
                    yield next(stream)
                except StopIteration as stop:
                    self.last_summary = stop.value or self.client.last_chat_summary or {}
                    break
        except DaemonError as exc:
            raise RuntimeError(str(exc)) from exc

    def health_check(self) -> bool:
        """Return True if the daemon is healthy."""
        try:
            self.client.health()
            return True
        except Exception:
            return False
