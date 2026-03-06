"""Engine: server lifecycle management and HTTP-based streaming chat.

Central module that owns the MLX server process.  Provides:
 - ensure_server(model)  – start / swap the running model
 - stop_server()         – gracefully stop the server
 - chat_stream(messages)  – stream tokens from /v1/chat/completions
"""

from __future__ import annotations

import http.client
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Generator, Optional

from .constants import DEFAULT_HOST, DEFAULT_PORT, LOGS_DIR, PIDS_DIR
from .config import get_profile, load_config
from .doctor import get_mlx_python


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _pid_file(port: int) -> Path:
    return PIDS_DIR / f"server-{port}.pid"


def _log_file(port: int) -> Path:
    return LOGS_DIR / f"server-{port}.log"


def _model_file(port: int) -> Path:
    """Stores the model repo currently served on a port."""
    return PIDS_DIR / f"server-{port}.model"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Engine:
    """Manages one MLX-LM server process and provides HTTP chat on top of it.

    Designed for use as a singleton within the TUI or CLI chat loop.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.host = host
        self.port = port
        self._proc: Optional[subprocess.Popen] = None
        self._model: Optional[str] = None

        # Try to recover state from a previous session
        self._recover_existing()

    # -- public API ---------------------------------------------------------

    def get_running_model(self) -> Optional[str]:
        """Return the currently loaded model repo, or ``None``."""
        if self._model and self._is_alive():
            return self._model
        return None

    def ensure_server(
        self,
        model: str,
        *,
        max_tokens: Optional[int] = None,
        on_status: Optional[callable] = None,
    ) -> None:
        """Guarantee that *model* is being served.

        * If the same model is already running, this is a no-op.
        * If a different model is running, it is stopped first.
        * The method blocks until the server passes a health-check
          (up to ~30 s).

        *on_status* is an optional callback ``(str) -> None`` for progress
        messages (e.g. "stopping old model…").
        """
        def _status(msg: str) -> None:
            if on_status:
                on_status(msg)

        # Already running the right model?
        if self._model == model and self._is_alive():
            _status(f"Server already running with {model}")
            return

        # Different model or dead server → stop whatever is there
        if self._model or self._proc:
            _status(f"Stopping server for {self._model or 'unknown'}…")
            self.stop_server()

        _status(f"Starting server for {model}…")
        self._start_server(model, max_tokens=max_tokens)

        # Wait for health
        _status("Waiting for server to become ready…")
        if not self._wait_healthy(timeout=60):
            _status("Server failed to start — check logs.")
            raise RuntimeError(
                f"MLX server did not become healthy within 60 s.  "
                f"Log: {_log_file(self.port)}"
            )

        self._model = model
        _status(f"Server ready — {model}")

    def stop_server(self) -> None:
        """Stop the running server, if any."""
        # 1. Try our tracked process
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)

        # 2. Try PID file (covers detached / orphan servers)
        pid_file = _pid_file(self.port)
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                # Give it a moment to exit
                for _ in range(20):
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.25)
                    except ProcessLookupError:
                        break
            except (ProcessLookupError, ValueError, OSError):
                pass
            pid_file.unlink(missing_ok=True)

        # 3. Last-resort: lsof kill on port
        self._lsof_kill()

        # Clean up state files
        _model_file(self.port).unlink(missing_ok=True)
        self._proc = None
        self._model = None

    def chat_stream(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> Generator[str, None, None]:
        """Stream chat-completion tokens from the running server.

        *messages* is the OpenAI-style ``[{role, content}, …]`` list.
        Yields individual text chunks as they arrive.
        """
        if not self._is_alive():
            raise RuntimeError("No server running — call ensure_server first.")

        body = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }).encode()

        conn = http.client.HTTPConnection(self.host, self.port, timeout=120)
        try:
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()

            if resp.status != 200:
                error_body = resp.read().decode(errors="replace")
                raise RuntimeError(
                    f"Server returned {resp.status}: {error_body[:500]}"
                )

            # --- SSE streaming ---
            for raw_line in self._iter_lines(resp):
                line = raw_line.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    payload = line[6:]
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        finally:
            conn.close()

    def health_check(self) -> bool:
        """Return ``True`` if the server responds on ``/v1/models``."""
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            resp.read()  # drain
            conn.close()
            return resp.status == 200
        except Exception:
            return False

    # -- private ------------------------------------------------------------

    def _start_server(
        self,
        model: str,
        max_tokens: Optional[int] = None,
    ) -> None:
        _ensure_dirs()
        config = load_config()
        profile = get_profile(config)

        if max_tokens is None:
            max_tokens = profile["max_tokens"] if profile else 2048

        python = get_mlx_python()
        cmd = [
            python, "-m", "mlx_lm.server",
            "--model", model,
            "--host", self.host,
            "--port", str(self.port),
            "--max-tokens", str(max_tokens),
        ]

        log_path = _log_file(self.port)
        log_fh = open(log_path, "w")
        self._proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        # Persist PID + model for recovery
        _pid_file(self.port).write_text(str(self._proc.pid))
        _model_file(self.port).write_text(model)
        self._model = model

    def _wait_healthy(self, timeout: int = 60) -> bool:
        """Poll the health endpoint until it succeeds or *timeout* elapses."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                # Server process exited unexpectedly
                return False
            if self.health_check():
                return True
            time.sleep(1)
        return False

    def _is_alive(self) -> bool:
        """Quick check: is our tracked process still running?"""
        if self._proc and self._proc.poll() is None:
            return True
        # Fallback: PID file
        pid_file = _pid_file(self.port)
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, ValueError, OSError):
                pass
        return False

    def _recover_existing(self) -> None:
        """Try to adopt an already-running server from a previous session."""
        pid_file = _pid_file(self.port)
        model_file = _model_file(self.port)
        if not pid_file.exists():
            return
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # check alive
        except (ProcessLookupError, ValueError, OSError):
            pid_file.unlink(missing_ok=True)
            model_file.unlink(missing_ok=True)
            return

        # Process exists — see if we know what model
        if model_file.exists():
            self._model = model_file.read_text().strip()
        else:
            # Try to read from /v1/models
            try:
                conn = http.client.HTTPConnection(self.host, self.port, timeout=3)
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                data = json.loads(resp.read())
                conn.close()
                models = data.get("data", [])
                if models:
                    self._model = models[0].get("id")
            except Exception:
                self._model = None

    def _lsof_kill(self) -> None:
        """Last resort: kill whatever is on our port via lsof."""
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{self.port}", "-sTCP:LISTEN"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split("\n"):
                pid_str = pid_str.strip()
                if pid_str:
                    try:
                        os.kill(int(pid_str), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass

    @staticmethod
    def _iter_lines(resp: http.client.HTTPResponse) -> Generator[str, None, None]:
        """Yield lines from a chunked / streaming HTTP response."""
        buf = ""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk.decode(errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                yield line
        if buf:
            yield buf
