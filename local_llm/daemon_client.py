"""HTTP client for the local-llm daemon."""

from __future__ import annotations

import http.client
import json
from typing import Generator, Optional
from urllib.parse import quote

from .constants import DEFAULT_HOST, DEFAULT_PORT


class DaemonError(RuntimeError):
    """Raised when the daemon returns an error."""


class DaemonClient:
    """Simple HTTP client for the local daemon API."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self.last_chat_summary: dict = {}

    def health(self) -> dict:
        return self._request_json("GET", "/v1/local/health", timeout=5)

    def ps(self) -> dict:
        return self._request_json("GET", "/v1/local/ps", timeout=5)

    def models(self) -> dict:
        return self._request_json("GET", "/v1/local/models", timeout=10)

    def warm(
        self,
        model: str,
        *,
        keep_alive_seconds: Optional[int] = None,
        profile: Optional[str] = None,
        safe_mode: bool = True,
    ) -> dict:
        payload = {
            "model": model,
            "keep_alive_seconds": keep_alive_seconds,
            "profile": profile,
            "safe_mode": safe_mode,
        }
        return self._request_json("POST", "/v1/local/warm", payload, timeout=600)

    def evict(self, model: Optional[str] = None) -> dict:
        return self._request_json("POST", "/v1/local/evict", {"model": model}, timeout=30)

    def cancel(self, request_id: Optional[str] = None) -> dict:
        return self._request_json("POST", "/v1/local/cancel", {"request_id": request_id}, timeout=5)

    def inspect(self, identifier: str) -> dict:
        encoded = quote(identifier, safe="")
        return self._request_json("GET", f"/v1/local/inspect?id={encoded}", timeout=5)

    def benchmark(self, payload: dict) -> dict:
        return self._request_json("POST", "/v1/local/benchmark", payload, timeout=1200)

    def chat_stream(self, payload: dict) -> Generator[str, None, dict]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            error_body = resp.read().decode(errors="replace")
            conn.close()
            raise DaemonError(f"Daemon returned {resp.status}: {error_body[:500]}")

        final_payload: dict = {}
        try:
            for raw_line in self._iter_lines(resp):
                line = raw_line.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                if payload.get("object") == "local_llm.summary":
                    final_payload = payload
                    continue
                delta = payload["choices"][0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
        finally:
            conn.close()
        self.last_chat_summary = final_payload
        return final_payload

    def chat(self, payload: dict) -> dict:
        return self._request_json("POST", "/v1/chat/completions", payload, timeout=600)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        *,
        timeout: int = 60,
    ) -> dict:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        body = None
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode(errors="replace")
        conn.close()

        if resp.status >= 400:
            raise DaemonError(raw or f"Daemon returned {resp.status}")
        if not raw:
            return {}
        return json.loads(raw)

    @staticmethod
    def _iter_lines(resp) -> Generator[str, None, None]:
        while True:
            line = resp.readline()
            if not line:
                break
            yield line.decode(errors="replace").rstrip("\r\n")
