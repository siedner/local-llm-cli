"""Long-lived local-llm daemon backed by MLX."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import signal
import threading
import time
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .budget import PromptBudgetError, validate_prompt_budget
from .config import (
    get_effective_profile,
    get_generation_settings,
    get_runtime_settings,
    load_config,
    record_benchmark,
)
from .constants import (
    CONFIG_SCHEMA_VERSION,
    DAEMON_JSONL_LOG_FILE,
    DAEMON_LOG_FILE,
    DAEMON_PID_FILE,
    DAEMON_STATE_FILE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOGS_DIR,
    PIDS_DIR,
    STATE_DIR,
)
from .memory import get_memory_pressure
from .mlx_runner import MLXRunner, RunnerMetrics
from .models import list_models


class BusyError(RuntimeError):
    """Raised when the runtime is already serving a request."""


class MemoryPressureError(RuntimeError):
    """Raised when the machine is under too much pressure for a new request."""


class RuntimeManager:
    """Stateful single-model runtime manager."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.config = load_config()
        self.runtime = get_runtime_settings(self.config)
        self.generation_defaults = get_generation_settings(self.config)
        self.profile_name, self.profile = get_effective_profile(self.config)
        self.runner = MLXRunner()

        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.runner_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.active_request_id: str | None = None
        self.active_cancel = threading.Event()
        self.active_session_id: str | None = None
        self.active_metrics: dict | None = None
        self.waiting_request_ids: list[str] = []
        self.sessions: dict[str, dict] = {}
        self.state = "stopped"
        self.last_error: str | None = None
        self.backoff_until = 0.0
        self.failure_count = 0
        self.loaded_model: str | None = None
        self.keep_alive_seconds = int(self.profile["keep_alive_seconds"])
        self.queue_limit = int(self.runtime["queue_limit"])
        self.request_timeout_seconds = int(self.runtime["request_timeout_seconds"])
        self.request_history: list[dict] = []

        self.logger = logging.getLogger("local_llm.daemon")
        self.json_logger = logging.getLogger("local_llm.daemon.json")
        self._reaper_thread = threading.Thread(target=self._reaper_loop, daemon=True)

    def start(self) -> None:
        self._configure_logging()
        self._refresh_config()
        self._reaper_thread.start()
        self._write_state_file()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.active_cancel.set()
        try:
            if self.runner.is_loaded:
                self.evict_model(reason="shutdown", force=True)
        finally:
            self._write_state_file()

    def snapshot(self) -> dict:
        with self.lock:
            self._prune_sessions_locked()
            memory = get_memory_pressure()
            session_count = len(self.sessions)
            return {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "host": self.host,
                "port": self.port,
                "profile": self.profile_name,
                "status": self.state,
                "loaded_model": self.loaded_model,
                "keep_alive_seconds": self.keep_alive_seconds,
                "queue_limit": self.queue_limit,
                "queue_depth": len(self.waiting_request_ids),
                "request_timeout_seconds": self.request_timeout_seconds,
                "active_request_id": self.active_request_id,
                "active_session_id": self.active_session_id,
                "failure_count": self.failure_count,
                "backoff_until": self.backoff_until,
                "last_error": self.last_error,
                "session_count": session_count,
                "memory_pressure": memory,
                "runner": {
                    "is_loaded": self.runner.is_loaded,
                    "loaded_at": self.runner.loaded_at,
                    "last_used_at": self.runner.last_used_at,
                    "load_duration_seconds": self.runner.load_duration_seconds,
                    "model_context_limit": self.runner.model_context_limit(),
                },
                "request_history": self.request_history[-10:],
                "active_metrics": self.active_metrics,
            }

    def available_models(self) -> dict:
        return {"models": list_models(disk=True, filter_relevant=False)}

    def inspect(self, identifier: str) -> dict:
        with self.lock:
            self._prune_sessions_locked()
            if identifier in self.sessions:
                return {"kind": "session", "session": self.sessions[identifier]}
            if self.loaded_model == identifier:
                return {"kind": "model", "snapshot": self.snapshot()}
        return {"kind": "unknown", "id": identifier}

    def warm_model(
        self,
        *,
        model: str,
        keep_alive_seconds: Optional[int] = None,
        profile_name: Optional[str] = None,
        safe_mode: bool = True,
    ) -> dict:
        with self.condition:
            self._guard_backoff()
            self._refresh_config(profile_name)
            self.keep_alive_seconds = int(keep_alive_seconds or self.profile["keep_alive_seconds"])
            if safe_mode:
                self.keep_alive_seconds = min(self.keep_alive_seconds, self.profile["keep_alive_seconds"])
            if self.active_request_id is not None:
                if self.loaded_model == model and self.runner.is_loaded:
                    return {
                        "ok": True,
                        "model": model,
                        "profile": self.profile_name,
                        "keep_alive_seconds": self.keep_alive_seconds,
                        "load_duration_seconds": self.runner.load_duration_seconds,
                        "reused": True,
                    }
                raise BusyError("Cannot switch models while a request is active.")
            self._set_state("loading")
        try:
            with self.runner_lock:
                info = self.runner.load_model(model)
        except Exception as exc:
            self._record_failure(exc)
            raise

        with self.lock:
            if self.loaded_model and self.loaded_model != model:
                self.sessions = {}
            self.loaded_model = model
            self._set_state("warm")
            self.last_error = None
            self.failure_count = 0
            self._write_state_file()
            self._log_event("warm_model", {"model": model, **info})
            return {
                "ok": True,
                "model": model,
                "profile": self.profile_name,
                "keep_alive_seconds": self.keep_alive_seconds,
                "load_duration_seconds": self.runner.load_duration_seconds,
                "reused": info["reused"],
            }

    def evict_model(self, *, reason: str = "manual", force: bool = False) -> dict:
        with self.condition:
            model = self.loaded_model
            if self.active_request_id is not None and not force:
                raise BusyError("Cannot evict the model while a request is active.")
            self._set_state("evicting")
        with self.runner_lock:
            self.runner.unload_model()
        with self.lock:
            self.loaded_model = None
            self.sessions = {}
            self._set_state("stopped")
            self._write_state_file()
            self._log_event("evict_model", {"model": model, "reason": reason})
            return {"ok": True, "evicted_model": model, "reason": reason}

    def cancel(self, request_id: Optional[str]) -> dict:
        with self.lock:
            if self.active_request_id is None:
                return {"ok": False, "cancelled": False}
            if request_id and request_id != self.active_request_id:
                return {"ok": False, "cancelled": False}
            self.active_cancel.set()
            self._log_event("cancel_request", {"request_id": self.active_request_id})
            return {"ok": True, "cancelled": True, "request_id": self.active_request_id}

    def run_chat(self, payload: dict):
        request = self._prepare_request(payload)
        request_id = str(uuid.uuid4())
        with self.condition:
            self._guard_backoff()
            while self.active_request_id is not None:
                if self.queue_limit <= 0:
                    raise BusyError("The runtime is busy with another request.")
                if request_id not in self.waiting_request_ids:
                    if len(self.waiting_request_ids) >= self.queue_limit:
                        raise BusyError("The runtime queue is full.")
                    self.waiting_request_ids.append(request_id)
                    self._write_state_file()
                remaining = request["request_timeout_seconds"] - (time.time() - request["requested_at"])
                if remaining <= 0:
                    if request_id in self.waiting_request_ids:
                        self.waiting_request_ids.remove(request_id)
                    self._write_state_file()
                    raise RuntimeError("Timed out while waiting for a free runtime slot.")
                self.condition.wait(timeout=min(0.5, remaining))
            if request_id in self.waiting_request_ids:
                self.waiting_request_ids.remove(request_id)
            self.active_request_id = request_id
            self.active_session_id = request["session_id"]
            self.active_cancel = threading.Event()
            started_at = time.time()
            self.active_metrics = {
                "requested_at": request["requested_at"],
                "started_at": started_at,
                "queue_wait_seconds": started_at - request["requested_at"],
                "timeout_seconds": request["request_timeout_seconds"],
                "prompt_tokens_estimate": request["prompt_tokens"],
                "memory_pressure": request["memory"]["state"],
                "model": request["model"],
                "session_id": request["session_id"],
            }
            self._set_state("busy")
            self._write_state_file()

        def iterator():
            full_text = ""
            final_metrics: RunnerMetrics | None = None
            timeout_triggered = threading.Event()

            def _cancel_for_timeout() -> None:
                timeout_triggered.set()
                self.active_cancel.set()

            timeout_timer = threading.Timer(request["request_timeout_seconds"], _cancel_for_timeout)
            timeout_timer.daemon = True
            timeout_timer.start()
            try:
                with self.runner_lock:
                    for event in self.runner.stream_chat(
                        messages=request["messages"],
                        temperature=request["temperature"],
                        top_p=request["top_p"],
                        top_k=request["top_k"],
                        max_tokens=request["max_output"],
                        cancel_event=self.active_cancel,
                        session_id=request["session_id"],
                    ):
                        if event["type"] == "delta":
                            full_text += event["text"]
                            yield {
                                "type": "delta",
                                "request_id": request_id,
                                "model": request["model"],
                                "text": event["text"],
                            }
                        else:
                            final_metrics = event["metrics"]
                if final_metrics is None:
                    raise RuntimeError("Generation completed without final metrics.")
                if timeout_triggered.is_set():
                    raise RuntimeError(
                        f"Request timed out after {request['request_timeout_seconds']} seconds."
                    )

                result = {
                    "id": request_id,
                    "model": request["model"],
                    "text": full_text,
                    "finish_reason": final_metrics.finish_reason,
                    "usage": {
                        "prompt_tokens": final_metrics.prompt_tokens,
                        "completion_tokens": final_metrics.generation_tokens,
                        "total_tokens": final_metrics.prompt_tokens + final_metrics.generation_tokens,
                    },
                    "metrics": asdict(final_metrics),
                    "session_id": request["session_id"],
                }
                result["metrics"]["queue_wait_seconds"] = self.active_metrics["queue_wait_seconds"]
                self._finalize_session(request, result)
                self._record_request(result)
                yield {"type": "final", "result": result}
            except Exception as exc:
                self._record_failure(exc)
                raise
            finally:
                timeout_timer.cancel()
                with self.condition:
                    self.active_request_id = None
                    self.active_session_id = None
                    self.active_cancel = threading.Event()
                    self.active_metrics = None
                    self._set_state("warm" if self.runner.is_loaded else "stopped")
                    self._write_state_file()
                    self.condition.notify_all()

        return request_id, iterator()

    def complete_chat(self, payload: dict) -> dict:
        _, iterator = self.run_chat(payload)
        final_result = None
        for event in iterator:
            if event["type"] == "final":
                final_result = event["result"]
        if final_result is None:
            raise RuntimeError("No completion produced.")
        return final_result

    def benchmark(self, payload: dict) -> dict:
        self._refresh_config(payload.get("profile"))
        model = payload["model"]
        runs = max(1, int(payload.get("runs", 5)))
        prompt = payload.get(
            "prompt",
            "Explain why predictable memory behavior matters for local Apple Silicon inference in two short sentences.",
        )
        max_tokens = int(payload.get("max_tokens", min(self.profile["max_tokens"], 128)))
        self.warm_model(model=model, profile_name=payload.get("profile"), safe_mode=True)

        times = []
        ttfts = []
        tps = []
        session_id = f"benchmark-{uuid.uuid5(uuid.NAMESPACE_URL, model)}"
        for _ in range(runs):
            result = self.complete_chat(
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.generation_defaults["system_prompt"]},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": max_tokens,
                    "stream": False,
                    "profile": payload.get("profile"),
                    "session": session_id,
                    "safe": True,
                }
            )
            metrics = result["metrics"]
            times.append(metrics["total_seconds"])
            ttfts.append(metrics["ttft_seconds"])
            tps.append(metrics["generation_tps"])

        benchmark = {
            "profile": payload.get("profile") or self.profile_name,
            "model": model,
            "runs": runs,
            "avg_total_seconds": sum(times) / len(times),
            "avg_ttft_seconds": sum(ttfts) / len(ttfts),
            "avg_generation_tps": sum(tps) / len(tps),
            "timestamp": time.time(),
        }
        record_benchmark(benchmark)
        return benchmark

    def _prepare_request(self, payload: dict) -> dict:
        model = payload.get("model")
        if not model:
            raise RuntimeError("model is required")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("messages are required")

        profile_name = payload.get("profile")
        self._refresh_config(profile_name)
        safe_mode = bool(payload.get("safe", self.runtime["safe_mode"]))
        self.warm_model(
            model=model,
            keep_alive_seconds=payload.get("keep_alive_seconds"),
            profile_name=profile_name,
            safe_mode=safe_mode,
        )

        profile_name, profile = get_effective_profile(self.config, profile_name)
        max_output = int(payload.get("max_tokens") or profile["max_tokens"])
        max_context = int(payload.get("max_context") or profile["default_context"])
        if safe_mode:
            max_context = min(max_context, profile["default_context"])
        max_context = min(max_context, profile["hard_context"])

        temperature = float(payload.get("temperature", self.generation_defaults["temp"]))
        top_p = float(payload.get("top_p", self.generation_defaults["top_p"]))
        top_k = int(payload.get("top_k", self.generation_defaults.get("top_k", 50)))

        memory = get_memory_pressure()
        if memory["state"] == "red":
            raise MemoryPressureError("Refusing new request while macOS memory pressure is red.")

        prompt_tokens = self.runner.estimate_prompt_tokens(messages)
        validate_prompt_budget(
            prompt_tokens=prompt_tokens,
            max_output_tokens=max_output,
            default_context=max_context,
            hard_context=profile["hard_context"],
            safe_mode=safe_mode,
        )

        keep_alive_seconds = int(payload.get("keep_alive_seconds") or profile["keep_alive_seconds"])
        if memory["state"] == "yellow":
            keep_alive_seconds = min(keep_alive_seconds, max(60, keep_alive_seconds // 2))

        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output": max_output,
            "max_context": max_context,
            "profile": profile_name,
            "prompt_tokens": prompt_tokens,
            "keep_alive_seconds": keep_alive_seconds,
            "safe_mode": safe_mode,
            "queue_limit": self.queue_limit,
            "request_timeout_seconds": self.request_timeout_seconds,
            "requested_at": time.time(),
            "memory": memory,
            "session_id": payload.get("session") or f"ephemeral-{uuid.uuid4().hex[:8]}",
        }

    def _finalize_session(self, request: dict, result: dict) -> None:
        with self.lock:
            self.keep_alive_seconds = request["keep_alive_seconds"]
            self.sessions[request["session_id"]] = {
                "id": request["session_id"],
                "model": request["model"],
                "updated_at": time.time(),
                "keep_alive_seconds": request["keep_alive_seconds"],
                "message_count": len(request["messages"]) + 1,
                "last_finish_reason": result["finish_reason"],
                "prompt_tokens": result["usage"]["prompt_tokens"],
                "completion_tokens": result["usage"]["completion_tokens"],
                "cache_hit": result["metrics"].get("cache_hit", False),
                "cached_prefix_tokens": result["metrics"].get("cached_prefix_tokens", 0),
            }
            self._prune_sessions_locked()
            self._write_state_file()

    def _record_request(self, result: dict) -> None:
        with self.lock:
            self._prune_sessions_locked()
            self.request_history.append(
                {
                    "id": result["id"],
                    "model": result["model"],
                    "session_id": result["session_id"],
                    "timestamp": time.time(),
                    "metrics": result["metrics"],
                }
            )
            self.request_history = self.request_history[-50:]
            self._log_event("request_complete", result)

    def _record_failure(self, exc: Exception) -> None:
        with self.lock:
            self.failure_count += 1
            self.last_error = str(exc)
            if self.failure_count >= 3:
                self.backoff_until = time.time() + 30
                self._set_state("crashed")
            else:
                self._set_state("failed")
            self._write_state_file()
            self._log_event("request_failed", {"error": str(exc), "failure_count": self.failure_count})

    def _refresh_config(self, profile_name: Optional[str] = None) -> None:
        self.config = load_config()
        self.runtime = get_runtime_settings(self.config)
        self.generation_defaults = get_generation_settings(self.config)
        self.profile_name, self.profile = get_effective_profile(self.config, profile_name)
        self.queue_limit = max(0, int(self.runtime.get("queue_limit", 0)))
        self.request_timeout_seconds = max(1, int(self.runtime.get("request_timeout_seconds", 300)))
        runtime_keep_alive = self.runtime.get("keep_alive_seconds")
        self.keep_alive_seconds = int(runtime_keep_alive or self.profile["keep_alive_seconds"])

    def _prune_sessions_locked(self) -> None:
        if not self.sessions:
            return

        now = time.time()
        expired: list[str] = []
        active_model = self.loaded_model
        for session_id, session in self.sessions.items():
            session_model = session.get("model")
            if active_model and session_model != active_model:
                expired.append(session_id)
                continue
            updated_at = float(session.get("updated_at", 0))
            ttl = int(session.get("keep_alive_seconds") or self.keep_alive_seconds or 0)
            if ttl > 0 and (now - updated_at) > ttl:
                expired.append(session_id)

        for session_id in expired:
            self.sessions.pop(session_id, None)
            self.runner.drop_session_cache(session_id)

    def _guard_backoff(self) -> None:
        if self.backoff_until and time.time() < self.backoff_until:
            remaining = int(self.backoff_until - time.time())
            raise RuntimeError(f"Runtime is cooling down after failures. Retry in {remaining}s.")

    def _set_state(self, state: str) -> None:
        self.state = state

    def _configure_logging(self) -> None:
        PIDS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        text_handler = logging.handlers.RotatingFileHandler(
            DAEMON_LOG_FILE,
            maxBytes=2_000_000,
            backupCount=3,
        )
        text_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(text_handler)

        self.json_logger.setLevel(logging.INFO)
        self.json_logger.handlers.clear()
        json_handler = logging.handlers.RotatingFileHandler(
            DAEMON_JSONL_LOG_FILE,
            maxBytes=2_000_000,
            backupCount=3,
        )
        json_handler.setFormatter(logging.Formatter("%(message)s"))
        self.json_logger.addHandler(json_handler)

    def _log_event(self, event: str, payload: dict) -> None:
        record = {
            "timestamp": time.time(),
            "event": event,
            "payload": payload,
        }
        self.logger.info("%s %s", event, payload)
        self.json_logger.info(json.dumps(record))

    def _write_state_file(self) -> None:
        DAEMON_STATE_FILE.write_text(json.dumps(self.snapshot(), indent=2) + "\n", encoding="utf-8")

    def _reaper_loop(self) -> None:
        while not self.stop_event.wait(5):
            with self.lock:
                self._prune_sessions_locked()
                if not self.runner.is_loaded or self.active_request_id is not None:
                    self._write_state_file()
                    continue
                last_used = self.runner.last_used_at or time.time()
                idle_seconds = time.time() - last_used
                memory = get_memory_pressure()
                if memory["state"] == "red":
                    reason = "memory_pressure_red"
                elif idle_seconds > self.keep_alive_seconds:
                    reason = "idle_timeout"
                else:
                    self._write_state_file()
                    continue
            try:
                self.evict_model(reason=reason)
            except Exception:
                continue


class DaemonHandler(BaseHTTPRequestHandler):
    """HTTP request handler bound to a RuntimeManager."""

    manager: RuntimeManager
    server_version = "local-llm-daemon/0.1"
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/v1/local/health":
                self._send_json(200, self.manager.snapshot())
            elif parsed.path == "/v1/local/ps":
                self._send_json(200, self.manager.snapshot())
            elif parsed.path == "/v1/local/models":
                self._send_json(200, self.manager.available_models())
            elif parsed.path == "/v1/local/inspect":
                params = parse_qs(parsed.query)
                identifier = params.get("id", [""])[0]
                self._send_json(200, self.manager.inspect(identifier))
            elif parsed.path == "/v1/models":
                snapshot = self.manager.snapshot()
                data = []
                if snapshot["loaded_model"]:
                    data.append({"id": snapshot["loaded_model"], "object": "model"})
                self._send_json(200, {"object": "list", "data": data})
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/v1/local/warm":
                self._send_json(
                    200,
                    self.manager.warm_model(
                        model=payload["model"],
                        keep_alive_seconds=payload.get("keep_alive_seconds"),
                        profile_name=payload.get("profile"),
                        safe_mode=bool(payload.get("safe_mode", True)),
                    ),
                )
            elif self.path == "/v1/local/evict":
                self._send_json(200, self.manager.evict_model(reason="manual"))
            elif self.path == "/v1/local/cancel":
                self._send_json(200, self.manager.cancel(payload.get("request_id")))
            elif self.path == "/v1/local/benchmark":
                self._send_json(200, self.manager.benchmark(payload))
            elif self.path == "/v1/chat/completions":
                if payload.get("stream", True):
                    self._send_streaming_completion(payload)
                else:
                    result = self.manager.complete_chat(payload)
                    self._send_json(200, _completion_response(result))
            else:
                self._send_json(404, {"error": "not found"})
        except BusyError as exc:
            self._send_json(429, {"error": str(exc)})
        except (RuntimeError, PromptBudgetError, MemoryPressureError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        self.manager.logger.info("%s - %s", self.client_address[0], format % args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode())

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_streaming_completion(self, payload: dict) -> None:
        request_id, iterator = self.manager.run_chat(payload)
        created = int(time.time())
        model = payload["model"]

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        final_result = None
        try:
            for event in iterator:
                if event["type"] == "delta":
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": event["text"]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self._write_sse(chunk)
                else:
                    final_result = event["result"]
        except (BrokenPipeError, ConnectionResetError):
            self.manager.cancel(request_id)
            return

        if final_result is None:
            raise RuntimeError("stream completed without final result")

        self._write_sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": final_result["finish_reason"]}],
            }
        )
        self._write_sse(
            {
                "id": request_id,
                "object": "local_llm.summary",
                "created": created,
                "model": model,
                "usage": final_result["usage"],
                "local_llm": {
                    "session_id": final_result["session_id"],
                    "metrics": final_result["metrics"],
                },
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _write_sse(self, payload: dict) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        self.send_header("Access-Control-Allow-Origin", origin or "*")
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")


def _completion_response(result: dict) -> dict:
    return {
        "id": result["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": result["usage"],
        "local_llm": {
            "session_id": result["session_id"],
            "metrics": result["metrics"],
        },
    }


def serve_forever(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    manager = RuntimeManager(host, port)
    manager.start()
    shutdown_started = threading.Event()

    class BoundHandler(DaemonHandler):
        pass

    BoundHandler.manager = manager

    httpd = ThreadingHTTPServer((host, port), BoundHandler)
    httpd.daemon_threads = True

    def _shutdown(*_args) -> None:
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        manager.stop_event.set()
        manager.active_cancel.set()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    PIDS_DIR.mkdir(parents=True, exist_ok=True)
    DAEMON_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    try:
        httpd.serve_forever()
    finally:
        manager.shutdown()
        DAEMON_PID_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="local-llm daemon")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    serve_forever(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
