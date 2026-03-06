"""MLX-backed runner for Apple Silicon local chat."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Generator, Optional

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.cache_prompt import make_prompt_cache
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler


@dataclass
class RunnerMetrics:
    """Summary metrics for a generation request."""

    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory_gb: float
    finish_reason: str
    ttft_seconds: float
    total_seconds: float
    cache_hit: bool
    cached_prefix_tokens: int


@dataclass
class SessionPrefixCache:
    """Prepared prompt cache for an exact session prefix."""

    session_id: str
    model_repo: str
    prefix_tokens: tuple[int, ...]
    prompt_cache: object
    updated_at: float


class MLXRunner:
    """Single-model MLX runner with warm model reuse."""

    def __init__(self) -> None:
        self.model_repo: Optional[str] = None
        self.model = None
        self.tokenizer = None
        self.model_config: dict | None = None
        self.loaded_at: float | None = None
        self.last_used_at: float | None = None
        self.load_duration_seconds: float | None = None
        self.session_caches: dict[str, SessionPrefixCache] = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.model_repo is not None

    def load_model(self, repo: str) -> dict:
        """Load a model into memory."""
        if self.model_repo == repo and self.is_loaded:
            return {
                "model": self.model_repo,
                "loaded_at": self.loaded_at,
                "load_duration_seconds": self.load_duration_seconds,
                "reused": True,
            }

        self.unload_model()
        started = time.perf_counter()
        model, tokenizer, config = load(repo, return_config=True)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = config if isinstance(config, dict) else {}
        self.model_repo = repo
        self.loaded_at = time.time()
        self.last_used_at = self.loaded_at
        self.load_duration_seconds = time.perf_counter() - started
        return {
            "model": self.model_repo,
            "loaded_at": self.loaded_at,
            "load_duration_seconds": self.load_duration_seconds,
            "reused": False,
        }

    def unload_model(self) -> None:
        """Unload the current model."""
        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.model_repo = None
        self.loaded_at = None
        self.last_used_at = None
        self.load_duration_seconds = None
        self.session_caches = {}
        gc.collect()
        mx.clear_cache()

    def estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate prompt tokens after applying the chat template."""
        return len(self.render_prompt_tokens(messages))

    def model_context_limit(self) -> Optional[int]:
        """Return the model's native context limit if the config exposes one."""
        if not self.model_config:
            return None
        for key in ("max_position_embeddings", "sliding_window", "context_length"):
            value = self.model_config.get(key)
            if isinstance(value, int) and value > 0:
                return value
        text_config = self.model_config.get("text_config")
        if isinstance(text_config, dict):
            for key in ("max_position_embeddings", "sliding_window", "context_length"):
                value = text_config.get(key)
                if isinstance(value, int) and value > 0:
                    return value
        return None

    def stream_chat(
        self,
        *,
        messages: list[dict],
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        cancel_event,
        session_id: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """Stream chat deltas and a final metrics event."""
        if not self.is_loaded:
            raise RuntimeError("No model is loaded.")

        full_prompt_tokens = self.render_prompt_tokens(messages)
        prompt_tokens = full_prompt_tokens
        prompt_cache = None
        cache_hit = False
        cached_prefix_tokens = 0
        if session_id:
            prompt_tokens, prompt_cache, cache_hit, cached_prefix_tokens = self._prepare_cached_prompt(
                session_id,
                full_prompt_tokens,
            )

        sampler = make_sampler(
            temp=max(temperature, 0.0),
            top_p=top_p,
            top_k=max(top_k, 0),
        )

        started = time.perf_counter()
        first_token_at: float | None = None
        full_text = ""
        last_resp = None
        cancelled = False

        try:
            for resp in stream_generate(
                self.model,
                self.tokenizer,
                prompt_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                prompt_cache=prompt_cache,
            ):
                if cancel_event.is_set():
                    cancelled = True
                    break

                last_resp = resp
                self.last_used_at = time.time()
                if resp.text:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    full_text += resp.text
                    yield {
                        "type": "delta",
                        "text": resp.text,
                        "generation_tokens": resp.generation_tokens,
                    }
        except Exception:
            if session_id:
                self.drop_session_cache(session_id)
            raise

        finished = time.perf_counter()
        ttft = (first_token_at - started) if first_token_at is not None else finished - started
        finish_reason = "cancelled" if cancelled else getattr(last_resp, "finish_reason", "stop")
        if session_id:
            if cancelled:
                self.drop_session_cache(session_id)
            else:
                self.store_session_prefix(session_id, messages, full_text)

        yield {
            "type": "final",
            "text": full_text,
            "metrics": RunnerMetrics(
                prompt_tokens=len(full_prompt_tokens),
                prompt_tps=float(getattr(last_resp, "prompt_tps", 0.0) or 0.0),
                generation_tokens=int(getattr(last_resp, "generation_tokens", 0) or 0),
                generation_tps=float(getattr(last_resp, "generation_tps", 0.0) or 0.0),
                peak_memory_gb=float(getattr(last_resp, "peak_memory", 0.0) or 0.0),
                finish_reason=finish_reason or "stop",
                ttft_seconds=ttft,
                total_seconds=finished - started,
                cache_hit=cache_hit,
                cached_prefix_tokens=cached_prefix_tokens,
            ),
        }

    def render_prompt_tokens(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        try:
            return list(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        except TypeError:
            return list(self.tokenizer.apply_chat_template(messages, tokenize=True))

    def store_session_prefix(
        self,
        session_id: str,
        messages: list[dict],
        assistant_text: str,
    ) -> None:
        """Build an exact prefix cache for the next turn in a session."""
        if not session_id or not self.is_loaded or self.model_repo is None:
            return

        prefix_messages = [*messages, {"role": "assistant", "content": assistant_text}]
        prefix_tokens = self.render_prompt_tokens(prefix_messages, add_generation_prompt=False)
        prompt_cache = self._build_prompt_cache(prefix_tokens)
        self.session_caches[session_id] = SessionPrefixCache(
            session_id=session_id,
            model_repo=self.model_repo,
            prefix_tokens=tuple(prefix_tokens),
            prompt_cache=prompt_cache,
            updated_at=time.time(),
        )
        self._trim_session_caches()

    def drop_session_cache(self, session_id: str) -> None:
        if session_id:
            self.session_caches.pop(session_id, None)

    def _prepare_cached_prompt(
        self,
        session_id: str,
        full_prompt_tokens: list[int],
    ) -> tuple[list[int], object | None, bool, int]:
        cached = self.session_caches.get(session_id)
        if not cached or cached.model_repo != self.model_repo:
            return full_prompt_tokens, None, False, 0

        prefix_tokens = list(cached.prefix_tokens)
        if len(full_prompt_tokens) < len(prefix_tokens):
            self.session_caches.pop(session_id, None)
            return full_prompt_tokens, None, False, 0
        if full_prompt_tokens[: len(prefix_tokens)] != prefix_tokens:
            self.session_caches.pop(session_id, None)
            return full_prompt_tokens, None, False, 0

        suffix_tokens = full_prompt_tokens[len(prefix_tokens):]
        if not suffix_tokens:
            return full_prompt_tokens, None, False, 0
        return suffix_tokens, cached.prompt_cache, True, len(prefix_tokens)

    def _build_prompt_cache(self, prompt_tokens: list[int]):
        if not self.is_loaded or self.model is None:
            raise RuntimeError("No model is loaded.")

        prompt_cache = make_prompt_cache(self.model)
        prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)
        for _ in generate_step(
            prompt_array,
            self.model,
            max_tokens=0,
            prompt_cache=prompt_cache,
        ):
            pass
        mx.eval([entry.state for entry in prompt_cache])
        return prompt_cache

    def _trim_session_caches(self, limit: int = 16) -> None:
        if len(self.session_caches) <= limit:
            return
        oldest = sorted(self.session_caches.items(), key=lambda item: item[1].updated_at)
        for session_id, _cache in oldest[: len(self.session_caches) - limit]:
            self.session_caches.pop(session_id, None)
