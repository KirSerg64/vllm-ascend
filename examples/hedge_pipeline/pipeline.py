# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
pipeline.py — Core async hedge pipeline logic.

The hedge pipeline runs two vLLM inference servers in parallel:

  1. Small model (Qwen3-0.6B): returns a filler answer quickly (~50 ms TTFT).
  2. Big model  (Qwen3-Omni-30B-A3B W8A8): returns the authoritative answer
     using an augmented prompt that includes the filler as a thinking hint.

Parallel tokenization
---------------------
When ``config.tokenizer.use_local_tokenizer=True`` (the default), the
pipeline loads the big model's HuggingFace tokenizer once at startup and
uses it to pre-tokenize the original user prompt *concurrently* with the
small model call.  After the small model finishes (~50 ms), only the short
filler-hint suffix needs to be tokenized (~1 ms).  The concatenated token-ID
list is then sent to the big model via ``POST /v1/completions`` with the
``prompt`` field as a list of integers, which bypasses vLLM's internal
tokenizer and lets NPU prefill begin immediately.

When ``use_local_tokenizer=False``, the pipeline falls back to the original
``/v1/chat/completions`` text-based path.

Usage (library)::

    from examples.hedge_pipeline.config import PipelineConfig
    from examples.hedge_pipeline.pipeline import HedgePipeline

    config = PipelineConfig()
    pipeline = HedgePipeline(config)

    async for event in pipeline.run("What is the capital of France?"):
        print(event)  # {"phase": "filler"|"final", "text": "...", "done": bool}

    await pipeline.close()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypedDict

import aiohttp

from .config import PipelineConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level tokenizer cache (shared across all HedgePipeline instances)
# ---------------------------------------------------------------------------

# Keys are model name/path strings; values are loaded tokenizer objects.
# Populated lazily on first use; protected by a per-key asyncio.Lock at
# load time so the same model is never loaded twice concurrently.
_TOKENIZER_CACHE: dict[str, Any] = {}
_TOKENIZER_LOAD_LOCK: asyncio.Lock | None = None  # created inside the event loop


def _get_load_lock() -> asyncio.Lock:
    global _TOKENIZER_LOAD_LOCK
    if _TOKENIZER_LOAD_LOCK is None:
        _TOKENIZER_LOAD_LOCK = asyncio.Lock()
    return _TOKENIZER_LOAD_LOCK


def _load_tokenizer_sync(model_name_or_path: str) -> Any:
    """Load a HuggingFace tokenizer synchronously (run in an executor)."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "The 'transformers' package is required for parallel tokenization. "
            "Install it with: pip install transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name_or_path)


async def _ensure_tokenizer(model_name_or_path: str) -> Any:
    """
    Return the tokenizer for *model_name_or_path*, loading it on first call.
    Concurrent callers wait for the load to finish rather than loading twice.
    """
    if model_name_or_path in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name_or_path]

    async with _get_load_lock():
        # Re-check after acquiring the lock (another coroutine may have loaded it).
        if model_name_or_path in _TOKENIZER_CACHE:
            return _TOKENIZER_CACHE[model_name_or_path]

        logger.info("Loading tokenizer '%s' …", model_name_or_path)
        t0 = time.monotonic()
        loop = asyncio.get_event_loop()
        tokenizer = await loop.run_in_executor(None, _load_tokenizer_sync, model_name_or_path)
        _TOKENIZER_CACHE[model_name_or_path] = tokenizer
        logger.info(
            "Tokenizer '%s' loaded in %.2f s (vocab_size=%d).",
            model_name_or_path,
            time.monotonic() - t0,
            tokenizer.vocab_size,
        )
        return tokenizer


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class HedgeEvent(TypedDict):
    """A single streaming event emitted by the hedge pipeline."""

    phase: str  # "filler" | "final"
    text: str  # incremental token text (delta, not accumulated)
    done: bool  # True on the last event of this phase


# ---------------------------------------------------------------------------
# HTTP streaming helpers
# ---------------------------------------------------------------------------


async def _stream_chat_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_id: str,
    timeout: float,
) -> AsyncIterator[str]:
    """
    Yield incremental text deltas from a vLLM ``/v1/chat/completions`` SSE
    stream.  Raises ``asyncio.TimeoutError`` if ``timeout`` is exceeded before
    the stream finishes.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
    client_timeout = aiohttp.ClientTimeout(total=timeout if timeout > 0 else None)

    async with session.post(url, json=payload, headers=headers, timeout=client_timeout) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


async def _stream_completions(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_id: str,
    timeout: float,
) -> AsyncIterator[str]:
    """
    Yield incremental text deltas from a vLLM ``/v1/completions`` SSE stream,
    supplying the prompt as a pre-tokenized list of integer token IDs.

    Passing token IDs instead of a text string causes vLLM to skip its
    internal tokenization step and feed the IDs directly to the scheduler,
    saving 5–25 ms on the critical path for typical prompt lengths.
    """
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt_token_ids,  # list[int] → vLLM bypasses tokenizer
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
    client_timeout = aiohttp.ClientTimeout(total=timeout if timeout > 0 else None)

    async with session.post(url, json=payload, headers=headers, timeout=client_timeout) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            # /v1/completions uses choices[n].text, not choices[n].delta.content
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text:
                yield text


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class HedgePipeline:
    """
    Async hedge pipeline that coordinates a small (filler) model and a large
    (thinking) model to minimise user-perceived latency.

    When ``config.tokenizer.use_local_tokenizer=True``, the pipeline
    pre-tokenizes the original user prompt concurrently with the small model
    call (using ``asyncio.create_task`` + ``run_in_executor``).  Once the
    small model finishes, only the short filler-hint suffix is tokenized
    (~1 ms), the two token-ID lists are concatenated, and the big model is
    called via ``/v1/completions`` with the combined IDs, skipping vLLM's
    internal tokenization step.

    Thread safety: instances are *not* thread-safe; use one instance per
    event loop (or one per asyncio Task if you guard with a lock).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        # Infinite round-robin iterator over big-model URLs.
        self._big_url_cycle = itertools.cycle(config.big.base_urls)
        # Thread pool dedicated to tokenization (CPU-bound, off event loop).
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Session / executor lifecycle
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.tokenizer.max_workers,
                thread_name_prefix="hedge_tokenizer",
            )
        return self._executor

    async def close(self) -> None:
        """Release the underlying HTTP session and thread pool."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    async def warm_tokenizer(self) -> None:
        """
        Pre-load the big model's tokenizer so the first request does not pay
        the startup cost.  Call this once during server startup.
        """
        if self.config.tokenizer.use_local_tokenizer:
            await _ensure_tokenizer(self.config.tokenizer.model_name_or_path)

    async def _tokenize_base_prompt(self, messages: list[dict]) -> list[int] | None:
        """
        Tokenize *messages* using the chat template but *without* the
        user-turn-close suffix or the generation prompt.  The returned IDs
        are later combined with the filler-hint suffix IDs.

        Runs in a thread executor so it does not block the event loop.
        Returns ``None`` on error so callers can fall back to the text path.
        """
        tok_cfg = self.config.tokenizer
        loop = asyncio.get_event_loop()

        def _do_tokenize() -> list[int]:
            tokenizer = _TOKENIZER_CACHE[tok_cfg.model_name_or_path]
            # Apply the chat template WITHOUT the generation prompt so we get
            # the user-turn text up to (and including) the turn-close marker.
            text: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Strip the turn-close suffix so we can append the filler hint
            # text *inside* the user turn before re-closing it.
            if tok_cfg.turn_end_str and text.endswith(tok_cfg.turn_end_str):
                text = text[: -len(tok_cfg.turn_end_str)]
            return tokenizer.encode(text, add_special_tokens=False)

        try:
            return await loop.run_in_executor(self._get_executor(), _do_tokenize)
        except Exception as exc:
            logger.warning(
                "Pre-tokenization of base prompt failed (%s); will fall back to /v1/chat/completions path.",
                exc,
            )
            return None

    async def _tokenize_close_and_gen_prompt(self) -> list[int]:
        """
        Tokenize the user-turn-close + generation-prompt string.
        Used when there is no filler text (fallback / timeout case).
        """
        tok_cfg = self.config.tokenizer
        loop = asyncio.get_event_loop()
        suffix = tok_cfg.turn_end_str + tok_cfg.generation_prompt_str

        def _do_tokenize() -> list[int]:
            tokenizer = _TOKENIZER_CACHE[tok_cfg.model_name_or_path]
            return tokenizer.encode(suffix, add_special_tokens=False)

        return await loop.run_in_executor(self._get_executor(), _do_tokenize)

    async def _tokenize_filler_suffix(self, filler_text: str) -> list[int]:
        """
        Tokenize the filler-hint suffix that is appended inside the user turn.

        The suffix consists of:
          1. The filler-hint text (built from ``config.filler_hint_template``)
          2. The user-turn-close marker  (e.g. ``\\n<|im_end|>\\n``)
          3. The generation-prompt string (e.g. ``<|im_start|>assistant\\n``)

        Running this after the small model finishes adds ~1 ms to the critical
        path (filler is at most ~80 tokens).
        """
        tok_cfg = self.config.tokenizer
        hint = self.config.build_augmented_suffix(filler_text)
        suffix = hint + tok_cfg.turn_end_str + tok_cfg.generation_prompt_str
        loop = asyncio.get_event_loop()

        def _do_tokenize() -> list[int]:
            tokenizer = _TOKENIZER_CACHE[tok_cfg.model_name_or_path]
            return tokenizer.encode(suffix, add_special_tokens=False)

        return await loop.run_in_executor(self._get_executor(), _do_tokenize)

    # ------------------------------------------------------------------
    # Round-robin URL selection
    # ------------------------------------------------------------------

    def _next_big_url(self) -> str:
        return next(self._big_url_cycle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        user_message: str,
        system_message: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[HedgeEvent]:
        """
        Run the hedge pipeline for a single user message.

        Yields :class:`HedgeEvent` dicts:

        * ``{"phase": "filler", "text": "<delta>", "done": False}`` — token
          deltas from the small model as they arrive (streamed).
        * ``{"phase": "filler", "text": "", "done": True}`` — signals end of
          filler phase.
        * ``{"phase": "final", "text": "<delta>", "done": False}`` — token
          deltas from the big model.
        * ``{"phase": "final", "text": "", "done": True}`` — signals end of
          final phase (end of response).

        When ``config.tokenizer.use_local_tokenizer=True``, the original
        prompt is tokenized concurrently with the small model call so that
        vLLM's tokenization step is removed from the critical path between
        the filler finishing and big-model prefill starting.

        Parameters
        ----------
        user_message:
            The user's input text.
        system_message:
            Optional system-level instruction prepended to both requests.
        request_id:
            Optional correlation ID; auto-generated if not supplied.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Build the base message list (used for both models).
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        session = await self._get_session()
        cfg = self.config
        tok_cfg = cfg.tokenizer

        # ── Launch background tokenization at T=0 ─────────────────────────
        # The task runs concurrently with the small model HTTP request so the
        # tokenization cost (~5–15 ms for a typical 500-token prompt) is fully
        # hidden behind the ~50 ms small model latency.
        tokenize_task: asyncio.Task[list[int] | None] | None = None
        if tok_cfg.use_local_tokenizer:
            try:
                # Ensure the tokenizer is loaded before kicking off the task.
                # If it was already loaded at startup, this returns instantly.
                await _ensure_tokenizer(tok_cfg.model_name_or_path)
                tokenize_task = asyncio.create_task(
                    self._tokenize_base_prompt(messages),
                    name=f"tokenize-{request_id}",
                )
            except Exception as exc:
                logger.warning(
                    "Could not load tokenizer for parallel tokenization (%s); "
                    "falling back to /v1/chat/completions path.",
                    exc,
                )

        # ── Phase 1: small model (filler) ─────────────────────────────────
        logger.debug("Starting filler phase (request_id=%s)", request_id)
        filler_tokens: list[str] = []
        t_filler_start = time.monotonic()

        small_cfg = cfg.small
        try:
            async for delta in _stream_chat_completion(
                session=session,
                base_url=small_cfg.base_url,
                model=small_cfg.model,
                messages=messages,
                max_tokens=small_cfg.max_tokens,
                temperature=small_cfg.temperature,
                top_p=small_cfg.top_p,
                request_id=f"{request_id}-small",
                timeout=small_cfg.timeout,
            ):
                filler_tokens.append(delta)
                yield HedgeEvent(phase="filler", text=delta, done=False)
        except asyncio.TimeoutError:
            logger.warning(
                "Filler timed out after %.2f s (request_id=%s); big model will use original prompt.",
                small_cfg.timeout,
                request_id,
            )
        except Exception as exc:
            logger.warning(
                "Filler error (request_id=%s): %s; big model will use original prompt.",
                request_id,
                exc,
            )

        filler_text = "".join(filler_tokens)
        filler_elapsed = time.monotonic() - t_filler_start
        logger.debug(
            "Filler phase done in %.3f s, %d tokens (request_id=%s)",
            filler_elapsed,
            len(filler_tokens),
            request_id,
        )

        # Signal end of filler phase.
        yield HedgeEvent(phase="filler", text="", done=True)

        # ── Phase 2: big model (final) ─────────────────────────────────────
        big_cfg = cfg.big
        big_url = self._next_big_url()
        logger.debug("Starting final phase via %s (request_id=%s)", big_url, request_id)

        t_final_start = time.monotonic()

        # Decide whether to use the fast token-ID path or the text path.
        use_token_path = False
        prompt_token_ids: list[int] = []

        if tokenize_task is not None:
            # By the time filler finishes (~50 ms), tokenization (~2–10 ms) is done.
            base_ids = await tokenize_task
            if base_ids is not None:
                try:
                    if filler_text.strip():
                        # Append filler hint + user-turn-close + gen-prompt tokens.
                        suffix_ids = await self._tokenize_filler_suffix(filler_text)
                    else:
                        # No filler — just close the user turn and add gen prompt.
                        suffix_ids = await self._tokenize_close_and_gen_prompt()
                    prompt_token_ids = base_ids + suffix_ids
                    use_token_path = True
                    logger.debug(
                        "Using /v1/completions with %d token IDs (base=%d, suffix=%d, request_id=%s).",
                        len(prompt_token_ids),
                        len(base_ids),
                        len(suffix_ids),
                        request_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Suffix tokenization failed (%s); falling back to text path.",
                        exc,
                    )

        if use_token_path:
            # ── Fast path: /v1/completions with pre-tokenized prompt ────────
            try:
                async for delta in _stream_completions(
                    session=session,
                    base_url=big_url,
                    model=big_cfg.model,
                    prompt_token_ids=prompt_token_ids,
                    max_tokens=big_cfg.max_tokens,
                    temperature=big_cfg.temperature,
                    top_p=big_cfg.top_p,
                    request_id=f"{request_id}-big",
                    timeout=big_cfg.timeout,
                ):
                    yield HedgeEvent(phase="final", text=delta, done=False)
            except asyncio.TimeoutError:
                logger.error(
                    "Big model timed out after %.2f s (request_id=%s).",
                    big_cfg.timeout,
                    request_id,
                )
            except Exception as exc:
                logger.error("Big model error (request_id=%s): %s", request_id, exc)
        else:
            # ── Fallback path: /v1/chat/completions with text prompt ────────
            if filler_text.strip():
                augmented_user_content = cfg.build_augmented_prompt(user_message, filler_text)
            else:
                augmented_user_content = user_message

            big_messages: list[dict] = []
            if system_message:
                big_messages.append({"role": "system", "content": system_message})
            big_messages.append({"role": "user", "content": augmented_user_content})

            try:
                async for delta in _stream_chat_completion(
                    session=session,
                    base_url=big_url,
                    model=big_cfg.model,
                    messages=big_messages,
                    max_tokens=big_cfg.max_tokens,
                    temperature=big_cfg.temperature,
                    top_p=big_cfg.top_p,
                    request_id=f"{request_id}-big",
                    timeout=big_cfg.timeout,
                ):
                    yield HedgeEvent(phase="final", text=delta, done=False)
            except asyncio.TimeoutError:
                logger.error(
                    "Big model timed out after %.2f s (request_id=%s).",
                    big_cfg.timeout,
                    request_id,
                )
            except Exception as exc:
                logger.error("Big model error (request_id=%s): %s", request_id, exc)

        final_elapsed = time.monotonic() - t_final_start
        logger.debug(
            "Final phase done in %.3f s (request_id=%s)",
            final_elapsed,
            request_id,
        )

        # Signal end of final phase.
        yield HedgeEvent(phase="final", text="", done=True)
