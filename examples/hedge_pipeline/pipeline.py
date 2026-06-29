# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
pipeline.py — Core async hedge pipeline logic.

The hedge pipeline runs two vLLM inference servers in parallel:

  1. Small model (Qwen3-0.6B): returns a filler answer quickly (~50 ms TTFT).
  2. Big model  (Qwen3-Omni-30B-A3B W8A8): returns the authoritative answer
     using an augmented prompt that includes the filler as a thinking hint.

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
import itertools
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import TypedDict

import aiohttp

from .config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class HedgeEvent(TypedDict):
    """A single streaming event emitted by the hedge pipeline."""

    phase: str  # "filler" | "final"
    text: str  # incremental token text (delta, not accumulated)
    done: bool  # True on the last event of this phase


# ---------------------------------------------------------------------------
# Helper: call a vLLM OpenAI-compatible streaming endpoint
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


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class HedgePipeline:
    """
    Async hedge pipeline that coordinates a small (filler) model and a large
    (thinking) model to minimise user-perceived latency.

    Thread safety: instances are *not* thread-safe; use one instance per
    event loop (or one per asyncio Task if you guard with a lock).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        # Infinite round-robin iterator over big-model URLs.
        self._big_url_cycle = itertools.cycle(config.big.base_urls)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=0)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        """Release the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_big_url(self) -> str:
        return next(self._big_url_cycle)

    async def _run_small_model(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict],
        request_id: str,
    ) -> tuple[str, float]:
        """
        Call the small model and collect its full output.

        Returns ``(filler_text, elapsed_seconds)``.
        On timeout or error, returns an empty string so the big model falls
        back to the original prompt.
        """
        cfg = self.config.small
        tokens: list[str] = []
        t0 = time.monotonic()
        try:
            async for delta in _stream_chat_completion(
                session=session,
                base_url=cfg.base_url,
                model=cfg.model,
                messages=messages,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                request_id=f"{request_id}-small",
                timeout=cfg.timeout,
            ):
                tokens.append(delta)
        except asyncio.TimeoutError:
            logger.warning(
                "Small model timed out after %.2f s (request_id=%s); falling back to original prompt.",
                cfg.timeout,
                request_id,
            )
        except Exception as exc:
            logger.warning(
                "Small model error (request_id=%s): %s; falling back to original prompt.",
                request_id,
                exc,
            )
        elapsed = time.monotonic() - t0
        return "".join(tokens), elapsed

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

        # Build the base message list.
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        session = await self._get_session()
        cfg = self.config

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
        # Build augmented prompt only if filler produced something useful.
        if filler_text.strip():
            augmented_user_content = cfg.build_augmented_prompt(user_message, filler_text)
        else:
            augmented_user_content = user_message

        big_messages: list[dict] = []
        if system_message:
            big_messages.append({"role": "system", "content": system_message})
        big_messages.append({"role": "user", "content": augmented_user_content})

        big_cfg = cfg.big
        big_url = self._next_big_url()
        logger.debug("Starting final phase via %s (request_id=%s)", big_url, request_id)

        t_final_start = time.monotonic()
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
            logger.error(
                "Big model error (request_id=%s): %s",
                request_id,
                exc,
            )

        final_elapsed = time.monotonic() - t_final_start
        logger.debug(
            "Final phase done in %.3f s (request_id=%s)",
            final_elapsed,
            request_id,
        )

        # Signal end of final phase.
        yield HedgeEvent(phase="final", text="", done=True)
