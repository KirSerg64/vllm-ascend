#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""LLM Orchestrator: the central coordinator between ASR and LLM.

Responsibilities
----------------
1. Consumes events from the session's ``token_queue``.
2. On ``asr_endpoint``: build the LLM prompt from conversation history and
   submit it to the AsyncLLMEngine.
3. On ``speech_start`` (barge-in): cancel the in-flight LLM request if
   ``enable_barge_in`` is set.
4. Stream LLM tokens to the WebSocket send queue as they arrive.
5. Maintain per-session conversation history.

Event protocol (token_queue messages)
--------------------------------------
Inbound (produced by audio_pipeline / asr_worker):
  {"type": "asr_partial",  "text": str, "stable_prefix": str}
  {"type": "asr_endpoint", "text": str}
  {"type": "speech_start"}
  {"type": "speech_end"}
  {"type": "_shutdown"}   — internal sentinel

Outbound (written to ws_send_queue for WebSocket delivery):
  {"type": "asr_partial",  "text": str}
  {"type": "token",        "text": str}
  {"type": "turn_end"}
  {"type": "error",        "message": str}
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from audio_pipeline import SPEECH_START_EVENT
from session_manager import SessionState

logger = logging.getLogger(__name__)

_SHUTDOWN_SENTINEL = {"type": "_shutdown"}


class Orchestrator:
    """Drives one session from ASR events to LLM token stream.

    Parameters
    ----------
    session:
        Mutable session state shared with the audio pipeline.
    engine:
        The shared ``vllm.AsyncLLMEngine`` instance.
    llm_config:
        The ``llm`` section of config.yaml.
    conv_config:
        The ``conversation`` section of config.yaml.
    enable_barge_in:
        Whether to cancel in-flight LLM generation when new speech starts.
    ws_send_queue:
        asyncio.Queue where outbound WebSocket messages are placed.
    """

    def __init__(
        self,
        session: SessionState,
        engine: Any,
        llm_config: dict[str, Any],
        conv_config: dict[str, Any],
        enable_barge_in: bool,
        ws_send_queue: asyncio.Queue,
    ) -> None:
        self._session = session
        self._engine = engine
        self._llm_cfg = llm_config
        self._conv_cfg = conv_config
        self._enable_barge_in = enable_barge_in
        self._ws_queue = ws_send_queue
        self._system_prompt: str = conv_config.get(
            "system_prompt",
            "You are a helpful customer support agent.",
        ).strip()
        self._max_history_turns: int = int(conv_config.get("max_history_turns", 10))

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Consume events from session.token_queue until shutdown."""
        queue = self._session.token_queue
        while True:
            event = await queue.get()
            try:
                await self._handle_event(event)
            except Exception as exc:
                logger.exception(
                    "[%s] Orchestrator error: %s",
                    self._session.session_id,
                    exc,
                )
                await self._ws_queue.put({"type": "error", "message": str(exc)})
            finally:
                queue.task_done()

            if event.get("type") == "_shutdown":
                break

    async def shutdown(self) -> None:
        """Request graceful shutdown of the event loop."""
        await self._session.token_queue.put(_SHUTDOWN_SENTINEL)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "asr_partial":
            # Forward partial ASR results to the client for display
            await self._ws_queue.put({"type": "asr_partial", "text": event.get("text", "")})

        elif event_type == "asr_endpoint":
            asr_text = event.get("text", "").strip()
            if not asr_text:
                return
            logger.info(
                "[%s] ASR endpoint committed: %r",
                self._session.session_id,
                asr_text,
            )
            await self._handle_asr_endpoint(asr_text)

        elif event_type == SPEECH_START_EVENT:
            if self._enable_barge_in and self._session.is_bot_speaking:
                await self._cancel_llm()

        elif event_type in ("speech_end", "_shutdown", "asr_partial"):
            pass  # handled above or ignored

        else:
            logger.debug(
                "[%s] Unhandled event type: %s",
                self._session.session_id,
                event_type,
            )

    async def _handle_asr_endpoint(self, asr_text: str) -> None:
        """Build LLM prompt and stream the response."""
        # Cancel any in-flight request (defensive; barge-in may have missed it)
        if self._session.is_bot_speaking:
            await self._cancel_llm()

        # Update conversation history
        self._session.append_user_turn(asr_text)

        # Build messages list for the LLM
        messages = self._build_messages()

        # Create a unique request ID for this generation
        request_id = f"{self._session.session_id}-{uuid.uuid4().hex[:8]}"
        async with self._session.lock:
            self._session.current_llm_request_id = request_id
            self._session.is_bot_speaking = True

        self._session.metrics.mark_llm_submit()
        logger.debug("[%s] Submitting LLM request %s", self._session.session_id, request_id)

        # Stream tokens
        full_response = await self._stream_llm(messages, request_id)

        # Persist assistant turn in history
        if full_response:
            self._session.append_assistant_turn(full_response)

        async with self._session.lock:
            self._session.is_bot_speaking = False
            self._session.current_llm_request_id = None

        self._session.metrics.mark_response_complete()
        await self._ws_queue.put({"type": "turn_end"})

    async def _stream_llm(self, messages: list[dict[str, str]], request_id: str) -> str:
        """Submit a generation request and stream tokens to ws_send_queue.

        Returns the full concatenated response text.
        """
        from vllm import SamplingParams  # type: ignore[import]

        sampling_params = SamplingParams(
            max_tokens=int(self._llm_cfg.get("max_tokens", 512)),
            temperature=float(self._llm_cfg.get("temperature", 0.7)),
            top_p=float(self._llm_cfg.get("top_p", 0.9)),
        )

        full_text = ""
        first_token = True

        try:
            async for output in self._engine.generate(  # type: ignore[union-attr]
                prompt=messages,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                if not output.outputs:
                    continue

                delta = output.outputs[0].text
                # vLLM streams cumulative text; compute the delta
                new_text = delta[len(full_text) :]
                if not new_text:
                    continue

                if first_token:
                    self._session.metrics.mark_first_token()
                    first_token = False

                full_text = delta
                # Mock TTS: send each token to the client
                await self._ws_queue.put({"type": "token", "text": new_text})

        except asyncio.CancelledError:
            logger.info(
                "[%s] LLM request %s was cancelled (barge-in).",
                self._session.session_id,
                request_id,
            )
        except Exception as exc:
            logger.exception(
                "[%s] LLM generation error: %s",
                self._session.session_id,
                exc,
            )
            await self._ws_queue.put({"type": "error", "message": f"LLM error: {exc}"})

        return full_text

    async def _cancel_llm(self) -> None:
        """Cancel the in-flight LLM request (barge-in Level 1)."""
        async with self._session.lock:
            request_id = self._session.current_llm_request_id
            self._session.is_bot_speaking = False
            self._session.current_llm_request_id = None

        if request_id is not None:
            try:
                await self._engine.abort(request_id)  # type: ignore[union-attr]
                logger.info(
                    "[%s] Cancelled LLM request %s (barge-in).",
                    self._session.session_id,
                    request_id,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Could not abort LLM request %s: %s",
                    self._session.session_id,
                    request_id,
                    exc,
                )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the message list for the LLM chat template."""
        messages: list[dict[str, str]] = [{"role": "system", "content": self._system_prompt}]
        history = self._session.get_history_window(self._max_history_turns)
        messages.extend(history)
        return messages
