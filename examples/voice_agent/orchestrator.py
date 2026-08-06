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
1. Consumes events from the session's ``token_queue`` (WebSocket path).
2. On ``asr_endpoint``: build a streaming prompt generator and submit it
   to vLLM's ``AsyncLLM.generate()``.
3. Stream LLM tokens to the WebSocket send queue as they arrive.
4. Maintain per-session conversation history.
5. Expose ``handle_openai_request()`` for the REST ``/v1/chat/completions``
   endpoint (text + async ASR queue path).

KV-cache continuation
---------------------
Each session keeps a **stable** ``llm_continuation_id`` that is reused as
the ``request_id`` for every ``engine.generate()`` call.  vLLM's AsyncLLM
retains the KV-cache across calls that share the same ``request_id``, so
only the *delta* for each new turn needs to be fed:

- **Turn 1**: ``[system_prompt, user_turn_1_text, user_turn_1_audio]``
- **Turn N**: ``[assistant_N-1_response, user_turn_N_text, user_turn_N_audio]``

For the **WebSocket path** the "user text" is the committed ASR transcript
(there is no separate text channel), so the prompt delta is simply the
transcript.  The system prompt is sent only on turn 1.

For the **REST path** the caller provides both a text portion (fed
immediately) and an optional async ASR queue (fed in parallel).

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
from typing import Any, AsyncGenerator, AsyncIterator

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
        The shared ``vllm.AsyncLLM`` instance.
    llm_config:
        The ``llm`` section of config.yaml.
    conv_config:
        The ``conversation`` section of config.yaml.
    ws_send_queue:
        asyncio.Queue where outbound WebSocket messages are placed.
    """

    def __init__(
        self,
        session: SessionState,
        engine: Any,
        llm_config: dict[str, Any],
        conv_config: dict[str, Any],
        ws_send_queue: asyncio.Queue,
    ) -> None:
        self._session = session
        self._engine = engine
        self._llm_cfg = llm_config
        self._conv_cfg = conv_config
        self._ws_queue = ws_send_queue
        self._system_prompt: str = conv_config.get(
            "system_prompt",
            "You are a helpful customer support agent.",
        ).strip()
        self._max_history_turns: int = int(conv_config.get("max_history_turns", 10))

    # ------------------------------------------------------------------
    # Main event loop (WebSocket path)
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
    # Event handlers (WebSocket path)
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

        elif event_type in ("speech_start", "speech_end", "_shutdown"):
            pass  # no action needed

        else:
            logger.debug(
                "[%s] Unhandled event type: %s",
                self._session.session_id,
                event_type,
            )

    async def _handle_asr_endpoint(self, asr_text: str) -> None:
        """Build LLM prompt from committed ASR text and stream the response.

        For the WebSocket path there is no separate "user text" channel —
        the ASR transcript is the entire user input.  We create a trivial
        one-shot async queue containing only ``asr_text`` and delegate to
        the shared ``_stream_llm_parallel`` helper.
        """
        # Build a single-item ASR queue so we can share the parallel helper
        asr_queue: asyncio.Queue[str | None] = asyncio.Queue()
        await asr_queue.put(asr_text)
        await asr_queue.put(None)  # sentinel

        self._session.metrics.mark_llm_submit()
        full_response = await self._stream_llm_parallel(
            user_text="",
            asr_queue=asr_queue,
            ws_send_queue=self._ws_queue,
        )

        if full_response:
            self._session.append_assistant_turn(full_response)

        self._session.metrics.mark_response_complete()
        await self._ws_queue.put({"type": "turn_end"})

    # ------------------------------------------------------------------
    # REST path: called from server.py /v1/chat/completions
    # ------------------------------------------------------------------

    async def handle_openai_request(
        self,
        user_text: str,
        asr_queue: asyncio.Queue,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM tokens for a REST request.

        Parameters
        ----------
        user_text:
            The text portion of the user's message (known immediately,
            before audio transcription completes).
        asr_queue:
            An ``asyncio.Queue`` that yields ``str`` transcript chunks
            produced by a concurrently running ASR task, terminated by
            a ``None`` sentinel.

        Yields
        ------
        str
            Raw LLM output token strings as they are generated.
        """
        self._session.metrics.mark_llm_submit()

        full_response = ""
        first_token = True

        async for token in self._generate_tokens(user_text, asr_queue):
            if first_token:
                self._session.metrics.mark_first_token()
                first_token = False
            full_response += token
            yield token

        if full_response:
            self._session.append_assistant_turn(full_response)

        self._session.metrics.mark_response_complete()

    # ------------------------------------------------------------------
    # Core LLM streaming helpers
    # ------------------------------------------------------------------

    async def _stream_llm_parallel(
        self,
        user_text: str,
        asr_queue: asyncio.Queue,
        ws_send_queue: asyncio.Queue,
    ) -> str:
        """Submit a generation request via a streaming prompt and forward
        tokens to *ws_send_queue*.

        Returns the full concatenated response text.
        """
        full_text = ""
        first_token = True

        async for token in self._generate_tokens(user_text, asr_queue):
            if first_token:
                self._session.metrics.mark_first_token()
                first_token = False
            full_text += token
            await ws_send_queue.put({"type": "token", "text": token})

        return full_text

    async def _generate_tokens(
        self,
        user_text: str,
        asr_queue: asyncio.Queue,
    ) -> AsyncGenerator[str, None]:
        """Core generator: build a streaming prompt and yield LLM tokens.

        The prompt sent to the engine for each turn is the *delta* only:

        - Turn 1  → system_prompt + user_text + asr_chunks
        - Turn N  → last_assistant_response + user_text + asr_chunks

        This leverages vLLM's KV-cache continuation: the engine retains the
        accumulated context under ``session.llm_continuation_id`` so the
        full history never needs to be re-transmitted.

        Parameters
        ----------
        user_text:
            Text portion of the user message; fed immediately before waiting
            for ASR output so the LLM can begin prefilling in parallel.
        asr_queue:
            Queue of ASR transcript strings, closed with a ``None`` sentinel.
        """
        from vllm import SamplingParams  # type: ignore[import]

        sampling_params = SamplingParams(
            max_tokens=int(self._llm_cfg.get("max_tokens", 512)),
            temperature=float(self._llm_cfg.get("temperature", 0.7)),
            top_p=float(self._llm_cfg.get("top_p", 0.9)),
        )

        prompt_gen = self._build_prompt_generator(user_text, asr_queue)

        full_text = ""
        try:
            async for output in self._engine.generate(
                inputs=prompt_gen,
                sampling_params=sampling_params,
                request_id=self._session.llm_continuation_id,
            ):
                if not output.outputs:
                    continue
                delta = output.outputs[0].text
                new_text = delta[len(full_text):]
                if not new_text:
                    continue
                full_text = delta
                yield new_text

        except asyncio.CancelledError:
            logger.info(
                "[%s] LLM generation cancelled.",
                self._session.session_id,
            )
        except Exception as exc:
            logger.exception(
                "[%s] LLM generation error: %s",
                self._session.session_id,
                exc,
            )
            raise

    async def _build_prompt_generator(
        self,
        user_text: str,
        asr_queue: asyncio.Queue,
    ) -> AsyncGenerator[str, None]:
        """Yield prompt tokens incrementally for KV-cache continuation.

        Turn 1  (no prior assistant response):
            system_prompt  →  user_text  →  asr chunks

        Turn N  (prior assistant response exists):
            last_assistant_response  →  user_text  →  asr chunks

        The system prompt is embedded in the first user turn via the chat
        template; subsequent turns feed only the delta so vLLM can append
        to its cached KV state.
        """
        last_assistant = self._session.last_assistant_response

        if last_assistant is None:
            # First turn: include system prompt
            yield self._system_prompt + "\n"
        else:
            # Subsequent turns: feed previous assistant response as prefix
            yield last_assistant + "\n"

        # User text is known immediately — feed it before waiting for ASR
        if user_text:
            yield user_text

        # ASR chunks arrive asynchronously; yield each as it is produced
        while True:
            chunk = await asr_queue.get()
            if chunk is None:  # sentinel: ASR complete
                break
            if chunk:
                yield chunk

        # Append the user turn in conversation history now that we have the
        # complete text (user_text + all asr chunks).  We do not store it
        # before generation so that the assistant turn can be stored in order.
        combined_user_text = user_text
        # Note: asr chunks are already consumed; the caller (handle_openai_request
        # / _handle_asr_endpoint) is responsible for saving the full user text
        # via append_user_turn if needed for history display purposes.
        # For KV-cache continuation we do not need to reconstruct the history.
        del combined_user_text  # unused; kept for clarity
