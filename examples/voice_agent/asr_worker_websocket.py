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
"""ASR worker using websocket connection to qwen-asr model on port 8008.

Design decisions
----------------
* A **single** :class:`ASRWorkerWebSocket` manages connections to the ASR service.
* Each session owns its own websocket connection to the ASR service.
* All websocket operations are async to avoid blocking the event loop.
* Two event types are emitted on the session's token_queue:
  ``{"type": "asr_partial", "text": "..."}``
  ``{"type": "asr_endpoint", "text": "..."}``
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import numpy as np
import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)

# Sentinel published to session queues
_ASR_PARTIAL = "asr_partial"
_ASR_ENDPOINT = "asr_endpoint"


def _common_prefix_length(a: str, b: str) -> int:
    """Return the length of the longest common prefix of two strings."""
    length = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        length += 1
    return length


class ASRStream:
    """Wrapper for a websocket connection to the ASR service for a single session."""

    def __init__(
        self,
        websocket: WebSocketClientProtocol,
        session_id: str,
    ) -> None:
        self.websocket = websocket
        self.session_id = session_id
        self.last_partial = ""
        self.is_endpoint = False
        self.accumulated_text = ""

    async def close(self) -> None:
        """Close the websocket connection."""
        try:
            await self.websocket.close()
        except Exception as exc:
            logger.warning(
                "[%s] Error closing ASR websocket: %s",
                self.session_id,
                exc,
            )


class ASRWorkerWebSocket:
    """WebSocket-based ASR worker connecting to qwen-asr on port 8008.

    Parameters
    ----------
    config:
        The ``asr`` section of config.yaml (as a dict).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._cfg = config
        self._asr_host = config.get("websocket_host", "localhost")
        self._asr_port = int(config.get("websocket_port", 8008))
        self._sample_rate = config.get("sample_rate", 16000)
        self._endpoint_silence_ms = config.get("endpoint_silence_ms", 200)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialize the websocket ASR client. Call once at startup."""
        logger.info(
            "WebSocket ASR client configured for ws://%s:%d",
            self._asr_host,
            self._asr_port,
        )

    async def create_stream(self) -> ASRStream:
        """Create a new per-session websocket connection to the ASR service."""
        uri = f"ws://{self._asr_host}:{self._asr_port}/ws"
        try:
            websocket = await websockets.connect(
                uri,
                ping_interval=30,
                ping_timeout=10,
            )
            logger.info("Created new ASR websocket connection to %s", uri)
            return ASRStream(websocket, "session")
        except Exception as exc:
            logger.error("Failed to connect to ASR service at %s: %s", uri, exc)
            raise RuntimeError(
                f"Could not connect to ASR service at {uri}. Ensure the qwen-asr model is running on port 8008."
            ) from exc

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    async def process_chunk(
        self,
        stream: ASRStream,
        pcm_float32: np.ndarray,
        session_id: str,
        event_queue: asyncio.Queue,
        last_partial_ref: list,  # mutable single-element list: [str]
    ) -> bool:
        """Feed one audio chunk to the ASR stream via websocket.

        Returns True if an endpoint was detected (phrase complete).
        The caller should then call :py:meth:`reset_stream`.
        """
        try:
            # Convert float32 PCM to int16 bytes for transmission
            pcm_int16 = (pcm_float32 * 32768.0).astype(np.int16)
            audio_bytes = pcm_int16.tobytes()

            # Send audio chunk to ASR service
            await stream.websocket.send(audio_bytes)

            # Try to receive response (non-blocking)
            try:
                response = await asyncio.wait_for(stream.websocket.recv(), timeout=0.1)

                # Parse the response
                if isinstance(response, str):
                    result = json.loads(response)
                    await self._process_asr_result(
                        result,
                        stream,
                        session_id,
                        event_queue,
                        last_partial_ref,
                    )
                else:
                    logger.warning(
                        "[%s] Received non-text response from ASR service",
                        session_id,
                    )

            except asyncio.TimeoutError:
                # No response yet, continue
                pass

            return stream.is_endpoint

        except Exception as exc:
            logger.error(
                "[%s] Error processing audio chunk: %s",
                session_id,
                exc,
            )
            return False

    async def _process_asr_result(
        self,
        result: dict[str, Any],
        stream: ASRStream,
        session_id: str,
        event_queue: asyncio.Queue,
        last_partial_ref: list,
    ) -> None:
        """Process ASR result from the websocket service."""
        result_type = result.get("type", "")
        text = result.get("text", "").strip()

        if result_type == "partial":
            # Partial transcription result
            old_partial = last_partial_ref[0]
            prefix_len = _common_prefix_length(old_partial, text)
            stable_text = text[:prefix_len] if prefix_len > 0 else ""

            if text and text != old_partial:
                last_partial_ref[0] = text
                stream.accumulated_text = text
                await event_queue.put(
                    {
                        "type": _ASR_PARTIAL,
                        "text": text,
                        "stable_prefix": stable_text,
                    }
                )

        elif result_type == "final" or result_type == "endpoint":
            # Final transcription or endpoint detected
            if text:
                stream.accumulated_text = text
                await event_queue.put(
                    {
                        "type": _ASR_ENDPOINT,
                        "text": text,
                    }
                )
                logger.debug("[%s] ASR endpoint: %r", session_id, text)
                stream.is_endpoint = True
            else:
                logger.debug(
                    "[%s] Empty text received for final/endpoint",
                    session_id,
                )

        elif result_type == "error":
            error_msg = result.get("message", "Unknown error")
            logger.error("[%s] ASR service error: %s", session_id, error_msg)

        else:
            logger.debug(
                "[%s] Unknown ASR result type: %s",
                session_id,
                result_type,
            )

    async def reset_stream(self, stream: ASRStream) -> None:
        """Reset the decoder state after an endpoint."""
        # Send reset signal to ASR service
        try:
            reset_msg = json.dumps({"type": "reset"})
            await stream.websocket.send(reset_msg)
            stream.last_partial = ""
            stream.is_endpoint = False
            stream.accumulated_text = ""
        except Exception as exc:
            logger.warning(
                "[%s] Error resetting ASR stream: %s",
                stream.session_id,
                exc,
            )

    def shutdown(self) -> None:
        """Shutdown the ASR worker."""
        logger.info("ASR worker shutdown")
