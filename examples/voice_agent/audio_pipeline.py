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
"""Audio ingestion pipeline: VAD → ASR chunk feed.

Responsibilities
----------------
1. Accept raw PCM bytes (16-bit signed, mono, 16 kHz) from the WebSocket
   handler or the file simulator.
2. Run silero-vad in a thread executor to detect speech activity.
3. On speech start: activate the ASR stream for the session and begin
   feeding audio chunks.
4. On speech end (VAD silence): signal the orchestrator that the user has
   stopped speaking (used for barge-in detection).
5. Emit ``{"type": "speech_start"}`` / ``{"type": "speech_end"}`` events
   into the session's event_queue so the orchestrator can react.

Note: The ASR endpoint detection (silence after speech) runs inside
:class:`~asr_worker.ASRWorker`.  VAD here is a *gating* mechanism that
avoids feeding silence to the ASR model and provides a fast barge-in
signal when new speech begins while the bot is still responding.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from asr_worker import ASRWorker
from session_manager import SessionState

logger = logging.getLogger(__name__)

SPEECH_START_EVENT = "speech_start"
SPEECH_END_EVENT = "speech_end"

# Sentinel used to signal the VAD buffer is complete
_FLUSH = object()


class AudioPipeline:
    """Processes raw audio bytes for a single session.

    One :class:`AudioPipeline` instance is created per WebSocket connection
    and torn down when the connection closes.

    Parameters
    ----------
    session:
        The :class:`~session_manager.SessionState` for this connection.
    asr_worker:
        The shared :class:`~asr_worker.ASRWorker` instance.
    vad_config:
        The ``vad`` section of config.yaml.
    asr_config:
        The ``asr`` section of config.yaml.
    """

    def __init__(
        self,
        session: SessionState,
        asr_worker: ASRWorker,
        vad_config: dict[str, Any],
        asr_config: dict[str, Any],
    ) -> None:
        self._session = session
        self._asr = asr_worker
        self._vad_cfg = vad_config
        self._asr_cfg = asr_config

        self._sample_rate: int = asr_config.get("sample_rate", 16000)
        self._chunk_size_ms: int = asr_config.get("chunk_size_ms", 20)
        self._chunk_samples: int = int(self._sample_rate * self._chunk_size_ms / 1000)

        # silero-vad state
        self._vad_model: Any | None = None
        self._vad_utils: Any | None = None
        self._vad_enabled: bool = vad_config.get("enabled", True)
        self._vad_threshold: float = float(vad_config.get("threshold", 0.5))
        self._vad_window_samples: int = int(self._sample_rate * vad_config.get("window_size_ms", 32) / 1000)
        self._min_speech_samples: int = int(self._sample_rate * vad_config.get("min_speech_duration_ms", 100) / 1000)
        self._speech_pad_samples: int = int(self._sample_rate * vad_config.get("speech_pad_ms", 50) / 1000)

        # State machine
        self._is_speech_active: bool = False
        self._speech_sample_count: int = 0
        self._silence_sample_count: int = 0

        # ASR stream for the current utterance (recreated per utterance)
        self._asr_stream: Any = None
        self._last_partial_ref: list = [""]

        # Executor for blocking VAD inference
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"vad-{session.session_id}")

        # Raw byte buffer for incomplete chunks
        self._byte_buffer = b""

    def load_vad(self) -> None:
        """Load the silero-vad model.  Must be called before processing."""
        if not self._vad_enabled:
            return
        try:
            import torch  # type: ignore[import]

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._vad_model = model
            self._vad_utils = utils
            logger.info("[%s] silero-vad loaded.", self._session.session_id)
        except Exception as exc:
            logger.warning(
                "[%s] Could not load silero-vad (%s). VAD disabled — all audio will be treated as speech.",
                self._session.session_id,
                exc,
            )
            self._vad_enabled = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def push_audio(self, pcm_bytes: bytes) -> None:
        """Receive raw PCM bytes and process them through the pipeline.

        Bytes are buffered internally and processed in ``chunk_size_ms``
        frames.  Incomplete frames are held until the next call.
        """
        self._session.metrics.mark_first_audio()
        self._session.touch()

        # Accumulate bytes
        self._byte_buffer += pcm_bytes

        samples_needed = self._chunk_samples * 2  # 16-bit = 2 bytes/sample
        while len(self._byte_buffer) >= samples_needed:
            chunk_bytes = self._byte_buffer[:samples_needed]
            self._byte_buffer = self._byte_buffer[samples_needed:]
            await self._process_chunk(chunk_bytes)

    async def flush(self) -> None:
        """Process any remaining buffered audio and signal end-of-stream."""
        if self._byte_buffer:
            # Pad to a full chunk with silence
            samples_needed = self._chunk_samples * 2
            padded = self._byte_buffer + b"\x00" * (samples_needed - len(self._byte_buffer))
            await self._process_chunk(padded)
            self._byte_buffer = b""

        # Force speech end if still active
        if self._is_speech_active:
            await self._on_speech_end()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_chunk(self, chunk_bytes: bytes) -> None:
        pcm_int16 = np.frombuffer(chunk_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0

        is_speech = await self._vad_decision(pcm_float)

        if is_speech:
            self._silence_sample_count = 0
            self._speech_sample_count += len(pcm_float)

            if not self._is_speech_active:
                if self._speech_sample_count >= self._min_speech_samples:
                    await self._on_speech_start()
            else:
                await self._feed_asr(pcm_float)
        else:
            if self._is_speech_active:
                self._silence_sample_count += len(pcm_float)
                # Feed silence to ASR so its endpoint rule can fire
                await self._feed_asr(pcm_float)
            else:
                self._speech_sample_count = 0

    async def _vad_decision(self, pcm_float: np.ndarray) -> bool:
        """Return True if the chunk contains speech."""
        if not self._vad_enabled or self._vad_model is None:
            return True  # treat everything as speech when VAD is off

        loop = asyncio.get_running_loop()
        confidence = await loop.run_in_executor(self._executor, self._run_vad_sync, pcm_float)
        return confidence >= self._vad_threshold

    def _run_vad_sync(self, pcm_float: np.ndarray) -> float:
        """Blocking silero-vad inference — runs in executor."""
        import torch  # type: ignore[import]

        # silero-vad expects a 1-D float32 tensor at 16 kHz
        tensor = torch.from_numpy(pcm_float)
        # Pad/trim to the expected window size
        if len(tensor) < self._vad_window_samples:
            tensor = torch.nn.functional.pad(tensor, (0, self._vad_window_samples - len(tensor)))
        else:
            tensor = tensor[: self._vad_window_samples]

        with torch.no_grad():
            confidence = self._vad_model(tensor, self._sample_rate).item()
        return float(confidence)

    async def _on_speech_start(self) -> None:
        self._is_speech_active = True
        # Handle both sync and async create_stream methods
        if asyncio.iscoroutinefunction(self._asr.create_stream):
            self._asr_stream = await self._asr.create_stream()
        else:
            self._asr_stream = self._asr.create_stream()
        self._last_partial_ref = [""]
        self._session.reset_turn()
        self._session.metrics.mark_speech_start()

        await self._session.token_queue.put({"type": SPEECH_START_EVENT})
        logger.debug("[%s] Speech start.", self._session.session_id)

    async def _on_speech_end(self) -> None:
        self._is_speech_active = False
        self._speech_sample_count = 0
        self._silence_sample_count = 0
        await self._session.token_queue.put({"type": SPEECH_END_EVENT})
        logger.debug("[%s] Speech end (VAD silence).", self._session.session_id)

    async def _feed_asr(self, pcm_float: np.ndarray) -> None:
        """Feed one float32 chunk to the ASR worker."""
        if self._asr_stream is None:
            return

        endpoint = await self._asr.process_chunk(
            stream=self._asr_stream,
            pcm_float32=pcm_float,
            session_id=self._session.session_id,
            event_queue=self._session.token_queue,
            last_partial_ref=self._last_partial_ref,
        )
        if endpoint:
            self._session.metrics.mark_asr_endpoint()
            # Handle both sync and async reset_stream methods
            if asyncio.iscoroutinefunction(self._asr.reset_stream):
                await self._asr.reset_stream(self._asr_stream)
            else:
                self._asr.reset_stream(self._asr_stream)
            self._is_speech_active = False
            self._speech_sample_count = 0
