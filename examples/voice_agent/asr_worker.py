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
"""ASR worker wrapping sherpa-onnx streaming Zipformer.

Design decisions
----------------
* A **single** :class:`ASRWorker` is shared across all sessions.  The
  ``sherpa_onnx.OnlineRecognizer`` is thread-safe for concurrent streams.
* Each session owns its own ``RecognitionStream`` (created via
  :py:meth:`create_stream`).  The stream is **reset** after each endpoint
  so that the decoder state is clean for the next utterance.
* All blocking sherpa-onnx calls run in a thread-pool executor so they
  never block the asyncio event loop.
* Two event types are emitted on the session's token_queue:
  ``{"type": "asr_partial", "text": "..."}``
  ``{"type": "asr_endpoint", "text": "..."}``
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import numpy as np

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


class ASRWorker:
    """Shared sherpa-onnx OnlineRecognizer wrapper.

    Parameters
    ----------
    config:
        The ``asr`` section of config.yaml (as a dict).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg = config
        self._recognizer: Any = None  # sherpa_onnx.OnlineRecognizer
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="asr-worker"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the sherpa-onnx model.  Call once at startup."""
        try:
            import sherpa_onnx  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed. "
                "Run: pip install sherpa-onnx"
            ) from exc

        model_dir = self._cfg.get("model_dir", "") or os.environ.get(
            "SHERPA_ONNX_MODEL_DIR", ""
        )
        model_size = self._cfg.get("model_size", "small")
        sample_rate = self._cfg.get("sample_rate", 16000)

        if not model_dir:
            raise ValueError(
                "asr.model_dir must be set in config.yaml or via the "
                "SHERPA_ONNX_MODEL_DIR environment variable. "
                "Download a streaming Zipformer model from "
                "https://github.com/k2-fsa/sherpa-onnx/releases"
            )

        logger.info(
            "Loading sherpa-onnx Zipformer-%s from %s", model_size, model_dir
        )

        # Build endpoint config
        endpoint_cfg = sherpa_onnx.EndpointConfig(
            rule1=sherpa_onnx.EndpointRule(
                must_contain_nonsilence=False,
                min_trailing_silence=float(
                    self._cfg.get("endpoint_silence_ms", 200)
                )
                / 1000.0,
                min_utterance_length=0.0,
            ),
            rule2=sherpa_onnx.EndpointRule(
                must_contain_nonsilence=True,
                min_trailing_silence=float(
                    self._cfg.get("endpoint_silence_ms", 200)
                )
                / 1000.0,
                min_utterance_length=0.0,
            ),
            rule3=sherpa_onnx.EndpointRule(
                must_contain_nonsilence=True,
                min_trailing_silence=0.0,
                min_utterance_length=20.0,
            ),
        )

        feat_config = sherpa_onnx.FeatureExtractorConfig(
            sampling_rate=sample_rate,
        )

        # Expect model files in model_dir with standard sherpa-onnx naming
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=os.path.join(model_dir, "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=os.path.join(model_dir, "decoder-epoch-99-avg-1.int8.onnx"),
            joiner=os.path.join(model_dir, "joiner-epoch-99-avg-1.int8.onnx"),
            tokens=os.path.join(model_dir, "tokens.txt"),
            num_threads=2,
            sample_rate=sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=float(
                self._cfg.get("endpoint_silence_ms", 200)
            )
            / 1000.0,
            rule2_min_trailing_silence=float(
                self._cfg.get("endpoint_silence_ms", 200)
            )
            / 1000.0,
            rule3_min_utterance_length=20.0,
            decoding_method="greedy_search",
        )
        logger.info("ASR model loaded successfully.")

    def create_stream(self) -> Any:
        """Create a new per-session recognition stream."""
        return self._recognizer.create_stream()

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    async def process_chunk(
        self,
        stream: Any,
        pcm_float32: np.ndarray,
        session_id: str,
        event_queue: asyncio.Queue,
        last_partial_ref: list,  # mutable single-element list: [str]
    ) -> bool:
        """Feed one audio chunk to the ASR stream.

        Returns True if an endpoint was detected (phrase complete).
        The caller should then call :py:meth:`reset_stream`.
        """
        loop = asyncio.get_running_loop()
        endpoint_detected = await loop.run_in_executor(
            self._executor,
            self._decode_chunk,
            stream,
            pcm_float32,
            session_id,
            event_queue,
            loop,
            last_partial_ref,
        )
        return endpoint_detected

    def _decode_chunk(
        self,
        stream: Any,
        pcm: np.ndarray,
        session_id: str,
        event_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
        last_partial_ref: list,
    ) -> bool:
        """Blocking decode — runs in executor thread."""
        self._recognizer.accept_waveform(stream, pcm)

        while self._recognizer.is_ready(stream):
            self._recognizer.decode(stream)

        result = self._recognizer.get_result(stream)
        text = result.text.strip()

        # Stable prefix tracking
        old_partial = last_partial_ref[0]
        prefix_len = _common_prefix_length(old_partial, text)
        stable_text = text[:prefix_len] if prefix_len > 0 else ""

        if text and text != old_partial:
            last_partial_ref[0] = text
            asyncio.run_coroutine_threadsafe(
                event_queue.put({"type": _ASR_PARTIAL, "text": text,
                                 "stable_prefix": stable_text}),
                loop,
            )

        endpoint = self._recognizer.is_endpoint(stream)
        if endpoint and text:
            asyncio.run_coroutine_threadsafe(
                event_queue.put({"type": _ASR_ENDPOINT, "text": text}),
                loop,
            )
            logger.debug("[%s] ASR endpoint: %r", session_id, text)

        return endpoint

    def reset_stream(self, stream: Any) -> None:
        """Reset the decoder state after an endpoint."""
        self._recognizer.reset(stream)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
