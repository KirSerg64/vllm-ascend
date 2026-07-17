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
"""TTFA (Time-To-First-Audio) metrics collection.

Each pipeline stage records a timestamp so end-to-end latency can be
computed and logged.  All times are wall-clock seconds from
``time.perf_counter()``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """Per-session pipeline timing.

    Timestamps are recorded once per *turn* (one user utterance → one
    assistant response).  Call :py:meth:`reset_turn` at the start of each
    new user phrase.
    """

    session_id: str

    # Timestamp of the first audio chunk received for the current turn
    first_audio_ts: float | None = field(default=None)
    # Timestamp when VAD confirmed speech start
    speech_start_ts: float | None = field(default=None)
    # Timestamp when ASR endpoint was detected (phrase complete)
    asr_endpoint_ts: float | None = field(default=None)
    # Timestamp when the LLM request was submitted
    llm_submit_ts: float | None = field(default=None)
    # Timestamp of the first LLM output token
    first_token_ts: float | None = field(default=None)
    # Timestamp when the full LLM response was delivered
    response_complete_ts: float | None = field(default=None)

    # Accumulated per-session totals
    total_turns: int = 0
    total_ttfa_ms: float = 0.0
    total_e2e_ms: float = 0.0

    def mark_first_audio(self) -> None:
        if self.first_audio_ts is None:
            self.first_audio_ts = time.perf_counter()

    def mark_speech_start(self) -> None:
        self.speech_start_ts = time.perf_counter()

    def mark_asr_endpoint(self) -> None:
        self.asr_endpoint_ts = time.perf_counter()

    def mark_llm_submit(self) -> None:
        self.llm_submit_ts = time.perf_counter()

    def mark_first_token(self) -> None:
        if self.first_token_ts is None:
            self.first_token_ts = time.perf_counter()
            self._log_ttfa()

    def mark_response_complete(self) -> None:
        self.response_complete_ts = time.perf_counter()
        self._log_e2e()

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def ttfa_ms(self) -> float | None:
        """Time from first audio chunk to first LLM token (ms)."""
        if self.first_audio_ts is not None and self.first_token_ts is not None:
            return (self.first_token_ts - self.first_audio_ts) * 1000.0
        return None

    @property
    def asr_latency_ms(self) -> float | None:
        """Time from first audio chunk to ASR endpoint (ms)."""
        if self.first_audio_ts is not None and self.asr_endpoint_ts is not None:
            return (self.asr_endpoint_ts - self.first_audio_ts) * 1000.0
        return None

    @property
    def llm_ttft_ms(self) -> float | None:
        """LLM time-to-first-token from request submission (ms)."""
        if self.llm_submit_ts is not None and self.first_token_ts is not None:
            return (self.first_token_ts - self.llm_submit_ts) * 1000.0
        return None

    @property
    def e2e_ms(self) -> float | None:
        """Time from first audio to full response completion (ms)."""
        if self.first_audio_ts is not None and self.response_complete_ts is not None:
            return (self.response_complete_ts - self.first_audio_ts) * 1000.0
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset_turn(self) -> None:
        """Reset per-turn timestamps at the beginning of a new phrase."""
        self.first_audio_ts = None
        self.speech_start_ts = None
        self.asr_endpoint_ts = None
        self.llm_submit_ts = None
        self.first_token_ts = None
        self.response_complete_ts = None

    def _log_ttfa(self) -> None:
        ttfa = self.ttfa_ms
        if ttfa is None:
            return
        status = "✅" if ttfa < 200 else "⚠️"
        logger.info(
            "[%s] %s TTFA=%.1f ms  (ASR=%.1f ms, LLM-TTFT=%.1f ms)",
            self.session_id,
            status,
            ttfa,
            self.asr_latency_ms or 0.0,
            self.llm_ttft_ms or 0.0,
        )
        self.total_turns += 1
        self.total_ttfa_ms += ttfa

    def _log_e2e(self) -> None:
        e2e = self.e2e_ms
        if e2e is None:
            return
        if self.total_turns > 0:
            self.total_e2e_ms += e2e
        logger.debug(
            "[%s] Turn complete  E2E=%.1f ms",
            self.session_id,
            e2e,
        )

    def log_summary(self) -> None:
        """Log aggregate statistics for this session."""
        if self.total_turns == 0:
            logger.info("[%s] No turns completed.", self.session_id)
            return
        avg_ttfa = self.total_ttfa_ms / self.total_turns
        avg_e2e = self.total_e2e_ms / self.total_turns if self.total_turns else 0
        logger.info(
            "[%s] Session summary: turns=%d  avg_TTFA=%.1f ms  avg_E2E=%.1f ms",
            self.session_id,
            self.total_turns,
            avg_ttfa,
            avg_e2e,
        )
