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
"""Per-session state management.

Each WebSocket client owns one :class:`SessionState` which holds:
- conversation history for multi-turn LLM context
- the live sherpa-onnx ASR stream
- barge-in tracking state
- TTFA metrics
- an asyncio lock for concurrent-safe mutation

:class:`SessionManager` is a singleton that owns all active sessions and
evicts idle ones based on a configurable timeout.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from metrics import SessionMetrics

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """All mutable state for a single connected client."""

    session_id: str

    # -----------------------------------------------------------------
    # Conversation history
    # -----------------------------------------------------------------
    # List of {"role": "user"|"assistant", "content": str} dicts.
    # The system prompt is prepended at prompt-build time, not stored here.
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # -----------------------------------------------------------------
    # ASR state
    # -----------------------------------------------------------------
    # The live sherpa-onnx RecognitionStream for this session.
    # Recreated after each endpoint to reset internal decoder state.
    asr_stream: Any = field(default=None)
    # Stable ASR prefix from the current utterance (for future chunked mode)
    stable_asr_prefix: str = ""
    # Full partial hypothesis from the last ASR decode
    last_partial: str = ""

    # -----------------------------------------------------------------
    # LLM / barge-in state
    # -----------------------------------------------------------------
    # Request ID of the currently in-flight LLM generation, or None
    current_llm_request_id: Optional[str] = None
    # True while the LLM is streaming tokens to the client
    is_bot_speaking: bool = False
    # asyncio Queue that the LLM orchestrator writes tokens into.
    # The WebSocket sender drains this queue.
    token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------
    metrics: SessionMetrics = field(init=False)

    # -----------------------------------------------------------------
    # Concurrency
    # -----------------------------------------------------------------
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Last activity timestamp (wall-clock seconds)
    last_active: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.metrics = SessionMetrics(session_id=self.session_id)

    def touch(self) -> None:
        """Update the last-activity timestamp."""
        self.last_active = time.monotonic()

    def is_idle(self, timeout_seconds: float) -> bool:
        return (time.monotonic() - self.last_active) > timeout_seconds

    def reset_turn(self) -> None:
        """Prepare for a new user utterance."""
        self.stable_asr_prefix = ""
        self.last_partial = ""
        self.metrics.reset_turn()

    def append_user_turn(self, text: str) -> None:
        self.conversation_history.append({"role": "user", "content": text})

    def append_assistant_turn(self, text: str) -> None:
        self.conversation_history.append(
            {"role": "assistant", "content": text}
        )

    def get_history_window(self, max_turns: int) -> List[Dict[str, str]]:
        """Return the last *max_turns* user+assistant pairs."""
        # Each "turn" = 1 user message + 1 assistant message = 2 entries.
        max_entries = max_turns * 2
        return self.conversation_history[-max_entries:]


class SessionManager:
    """Thread-safe registry of all active :class:`SessionState` objects.

    Usage::

        manager = SessionManager(session_timeout_minutes=30)
        state = manager.get_or_create("abc123")
        ...
        await manager.evict_idle_sessions()
    """

    def __init__(self, session_timeout_minutes: int = 30) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = asyncio.Lock()
        self._timeout_seconds = session_timeout_minutes * 60

    async def get_or_create(self, session_id: str) -> SessionState:
        async with self._lock:
            if session_id not in self._sessions:
                logger.info("Creating new session: %s", session_id)
                self._sessions[session_id] = SessionState(
                    session_id=session_id
                )
            state = self._sessions[session_id]
            state.touch()
            return state

    async def get(self, session_id: str) -> Optional[SessionState]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            if state is not None:
                state.metrics.log_summary()
                logger.info("Session removed: %s", session_id)

    async def evict_idle_sessions(self) -> None:
        """Remove sessions that have been idle longer than the timeout."""
        async with self._lock:
            idle = [
                sid
                for sid, s in self._sessions.items()
                if s.is_idle(self._timeout_seconds)
            ]
            for sid in idle:
                state = self._sessions.pop(sid)
                state.metrics.log_summary()
                logger.info(
                    "Evicting idle session: %s (idle > %ds)",
                    sid,
                    self._timeout_seconds,
                )

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)
