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
"""FastAPI server: WebSocket + REST endpoints.

Endpoints
---------
GET  /health
    Liveness probe.  Returns ``{"status": "ok"}``.

WebSocket  /ws/{session_id}
    Real-time audio streaming endpoint.

    Client → Server messages:
      Binary frames: raw PCM audio (16-bit signed, mono, 16 kHz)
      Text frames (JSON): {"type": "end"} — client signals end of audio stream

    Server → Client messages (JSON):
      {"type": "asr_partial", "text": str}   — partial ASR hypothesis
      {"type": "token",       "text": str}   — LLM output token
      {"type": "turn_end"}                    — LLM response complete
      {"type": "error",       "message": str}

POST /simulate
    Trigger an offline simulation from a WAV file.
    Body: {"audio_file": str, "session_id": str}  (both optional)
    Returns 202 Accepted immediately; simulation runs in background.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

from audio_pipeline import AudioPipeline
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from orchestrator import Orchestrator
from pydantic import BaseModel
from session_manager import SessionManager
from simulation import simulate_session

logger = logging.getLogger(__name__)


def create_app(
    engine: Any,
    session_manager: SessionManager,
    asr_worker: Any,
    config: dict[str, Any],
) -> FastAPI:
    """Factory that wires up the FastAPI application.

    Parameters
    ----------
    engine:
        The shared ``vllm.AsyncLLMEngine`` instance.
    session_manager:
        Shared :class:`~session_manager.SessionManager`.
    asr_worker:
        Shared :class:`~asr_worker.ASRWorker`.
    config:
        Full parsed config.yaml dict.
    """
    server_cfg = config.get("server", {})
    vad_cfg = config.get("vad", {})
    asr_cfg = config.get("asr", {})
    llm_cfg = config.get("llm", {})
    conv_cfg = config.get("conversation", {})
    sim_cfg = config.get("simulation", {})
    enable_barge_in: bool = bool(server_cfg.get("enable_barge_in", True))

    app = FastAPI(title="Voice Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_cfg.get("cors_origins", ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Background task: evict idle sessions every 60 s
    @app.on_event("startup")
    async def _start_eviction_loop() -> None:
        async def _evict_loop() -> None:
            while True:
                await asyncio.sleep(60)
                await session_manager.evict_idle_sessions()

        asyncio.create_task(_evict_loop())

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "active_sessions": session_manager.active_session_count,
        }

    # ------------------------------------------------------------------
    # WebSocket endpoint
    # ------------------------------------------------------------------

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        logger.info("WebSocket connected: session=%s", session_id)

        state = await session_manager.get_or_create(session_id)

        # Per-connection outbound queue (orchestrator → WebSocket sender)
        ws_send_queue: asyncio.Queue = asyncio.Queue()

        pipeline = AudioPipeline(
            session=state,
            asr_worker=asr_worker,
            vad_config=vad_cfg,
            asr_config=asr_cfg,
        )
        pipeline.load_vad()

        orch = Orchestrator(
            session=state,
            engine=engine,
            llm_config=llm_cfg,
            conv_config=conv_cfg,
            enable_barge_in=enable_barge_in,
            ws_send_queue=ws_send_queue,
        )

        # Start the orchestrator event loop and WebSocket sender as tasks
        orch_task = asyncio.create_task(orch.run())
        sender_task = asyncio.create_task(_ws_sender(websocket, ws_send_queue))

        try:
            async for message in websocket.iter_bytes():
                if isinstance(message, bytes):
                    await pipeline.push_audio(message)
                # Text frames: check for end-of-stream signal
            # Client closed connection normally
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: session=%s", session_id)
        except Exception as exc:
            logger.exception("WebSocket error session=%s: %s", session_id, exc)
        finally:
            await pipeline.flush()
            pipeline.shutdown()
            await orch.shutdown()
            sender_task.cancel()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(orch_task, timeout=5.0)
            await session_manager.remove(session_id)

    # ------------------------------------------------------------------
    # Simulate endpoint
    # ------------------------------------------------------------------

    class SimulateRequest(BaseModel):
        audio_file: str | None = None
        session_id: str | None = None

    @app.post("/simulate", status_code=202)
    async def simulate_endpoint(req: SimulateRequest) -> dict[str, str]:
        audio_file = req.audio_file or sim_cfg.get("audio_file", "test_audio.wav")
        session_id = req.session_id or sim_cfg.get("session_id", "sim-session-001")
        playback_speed = float(sim_cfg.get("playback_speed", 1.0))
        chunk_ms = int(asr_cfg.get("chunk_size_ms", 20))
        sample_rate = int(asr_cfg.get("sample_rate", 16000))

        state = await session_manager.get_or_create(session_id)
        ws_send_queue: asyncio.Queue = asyncio.Queue()

        pipeline = AudioPipeline(
            session=state,
            asr_worker=asr_worker,
            vad_config=vad_cfg,
            asr_config=asr_cfg,
        )
        pipeline.load_vad()

        orch = Orchestrator(
            session=state,
            engine=engine,
            llm_config=llm_cfg,
            conv_config=conv_cfg,
            enable_barge_in=enable_barge_in,
            ws_send_queue=ws_send_queue,
        )

        async def _run_simulation() -> None:
            orch_task = asyncio.create_task(orch.run())
            # Log outbound messages to stdout (mock TTS)
            log_task = asyncio.create_task(_log_token_stream(ws_send_queue, session_id))
            try:
                await simulate_session(
                    wav_path=audio_file,
                    push_audio_fn=pipeline.push_audio,
                    flush_fn=pipeline.flush,
                    chunk_ms=chunk_ms,
                    sample_rate=sample_rate,
                    playback_speed=playback_speed,
                )
            finally:
                await pipeline.flush()
                pipeline.shutdown()
                await orch.shutdown()
                log_task.cancel()
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(orch_task, timeout=30.0)
                await session_manager.remove(session_id)

        asyncio.create_task(_run_simulation())
        return {"session_id": session_id, "status": "started"}

    return app


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


async def _ws_sender(websocket: WebSocket, send_queue: asyncio.Queue) -> None:
    """Drain the send queue and push JSON messages to the WebSocket."""
    try:
        while True:
            msg = await send_queue.get()
            try:
                await websocket.send_text(json.dumps(msg))
            except Exception:
                break
            finally:
                send_queue.task_done()
    except asyncio.CancelledError:
        pass


async def _log_token_stream(send_queue: asyncio.Queue, session_id: str) -> None:
    """Log outbound messages to stdout (used in simulation mode)."""
    current_turn = []
    try:
        while True:
            msg = await send_queue.get()
            msg_type = msg.get("type")
            if msg_type == "asr_partial":
                print(f"\r[{session_id}] ASR: {msg.get('text', '')}", end="", flush=True)
            elif msg_type == "token":
                text = msg.get("text", "")
                current_turn.append(text)
                print(text, end="", flush=True)
            elif msg_type == "turn_end":
                print()  # newline after response
                current_turn.clear()
            elif msg_type == "error":
                print(f"\n[{session_id}] ERROR: {msg.get('message', '')}")
            send_queue.task_done()
    except asyncio.CancelledError:
        if current_turn:
            print()
