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

    Server → Client messages (JSON):
      {"type": "asr_partial", "text": str}   — partial ASR hypothesis
      {"type": "token",       "text": str}   — LLM output token
      {"type": "turn_end"}                    — LLM response complete
      {"type": "error",       "message": str}

POST /v1/chat/completions
    OpenAI-compatible chat completion endpoint.

    Each user message may contain a text body and/or a base64-encoded
    audio field ``audio_b64``.  When audio is present it is transcribed
    concurrently with LLM prefill of the text portion, minimising latency.

    Multi-turn sessions are identified by the ``user`` field of the request
    body (used as session_id).  Omit it for stateless single-turn calls.

    Supports both streaming (``"stream": true``) and non-streaming responses
    in standard OpenAI JSON format.

POST /simulate
    Trigger an offline simulation from a WAV file.
    Body: {"audio_file": str, "session_id": str}  (both optional)
    Returns 202 Accepted immediately; simulation runs in background.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

import numpy as np
from audio_pipeline import AudioPipeline
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from orchestrator import Orchestrator
from pydantic import BaseModel, Field
from session_manager import SessionManager
from simulation import simulate_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response schemas
# ---------------------------------------------------------------------------


class ContentPart(BaseModel):
    """A single part within a multi-part message content array."""

    type: str  # "text" | "audio"
    text: Optional[str] = None
    # Base64-encoded raw 16-bit signed PCM audio (16 kHz, mono)
    audio_b64: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    # content may be a plain string or a list of ContentPart objects
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # Used as session_id for multi-turn KV-cache continuation.
    # Clients should send a stable identifier (e.g. a UUID) across turns.
    user: Optional[str] = None


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: ChoiceDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


# ---------------------------------------------------------------------------
# ASR helper for the REST path
# ---------------------------------------------------------------------------


async def _run_asr_to_queue(
    asr_worker: Any,
    audio_bytes: bytes,
    queue: asyncio.Queue,
    session_id: str,
) -> None:
    """Transcribe raw PCM bytes and push transcript chunks into *queue*.

    The function is designed to run as a background ``asyncio.Task`` so that
    ASR and LLM prefill proceed concurrently.  A ``None`` sentinel is put
    into the queue when transcription is complete.

    Parameters
    ----------
    asr_worker:
        Shared ASR worker instance (ASRWorker or ASRWorkerWebSocket).
    audio_bytes:
        Raw 16-bit signed PCM, mono, 16 kHz.
    queue:
        Destination queue; caller reads from this to get transcript chunks.
    session_id:
        Used for logging only.
    """
    try:
        # Create a temporary event queue to receive asr_endpoint events
        event_queue: asyncio.Queue = asyncio.Queue()

        # Handle both sync and async create_stream
        if asyncio.iscoroutinefunction(asr_worker.create_stream):
            stream = await asr_worker.create_stream()
        else:
            stream = asr_worker.create_stream()

        sample_rate = 16000
        chunk_samples = int(sample_rate * 20 / 1000)  # 20 ms chunks
        pcm_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0

        last_partial_ref: list = [""]
        n_chunks = (len(pcm_float) + chunk_samples - 1) // chunk_samples

        for i in range(n_chunks):
            chunk = pcm_float[i * chunk_samples : (i + 1) * chunk_samples]
            if len(chunk) == 0:
                continue
            endpoint = await asr_worker.process_chunk(
                stream=stream,
                pcm_float32=chunk,
                session_id=session_id,
                event_queue=event_queue,
                last_partial_ref=last_partial_ref,
            )
            if endpoint:
                break

        # Drain the event_queue and push asr_endpoint text into the output queue
        transcript = ""
        while not event_queue.empty():
            event = event_queue.get_nowait()
            if event.get("type") == "asr_endpoint":
                transcript = event.get("text", "")

        if not transcript:
            # Fall back to last partial if no endpoint was detected
            transcript = last_partial_ref[0]

        if transcript:
            await queue.put(transcript)

    except Exception as exc:
        logger.exception("[%s] ASR REST task error: %s", session_id, exc)
    finally:
        await queue.put(None)  # sentinel: ASR complete


def _extract_text_and_audio(message: ChatMessage) -> tuple[str, bytes | None]:
    """Extract the text and optional raw PCM audio from a ChatMessage.

    Content may be a plain string or a list of ContentPart dicts/objects.
    Audio is expected as a base64-encoded raw PCM field ``audio_b64``.
    """
    content = message.content

    if isinstance(content, str):
        return content, None

    if isinstance(content, list):
        text_parts: list[str] = []
        audio_bytes: bytes | None = None
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type", "")
                if part_type == "text":
                    text_parts.append(part.get("text", ""))
                elif part_type == "audio" and part.get("audio_b64"):
                    audio_bytes = base64.b64decode(part["audio_b64"])
            elif isinstance(part, ContentPart):
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "audio" and part.audio_b64:
                    audio_bytes = base64.b64decode(part.audio_b64)
        return " ".join(text_parts), audio_bytes

    return str(content), None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


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
        The shared ``vllm.AsyncLLM`` instance.
    session_manager:
        Shared :class:`~session_manager.SessionManager`.
    asr_worker:
        Shared ASR worker instance.
    config:
        Full parsed config.yaml dict.
    """
    server_cfg = config.get("server", {})
    vad_cfg = config.get("vad", {})
    asr_cfg = config.get("asr", {})
    llm_cfg = config.get("llm", {})
    conv_cfg = config.get("conversation", {})
    sim_cfg = config.get("simulation", {})

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
    # OpenAI-compatible chat completions endpoint
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Any:
        """Multi-turn voice-aware chat completion.

        The last message in ``messages`` is treated as the current user
        turn.  Its ``content`` may be:
        - A plain string (text only)
        - A list of ContentPart objects with ``type: "text"`` and/or
          ``type: "audio"`` (base64 PCM)

        When audio is present a background ASR task is started immediately.
        The text portion (if any) is fed to the LLM at once, so LLM prefill
        of the system prompt + text begins before ASR finishes.

        The ``user`` field of the request is used as the session_id.  Reuse
        the same value across turns to enable KV-cache continuation.
        """
        session_id = request.user or f"rest-{uuid.uuid4().hex[:8]}"
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_name = request.model

        state = await session_manager.get_or_create(session_id)
        state.touch()

        # Extract the current user turn (last message)
        if not request.messages:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail="messages must not be empty")

        last_msg = request.messages[-1]
        user_text, audio_bytes = _extract_text_and_audio(last_msg)

        # Override sampling params from the request if provided
        effective_llm_cfg = dict(llm_cfg)
        if request.max_tokens is not None:
            effective_llm_cfg["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            effective_llm_cfg["temperature"] = request.temperature
        if request.top_p is not None:
            effective_llm_cfg["top_p"] = request.top_p

        # Build the ASR queue and start the ASR task in parallel
        asr_queue: asyncio.Queue[str | None] = asyncio.Queue()
        if audio_bytes:
            asyncio.create_task(
                _run_asr_to_queue(asr_worker, audio_bytes, asr_queue, session_id)
            )
        else:
            # No audio: put sentinel immediately so the prompt generator
            # does not block waiting for ASR output.
            await asr_queue.put(None)

        # Save the user turn text now (audio transcript appended after generation)
        if user_text:
            state.append_user_turn(user_text)

        orch = Orchestrator(
            session=state,
            engine=engine,
            llm_config=effective_llm_cfg,
            conv_config=conv_cfg,
            ws_send_queue=asyncio.Queue(),  # unused in REST path
        )

        if request.stream:
            async def _sse_stream() -> AsyncIterator[str]:
                # Send the role delta first
                first_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[StreamChoice(delta=ChoiceDelta(role="assistant"))],
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

                async for token in orch.handle_openai_request(user_text, asr_queue):
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[StreamChoice(delta=ChoiceDelta(content=token))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # Final chunk with finish_reason
                finish_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[StreamChoice(delta=ChoiceDelta(), finish_reason="stop")],
                )
                yield f"data: {finish_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                _sse_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming: collect all tokens
            tokens: list[str] = []
            async for token in orch.handle_openai_request(user_text, asr_queue):
                tokens.append(token)
            full_text = "".join(tokens)

            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=model_name,
                choices=[
                    Choice(
                        message=ChatMessage(role="assistant", content=full_text),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=0,
                    completion_tokens=len(tokens),
                    total_tokens=len(tokens),
                ),
            )
            return response

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
            ws_send_queue=ws_send_queue,
        )

        # Start the orchestrator event loop and WebSocket sender as tasks
        orch_task = asyncio.create_task(orch.run())
        sender_task = asyncio.create_task(_ws_sender(websocket, ws_send_queue))

        try:
            async for message in websocket.iter_bytes():
                if isinstance(message, bytes):
                    await pipeline.push_audio(message)
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
            ws_send_queue=ws_send_queue,
        )

        async def _run_simulation() -> None:
            orch_task = asyncio.create_task(orch.run())
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    current_turn: list[str] = []
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
