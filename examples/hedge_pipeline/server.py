# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
server.py — FastAPI proxy server for the hedge pipeline.

Exposes a single endpoint::

    POST / v1 / hedge / chat / completions

that accepts OpenAI-style chat-completions request bodies and returns a
Server-Sent Events (SSE) stream with two event types:

    event: filler   — incremental tokens from the small model
    event: final    — incremental tokens from the big model

Start the server::

    python -m examples.hedge_pipeline.server [--host 0.0.0.0] [--port 8000]

Or via uvicorn directly::

    uvicorn examples.hedge_pipeline.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_config
from .pipeline import HedgePipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("hedge_server")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hedge Pipeline Server",
    description=(
        "Async dual-model inference proxy that returns a fast filler answer "
        "from a small model followed by the authoritative answer from a large "
        "thinking model."
    ),
    version="1.0.0",
)

# Global pipeline instance — initialised in the lifespan event.
_pipeline: HedgePipeline | None = None


@app.on_event("startup")
async def startup_event() -> None:
    global _pipeline
    config = get_config()
    logging.getLogger().setLevel(config.log_level)
    _pipeline = HedgePipeline(config)
    logger.info(
        "Hedge pipeline ready.  small=%s  big=%s",
        config.small.base_url,
        config.big.base_urls,
    )
    tok_cfg = config.tokenizer
    if tok_cfg.use_local_tokenizer:
        logger.info(
            "Warming tokenizer '%s' for parallel pre-tokenization …",
            tok_cfg.model_name_or_path,
        )
        await _pipeline.warm_tokenizer()
        logger.info("Tokenizer ready (parallel tokenization enabled).")
    else:
        logger.info("Parallel tokenization disabled (HEDGE_USE_LOCAL_TOKENIZER=0).")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    if _pipeline is not None:
        await _pipeline.close()
    logger.info("Hedge pipeline shut down.")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class HedgeRequest(BaseModel):
    messages: list[ChatMessage]
    system: str | None = Field(None, description="Optional system prompt override.")
    # Forwarded to both models when supplied; otherwise model defaults apply.
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    request_id: str | None = Field(None, description="Client-supplied correlation ID (auto-generated if absent).")


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_line(event: str, data: dict) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _generate_sse(
    pipeline: HedgePipeline,
    user_message: str,
    system_message: str | None,
    request_id: str,
) -> AsyncIterator[str]:
    """Drive the hedge pipeline and yield SSE-formatted strings."""
    try:
        async for event in pipeline.run(
            user_message=user_message,
            system_message=system_message,
            request_id=request_id,
        ):
            yield _sse_line(event["phase"], event)
    except asyncio.CancelledError:
        logger.info("Client disconnected (request_id=%s).", request_id)
    except Exception as exc:
        logger.exception("Unexpected error in SSE generator (request_id=%s): %s", request_id, exc)
        # Return a generic error message to the client to avoid leaking
        # internal details such as file paths or stack frames.
        yield _sse_line("error", {"message": "Internal server error. See server logs for details."})
    finally:
        yield "event: done\ndata: {}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe."""
    return JSONResponse({"status": "ok"})


@app.post("/v1/hedge/chat/completions")
async def hedge_chat_completions(request: HedgeRequest, http_request: Request) -> StreamingResponse:
    """
    Submit a chat message and receive a two-phase SSE stream.

    **SSE event types**

    | Type     | Meaning                                              |
    |----------|------------------------------------------------------|
    | filler   | Incremental delta from the small (fast) model        |
    | final    | Incremental delta from the big (thinking) model      |
    | done     | End-of-stream sentinel (no data payload)             |
    | error    | Error payload ``{"message": "..."}``                 |

    Each ``filler`` / ``final`` event carries::

        {"phase": "filler" | "final", "text": "<delta>", "done": bool}
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    # Extract the last user turn from the messages list.
    user_turns = [m for m in request.messages if m.role == "user"]
    if not user_turns:
        raise HTTPException(status_code=400, detail="No user message found in 'messages'.")
    user_message = user_turns[-1].content

    # Resolve system message: explicit field > first system turn > None.
    system_message = request.system
    if system_message is None:
        system_turns = [m for m in request.messages if m.role == "system"]
        if system_turns:
            system_message = system_turns[0].content

    request_id = request.request_id or str(uuid.uuid4())

    # Allow per-request overrides of generation params by updating a
    # *copy* of the pipeline's config so the global state is not mutated.
    pipeline = _pipeline
    if any(v is not None for v in (request.max_tokens, request.temperature, request.top_p)):
        from copy import deepcopy

        config_copy = deepcopy(pipeline.config)
        if request.max_tokens is not None:
            config_copy.big.max_tokens = request.max_tokens
        if request.temperature is not None:
            config_copy.small.temperature = request.temperature
            config_copy.big.temperature = request.temperature
        if request.top_p is not None:
            config_copy.small.top_p = request.top_p
            config_copy.big.top_p = request.top_p
        pipeline = HedgePipeline(config_copy)

    logger.info(
        "Serving hedge request (request_id=%s, user_msg_len=%d)",
        request_id,
        len(user_message),
    )

    return StreamingResponse(
        _generate_sse(pipeline, user_message, system_message, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )


@app.get("/v1/hedge/config")
async def get_pipeline_config() -> JSONResponse:
    """Return the active pipeline configuration (no secrets)."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")
    cfg = _pipeline.config
    return JSONResponse(
        {
            "small_model": {
                "base_url": cfg.small.base_url,
                "model": cfg.small.model,
                "max_tokens": cfg.small.max_tokens,
                "temperature": cfg.small.temperature,
                "timeout": cfg.small.timeout,
            },
            "big_model": {
                "base_urls": cfg.big.base_urls,
                "model": cfg.big.model,
                "max_tokens": cfg.big.max_tokens,
                "temperature": cfg.big.temperature,
                "timeout": cfg.big.timeout,
            },
            "tokenizer": {
                "model_name_or_path": cfg.tokenizer.model_name_or_path,
                "use_local_tokenizer": cfg.tokenizer.use_local_tokenizer,
                "max_workers": cfg.tokenizer.max_workers,
            },
            "stream_filler": cfg.stream_filler,
        }
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hedge Pipeline Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    parser.add_argument("--log-level", default="info", help="Log level (default: info)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    uvicorn.run(
        "examples.hedge_pipeline.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
