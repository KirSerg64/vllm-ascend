# Hedge Pipeline

A fast-filler + thinking-model inference pipeline for vLLM-Ascend that reduces
user-perceived latency by streaming a quick preliminary answer from a small
model while the authoritative answer from a large thinking model is computed.

## Architecture

```
User Request
     │
     ├─────────────────────────────────────────┐
     ▼                                         ▼
[Small model: Qwen3-0.6B]               [Big model: Qwen3-Omni-30B-A3B W8A8]
 ~50 ms TTFT                             ~950 ms TTFT
 Streams quick filler to user            Receives original_prompt + filler hint
 Output injected into big-model prompt   Streams final answer to user
     │                                         │
     ▼                                         ▼
[User sees fast response]          [User sees corrected/full response]
```

### Flow

1. User sends a message to the hedge server.
2. Small model (Qwen3-0.6B) generates a filler answer (~50 ms TTFT) and
   streams it to the client immediately.
3. After the filler is complete, the big model receives an augmented prompt
   (original question + filler answer as a thinking hint).
4. Big model streams the authoritative answer to the client (~950 ms TTFT, but
   the user already has a preliminary response).

### Key features

| Feature | Details |
|---|---|
| **Graceful degradation** | Small model timeout → big model uses original prompt |
| **Round-robin load balancing** | Multiple big-model instances supported |
| **SSE streaming** | Two event types (`filler`, `final`) for client-side rendering |
| **Async/await** | Full async implementation for high concurrency |
| **NPU-optimised** | No unnecessary CPU-NPU synchronisation |
| **vllm-ascend compatible** | Works with speculative decoding, quantisation, etc. |

---

## Prerequisites

```bash
pip install fastapi uvicorn aiohttp httpx pydantic
```

---

## Quick Start

### 1. Start the small model server (Qwen3-0.6B)

```bash
vllm serve Qwen/Qwen3-0.6B \
    --host 0.0.0.0 \
    --port 8100 \
    --max-model-len 4096 \
    --dtype float16
```

### 2. Start one or more big model servers (Qwen3-Omni-30B-A3B W8A8)

```bash
# Instance 1
vllm serve Qwen/Qwen3-30B-A3B \
    --host 0.0.0.0 \
    --port 8200 \
    --quantization fp8 \
    --tensor-parallel-size 4

# Instance 2 (optional, for load balancing)
vllm serve Qwen/Qwen3-30B-A3B \
    --host 0.0.0.0 \
    --port 8201 \
    --quantization fp8 \
    --tensor-parallel-size 4
```

### 3. Start the hedge server

```bash
# Single big-model instance
SMALL_MODEL_BASE_URL=http://localhost:8100 \
SMALL_MODEL_NAME=Qwen/Qwen3-0.6B \
BIG_MODEL_BASE_URLS=http://localhost:8200 \
BIG_MODEL_NAME=Qwen/Qwen3-30B-A3B \
python -m examples.hedge_pipeline.server --host 0.0.0.0 --port 8000

# Two big-model instances (round-robin)
BIG_MODEL_BASE_URLS="http://localhost:8200,http://localhost:8201" \
python -m examples.hedge_pipeline.server --host 0.0.0.0 --port 8000
```

### 4. Run the example client

```bash
python examples/hedge_pipeline/client_example.py \
    --url http://localhost:8000 \
    --message "Explain the concept of quantum entanglement."
```

---

## Configuration

All configuration is driven by environment variables.

### Small model

| Variable | Default | Description |
|---|---|---|
| `SMALL_MODEL_BASE_URL` | `http://localhost:8100` | vLLM server URL for the small model |
| `SMALL_MODEL_NAME` | `Qwen/Qwen3-0.6B` | Model name as registered in the vLLM server |
| `SMALL_MODEL_MAX_TOKENS` | `80` | Maximum tokens the filler may generate |
| `SMALL_MODEL_TEMPERATURE` | `0.3` | Sampling temperature (lower = more deterministic) |
| `SMALL_MODEL_TIMEOUT` | `0.5` | Hard timeout in seconds; exceeding it falls back gracefully |
| `SMALL_MODEL_TOP_P` | `0.9` | Top-p nucleus sampling |

### Big model

| Variable | Default | Description |
|---|---|---|
| `BIG_MODEL_BASE_URLS` | `http://localhost:8200` | Comma-separated list of vLLM server URLs |
| `BIG_MODEL_NAME` | `Qwen/Qwen3-30B-A3B` | Model name as registered in the vLLM server |
| `BIG_MODEL_MAX_TOKENS` | `1024` | Maximum tokens the big model may generate |
| `BIG_MODEL_TEMPERATURE` | `0.6` | Sampling temperature |
| `BIG_MODEL_TOP_P` | `0.95` | Top-p nucleus sampling |
| `BIG_MODEL_TIMEOUT` | `60.0` | Hard timeout in seconds (0 = no timeout) |

### Pipeline

| Variable | Default | Description |
|---|---|---|
| `HEDGE_STREAM_FILLER` | `1` | `1` = stream filler tokens as they arrive; `0` = buffer first |
| `HEDGE_LOG_LEVEL` | `INFO` | Python log level for the pipeline and server |

---

## API Reference

### `POST /v1/hedge/chat/completions`

Submit a user message and receive a two-phase SSE stream.

**Request body**

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "system": "You are a helpful assistant.",
  "max_tokens": 512,
  "temperature": 0.6,
  "request_id": "my-correlation-id-123"
}
```

All fields except `messages` are optional.

**Response** — `text/event-stream`

```
event: filler
data: {"phase": "filler", "text": "Paris", "done": false}

event: filler
data: {"phase": "filler", "text": " is", "done": false}

event: filler
data: {"phase": "filler", "text": "", "done": true}

event: final
data: {"phase": "final", "text": "The capital of France is Paris", "done": false}

...

event: final
data: {"phase": "final", "text": "", "done": true}

event: done
data: {}
```

| Event | Payload | Meaning |
|---|---|---|
| `filler` | `{"phase": "filler", "text": "<delta>", "done": bool}` | Token delta from the small model |
| `final` | `{"phase": "final", "text": "<delta>", "done": bool}` | Token delta from the big model |
| `done` | `{}` | End-of-stream sentinel |
| `error` | `{"message": "..."}` | Error description |

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

### `GET /v1/hedge/config`

Returns the active pipeline configuration.

---

## Using the pipeline library directly

You can also use the pipeline without the HTTP server:

```python
import asyncio
from examples.hedge_pipeline.config import PipelineConfig
from examples.hedge_pipeline.pipeline import HedgePipeline

async def main():
    config = PipelineConfig()
    pipeline = HedgePipeline(config)

    async for event in pipeline.run("What is quantum entanglement?"):
        print(f"[{event['phase']}] {event['text']}", end="", flush=True)

    await pipeline.close()

asyncio.run(main())
```

---

## Expected Performance

| Metric | Value |
|---|---|
| Filler TTFT | ~50 ms |
| Big model TTFT | ~950 ms (user already has filler) |
| User-perceived latency | ~50 ms |
| Filler quality | Preliminary answer from Qwen3-0.6B |
| Final quality | Full reasoning from Qwen3-Omni-30B-A3B-Thinking |

---

## Files

| File | Description |
|---|---|
| `__init__.py` | Package marker |
| `config.py` | Configuration dataclasses; all params from env vars |
| `pipeline.py` | Core async dual-model orchestration |
| `server.py` | FastAPI proxy with SSE streaming and round-robin load balancing |
| `client_example.py` | CLI client that prints filler + final responses |
| `README.md` | This file |
