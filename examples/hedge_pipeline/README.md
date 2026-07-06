# Hedge Pipeline

A fast-filler + thinking-model inference pipeline for vLLM-Ascend that reduces
user-perceived latency by streaming a quick preliminary answer from a small
model while the authoritative answer from a large thinking model is computed.

## Architecture

```
T=0ms   User Request arrives
         │
         ├─── small model call (HTTP) ──────────────────────────────────────────┐
         │                                                                      │
         ├─── tokenize(original_prompt) in thread executor ─── done ~2–10ms   │
         │                                                                      │
         │    ← filler tokens streamed to client as they arrive (~50ms TTFT)   │
         │                                                                      │
T=50ms  ← small model done; tokenize(filler_suffix) ~1ms ──────────────────────┘
         │
         ├─── concatenate token IDs: [original] + [filler_suffix]
         │
         └─── POST /v1/completions  {"prompt": [int, int, ...]}
                  └─ vLLM skips tokenization → prefill starts immediately
                  └─ big model streams final answer (~950ms TTFT)
```

### Flow

1. User sends a message to the hedge server.
2. Two tasks start **concurrently at T=0**:
   - The small model (Qwen3-0.6B) generates a filler answer and streams it
     to the client (~50 ms TTFT).
   - The proxy tokenizes the original prompt in a thread executor (~2–10 ms).
3. When the small model finishes, only the short filler-hint suffix needs to
   be tokenized (~1 ms; ≤80 tokens).
4. The concatenated token-ID list is sent directly to the big model via
   `POST /v1/completions` with `"prompt": [int, ...]`, bypassing vLLM's
   internal tokenisation step and letting NPU prefill begin immediately.
5. The big model streams the authoritative answer (~950 ms TTFT, but the
   user already has a preliminary response).

**Net saving vs. the previous text-based path: ~10–20 ms per request**, fully
hidden by existing latency and with zero hardware cost.

### Key features

| Feature | Details |
|---|---|
| **Parallel tokenization** | Original prompt tokenized concurrently with small model call |
| **`/v1/completions` token-ID path** | Bypasses vLLM's tokenizer on the big model server |
| **Graceful degradation** | Any failure falls back to `/v1/chat/completions` text path |
| **Round-robin load balancing** | Multiple big-model instances supported |
| **SSE streaming** | Two event types (`filler`, `final`) for client-side rendering |
| **Async/await** | Full async implementation for high concurrency |
| **NPU-optimised** | No unnecessary CPU-NPU synchronisation |
| **vllm-ascend compatible** | Works with speculative decoding, quantisation, etc. |

---

## Prerequisites

```bash
pip install fastapi uvicorn aiohttp httpx pydantic transformers
```

> `transformers` is required for local tokenization (parallel pre-tokenization
> feature).  Set `HEDGE_USE_LOCAL_TOKENIZER=0` to skip it.

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

### Tokenizer (parallel pre-tokenization)

| Variable | Default | Description |
|---|---|---|
| `HEDGE_USE_LOCAL_TOKENIZER` | `1` | `1` = enable parallel tokenization + `/v1/completions` path; `0` = use original text path |
| `TOKENIZER_MODEL_NAME` | same as `BIG_MODEL_NAME` | HuggingFace model name/path for the tokenizer loaded in the proxy |
| `TOKENIZER_MAX_WORKERS` | `2` | Thread-pool size for tokenization (rarely needs increasing) |
| `TOKENIZER_TURN_END_STR` | `\n<\|im_end\|>\n` | Chat-template user-turn-close string (ChatML / Qwen3 default) |
| `TOKENIZER_GEN_PROMPT_STR` | `<\|im_start\|>assistant\n` | Chat-template generation-prompt string (ChatML / Qwen3 default) |

> **Note on `TOKENIZER_TURN_END_STR` / `TOKENIZER_GEN_PROMPT_STR`**: these
> must match the chat template of your big model.  The defaults work for any
> ChatML-format model (Qwen3, Qwen2, Mistral-instruct, etc.).  Override them
> if you use a model with a different template (e.g. LLaMA-3's `<|eot_id|>`).

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
    # Warm the tokenizer once before serving requests.
    await pipeline.warm_tokenizer()

    async for event in pipeline.run("What is quantum entanglement?"):
        print(f"[{event['phase']}] {event['text']}", end="", flush=True)

    await pipeline.close()

asyncio.run(main())
```

---

## Expected Performance

| Metric | Without parallel tokenization | With parallel tokenization |
|---|---|---|
| Filler TTFT | ~50 ms | ~50 ms |
| Big model TTFT | ~950 ms | ~930–940 ms |
| Tokenization on critical path | ~10–25 ms | ~1 ms (suffix only) |
| User-perceived latency | ~50 ms | ~50 ms |
| Filler quality | Preliminary answer from Qwen3-0.6B | ← same |
| Final quality | Full reasoning from Qwen3-Omni-30B-A3B-Thinking | ← same |

---

## Files

| File | Description |
|---|---|
| `__init__.py` | Package marker |
| `config.py` | Configuration dataclasses including `TokenizerConfig`; all params from env vars |
| `pipeline.py` | Core async dual-model orchestration with parallel tokenization and `/v1/completions` path |
| `server.py` | FastAPI proxy with SSE streaming, round-robin load balancing, and tokenizer warm-up |
| `client_example.py` | CLI client that prints filler + final responses |
| `README.md` | This file |
