# Voice Agent — ASR → LLM Streaming Pipeline on Ascend NPU

A low-latency, chunked streaming Voice Agent that connects a real-time ASR
model to an LLM, targeting **TTFA (Time-To-First-Audio) < 200 ms** on Ascend
910B NPU.

```text
Audio → [VAD] → [ASR: WebSocket/Qwen-ASR or sherpa-onnx] → [LLM: Qwen3-0.6B via AsyncLLM] → Token stream
```

## Features

- **Flexible ASR Backend**: Supports both WebSocket-based ASR (e.g., qwen-asr) and local sherpa-onnx models
- **Parallel text + audio feeding**: User text is fed to the LLM immediately while ASR
  transcription of the audio portion runs concurrently, minimising first-token latency
- **KV-cache continuation**: vLLM's `AsyncLLM` retains the KV cache across turns under a
  stable `request_id`.  Only the delta (previous assistant response + new user input) is
  fed each turn — the full history is never re-transmitted to the engine
- **System prompt pre-warming**: system prompt tokens are prefilled on the very first turn
  before any ASR output is available
- **Multi-turn conversation memory**: per-session history managed by `SessionManager`
- **Concurrent sessions**: multiple users served by a single `AsyncLLM` via continuous batching
- **OpenAI-compatible REST endpoint**: `POST /v1/chat/completions` accepts OpenAI chat format
  with an optional `audio_b64` field for voice input
- **WebSocket endpoint**: low-latency real-time audio streaming for voice clients
- **Offline simulation**: feed a WAV file through the exact same pipeline for TTFA benchmarking
- **TTFA metrics**: automatic timing log per turn with per-session aggregates

---

## Prerequisites

- Ascend 910B NPU with CANN 8.x installed
- Python 3.10+
- vllm-ascend installed (provides `vllm`)

### ASR Backend Options

- **Option 1: WebSocket-based ASR (qwen-asr)**
    - A qwen-asr model running on port 8008 (or configure another port)
    - No additional model downloads required

- **Option 2: Local sherpa-onnx**
    - A [sherpa-onnx streaming Zipformer model](#sherpa-onnx-backend)

---

## Installation

```bash
cd examples/voice_agent
pip install -r requirements.txt
```

---

## ASR Configuration

### WebSocket Backend (qwen-asr)

Set in `config.yaml`:

```yaml
asr:
  backend: "websocket"
  websocket_host: "localhost"
  websocket_port: 8008
  sample_rate: 16000
  chunk_size_ms: 20
  endpoint_silence_ms: 200
```

Ensure the qwen-asr model is running on the configured port before starting the voice agent.

### sherpa-onnx Backend

Download a streaming Zipformer model from the
[sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) page.

Example (English, small):

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/\
sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
```

Then set in `config.yaml`:

```yaml
asr:
  backend: "sherpa-onnx"
  model: "zipformer"
  model_size: "small"
  model_dir: "/path/to/sherpa-onnx-streaming-zipformer-en-2023-06-26"
  sample_rate: 16000
  chunk_size_ms: 20
  endpoint_silence_ms: 200
```

Or export `SHERPA_ONNX_MODEL_DIR=/path/to/model`.

Expected directory layout:

```text
<model_dir>/
  encoder-epoch-99-avg-1.int8.onnx
  decoder-epoch-99-avg-1.int8.onnx
  joiner-epoch-99-avg-1.int8.onnx
  tokens.txt
```

---

## Configuration

All parameters are in `config.yaml`.  Key tuning levers:

| Parameter | Default | Effect |
|---|---|---|
| `asr.backend` | websocket | Backend type: `websocket` or `sherpa-onnx` |
| `asr.websocket_host` | localhost | Host for websocket ASR service (websocket backend only) |
| `asr.websocket_port` | 8008 | Port for websocket ASR service (websocket backend only) |
| `asr.endpoint_silence_ms` | 200 | Lower = faster TTFA but more false endpoints |
| `asr.model_size` | small | `tiny` / `small` / `large` trade accuracy vs latency (sherpa-onnx only) |
| `llm.max_tokens` | 512 | Caps response length |
| `llm.enforce_eager` | false | Set `true` to skip CANN graph capture (debug) |
| `conversation.max_history_turns` | 10 | Older turns evicted FIFO; affects prompt length |
| `simulation.playback_speed` | 1.0 | `2.0` = 2× real-time for fast benchmarking |

---

## Usage

### Server mode (real WebSocket / REST clients)

```bash
python main.py --config config.yaml
```

### REST API: `/v1/chat/completions`

Text-only request:

```bash
curl -X POST http://localhost:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voice-agent",
    "user": "session-alice",
    "messages": [{"role": "user", "content": "Hello, how can you help me?"}],
    "stream": false
  }'
```

Voice request (text + audio, streaming):

```bash
AUDIO_B64=$(base64 -w0 my_audio.pcm)
curl -X POST http://localhost:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"voice-agent\",
    \"user\": \"session-alice\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\",  \"text\": \"Can you help with my order?\"},
        {\"type\": \"audio\", \"audio_b64\": \"$AUDIO_B64\"}
      ]
    }],
    \"stream\": true
  }"
```

> **Note**: `audio_b64` must be base64-encoded raw 16-bit signed PCM, mono, 16 kHz.
> The text portion and audio portion are processed in parallel — the LLM begins
> generating while ASR is still transcribing.

Multi-turn conversations use the `user` field as a stable session identifier.
Reuse the same value across turns to benefit from KV-cache continuation.

### WebSocket endpoint

Connect to `ws://localhost:8765/ws/<session_id>` and send
raw 16-bit signed PCM audio at 16 kHz as binary frames.

### Offline simulation mode (WAV file)

```bash
python main.py --simulate --audio-file test_audio.wav
```

Or trigger via REST while the server is running:

```bash
curl -X POST http://localhost:8765/simulate \
  -H 'Content-Type: application/json' \
  -d '{"audio_file": "test_audio.wav", "session_id": "test-001"}'
```

### Health check

```bash
curl http://localhost:8765/health
```

---

## WebSocket Message Protocol

### Client → Server

| Frame type | Content | Meaning |
|---|---|---|
| Binary | Raw PCM bytes | Audio chunk |

### Server → Client

| JSON field `type` | Additional fields | Meaning |
|---|---|---|
| `asr_partial` | `text` | Partial ASR hypothesis (display only) |
| `token` | `text` | One LLM output token (mock TTS) |
| `turn_end` | — | LLM response complete |
| `error` | `message` | Pipeline error |

---

## Architecture: Parallel Text + Audio Feeding

```
POST /v1/chat/completions
        │
        ├─ Task A: _run_asr_to_queue(audio → asr_queue)   [non-blocking]
        │
        └─ Orchestrator._generate_tokens():
               prompt_generator():
                 yield system_prompt (turn 1) or last_assistant (turn N)
                 yield user_text                ← fed immediately
                 while True:
                   chunk = await asr_queue.get()  ← waits for Task A
                   if chunk is None: break
                   yield chunk
               engine.generate(inputs=prompt_generator(), request_id=stable_id)
                 ↓ prefilling starts immediately on system + user_text
                 ↓ continues as ASR chunks arrive
                 ↓ tokens streamed → SSE to client
```

The KV cache for each session lives inside vLLM under the stable
`llm_continuation_id`.  Turn N feeds only the delta:
`last_assistant_response + user_text + asr_transcript`.

---

## TTFA Budget (Ascend 910B)

| Stage | Expected latency |
|---|---|
| Audio buffering | ~20 ms |
| VAD decision | ~5 ms |
| ASR endpoint (small model) | 80–150 ms |
| LLM TTFT (Qwen3-0.6B, warm) | 30–80 ms |
| **Total (best case)** | **~135–255 ms** |

Set `asr.endpoint_silence_ms: 200` and `llm.prewarm_on_startup: true` for best
results.

---

## File Structure

```text
examples/voice_agent/
├── config.yaml                — all tunables
├── main.py                    — entry point (server or simulate)
├── server.py                  — FastAPI WebSocket + REST endpoints
├── session_manager.py         — per-session state + idle eviction
├── orchestrator.py            — prompt generator, LLM calls, KV-cache continuation
├── asr_worker.py              — sherpa-onnx Zipformer wrapper
├── asr_worker_websocket.py    — WebSocket ASR client for qwen-asr
├── audio_pipeline.py          — VAD + audio chunk routing
├── simulation.py              — WAV file → pipeline coroutine
├── metrics.py                 — TTFA timing dataclass
└── requirements.txt           — Python dependencies
```
