# Voice Agent — ASR → LLM Streaming Pipeline on Ascend NPU

A low-latency, chunked streaming Voice Agent that connects a real-time ASR
model to an LLM, targeting **TTFA (Time-To-First-Audio) < 200 ms** on Ascend
910B NPU.

```
Audio → [VAD] → [ASR: Zipformer/sherpa-onnx] → [LLM: Qwen3-0.6B via AsyncLLMEngine] → Token stream (mock TTS)
```

## Features

- **Chunked streaming**: ASR streams partial hypotheses in real time; LLM starts
  as soon as a phrase endpoint is detected
- **Barge-in**: Configurable soft-cancel of LLM generation when user starts
  speaking again
- **Multi-turn conversation memory**: Per-session history with a configurable
  window kept in-process
- **Concurrent sessions**: Multiple users served by a single
  `AsyncLLMEngine` via continuous batching
- **Offline simulation**: Feed a WAV file through the exact same pipeline for
  TTFA benchmarking without hardware microphones
- **TTFA metrics**: Automatic timing log per turn with per-session aggregates

---

## Prerequisites

- Ascend 910B NPU with CANN 8.x installed
- Python 3.10+
- vllm-ascend installed (provides `vllm`)
- A [sherpa-onnx streaming Zipformer model](#asr-model-download)

---

## Installation

```bash
cd examples/voice_agent
pip install -r requirements.txt
```

---

## ASR Model Download

Download a streaming Zipformer model from the
[sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases) page.

Example (English, small):

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/\
sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
```

Then set `asr.model_dir` in `config.yaml` (or export
`SHERPA_ONNX_MODEL_DIR=/path/to/model`).

Expected directory layout:

```
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
| `asr.endpoint_silence_ms` | 200 | Lower = faster TTFA but more false endpoints |
| `asr.model_size` | small | `tiny` / `small` / `large` trade accuracy vs latency |
| `llm.max_tokens` | 512 | Caps response length |
| `llm.enforce_eager` | false | Set `true` to skip CANN graph capture (debug) |
| `conversation.max_history_turns` | 10 | Older turns evicted FIFO; affects prompt length |
| `server.enable_barge_in` | true | Cancel LLM on new user speech |
| `simulation.playback_speed` | 1.0 | `2.0` = 2× real-time for fast benchmarking |

---

## Usage

### Server mode (real WebSocket clients)

```bash
python main.py --config config.yaml
```

Connect a WebSocket client to `ws://localhost:8765/ws/<session_id>` and send
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

```
examples/voice_agent/
├── config.yaml         — all tunables
├── main.py             — entry point (server or simulate)
├── server.py           — FastAPI WebSocket + REST endpoints
├── session_manager.py  — per-session state + idle eviction
├── orchestrator.py     — commit logic, LLM calls, history
├── asr_worker.py       — sherpa-onnx Zipformer wrapper
├── audio_pipeline.py   — VAD + audio chunk routing
├── simulation.py       — WAV file → pipeline coroutine
├── metrics.py          — TTFA timing dataclass
└── requirements.txt    — Python dependencies
```
