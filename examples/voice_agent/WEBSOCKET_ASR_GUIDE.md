# WebSocket ASR Integration Guide

This guide explains how to use the websocket-based ASR backend to connect to a qwen-asr model running on port 8008.

## Overview

The voice agent now supports two ASR backends:

1. **sherpa-onnx** (local model) - Original implementation
2. **websocket** (remote service) - New implementation for qwen-asr

## Configuration

### Enable WebSocket Backend

Edit `config.yaml`:

```yaml
asr:
  backend: "websocket"
  websocket_host: "localhost"
  websocket_port: 8008
  sample_rate: 16000
  chunk_size_ms: 20
  endpoint_silence_ms: 200
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | sherpa-onnx | Backend type: `websocket` or `sherpa-onnx` |
| `websocket_host` | localhost | Host address of the ASR service |
| `websocket_port` | 8008 | Port number of the ASR service |
| `sample_rate` | 16000 | Audio sample rate (Hz) |
| `chunk_size_ms` | 20 | Audio chunk duration (milliseconds) |
| `endpoint_silence_ms` | 200 | Silence duration before endpoint detection (milliseconds) |

## WebSocket Protocol

The ASR worker communicates with the qwen-asr service using the following protocol:

### Client → Server

#### Audio Data

- **Type**: Binary frames
- **Format**: 16-bit signed integer PCM audio
- **Sample Rate**: 16 kHz (configurable)
- **Channels**: Mono

#### Control Messages

```json
{
  "type": "reset"
}
```

Sent to reset the decoder state after an endpoint.

### Server → Client

The ASR service should respond with JSON messages:

#### Partial Transcription

```json
{
  "type": "partial",
  "text": "partial transcription text"
}
```

Sent for intermediate recognition results.

#### Final Transcription / Endpoint

```json
{
  "type": "final",
  "text": "final transcription text"
}
```

or

```json
{
  "type": "endpoint",
  "text": "final transcription text"
}
```

Sent when a phrase endpoint is detected.

#### Error

```json
{
  "type": "error",
  "message": "error description"
}
```

Sent when an error occurs.

## Running the ASR Service

### Prerequisites

Ensure the qwen-asr model is running and accessible:

```bash
# The service should be listening on the configured port (default: 8008)
# Example: ws://localhost:8008/ws
```

### Start the Voice Agent

```bash
cd examples/voice_agent
python main.py --config config.yaml
```

The voice agent will:

1. Connect to the ASR service at `ws://{websocket_host}:{websocket_port}/ws`
2. Create a new websocket connection for each session
3. Stream audio chunks to the ASR service
4. Receive transcription results in real-time

### Simulation Mode

Test with a WAV file:

```bash
python main.py --simulate --audio-file test_audio.wav
```

## Implementation Details

### ASRWorkerWebSocket

The `asr_worker_websocket.py` module provides:

- **ASRWorkerWebSocket**: Main worker class that manages connections
- **ASRStream**: Per-session websocket connection wrapper

Key features:

- Async websocket operations (non-blocking)
- Automatic connection management
- Error handling and recovery
- Compatible interface with existing `ASRWorker`

### Audio Pipeline Integration

The `audio_pipeline.py` automatically detects and handles both sync and async ASR workers:

```python
# Handles both sherpa-onnx (sync) and websocket (async) backends
if asyncio.iscoroutinefunction(self._asr.create_stream):
    self._asr_stream = await self._asr.create_stream()
else:
    self._asr_stream = self._asr.create_stream()
```

## Troubleshooting

### Connection Errors

If you see:

```text
Failed to connect to ASR service at ws://localhost:8008/ws
```

**Solutions**:

1. Verify the qwen-asr service is running
2. Check the host and port in `config.yaml`
3. Ensure no firewall is blocking the connection

### No Transcription Results

If audio is being sent but no results are received:

1. Check ASR service logs for errors
2. Verify the audio format matches expectations (16-bit PCM, 16 kHz, mono)
3. Increase logging level to debug:

```python
logging.getLogger("asr_worker_websocket").setLevel(logging.DEBUG)
```

### Endpoint Detection Issues

If endpoints are detected too early or too late:

1. Adjust `endpoint_silence_ms` in `config.yaml`
2. Ensure the ASR service is sending proper endpoint signals

## Performance Considerations

### Latency

- **Network latency**: Add 5-20ms for websocket communication
- **ASR processing**: Depends on qwen-asr model and hardware
- **Total TTFA**: Network + ASR + LLM latency

### Throughput

- Each session maintains its own websocket connection
- The ASR service must support concurrent connections
- Consider connection pooling for high-load scenarios

## Migration from sherpa-onnx

To switch from sherpa-onnx to websocket backend:

1. Start the qwen-asr service on port 8008
2. Update `config.yaml`:

```yaml
asr:
  backend: "websocket"  # Changed from "sherpa-onnx"
  websocket_host: "localhost"
  websocket_port: 8008
```

3. Remove sherpa-onnx dependencies (optional):

```bash
pip uninstall sherpa-onnx
```

All other components remain unchanged.

## Example: Custom ASR Service

If you're implementing your own ASR service, ensure it:

1. Accepts websocket connections at `/ws`
2. Receives binary audio frames (16-bit PCM)
3. Sends JSON responses with `type` and `text` fields
4. Handles `{"type": "reset"}` control messages
5. Supports concurrent connections (one per session)

Minimal example structure:

```python
import asyncio
import websockets

async def handle_asr(websocket, path):
    async for message in websocket:
        if isinstance(message, bytes):
            # Process audio chunk
            text = await transcribe(message)
            await websocket.send(json.dumps({
                "type": "partial",
                "text": text
            }))
        elif isinstance(message, str):
            msg = json.loads(message)
            if msg.get("type") == "reset":
                # Reset decoder state
                reset_decoder()

start_server = websockets.serve(handle_asr, "localhost", 8008)
asyncio.get_event_loop().run_until_complete(start_server)
```

## References

- [qwen-asr Repository](https://github.com/QwenLM/Qwen2-Audio)
- [Voice Agent README](README.md)
- [WebSocket Protocol Specification](https://datatracker.ietf.org/doc/html/rfc6455)
