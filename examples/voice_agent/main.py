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
"""Entry point for the Voice Agent server.

Usage
-----
Start the server::

    python main.py --config config.yaml

Start in offline simulation mode (no WebSocket client required)::

    python main.py --config config.yaml --simulate

Start simulation with a custom audio file::

    python main.py --config config.yaml --simulate --audio-file path/to/audio.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _build_engine(config: dict[str, Any]) -> Any:
    """Construct and return a vllm.AsyncLLMEngine."""
    # isort: skip_file
    from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore[import]

    llm_cfg = config.get("llm", {})
    engine_args = AsyncEngineArgs(
        model=llm_cfg.get("model", "Qwen/Qwen3-0.6B-W8A8"),
        tensor_parallel_size=int(llm_cfg.get("tensor_parallel_size", 1)),
        max_num_seqs=int(llm_cfg.get("max_num_seqs", 32)),
        enforce_eager=bool(llm_cfg.get("enforce_eager", False)),
        disable_log_requests=True,
    )
    logger.info("Initialising AsyncLLMEngine: %s", engine_args.model)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


async def _prewarm_engine(engine: Any, system_prompt: str) -> None:
    """Submit a short dummy request to warm up the NPU graph."""
    from vllm import SamplingParams  # type: ignore[import]

    logger.info("Pre-warming LLM engine…")
    params = SamplingParams(max_tokens=1, temperature=0.0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "hi"},
    ]
    async for _ in engine.generate(
        prompt=messages,
        sampling_params=params,
        request_id="prewarm-0001",
    ):
        pass
    logger.info("LLM engine pre-warm complete.")


async def _run_server(config: dict[str, Any], engine: Any) -> None:
    import uvicorn  # type: ignore[import]

    from server import create_app
    from session_manager import SessionManager

    # Load ASR model based on backend configuration
    asr_cfg = config.get("asr", {})
    backend = asr_cfg.get("backend", "sherpa-onnx")

    if backend == "websocket":
        from asr_worker_websocket import ASRWorkerWebSocket

        asr_worker = ASRWorkerWebSocket(asr_cfg)
    else:
        from asr_worker import ASRWorker

        asr_worker = ASRWorker(asr_cfg)

    asr_worker.load()

    session_manager = SessionManager(
        session_timeout_minutes=int(config.get("server", {}).get("session_timeout_minutes", 30))
    )

    app = create_app(
        engine=engine,
        session_manager=session_manager,
        asr_worker=asr_worker,
        config=config,
    )

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = int(server_cfg.get("port", 8765))

    logger.info("Starting Voice Agent server on %s:%d", host, port)

    uvicorn_config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(uvicorn_config)
    await server.serve()


async def _run_simulate(config: dict[str, Any], engine: Any, audio_file: str) -> None:
    """Run a single offline simulation session without starting a web server."""
    from audio_pipeline import AudioPipeline
    from orchestrator import Orchestrator
    from server import _log_token_stream
    from session_manager import SessionManager
    from simulation import simulate_session

    asr_cfg = config.get("asr", {})
    vad_cfg = config.get("vad", {})
    llm_cfg = config.get("llm", {})
    conv_cfg = config.get("conversation", {})
    sim_cfg = config.get("simulation", {})
    server_cfg = config.get("server", {})

    session_id = sim_cfg.get("session_id", "sim-session-001")
    playback_speed = float(sim_cfg.get("playback_speed", 1.0))
    chunk_ms = int(asr_cfg.get("chunk_size_ms", 20))
    sample_rate = int(asr_cfg.get("sample_rate", 16000))
    enable_barge_in = bool(server_cfg.get("enable_barge_in", True))

    # Load ASR based on backend configuration
    backend = asr_cfg.get("backend", "sherpa-onnx")

    if backend == "websocket":
        from asr_worker_websocket import ASRWorkerWebSocket

        asr_worker = ASRWorkerWebSocket(asr_cfg)
    else:
        from asr_worker import ASRWorker

        asr_worker = ASRWorker(asr_cfg)

    asr_worker.load()

    session_manager = SessionManager()
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

    print(f"\n{'=' * 60}")
    print("  Voice Agent — Simulation Mode")
    print(f"  Audio: {audio_file}")
    print(f"  Session: {session_id}")
    print(f"{'=' * 60}\n")

    orch_task = asyncio.create_task(orch.run())
    log_task = asyncio.create_task(_log_token_stream(ws_send_queue, session_id))

    await simulate_session(
        wav_path=audio_file,
        push_audio_fn=pipeline.push_audio,
        flush_fn=pipeline.flush,
        chunk_ms=chunk_ms,
        sample_rate=sample_rate,
        playback_speed=playback_speed,
    )

    await pipeline.flush()
    pipeline.shutdown()
    await orch.shutdown()
    log_task.cancel()

    try:
        await asyncio.wait_for(orch_task, timeout=30.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass

    state.metrics.log_summary()
    await session_manager.remove(session_id)
    asr_worker.shutdown()


async def _async_main(args: argparse.Namespace) -> None:
    config = _load_config(args.config)

    # Override audio file from CLI if provided
    if args.audio_file:
        config.setdefault("simulation", {})["audio_file"] = args.audio_file

    engine = _build_engine(config)

    # Pre-warm engine if configured
    if config.get("llm", {}).get("prewarm_on_startup", True):
        system_prompt = config.get("conversation", {}).get("system_prompt", "You are a helpful assistant.")
        await _prewarm_engine(engine, system_prompt)

    if args.simulate:
        audio_file = args.audio_file or config.get("simulation", {}).get("audio_file", "test_audio.wav")
        if not Path(audio_file).exists():
            logger.error("Audio file not found: %s", audio_file)
            sys.exit(1)
        await _run_simulate(config, engine, audio_file)
    else:
        await _run_server(config, engine)


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Voice Agent: ASR → LLM streaming pipeline on Ascend NPU")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: config.yaml in script directory)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in offline simulation mode from a WAV file",
    )
    parser.add_argument(
        "--audio-file",
        default=None,
        help="Path to WAV file for simulation (overrides config.yaml)",
    )
    args = parser.parse_args()

    # Ensure the script's directory is on sys.path so local modules resolve
    sys.path.insert(0, str(Path(__file__).parent))

    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
