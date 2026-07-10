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
"""Audio file simulation: replaces a real microphone for testing.

:func:`simulate_session` reads a WAV file, resamples it to 16 kHz mono
if necessary, and feeds 20 ms PCM chunks into the pipeline at real-time
speed — exactly as a WebSocket client would.  This allows end-to-end
TTFA measurement without hardware.

Usage
-----
From the CLI (via ``main.py --simulate``) or directly::

    asyncio.run(simulate_session(
        wav_path="test_audio.wav",
        push_audio_fn=pipeline.push_audio,
        flush_fn=pipeline.flush,
        chunk_ms=20,
        playback_speed=1.0,
    ))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Coroutine

import numpy as np

logger = logging.getLogger(__name__)


async def simulate_session(
    wav_path: str,
    push_audio_fn: Callable[[bytes], Coroutine],
    flush_fn: Callable[[], Coroutine],
    chunk_ms: int = 20,
    sample_rate: int = 16000,
    playback_speed: float = 1.0,
) -> None:
    """Stream a WAV file through the audio pipeline at simulated real-time pace.

    Parameters
    ----------
    wav_path:
        Path to the source WAV file (any sample rate, mono or stereo).
    push_audio_fn:
        Coroutine that accepts raw 16-bit PCM bytes for one chunk.
    flush_fn:
        Coroutine called after the last chunk to flush buffered audio.
    chunk_ms:
        Duration of each audio chunk in milliseconds.
    sample_rate:
        Target sample rate expected by the ASR model (default 16 kHz).
    playback_speed:
        Speed multiplier.  1.0 = real-time; 2.0 = twice as fast.
    """
    try:
        import soundfile as sf  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "soundfile is not installed. Run: pip install soundfile"
        ) from exc

    logger.info("Loading audio file: %s", wav_path)
    data, file_sr = sf.read(wav_path, dtype="float32", always_2d=False)

    # Convert to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if file_sr != sample_rate:
        try:
            from scipy.signal import resample_poly  # type: ignore[import]
            import math

            gcd = math.gcd(sample_rate, file_sr)
            up = sample_rate // gcd
            down = file_sr // gcd
            data = resample_poly(data, up, down).astype(np.float32)
            logger.info(
                "Resampled audio: %d Hz → %d Hz", file_sr, sample_rate
            )
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required for audio resampling. "
                "Run: pip install scipy"
            ) from exc

    # Convert float32 [-1, 1] to int16 PCM
    pcm_int16 = (data * 32767.0).clip(-32768, 32767).astype(np.int16)

    # Chunk parameters
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2  # 16-bit = 2 bytes per sample
    sleep_seconds = (chunk_ms / 1000.0) / playback_speed

    total_samples = len(pcm_int16)
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    logger.info(
        "Simulating %.2f s of audio (%d chunks × %d ms, speed=%.1fx)",
        total_samples / sample_rate,
        n_chunks,
        chunk_ms,
        playback_speed,
    )

    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, total_samples)
        chunk = pcm_int16[start:end].tobytes()

        # Pad final chunk to full size
        if len(chunk) < chunk_bytes:
            chunk = chunk + b"\x00" * (chunk_bytes - len(chunk))

        await push_audio_fn(chunk)
        await asyncio.sleep(sleep_seconds)

    await flush_fn()
    logger.info("Simulation complete.")
