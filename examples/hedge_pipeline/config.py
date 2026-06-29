# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
config.py — Configuration management for the hedge pipeline.

Small model  : Qwen3-0.6B  (~50 ms TTFT, quick filler answer)
Big model    : Qwen3-Omni-30B-A3B W8A8  (~950 ms TTFT, authoritative answer)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SmallModelConfig:
    """Configuration for the fast filler (small) model server."""

    # Base URL of the running vLLM server for the small model.
    base_url: str = os.getenv("SMALL_MODEL_BASE_URL", "http://localhost:8100")

    # Model name/path as registered in the vLLM server.
    model: str = os.getenv("SMALL_MODEL_NAME", "Qwen/Qwen3-0.6B")

    # Maximum tokens the filler answer may generate.
    max_tokens: int = int(os.getenv("SMALL_MODEL_MAX_TOKENS", "80"))

    # Sampling temperature — lower = more deterministic filler.
    temperature: float = float(os.getenv("SMALL_MODEL_TEMPERATURE", "0.3"))

    # Hard timeout (seconds) to wait for the small model to finish.
    # If exceeded, the big model is called with the original prompt only.
    timeout: float = float(os.getenv("SMALL_MODEL_TIMEOUT", "0.5"))

    # top_p for nucleus sampling.
    top_p: float = float(os.getenv("SMALL_MODEL_TOP_P", "0.9"))


@dataclass
class BigModelConfig:
    """Configuration for the authoritative (big) model server(s)."""

    # Comma-separated list of base URLs for big-model vLLM instances.
    # Multiple URLs enable round-robin load balancing across replicas.
    base_urls: list[str] = field(
        default_factory=lambda: [
            u.strip() for u in os.getenv("BIG_MODEL_BASE_URLS", "http://localhost:8200").split(",") if u.strip()
        ]
    )

    # Model name/path as registered in the vLLM server.
    model: str = os.getenv("BIG_MODEL_NAME", "Qwen/Qwen3-30B-A3B")

    # Maximum tokens the big model may generate.
    max_tokens: int = int(os.getenv("BIG_MODEL_MAX_TOKENS", "1024"))

    # Sampling temperature.
    temperature: float = float(os.getenv("BIG_MODEL_TEMPERATURE", "0.6"))

    # top_p for nucleus sampling.
    top_p: float = float(os.getenv("BIG_MODEL_TOP_P", "0.95"))

    # Hard timeout (seconds) to wait for the big model to complete.
    # 0 means no timeout.
    timeout: float = float(os.getenv("BIG_MODEL_TIMEOUT", "60.0"))


@dataclass
class PipelineConfig:
    """Top-level configuration for the hedge pipeline."""

    small: SmallModelConfig = field(default_factory=SmallModelConfig)
    big: BigModelConfig = field(default_factory=BigModelConfig)

    # Template used to inject the filler answer into the big-model prompt.
    # {original_prompt} and {filler_text} are substituted at runtime.
    augmented_prompt_template: str = (
        "{original_prompt}\n\n"
        "[Preliminary answer from a fast model: {filler_text}]\n\n"
        "Now provide the correct, complete answer:"
    )

    # When True, stream the filler answer token-by-token to the client as
    # it is produced; when False, buffer the full filler before forwarding.
    stream_filler: bool = bool(int(os.getenv("HEDGE_STREAM_FILLER", "1")))

    # Log level used by the pipeline and server modules.
    log_level: str = os.getenv("HEDGE_LOG_LEVEL", "INFO")

    def build_augmented_prompt(self, original_prompt: str, filler_text: str) -> str:
        """Return the big-model prompt that incorporates the filler answer."""
        return self.augmented_prompt_template.format(
            original_prompt=original_prompt,
            filler_text=filler_text.strip(),
        )

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Construct a PipelineConfig entirely from environment variables."""
        return cls(
            small=SmallModelConfig(),
            big=BigModelConfig(),
        )


# ---------------------------------------------------------------------------
# Convenience singleton — import and use directly, or override in tests.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    """Return (and lazily create) the default pipeline configuration."""
    global DEFAULT_CONFIG
    if DEFAULT_CONFIG is None:
        DEFAULT_CONFIG = PipelineConfig.from_env()
    return DEFAULT_CONFIG
