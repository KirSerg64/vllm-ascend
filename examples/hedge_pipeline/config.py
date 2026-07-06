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
class TokenizerConfig:
    """
    Configuration for the proxy-side HuggingFace tokenizer used to
    pre-tokenize prompts in parallel with the small model call.

    When ``use_local_tokenizer=True`` (the default), the pipeline loads the
    big model's tokenizer once at startup.  At request time it tokenizes the
    original user prompt concurrently with the small (filler) model call so
    that the full tokenization cost is off the critical path.  After the
    filler completes, only the short filler-hint suffix (~80 tokens) needs
    to be tokenized.  The concatenated token-ID list is then sent directly
    to the big model via ``POST /v1/completions``, which bypasses vLLM's
    internal tokenization step and lets prefill start immediately.

    When ``use_local_tokenizer=False``, the pipeline falls back to the
    original ``/v1/chat/completions`` text path (no parallel tokenization).
    """

    # Tokenizer model name or local path.  Must match the big model to ensure
    # token IDs are valid.  Defaults to the same value as BIG_MODEL_NAME.
    model_name_or_path: str = os.getenv(
        "TOKENIZER_MODEL_NAME",
        os.getenv("BIG_MODEL_NAME", "Qwen/Qwen3-30B-A3B"),
    )

    # Enable parallel pre-tokenization and /v1/completions path.
    # Set to 0 to disable and use the original /v1/chat/completions path.
    use_local_tokenizer: bool = bool(int(os.getenv("HEDGE_USE_LOCAL_TOKENIZER", "1")))

    # Number of threads reserved for tokenization via run_in_executor.
    # 2 is enough since tokenization is fast and rarely concurrent per process.
    max_workers: int = int(os.getenv("TOKENIZER_MAX_WORKERS", "2"))

    # The string that closes each chat turn in the model's chat template.
    # For Qwen3 / ChatML this is "\n<|im_end|>\n".
    # Used to strip the user-turn close from the pre-tokenized base so the
    # filler hint can be appended inside the user turn before re-closing it.
    turn_end_str: str = os.getenv("TOKENIZER_TURN_END_STR", "\n<|im_end|>\n")

    # The generation-prompt string inserted after the last user turn.
    # For Qwen3 / ChatML this is "<|im_start|>assistant\n".
    generation_prompt_str: str = os.getenv("TOKENIZER_GEN_PROMPT_STR", "<|im_start|>assistant\n")


@dataclass
class PipelineConfig:
    """Top-level configuration for the hedge pipeline."""

    small: SmallModelConfig = field(default_factory=SmallModelConfig)
    big: BigModelConfig = field(default_factory=BigModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    # Filler-hint suffix template.  ``{filler_text}`` is substituted at
    # runtime and this string is inserted *inside* the user turn after the
    # original user message, before the generation prompt.
    filler_hint_template: str = (
        "\n\n[Preliminary answer from a fast model: {filler_text}]\n\nNow provide the correct, complete answer:"
    )

    # When True, stream the filler answer token-by-token to the client as
    # it is produced; when False, buffer the full filler before forwarding.
    stream_filler: bool = bool(int(os.getenv("HEDGE_STREAM_FILLER", "1")))

    # Log level used by the pipeline and server modules.
    log_level: str = os.getenv("HEDGE_LOG_LEVEL", "INFO")

    def build_augmented_prompt(self, original_prompt: str, filler_text: str) -> str:
        """
        Return the full augmented user-message string for the text-based
        ``/v1/chat/completions`` path (used when ``use_local_tokenizer=False``
        or as a fallback).
        """
        return original_prompt + self.build_augmented_suffix(filler_text)

    def build_augmented_suffix(self, filler_text: str) -> str:
        """
        Return only the filler-hint suffix that is appended to the original
        user message *inside* the user turn.  Does not include the turn-close
        or generation-prompt tokens — those are added by
        ``_tokenize_filler_suffix`` when building the token-ID list.
        """
        return self.filler_hint_template.format(filler_text=filler_text.strip())

    @classmethod
    def from_env(cls) -> PipelineConfig:
        """Construct a PipelineConfig entirely from environment variables."""
        return cls(
            small=SmallModelConfig(),
            big=BigModelConfig(),
            tokenizer=TokenizerConfig(),
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
