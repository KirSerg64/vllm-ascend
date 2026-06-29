# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
client_example.py — Python client that consumes the hedge pipeline SSE stream.

Usage::

    python examples/hedge_pipeline/client_example.py \
        --url http://localhost:8000 \
        --message "What is the capital of France?"

The client prints the filler answer as it streams in, then replaces/appends
the final authoritative answer from the big model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def stream_hedge(
    server_url: str,
    message: str,
    system: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> None:
    """
    Connect to the hedge pipeline server, print the filler and final answers.

    The filler tokens are streamed and printed with a ``[Filler]`` prefix;
    the final tokens follow with a ``[Final]`` prefix.
    """
    url = f"{server_url.rstrip('/')}/v1/hedge/chat/completions"
    payload: dict = {
        "messages": [{"role": "user", "content": message}],
    }
    if system:
        payload["system"] = system
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    filler_buf: list[str] = []
    final_buf: list[str] = []
    t_start = time.monotonic()
    t_filler_first: float | None = None
    t_final_first: float | None = None

    print(f"\n{'=' * 60}")
    print(f"Question: {message}")
    print(f"{'=' * 60}")

    with httpx.stream("POST", url, json=payload, timeout=120) as response:
        response.raise_for_status()

        current_phase: str | None = None
        event_type: str | None = None

        for line in response.iter_lines():
            line = line.strip()
            if not line:
                # Blank line → dispatch the buffered event.
                event_type = None
                continue

            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
                continue

            if line.startswith("data:"):
                raw = line[len("data:") :].strip()
                if raw == "{}" or not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                phase = data.get("phase")
                text = data.get("text", "")
                done = data.get("done", False)

                if event_type == "done":
                    break

                if event_type == "error":
                    print(f"\n[ERROR] {data.get('message', 'unknown error')}", file=sys.stderr)
                    return

                # ── Filler phase ──────────────────────────────────────────
                if phase == "filler":
                    if current_phase != "filler":
                        current_phase = "filler"
                        print("\n[Filler] ", end="", flush=True)
                    if text:
                        if t_filler_first is None:
                            t_filler_first = time.monotonic() - t_start
                        filler_buf.append(text)
                        print(text, end="", flush=True)
                    if done:
                        elapsed = time.monotonic() - t_start
                        print(f"\n[Filler done — first token: {t_filler_first:.3f}s, full filler: {elapsed:.3f}s]")

                # ── Final phase ───────────────────────────────────────────
                elif phase == "final":
                    if current_phase != "final":
                        current_phase = "final"
                        print("\n[Final]  ", end="", flush=True)
                    if text:
                        if t_final_first is None:
                            t_final_first = time.monotonic() - t_start
                        final_buf.append(text)
                        print(text, end="", flush=True)
                    if done:
                        elapsed = time.monotonic() - t_start
                        print(f"\n[Final done — first token: {t_final_first:.3f}s, total: {elapsed:.3f}s]")

    print(f"\n{'=' * 60}")
    print("[Summary]")
    print(f"  Filler first token : {t_filler_first:.3f}s" if t_filler_first else "  Filler : (none)")
    print(f"  Final first token  : {t_final_first:.3f}s" if t_final_first else "  Final  : (none)")
    print(f"  Total elapsed      : {time.monotonic() - t_start:.3f}s")
    print(f"{'=' * 60}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Hedge Pipeline Client Example")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Hedge server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--message",
        "-m",
        default="What is the capital of France and why is it historically significant?",
        help="User message to send",
    )
    parser.add_argument("--system", "-s", default=None, help="Optional system prompt")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens override")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature override")
    args = parser.parse_args(argv)

    stream_hedge(
        server_url=args.url,
        message=args.message,
        system=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
