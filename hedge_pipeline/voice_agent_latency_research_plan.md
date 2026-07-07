# Deep Engineering Research: Low-Latency Voice Agent Optimization

## 1. End-to-End Latency Breakdown

### Latency Timeline (Typical Production Baseline)

```text
Timeline (ms):
0ms   ──────── VAD endpoint detection         ~20–80ms
80ms  ──────── ASR streaming (partial)        ~100–300ms (TTFT of ASR)
380ms ──────── Router decision                ~1–5ms
385ms ──────── Tokenization + prompt build    ~5–15ms
400ms ──────── KV cache lookup/creation       ~10–50ms
450ms ──────── Prefill (30B-A3B)              ~100–300ms (seq-len dependent)
750ms ──────── First token (LLM)              target: <200ms from ASR end
800ms ──────── First TTS chunk                ~50–150ms (CosyVoice TTFA)
950ms ──────── First audio to user            ~50–100ms (buffering)
────────────────────────────────────────────────────────
Total TTFA (Time-To-First-Audio): ~700–1200ms
Perceived latency target: <600ms
```

### Per-Stage Analysis

| Stage | Typical Latency | Primary Bottleneck | Optimization Opportunity |
|---|---|---|---|
| VAD | 20–80ms | Audio chunking, endpoint sensitivity | Parallel VAD+ASR start; reduce chunk size |
| ASR | 100–300ms | Model size, CPU/NPU bandwidth | Streaming CTC (emit tokens continuously); partial hypothesis passing |
| Router | 1–5ms | Sequential dependency | Predictive routing from partial ASR; zero-copy hand-off |
| Tokenization | 5–15ms | Single-threaded tokenizer | Pre-tokenize system prompt; background thread |
| Prompt build | 5–15ms | String concatenation, history retrieval | Prefix cache; pre-built prompt tensors |
| KV creation | 10–50ms | Memory allocation | Paged attention pre-allocation; KV prefix reuse |
| Prefill | 100–300ms | Memory bandwidth (MoE experts) | Chunked prefill; disaggregated prefill; overlap with small-model run |
| First token | 0–10ms | TTFT after prefill | Speculative decoding reduces this to near-zero |
| Generation | 50–500ms | Autoregressive decode | Speculative decoding; continuous batching |
| Detokenization | 1–5ms | Negligible | Stream tokens directly |
| TTS | 50–150ms | CosyVoice synthesis latency | Chunk at punctuation; stream vocoder; predictive prosody |
| Audio playback | 50–100ms | Buffering/jitter | Adaptive buffer; client-side VAD-aware playback |

## 2. Parallelization Opportunities

### Opportunity Map

```text
Time →
ASR:        [═══════════════════] emit partial tokens at t=50ms intervals
                    ↓ partial text
Router:             [═] predict from partial at t=100ms
                         ↓
Small 0.6B:              [═════] generate fast reply / plan
                    ↓ simultaneously
Prompt build:       [═══] construct prompt incrementally as ASR emits tokens
                              ↓
KV prefill:                   [═══════] begin prefill as tokens arrive (chunked)
                                        ↓
Large 30B:                              [════════] prefill + decode
                                              ↓ first sentence tokens
TTS:                                          [═══════] synthesize chunk 1
                                              ↓
Audio:                                              [══] playback
```

### Key Parallelization Ideas

1. **Streaming ASR → Incremental Prompt Build (Immediate)**
   - As ASR emits partial hypotheses every 50–100ms, begin constructing the LLM prompt.
   - Expected gain: 100–200ms overlap.
   - Engineering complexity: Medium.

2. **Small Model Pre-generation During Prefill (High Priority)**
   - While large model does prefill (100–300ms), small 0.6B generates first 10–20 tokens.
   - Use speculative decoding to accept/reject draft tokens.
   - Expected gain: 50–150ms TTFA reduction.
   - Engineering complexity: Medium-High.

3. **Async TTS Start on Sentence Fragments**
   - Start TTS at first clause boundary instead of waiting for full sentence.
   - Expected gain: 100–300ms perceived latency.
   - Engineering complexity: Low-Medium.

4. **Hedged / Race Execution**
   - Send request to both NPUs and keep first responder.
   - Expected gain: significant P99 reduction.
   - Engineering complexity: Low.

5. **Overlap Prefill and KV Cache Population**
   - Allocate/extend KV in background during prompt assembly.
   - Engineering complexity: High.

## 3. Small Model + Large Model Collaboration

### A. Speculative Decoding (Most Production-Ready)

```text
Small 0.6B (draft):  [t1][t2][t3][t4][t5]  → propose γ=5 tokens
Large 30B (verifier): [verify all 5 in one forward pass]
                       Accept: t1, t2, t3  (reject t4, t5)
```

- Candidate methods: EAGLE-2, Medusa, Hydra, Lookahead variants.
- Expected gain: 2–4x decode throughput; 30–60% latency reduction on longer responses.
- vLLM integration: use native speculative decoding path.

### B. Draft First Sentence

- Small model emits first sentence for immediate TTS while large model continues.
- Risk: semantic mismatch; mitigate with confidence thresholds and correction policy.

### C. Answer Planning / Outline Injection

- Small model emits compact plan appended to large-model prompt.
- Expected gain: ~20–50ms first-token improvement.

### D. Context Compression and Retrieval Query Drafting

- Small model compresses history and drafts retrieval queries while large model prefills.

## 4. Streaming Optimizations

- **Streaming ASR**: partial hypotheses with confidence gating.
- **Streaming LLM**: chunked prefill, incremental KV extension, immediate token stream.
- **Streaming TTS**: clause-level synthesis with low initial buffer.
- **Rolling context**: sliding window + periodic compression for old turns.

## 5. Prompt Engineering for Low Latency

- Prefix caching for stable system/instruction sections.
- Semantic cache for frequent intents.
- Prompt compression for conversation history.
- Dynamic prompt pruning by intent.

## 6. vLLM Optimizations

- Continuous batching and scheduler tuning for voice workloads.
- Chunked prefill and prefix caching.
- KV reuse and cache hygiene.
- Speculative decoding support and draft-model tuning.
- Graph capture where supported in Ascend path.

## 7. NPU Scheduling (2 NPUs)

- Adaptive routing with queue-depth and estimated prefill cost.
- Hedged requests for latency-critical turns.
- Prefill/decode role split (advanced).
- Work stealing and QoS priority queues.

## 8. ASR Optimizations

- Evaluate SenseVoice / Paraformer / Whisper-streaming variants by WER-latency frontier.
- Early endpoint detection.
- Incremental punctuation and confidence estimation.
- Optional speaker/language detection in parallel sidecar.

## 9. TTS Optimizations

- CosyVoice low-buffer streaming profile.
- Parallel chunk synthesis with playback overlap.
- Predictive prosody.
- Causal vocoder path for lower TTFA.

## 10. User Experience Optimizations

- Instant acknowledgement/backchannel audio.
- Interruptibility (barge-in) with hard cancellation.
- Progressive refinement responses.
- Adaptive speaking pace and pause strategy.

## 11. Speculative Execution

- Speculative prompting, retrieval, and tool prefetching from partial ASR.
- Parallel hypothesis handling (top-k partial transcripts).
- Conversation-state prediction for prewarming likely next turn.

## 12. State-of-the-Art Research Focus Areas

Track and map practical findings from major labs and companies on:

- speculative decoding,
- disaggregated serving,
- streaming multimodal inference,
- real-time speech interaction,
- low-latency TTS/ASR.

## 13. Open Source Projects to Mine

- vLLM, SGLang, TensorRT-LLM, MLC-LLM, llama.cpp, DeepSpeed, FlashInfer,
- Whisper.cpp, CosyVoice, Moonshine,
- Medusa, EAGLE, Lookahead, Hydra.

Extract reusable scheduler, caching, and streaming patterns.

## 14. Optimization Roadmap

### Short-term (2–3 weeks)

| Idea | Difficulty | Risk | Expected Latency Reduction | Priority |
|---|---|---|---|---|
| Prefix caching enable/tune | Low | Low | High | P0 |
| Async TTS start on clauses | Low | Low | High | P0 |
| Streaming ASR → prompt overlap | Medium | Medium | Medium-High | P0 |
| vLLM speculative decoding | Medium | Medium | High | P0 |
| Barge-in cancellation path | Medium | Medium | UX-critical | P1 |

### Long-term (1–2 months)

| Idea | Difficulty | Risk | Expected Latency Reduction | Priority |
|---|---|---|---|---|
| Disaggregated prefill/decode across NPUs | High | High | Very High | P0 |
| EAGLE/Hydra-style advanced speculation | High | Medium | Very High | P0 |
| Semantic streaming and context compression | Medium-High | Medium | High | P1 |
| Custom Ascend kernels for hot paths | Very High | High | High | P2 |

## 15. Prioritized ROI Table

| Idea | Difficulty | Time | Risk | Latency Gain | Throughput Gain | User Impact | References |
|---|---|---|---|---|---|---|---|
| Prefix caching | Low | Short | Low | High | High | High | vLLM docs |
| Async clause-level TTS | Low | Short | Low | High | Medium | Very High | CosyVoice |
| Speculative decoding | Medium | Short | Medium | High | High | High | EAGLE/Medusa/Hydra |
| Streaming ASR overlap | Medium | Short | Medium | Medium-High | Medium | High | ASR literature |
| Disaggregated serving | High | Medium | High | Very High | Very High | High | Splitwise/DistServe |

## 16. Challenge Existing Assumptions

- Two large models may be suboptimal versus role-specialized NPUs (prefill vs decode).
- Running draft models continuously can improve readiness and reduce cold-start penalties.
- ASR text may not be the only interface; semantic/audio-native representations may cut pipeline overhead.
- TTS should start before sentence completion when confidence/prosody permits.
- Predictive routing can outperform reactive routing under bursty traffic.

## Expected Aggregate Impact (Phased)

- **Phase 1 (2–3 weeks):** strong perceived-latency drop and meaningful TTFA reduction.
- **Phase 2 (1–2 months):** architecture-level gains with substantial P50/P99 improvements and higher hardware utilization.
