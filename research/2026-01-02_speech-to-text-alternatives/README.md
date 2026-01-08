# Speech-to-Text (STT/ASR) Alternatives to Whisper - 2025

> **Research Date**: 2026-01-02
> **Topic**: Commercial and open-source speech recognition alternatives
> **Status**: Complete

---

## Executive Summary

The speech-to-text landscape in 2025 has evolved significantly beyond OpenAI's Whisper. Commercial APIs now offer sub-300ms latency with 5-15% WER, while open-source models like NVIDIA Canary-Qwen-2.5B have matched or exceeded commercial accuracy (5.63% WER). Key trends include real-time streaming capabilities, multilingual support, and hybrid ASR-LLM architectures.

---

## Quick Reference

| Provider | Model | WER | Latency | Pricing | Best For |
|----------|-------|-----|---------|---------|----------|
| Deepgram | Nova-3 | ~8-18% | <300ms | $0.0077/min | Real-time, streaming |
| AssemblyAI | Universal-2 | 14.5% (streaming) | 300ms | $0.0025/min | Production reliability |
| Google | Chirp 3 | 11.6% (batch) | High | $0.016/min | Multilingual batch |
| AWS | Transcribe | ~15-20% | Variable | $0.024/min | AWS ecosystem |
| Azure | Speech | ~12-18% | Variable | $0.017/min | Enterprise, 100+ langs |
| NVIDIA | Canary-Qwen-2.5B | 5.63% | Fast | Open source | Best accuracy |
| OpenAI | Whisper Large v3 | 7.88% | Variable | $0.006/min | Multilingual |

---

## Key Findings

### 1. Commercial APIs Lead in Production Reliability

- **Deepgram Nova-3**: 54% WER reduction over competitors, real-time multilingual
- **AssemblyAI Universal-2**: 99.95% uptime SLA, immutable streaming transcripts
- **Google Chirp 3**: Best batch accuracy at 11.6% WER, 100+ languages

### 2. Open Source Has Caught Up

- **NVIDIA Canary-Qwen-2.5B**: Tops Hugging Face leaderboard (5.63% WER)
- **Whisper Large v3 Turbo**: 6x faster than v3, minimal accuracy loss
- **Vosk**: Best for offline/edge deployment (50MB models)

### 3. Real-Time Streaming is Mature

- Sub-300ms latency now standard for commercial APIs
- AssemblyAI leads streaming accuracy (14.5% WER)
- NVIDIA Parakeet TDT achieves 2000+ RTFx for self-hosted streaming

---

## Detailed Analysis

See companion files:
- [sources.md](./sources.md) - All reference URLs
- [findings.md](./findings.md) - Detailed technical analysis
