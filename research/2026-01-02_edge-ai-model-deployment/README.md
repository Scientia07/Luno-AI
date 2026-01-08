# Edge AI and Model Deployment Research (2025)

> **Date**: 2026-01-02
> **Topic**: Edge Inference, Quantization, Optimization, and Cloud AI Platforms
> **Status**: Complete

---

## Overview

This research session covers the complete landscape of AI model deployment in 2025, from edge inference frameworks to cloud AI platforms and GPU providers. The guide provides structured recommendations for different use cases.

---

## Key Findings Summary

| Category | Top Recommendations |
|----------|---------------------|
| **NVIDIA GPU Inference** | TensorRT or TensorRT-LLM |
| **Cross-Platform Inference** | ONNX Runtime with execution providers |
| **Mobile/Embedded** | TFLite (LiteRT), CoreML, ARM NN |
| **LLM Quantization (GPU)** | AWQ or GPTQ |
| **LLM Quantization (CPU)** | GGUF with K-quants |
| **Cloud ML Platform** | Based on data location (see below) |
| **Budget GPU Cloud** | Vast.ai or RunPod |
| **Production GPU Cloud** | Lambda Labs or CoreWeave |

---

## Contents

1. [findings.md](./findings.md) - Detailed technical analysis
2. [sources.md](./sources.md) - All references and links
3. [use-cases.md](./use-cases.md) - Scenario-based recommendations

---

## Quick Decision Matrix

### Choose Your Edge Inference Framework

```
Do you have NVIDIA hardware?
├── Yes → TensorRT (best performance)
│         └── Or ONNX Runtime + TensorRT EP (flexibility + performance)
└── No
    ├── Intel hardware → OpenVINO
    ├── Apple Silicon → CoreML + MLX
    ├── Mobile/Embedded → TFLite (LiteRT)
    └── Multi-platform → ONNX Runtime
```

### Choose Your Quantization Method

```
What's your primary hardware?
├── GPU with ample VRAM
│   ├── Need best quality → AWQ (95% quality retention)
│   └── Maximum speed → GPTQ + Marlin kernels
├── CPU or Apple Silicon
│   └── GGUF with K-quants (Q4_K_M → Q5_K_M → Q8_0)
└── Hugging Face Integration needed
    └── bitsandbytes (NF4/FP4)
```

### Choose Your Cloud Platform

```
Where is your data?
├── S3/AWS ecosystem → SageMaker
├── BigQuery/GCP → Vertex AI
├── Azure Data Lake → Azure ML
└── Need cheapest option → GPU cloud providers
```

---

## Related Sessions

- [2026-01-01: AI Technology Stack Overview](../2026-01-01_ai-technology-stack-overview/)

---

*Research conducted using web search and 2025 documentation.*
