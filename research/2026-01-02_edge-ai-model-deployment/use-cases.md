# Use Case Recommendations

This guide provides specific recommendations based on common deployment scenarios.

---

## Scenario 1: Deploy LLM on Consumer Hardware

**Goal**: Run a 7B-70B parameter LLM locally on consumer GPU/CPU

### Recommended Stack

| Component | Recommendation | Reason |
|-----------|----------------|--------|
| **Quantization** | GGUF Q4_K_M or Q5_K_M | Best CPU/hybrid support |
| **Runtime** | llama.cpp or Ollama | Optimized for GGUF |
| **If GPU-only** | AWQ + vLLM | Maximum GPU throughput |

### Hardware Requirements by Model Size

| Model | Q4_K_M Size | Minimum GPU | CPU RAM |
|-------|-------------|-------------|---------|
| 7B | ~4GB | 6GB VRAM | 16GB |
| 13B | ~8GB | 10GB VRAM | 24GB |
| 34B | ~20GB | 24GB VRAM | 48GB |
| 70B | ~40GB | 48GB VRAM | 96GB |

### Quality vs Speed Decision

```
Priority: Quality → Use Q5_K_M or Q8_0
Priority: Speed → Use Q4_K_M
Priority: Minimal RAM → Use Q3_K_S (quality trade-off)
```

---

## Scenario 2: Mobile App with On-Device AI

**Goal**: Deploy computer vision or small language model on iOS/Android

### iOS Deployment

| Component | Recommendation | Notes |
|-----------|----------------|-------|
| **Framework** | CoreML | Native Apple integration |
| **Conversion** | coremltools | From PyTorch/TF |
| **For LLMs** | MLX + GGUF | Apple Silicon optimized |

### Android Deployment

| Component | Recommendation | Notes |
|-----------|----------------|-------|
| **Framework** | TFLite (LiteRT) | Smallest binary |
| **Alternative** | ONNX Runtime Mobile | More flexibility |
| **Acceleration** | NNAPI delegate | Uses NPU |

### Cross-Platform

Use ONNX Runtime with platform-specific execution providers:
- iOS: CoreML EP
- Android: NNAPI EP

---

## Scenario 3: Real-Time Computer Vision at Edge

**Goal**: Object detection/segmentation with <50ms latency

### NVIDIA Edge (Jetson)

| Component | Recommendation | Performance |
|-----------|----------------|-------------|
| **Model** | YOLOv8/v11 | Best speed/accuracy |
| **Framework** | TensorRT | 5x faster than PyTorch |
| **Precision** | FP16 or INT8 | With calibration |

### Intel Edge (NUC, Core)

| Component | Recommendation | Performance |
|-----------|----------------|-------------|
| **Framework** | OpenVINO | 3x CPU speedup |
| **Precision** | INT8 | With NNCF |

### ARM Edge (Raspberry Pi, Coral)

| Component | Recommendation | Notes |
|-----------|----------------|-------|
| **Framework** | TFLite | Best ARM support |
| **Acceleration** | Edge TPU | If using Coral |

---

## Scenario 4: Production LLM API Service

**Goal**: Serve LLM with high throughput and low latency

### Cloud Deployment

| Scale | Recommendation | Cost |
|-------|----------------|------|
| **Startup** | Modal + AWQ model | Pay-per-use |
| **Medium** | RunPod Serverless | Low hourly |
| **Enterprise** | Lambda Labs + vLLM | Predictable |

### Optimization Stack

```
Model: AWQ or GPTQ quantized
Runtime: vLLM or TensorRT-LLM
Hardware: A100 80GB or H100
Features:
  - Continuous batching
  - PagedAttention
  - Speculative decoding
```

### Cost Optimization Tips

1. Use per-second billing (RunPod, Thunder Compute)
2. Implement request batching
3. Use model caching / persistent endpoints
4. Consider spot instances for non-critical traffic

---

## Scenario 5: Enterprise ML Platform

**Goal**: Full MLOps pipeline for organization

### Platform Selection

| If Your Data Lives In | Choose | Why |
|----------------------|--------|-----|
| AWS S3 | SageMaker | Native integration |
| BigQuery | Vertex AI | Fastest path |
| Azure Data Lake | Azure ML | Best governance |
| Multi-cloud | Kubeflow | Portability |

### Recommended Architecture

```
Training:
  Cloud: Managed (SageMaker/Vertex/Azure ML)
  Alternative: Lambda Labs (cost-effective)

Inference:
  Production: Cloud endpoints
  Edge: TensorRT/OpenVINO exports

MLOps:
  Experiments: MLflow or built-in
  Monitoring: Built-in + custom alerts
  CI/CD: Cloud-native pipelines
```

---

## Scenario 6: Research and Experimentation

**Goal**: Train and iterate quickly on ML models

### GPU Resources

| Budget | Provider | GPU | Cost/hour |
|--------|----------|-----|-----------|
| Low | Vast.ai | RTX 3090 | ~$0.16 |
| Medium | RunPod | A100 40GB | ~$1.50 |
| High | Lambda Labs | H100 | ~$2.50 |

### Experiment Workflow

```
1. Prototype: Local GPU or Colab
2. Scale: RunPod or Vast.ai
3. Production: Lambda Labs / Cloud
```

### Save Money Tips

1. Use Vast.ai for overnight training (checkpoint frequently)
2. Lambda Labs 50% academic discount
3. Reserve instances on cloud platforms
4. Use spot instances when possible

---

## Scenario 7: Multi-Modal AI Application

**Goal**: Vision + Language model deployment

### Model Options

| Task | Model | Quantization |
|------|-------|--------------|
| Image understanding | LLaVA, Qwen-VL | AWQ |
| Document processing | DocTR + LLM | Mixed |
| Video analysis | Custom pipeline | Per-component |

### Deployment Stack

```
Vision Encoder: TensorRT FP16
LLM: AWQ + vLLM
Orchestration: LangGraph or custom
```

---

## Scenario 8: Hybrid Cloud-Edge Deployment

**Goal**: Process at edge, fallback to cloud

### Architecture

```
Edge Device:
  - TFLite/TensorRT for fast inference
  - Small/quantized models
  - Cache common responses

Cloud:
  - Full model for complex queries
  - Batch processing
  - Training and updates

Decision Logic:
  - Latency requirement
  - Query complexity
  - Network availability
```

### Recommended Tools

| Layer | Tool | Purpose |
|-------|------|---------|
| Edge Runtime | TensorRT/TFLite | Local inference |
| Sync | MQTT/gRPC | Communication |
| Cloud Backend | SageMaker Endpoint | Fallback |
| Orchestration | AWS IoT Greengrass | Management |

---

## Quick Reference: Tool Selection

### By Hardware

| Hardware | Best Framework |
|----------|----------------|
| NVIDIA GPU (datacenter) | TensorRT |
| NVIDIA GPU (consumer) | ONNX Runtime + CUDA |
| NVIDIA Jetson | TensorRT |
| Intel CPU/GPU | OpenVINO |
| Apple Silicon | CoreML + MLX |
| Mobile (any) | TFLite |
| AMD GPU | ONNX Runtime + ROCm |

### By Model Type

| Model Type | Quantization | Runtime |
|------------|--------------|---------|
| Vision CNN | INT8 (PTQ) | TensorRT |
| Transformer | FP16 | TensorRT/ONNX |
| LLM (GPU) | AWQ/GPTQ | vLLM |
| LLM (CPU) | GGUF | llama.cpp |

### By Priority

| Priority | Choose |
|----------|--------|
| Maximum speed | TensorRT + INT8 |
| Maximum portability | ONNX Runtime |
| Minimum cost | Vast.ai + GGUF |
| Best DX | Modal |
| Enterprise features | Cloud ML platforms |

---

## Cost Comparison Summary

### GPU Hour Rates (Approximate 2025)

| Provider | A100 80GB | H100 | RTX 4090 |
|----------|-----------|------|----------|
| AWS | ~$4.00 | ~$5.50 | N/A |
| GCP | ~$3.50 | ~$5.00 | N/A |
| Azure | ~$3.50 | ~$5.00 | N/A |
| Lambda Labs | ~$1.10 | ~$2.50 | N/A |
| RunPod | ~$1.50 | ~$2.00 | ~$0.34 |
| Vast.ai | ~$0.50 | ~$1.20 | ~$0.20 |

### Monthly Estimates (24/7 Operation)

| Use Case | Provider | Monthly Cost |
|----------|----------|--------------|
| LLM API (A100) | Vast.ai | ~$360 |
| LLM API (A100) | Lambda | ~$800 |
| LLM API (A100) | AWS | ~$2,900 |
| Training (H100) | Lambda | ~$1,800 |
| Training (H100) | AWS | ~$4,000 |

*Costs vary based on utilization, region, and additional services.*

---

*Last updated: January 2, 2026*
