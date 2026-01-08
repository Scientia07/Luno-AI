# Edge AI and Model Deployment: Detailed Findings

---

## 1. Edge Inference Frameworks

### TensorRT (NVIDIA)

**Overview**: NVIDIA's flagship inference optimizer, delivering industry-leading performance on NVIDIA GPUs through graph optimizations, kernel fusion, and precision calibration.

**Key Features**:
- FP8/FP16/INT8 precision support
- Dynamic shape handling
- Kernel auto-tuning
- Up to 5x GPU speedup over native PyTorch

**Best For**:
- Computer vision at scale
- Generative AI and LLMs on NVIDIA hardware
- Datacenter and edge GPUs (Jetson, RTX, A100, H100)

**Limitations**:
- NVIDIA-only (no portability)
- Complex conversion pipeline for some models

**Performance**: Consistently achieves lowest latency and highest throughput on NVIDIA hardware. Benchmark studies on Jetson AGX Orin show TensorRT outperforms alternatives for stable, well-bounded model graphs.

---

### ONNX Runtime

**Overview**: Microsoft's cross-platform inference engine with broad hardware support through execution providers (EPs).

**Supported Hardware via EPs**:
| Execution Provider | Hardware |
|--------------------|----------|
| CUDA EP | NVIDIA GPUs |
| TensorRT EP | NVIDIA GPUs (optimized) |
| OpenVINO EP | Intel CPUs/GPUs/VPUs |
| DirectML EP | Windows devices |
| ROCm EP | AMD GPUs |
| CoreML EP | Apple Silicon |
| QNN EP | Qualcomm SoCs |
| NNAPI EP | Android devices |

**Best For**:
- Cross-vendor portability
- Multi-cloud deployments
- Hybrid strategies (ONNX as primary, TensorRT EP for NVIDIA)

**Performance**: Approaches TensorRT performance when using TensorRT EP, especially with FP16/INT8 quantization. Excellent flexibility for heterogeneous environments.

---

### OpenVINO (Intel)

**Overview**: Intel's toolkit optimized for AI inference on Intel hardware.

**Supported Hardware**:
- Intel CPUs (Core, Xeon)
- Intel GPUs (integrated and discrete)
- Intel VPUs (Movidius)
- Intel FPGAs

**Best For**:
- Industrial IoT
- Smart cameras and retail
- Edge devices with Intel processors
- Up to 3x CPU speedup

**Key Features**:
- Model optimization and compression
- Post-training quantization
- Neural network compression framework

---

### CoreML (Apple)

**Overview**: Apple's framework for on-device ML, paired with MLX for Apple Silicon acceleration.

**Best For**:
- iOS/macOS applications
- Apple Silicon Macs (M1/M2/M3/M4)
- On-device privacy-focused apps

**Tools**: coremltools for model conversion from PyTorch/TensorFlow.

**Limitations**:
- Apple ecosystem only
- Some model conversion nuances

---

### TFLite / LiteRT

**Overview**: Google's lightweight runtime (formerly TensorFlow Lite) for mobile and embedded devices.

**Key Features**:
- Ultra-small binary (~300KB minimum)
- Edge TPU support
- Microcontroller support (TFLite Micro)
- Delegate system for hardware acceleration

**Best For**:
- Mobile apps (Android/iOS)
- IoT and embedded devices
- Microcontrollers
- Google Edge TPU deployment

---

### Framework Comparison Table

| Framework | Hardware | Latency | Portability | Ease of Use |
|-----------|----------|---------|-------------|-------------|
| TensorRT | NVIDIA only | Excellent | Low | Medium |
| ONNX Runtime | Cross-platform | Very Good | Excellent | High |
| OpenVINO | Intel only | Very Good | Low | High |
| CoreML | Apple only | Very Good | Low | High |
| TFLite | Mobile/Embedded | Good | Medium | High |

---

## 2. Quantization Methods

### GPTQ (Generative Pre-trained Transformer Quantization)

**Type**: Post-training quantization (PTQ) for 4-bit

**How It Works**:
- One-shot weight quantization using approximate second-order information
- Requires calibration dataset
- Optimized for GPU inference

**Performance**:
- 5x faster than GGUF on pure GPU with Marlin kernels
- ~90% quality retention
- Best for static, GPU-only deployments

**Pros**:
- Excellent GPU inference speed
- Highly accurate at 4-bit

**Cons**:
- Requires calibration data
- GPU-only optimization

---

### AWQ (Activation-aware Weight Quantization)

**Type**: Activation-aware 4-bit quantization

**How It Works**:
- Protects salient weights based on activation distributions
- Does NOT require backpropagation or reconstruction
- Skips unimportant weights during quantization

**Performance**:
- ~95% quality retention (highest among 4-bit methods)
- Often outperforms GPTQ in speed
- Faster quantization process

**Best For**:
- Instruction-tuned LLMs
- Multi-modal models
- Production deployments requiring quality

---

### bitsandbytes

**Type**: Runtime quantization library

**Methods Available**:
- LLM.int8 (8-bit inference)
- NF4/FP4 (4-bit inference and fine-tuning)
- Double quantization option

**Key Advantage**: No calibration data required. Direct Hugging Face integration.

**Best For**:
- Quick experimentation
- QLoRA fine-tuning
- Hugging Face workflows

---

### GGUF

**Type**: File format with K-quant quantization

**How It Works**:
- Self-contained format (model + metadata in one file)
- K-quants use improved blockwise quantization
- Designed for CPU and hybrid CPU/GPU inference

**Quantization Levels**:

| Level | Bits | Size (7B) | Quality Loss | Use Case |
|-------|------|-----------|--------------|----------|
| Q8_0 | 8 | ~8GB | Near-zero | Maximum quality |
| Q6_K | 6 | ~6GB | Minimal | High-quality reasoning |
| Q5_K_M | 5 | ~5GB | Small | Recommended balance |
| Q4_K_M | 4 | ~4GB | Moderate | Memory-constrained |
| Q3_K_S | 3 | ~3GB | Noticeable | Extreme compression |
| Q2_K | 2 | ~2GB | Significant | Testing only |

**Recommended Ladder**: Q4_K_M -> Q5_K_M -> Q8_0

**Use Case Guidelines**:
- Chat/Assistant: Q4_K_M (or Q5_K_M for consistency)
- Reasoning/Math: Q5_K_M or higher
- Coding: Q5_K_M or Q8_0
- Local RAG: Q5_K_M or Q8_0

---

### Quantization Method Comparison

| Method | Quality | GPU Speed | CPU Support | Calibration |
|--------|---------|-----------|-------------|-------------|
| AWQ | 95% | Excellent | No | Minimal |
| GGUF | 92% | Good | Excellent | No |
| GPTQ | 90% | Excellent | No | Required |
| bitsandbytes | 93% | Good | Limited | No |

---

## 3. Model Optimization Techniques

### Pruning

**Unstructured Pruning**:
- Removes individual weights
- Creates sparse matrices
- High compression ratios possible
- Requires specialized hardware for speedups

**Structured Pruning**:
- Removes entire filters/channels/layers
- Maintains dense structure
- Direct acceleration on standard hardware
- More practical for deployment

**N:M Structured Sparsity**:
- Example: 2:4 sparsity (2 zeros per 4 elements)
- 2x speedup on NVIDIA A100+ without quality drop
- Best of both worlds: structured + fine-grained

**Notable Methods**:
- **WANDA**: Importance-aware pruning using weight magnitudes + activation statistics
- **MAMA**: Movement and magnitude analysis for identifying important connections

---

### Knowledge Distillation

**How It Works**:
1. Train large "teacher" model
2. Train smaller "student" to match teacher's output distributions
3. Student learns soft labels, not just hard predictions

**Benefits**:
- Significant size reduction (often 10x)
- Maintains most capabilities
- Works with any architecture

**Example**: NVIDIA's TensorRT Model Optimizer pipeline prunes Qwen3-8B from 36 to 24 layers (~6B parameters) using depth pruning + distillation.

---

### Combined Optimization Strategy

The optimal pipeline combines multiple techniques:

```
1. Pruning (remove structural redundancy)
   ↓
2. Knowledge Distillation (transfer to smaller model)
   ↓
3. Quantization (reduce precision)
   ↓
4. Hardware-specific compilation (TensorRT, etc.)
```

---

## 4. Cloud AI Platforms

### AWS SageMaker

**Market Position**: 34% market share (leader)

**Strengths**:
- Comprehensive MLOps (Studio, Pipelines, Endpoints)
- Wide range of built-in algorithms
- Tight AWS ecosystem integration
- Industrial-grade endpoint scaling

**Pricing**:
- Training: $0.302 - $3.825/hour
- Serverless inference options available

**Best For**:
- Teams with more engineers than analysts
- Complex conversational AI
- Existing AWS customers

---

### GCP Vertex AI

**Market Position**: 22% market share

**Strengths**:
- Fastest path from dataset to results
- TPU v5p cluster access
- Excellent BigQuery integration
- Advanced foundation model access

**Pricing**:
- AutoML training: $1.375/node-hour
- Generative AI: from $0.0001/1K characters
- $300 free credits for new users

**Best For**:
- ML training and deployment
- NLP workloads
- Existing GCP/BigQuery users

---

### Azure Machine Learning

**Market Position**: 29% market share

**Strengths**:
- Exceptional Microsoft ecosystem integration
- Robust governance and compliance
- Hybrid cloud capabilities
- Strong CI/CD with Azure DevOps
- Responsible AI features

**Pricing**:
- Compute from $0.042/hour
- 1-year and 3-year reserved savings

**Best For**:
- Enterprise/regulated industries
- Microsoft shops
- Hybrid on-prem + cloud

---

### Cloud Platform Decision Matrix

| Factor | SageMaker | Vertex AI | Azure ML |
|--------|-----------|-----------|----------|
| Data Location | S3 | BigQuery | Azure Data Lake |
| Hardware Edge | Inferentia3 | TPU v5p | Confidential computing |
| Best MLOps | Model Monitor | Experiments | Azure DevOps integration |
| Pricing Model | Usage-based | Per-node | Reserved instances |

---

## 5. GPU Cloud Providers

### Lambda Labs

**Pricing**: A100 80GB ~$1.10/hour

**Strengths**:
- Zero egress fees
- InfiniBand networking
- Deep learning optimized
- Lambda Stack pre-installed
- 50% academic discount

**Best For**: Large-scale LLM training, research

---

### RunPod

**Pricing**:
- RTX 4090: ~$0.34/hour
- H100 PCIe: ~$1.99/hour

**Features**:
- Per-second billing
- Secure Cloud + Community Cloud tiers
- Containerized workflows
- Serverless GPU endpoints

**Best For**: Real-time iteration, startups, containerized workloads

**Note**: Some reliability concerns with Community Cloud

---

### Vast.ai

**Pricing**:
- A100 80GB: ~$0.50/hour
- RTX 3090: ~$0.16/hour
- 50-70% cheaper than hyperscalers

**Model**: Marketplace (like Airbnb for GPUs)

**Best For**:
- Cost-sensitive workloads
- Hobby projects
- Non-critical training that can checkpoint

**Note**: Variable reliability, depends on host

---

### Modal

**Pricing**: Premium (most expensive per-hour)

**Strengths**:
- Exceptional developer experience
- Sub-second cold starts
- Python-native deployment
- Automatic scaling
- Built on Oracle Cloud (AWS/GCP/Azure support)

**Best For**: Production apps, developers who value DX

---

### GPU Provider Comparison

| Provider | Cost | Reliability | Dev Experience | Best Use |
|----------|------|-------------|----------------|----------|
| Lambda Labs | Medium | High | Good | LLM training |
| RunPod | Low | Medium | Good | Iteration/testing |
| Vast.ai | Lowest | Variable | Basic | Budget training |
| Modal | High | High | Excellent | Production apps |

### Hidden Costs to Watch

1. **Storage fees**: Can exceed GPU costs for large datasets
2. **Egress fees**: Lambda Labs has zero egress (unique)
3. **Support plans**: Enterprise support adds significant cost
4. **Billing granularity**: Per-minute ~40% cheaper than hourly for bursty workloads

---

## Summary Recommendations

### For Startups
1. Start with RunPod or Modal for fast iteration
2. Use GGUF quantization for local testing
3. Deploy with ONNX Runtime for flexibility

### For Enterprises
1. Choose cloud platform based on data location
2. Use TensorRT for NVIDIA deployments
3. Implement AWQ for production LLMs

### For Research
1. Lambda Labs for training (academic discount + InfiniBand)
2. Vast.ai for budget experiments
3. Use multiple quantization methods for comparison

### For Edge Deployment
1. TFLite for mobile
2. CoreML for Apple devices
3. OpenVINO for Intel edge devices
4. TensorRT for Jetson
