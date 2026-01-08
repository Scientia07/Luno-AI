# vLLM Integration

> **High-throughput LLM serving with PagedAttention**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Fast LLM inference engine with PagedAttention |
| **Why** | 2-24x higher throughput than HuggingFace |
| **Best For** | Production serving, high concurrency |
| **Key Feature** | PagedAttention for efficient memory management |

### vLLM vs Alternatives

| Engine | Throughput | Ease | Features |
|--------|------------|------|----------|
| **vLLM** | ⭐⭐⭐⭐⭐ | Medium | Production |
| **Ollama** | ⭐⭐⭐ | Easy | Local dev |
| **TGI** | ⭐⭐⭐⭐ | Medium | HF ecosystem |
| **llama.cpp** | ⭐⭐⭐ | Medium | CPU/edge |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA with 16GB+ VRAM |
| **CUDA** | 12.1+ |
| **Python** | 3.9-3.12 |
| **RAM** | 32GB+ recommended |

---

## Quick Start (15 min)

```bash
pip install vllm
```

### Start Server

```bash
# Serve Llama 3.1 8B
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000

# With quantization (less VRAM)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --quantization awq \
    --max-model-len 4096
```

### Query with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=500
)
print(response.choices[0].message.content)
```

### Direct Python Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling = SamplingParams(temperature=0.7, max_tokens=500)

outputs = llm.generate(["What is AI?"], sampling)
print(outputs[0].outputs[0].text)
```

---

## Learning Path

### L0: Basic Serving (1 hour)
- [ ] Install vLLM
- [ ] Start server with default model
- [ ] Query via OpenAI client
- [ ] Test streaming responses

### L1: Configuration (2-3 hours)
- [ ] Configure GPU memory usage
- [ ] Enable quantization (AWQ/GPTQ)
- [ ] Set up tensor parallelism
- [ ] Configure batch sizes

### L2: Production (4-6 hours)
- [ ] Deploy with Docker
- [ ] Set up load balancing
- [ ] Configure rate limiting
- [ ] Add monitoring (Prometheus)

### L3: Advanced (1 day)
- [ ] Multi-GPU serving
- [ ] Custom model loading
- [ ] Speculative decoding
- [ ] LoRA adapter serving

---

## Code Examples

### Streaming Responses

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Batch Processing

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling = SamplingParams(temperature=0.7, max_tokens=200)

prompts = [
    "Summarize machine learning",
    "Explain neural networks",
    "What is deep learning?"
]

# Batch processing is automatic and efficient
outputs = llm.generate(prompts, sampling)

for prompt, output in zip(prompts, outputs):
    print(f"Q: {prompt}")
    print(f"A: {output.outputs[0].text}\n")
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM vllm/vllm-openai:latest

# Pre-download model
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('meta-llama/Llama-3.1-8B-Instruct')"

CMD ["--model", "meta-llama/Llama-3.1-8B-Instruct", "--port", "8000"]
```

```bash
docker run --gpus all -p 8000:8000 my-vllm-server
```

### Multi-GPU Serving

```bash
# 2 GPUs with tensor parallelism
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9
```

---

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu-memory-utilization` | 0.9 | GPU memory fraction |
| `--max-model-len` | Model default | Max context length |
| `--quantization` | None | awq, gptq, squeezellm |
| `--tensor-parallel-size` | 1 | Number of GPUs |
| `--max-num-seqs` | 256 | Max concurrent requests |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Reduce `--gpu-memory-utilization`, enable quantization |
| Slow startup | Model downloading, use local cache |
| Connection refused | Check port, firewall settings |
| Low throughput | Increase `--max-num-seqs`, check GPU util |

---

## Resources

- [vLLM Docs](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

---

*Part of [Luno-AI](../../README.md) | LLM Track*
