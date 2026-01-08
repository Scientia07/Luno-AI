# llama.cpp Integration

> **Run LLMs efficiently on CPU and consumer hardware**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | C++ inference engine for LLMs |
| **Why** | CPU inference, low memory, quantization |
| **Models** | Llama, Mistral, Qwen, Phi, many more |
| **Best For** | Local deployment, edge devices, CPU-only systems |

### llama.cpp vs Alternatives

| Feature | llama.cpp | Ollama | vLLM |
|---------|-----------|--------|------|
| CPU Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| GPU Support | Yes | Yes | Yes |
| Memory Usage | Lowest | Low | High |
| Ease of Use | Medium | Easy | Medium |
| Customization | High | Low | High |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **CPU** | AVX2 support recommended |
| **RAM** | 8GB+ (model dependent) |
| **GPU** | Optional, CUDA/Metal/Vulkan |

---

## Quick Start (20 min)

### Python Bindings

```bash
pip install llama-cpp-python
```

```python
from llama_cpp import Llama

# Load quantized model
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=4096,        # Context length
    n_threads=8,       # CPU threads
    n_gpu_layers=0     # 0 for CPU only
)

# Generate
output = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=100,
    stop=["Q:", "\n"],
    echo=False
)

print(output["choices"][0]["text"])
```

### With GPU Acceleration

```python
llm = Llama(
    model_path="models/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=35,  # Offload layers to GPU
    n_threads=4
)
```

---

## Learning Path

### L0: Basic Usage (1-2 hours)
- [ ] Install llama-cpp-python
- [ ] Download GGUF model
- [ ] Run first inference
- [ ] Understand quantization levels

### L1: Optimization (2-3 hours)
- [ ] GPU layer offloading
- [ ] Context length tuning
- [ ] Batch processing
- [ ] Memory mapping

### L2: Advanced (4-6 hours)
- [ ] Custom quantization
- [ ] Server deployment
- [ ] Embedding models
- [ ] Multi-model serving

---

## Code Examples

### Chat Completions

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/mistral-7b-instruct.Q4_K_M.gguf",
    n_ctx=4096,
    chat_format="mistral-instruct"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]

response = llm.create_chat_completion(
    messages=messages,
    max_tokens=500,
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])
```

### Streaming

```python
from llama_cpp import Llama

llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf", n_ctx=2048)

for chunk in llm(
    "Write a poem about coding:",
    max_tokens=200,
    stream=True
):
    print(chunk["choices"][0]["text"], end="", flush=True)
```

### Embeddings

```python
from llama_cpp import Llama

# Load embedding model
llm = Llama(
    model_path="models/nomic-embed-text-v1.5.Q4_K_M.gguf",
    embedding=True,
    n_ctx=2048
)

# Generate embeddings
text = "The quick brown fox jumps over the lazy dog"
embedding = llm.embed(text)

print(f"Embedding dimension: {len(embedding)}")
```

### OpenAI-Compatible Server

```bash
# Start server
python -m llama_cpp.server \
    --model models/mistral-7b-instruct.Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --n_ctx 4096 \
    --n_gpu_layers 35
```

```python
# Use with OpenAI client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Batch Processing

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/llama-2-7b.Q4_K_M.gguf",
    n_ctx=2048,
    n_batch=512  # Batch size for prompt processing
)

prompts = [
    "Summarize: The quick brown fox...",
    "Translate to French: Hello world",
    "Explain: What is AI?"
]

for prompt in prompts:
    output = llm(prompt, max_tokens=100)
    print(f"Input: {prompt[:30]}...")
    print(f"Output: {output['choices'][0]['text']}\n")
```

### Memory-Mapped Models

```python
# Use memory mapping for large models
llm = Llama(
    model_path="models/llama-2-70b.Q4_K_M.gguf",
    n_ctx=2048,
    use_mmap=True,    # Memory map model file
    use_mlock=False,  # Don't lock in RAM
    n_gpu_layers=60
)
```

---

## Quantization Levels

| Level | Size | Quality | Speed |
|-------|------|---------|-------|
| Q2_K | Smallest | ⭐⭐ | Fastest |
| Q3_K_M | Small | ⭐⭐⭐ | Fast |
| Q4_K_M | Medium | ⭐⭐⭐⭐ | Medium |
| Q5_K_M | Large | ⭐⭐⭐⭐⭐ | Slower |
| Q6_K | Larger | ⭐⭐⭐⭐⭐ | Slow |
| Q8_0 | Largest | ⭐⭐⭐⭐⭐ | Slowest |

**Recommended**: Q4_K_M for most use cases

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow inference | Enable GPU layers, reduce context |
| OOM error | Use smaller quant, enable mmap |
| Poor quality | Use larger quant (Q5/Q6) |
| CUDA errors | Rebuild with CUDA support |

---

## Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [GGUF Models](https://huggingface.co/TheBloke)
- [Quantization Guide](https://github.com/ggerganov/llama.cpp/discussions/2948)

---

*Part of [Luno-AI](../../README.md) | LLM Track*
