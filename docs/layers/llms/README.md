# Large Language Models (LLMs)

> **The intelligence behind modern AI** - from transformers to agents.

---

## Layer Navigation

| Layer | Content | Status |
|-------|---------|--------|
| L0 | [Overview](#overview) | This file |
| L1 | [Concepts](./concepts.md) | Pending |
| L2 | [Deep Dive](./deep-dive.md) | Pending |
| L3 | [Labs](../../labs/llms/) | Pending |
| L4 | [Advanced](./advanced.md) | Pending |

---

## Overview

Large Language Models are neural networks trained on vast amounts of text that can understand, generate, and reason about language. They power chatbots, code assistants, and increasingly, autonomous agents.

```
              THE LLM LANDSCAPE (2025)

    CLOSED SOURCE                 OPEN WEIGHTS
    ┌─────────────┐               ┌─────────────┐
    │ GPT-4/4o    │               │ Llama 3.x   │
    │ Claude 3.5  │               │ Mistral     │
    │ Gemini      │               │ Qwen 2.5    │
    │ Grok        │               │ DeepSeek    │
    └─────────────┘               └─────────────┘
           │                             │
           └──────────┬──────────────────┘
                      │
              ┌───────▼───────┐
              │   USE CASES   │
              ├───────────────┤
              │ Chat/Q&A      │
              │ Code Gen      │
              │ Analysis      │
              │ Agents        │
              │ RAG           │
              └───────────────┘
```

---

## Model Families

### Closed Source (API Access)

| Model | Provider | Strengths |
|-------|----------|-----------|
| **GPT-4o** | OpenAI | Best all-around, multimodal |
| **Claude 3.5 Opus/Sonnet** | Anthropic | Best for code, long context |
| **Gemini Pro/Ultra** | Google | Native multimodal |
| **Grok** | xAI | Real-time knowledge |

### Open Weights (Run Locally)

| Model | Params | Strengths |
|-------|--------|-----------|
| **Llama 3.3** | 70B | Best open all-around |
| **Qwen 2.5** | 7B-72B | Excellent multilingual |
| **Mistral** | 7B | Efficient, strong |
| **DeepSeek-V3** | 671B MoE | Near GPT-4 quality |
| **Phi-4** | 14B | Small but capable |
| **Gemma 2** | 2B-27B | Google's open model |

### Specialized Models

| Type | Models |
|------|--------|
| **Code** | Codestral, DeepSeek-Coder, Qwen-Coder |
| **Math** | DeepSeek-Math, Qwen-Math |
| **Embeddings** | BGE, E5, nomic-embed |
| **Vision** | LLaVA, Qwen-VL, InternVL |

---

## MoE (Mixture of Experts)

**Efficient scaling by routing to specialized "experts"**

```
Input Token ──▶ Router ──▶ Expert 1 ──┐
                  │                   │
                  ├──▶ Expert 2 ◀─────┤──▶ Output
                  │        ↑          │
                  └──▶ Expert N ──────┘

Only 2-4 experts active per token (out of 8-160)
= Same quality, less compute
```

| Model | Experts | Active | Total Params |
|-------|---------|--------|--------------|
| Mixtral 8x7B | 8 | 2 | 46.7B |
| DeepSeek-V3 | 256 | 8 | 671B |
| Grok-1 | 8 | 2 | 314B |

---

## Running LLMs Locally

### Ollama (Easiest)

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Run a model
ollama run llama3.2

# Use in code
import ollama
response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Hello!'}
])
```

### llama.cpp (Most Efficient)

```bash
# CPU inference with GGUF models
./main -m model.gguf -p "Hello"
```

### vLLM (Production)

```bash
# High-throughput serving
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

---

## Key Concepts

### 1. Context Window

The amount of text the model can "see" at once:

| Model | Context |
|-------|---------|
| GPT-4 Turbo | 128K tokens |
| Claude 3.5 | 200K tokens |
| Gemini 1.5 | 1M+ tokens |
| Llama 3.1 | 128K tokens |

### 2. Tokenization

```
"Hello, world!" ──▶ [15496, 11, 995, 0]
                          ↓
                    ~1 token ≈ 4 chars
```

### 3. Temperature

```
temp=0.0  ──▶ Deterministic, focused
temp=0.7  ──▶ Creative, varied (default)
temp=1.0+ ──▶ Very random, chaotic
```

### 4. Inference Optimization

| Technique | Purpose |
|-----------|---------|
| **Quantization** | Reduce memory (8-bit, 4-bit) |
| **KV Cache** | Speed up generation |
| **Speculative Decoding** | Faster with draft model |
| **Flash Attention** | Memory-efficient attention |

---

## Fine-tuning

### When to Fine-tune

- Specific domain knowledge
- Custom output format
- Specialized behavior
- Cost optimization (use smaller model)

### Methods

| Method | VRAM | Quality | Speed |
|--------|------|---------|-------|
| Full Fine-tune | 40GB+ | Best | Slow |
| LoRA | 8-16GB | Good | Fast |
| QLoRA | 4-8GB | Good | Fast |

```python
# QLoRA example with PEFT
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, lora_alpha=32)
model = get_peft_model(base_model, lora_config)
```

---

## RAG (Retrieval Augmented Generation)

**Enhance LLMs with external knowledge**

```
Query: "What's our return policy?"
         │
         ▼
┌─────────────────────┐
│   Vector Search     │──▶ Find relevant docs
│   (Embeddings)      │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   LLM + Context     │──▶ Generate answer
│                     │
└─────────────────────┘
         │
         ▼
"Our return policy is 30 days..."
```

---

## Use Case Decision Tree

```
What do you need?
│
├── Best quality, cost OK? ──────────▶ GPT-4o / Claude Opus
│
├── Fast, cheap API? ────────────────▶ GPT-4o-mini / Claude Haiku
│
├── Run locally, best quality? ──────▶ Llama 3.3 70B / DeepSeek-V3
│
├── Run locally, fast? ──────────────▶ Llama 3.2 8B / Mistral 7B
│
├── Code generation? ────────────────▶ Claude / Codestral
│
├── Long documents? ─────────────────▶ Claude / Gemini
│
└── Embeddings for search? ──────────▶ BGE-large / E5-large
```

---

## Labs

| Notebook | Focus |
|----------|-------|
| `01-ollama-quickstart.ipynb` | Local LLMs |
| `02-openai-api.ipynb` | API usage |
| `03-embeddings.ipynb` | Vector representations |
| `04-rag-basics.ipynb` | Retrieval augmented |
| `05-lora-finetuning.ipynb` | Fine-tuning |
| `06-agents-intro.ipynb` | Tool use |

---

## Next Steps

- L1: [How Transformers Work](./concepts.md)
- L2: [Attention Mechanisms](./deep-dive.md)
- Related: [Agents](../agents/README.md)

---

*"Language is the interface to intelligence."*
