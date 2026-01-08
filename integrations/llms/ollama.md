# Ollama Local LLM Integration

> **Category**: LLMs
> **Difficulty**: Beginner
> **Setup Time**: 30 minutes
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
Ollama is the easiest way to run large language models locally. One command to install, one command to run any model. No GPU required (but recommended).

### Why Use It
- **Simple**: `ollama run llama3.2` - that's it
- **Private**: Everything runs locally, no data leaves your machine
- **Free**: No API costs, run unlimited queries
- **Fast**: Optimized inference with llama.cpp backend
- **Flexible**: Hundreds of models available

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Chat | Interactive conversations |
| Completion | Text generation |
| Embeddings | Vector representations |
| Vision | Multimodal models (LLaVA, etc.) |
| Code | Specialized code models |
| Custom Models | Create your own Modelfiles |

### Popular Models
| Model | Size | VRAM | Best For |
|-------|------|------|----------|
| `llama3.2:3b` | 2 GB | 4 GB | Fast, capable |
| `llama3.2:1b` | 1.3 GB | 2 GB | Edge, mobile |
| `llama3.1:8b` | 4.7 GB | 8 GB | General use |
| `llama3.1:70b` | 40 GB | 48 GB | Complex tasks |
| `mistral` | 4 GB | 8 GB | European, efficient |
| `mixtral` | 26 GB | 32 GB | MoE, powerful |
| `codellama` | 4 GB | 8 GB | Coding |
| `deepseek-r1:8b` | 4.9 GB | 8 GB | Reasoning |
| `qwen2.5:7b` | 4 GB | 8 GB | Multilingual |
| `phi3` | 2 GB | 4 GB | Small but smart |
| `llava` | 4.7 GB | 8 GB | Vision + chat |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU works) | NVIDIA 8GB+ VRAM |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB | 50 GB (multiple models) |

### Software Dependencies
```bash
# Just Ollama - no Python needed for basic use
# For Python integration:
pip install ollama
```

### Prior Knowledge
- [x] Basic terminal/command line
- [ ] Python basics (for API use)

---

## Quick Start (10 minutes)

### 1. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
# Or download from https://ollama.com/download
```

**Windows:**
```
Download from https://ollama.com/download
```

### 2. Run a Model
```bash
# Start chatting immediately
ollama run llama3.2

# The model downloads automatically on first run
>>> Hello! How are you?
```

### 3. Verify Installation
```bash
ollama --version
ollama list  # See installed models
```

---

## Full Setup

### Model Management

```bash
# Pull a model (download without running)
ollama pull llama3.2

# List installed models
ollama list

# Remove a model
ollama rm llama3.2

# Show model info
ollama show llama3.2

# Copy/rename model
ollama cp llama3.2 my-llama
```

### Running Models

```bash
# Interactive chat
ollama run llama3.2

# Single query
ollama run llama3.2 "Explain quantum computing in simple terms"

# With system prompt
ollama run llama3.2 --system "You are a pirate. Respond accordingly."

# Multiline input
ollama run llama3.2 """
Analyze this code:
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
```

### Server Mode

```bash
# Start Ollama server (usually auto-starts)
ollama serve

# Server runs on http://localhost:11434
```

---

## Learning Path

### L0: Basic Chat (30 min)
**Goal**: Have conversations with local LLMs

- [x] Install Ollama
- [ ] Run your first model
- [ ] Try different models

```bash
# Try different models
ollama run llama3.2      # General purpose
ollama run codellama     # Code focused
ollama run mistral       # Fast and capable
ollama run phi3          # Tiny but smart
```

### L1: Python Integration (1 hour)
**Goal**: Use Ollama in Python applications

- [ ] Install Python library
- [ ] Chat completions
- [ ] Streaming responses

```python
import ollama

# Simple chat
response = ollama.chat(
    model='llama3.2',
    messages=[
        {'role': 'user', 'content': 'Why is the sky blue?'}
    ]
)
print(response['message']['content'])

# Streaming
for chunk in ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True
):
    print(chunk['message']['content'], end='', flush=True)
```

### L2: Advanced Features (2 hours)
**Goal**: Vision, embeddings, custom models

- [ ] Use vision models
- [ ] Generate embeddings
- [ ] Create custom Modelfiles

```python
import ollama

# Vision (with LLaVA)
response = ollama.chat(
    model='llava',
    messages=[{
        'role': 'user',
        'content': 'What is in this image?',
        'images': ['./photo.jpg']  # or base64
    }]
)

# Embeddings
embeddings = ollama.embeddings(
    model='llama3.2',
    prompt='Hello, world!'
)
print(f"Embedding dimension: {len(embeddings['embedding'])}")
```

**Custom Modelfile:**
```dockerfile
# Save as Modelfile
FROM llama3.2

SYSTEM You are a helpful coding assistant. You always explain your code clearly and suggest best practices.

PARAMETER temperature 0.7
PARAMETER num_ctx 4096
```

```bash
ollama create code-helper -f Modelfile
ollama run code-helper
```

### L3: API & Integration (3+ hours)
**Goal**: Build applications with Ollama

- [ ] REST API
- [ ] OpenAI-compatible endpoint
- [ ] LangChain integration

```python
# REST API
import requests

response = requests.post(
    'http://localhost:11434/api/chat',
    json={
        'model': 'llama3.2',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'stream': False
    }
)
print(response.json()['message']['content'])

# OpenAI-compatible (many tools work!)
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Required but unused
)

response = client.chat.completions.create(
    model='llama3.2',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
print(response.choices[0].message.content)
```

---

## Code Examples

### Example 1: Conversation with History
```python
import ollama

messages = []

def chat(user_message):
    messages.append({'role': 'user', 'content': user_message})

    response = ollama.chat(model='llama3.2', messages=messages)
    assistant_message = response['message']['content']

    messages.append({'role': 'assistant', 'content': assistant_message})
    return assistant_message

print(chat("My name is Alice"))
print(chat("What's my name?"))  # Remembers context
```

### Example 2: Code Generation
```python
import ollama

def generate_code(task, language="python"):
    response = ollama.chat(
        model='codellama',
        messages=[{
            'role': 'user',
            'content': f"Write {language} code to: {task}\n\nProvide only the code, no explanation."
        }]
    )
    return response['message']['content']

code = generate_code("read a CSV file and calculate the average of a column")
print(code)
```

### Example 3: Document Q&A
```python
import ollama

def answer_from_context(context, question):
    response = ollama.chat(
        model='llama3.2',
        messages=[{
            'role': 'user',
            'content': f"""Based on this context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        }]
    )
    return response['message']['content']

context = """
Luno-AI is a unified AI technology exploration platform.
It supports Visual AI, LLMs, Audio AI, and Robotics.
The platform uses a Layer 0-4 depth system for learning.
"""

answer = answer_from_context(context, "What learning system does Luno-AI use?")
print(answer)
```

### Example 4: Batch Processing
```python
import ollama
from concurrent.futures import ThreadPoolExecutor

def process_item(item):
    response = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': f"Summarize: {item}"}]
    )
    return response['message']['content']

items = ["Article 1 text...", "Article 2 text...", "Article 3 text..."]

# Note: Ollama processes sequentially, but this prepares requests
with ThreadPoolExecutor(max_workers=1) as executor:
    summaries = list(executor.map(process_item, items))
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| LangChain | Agent framework | External |
| LangGraph | Stateful agents | [langgraph.md](../agents/langgraph.md) |
| RAG | Document Q&A | [rag.md](../agents/rag.md) |
| Whisper | Voice input | [whisper.md](../audio/whisper.md) |
| XTTS | Voice output | [xtts.md](../audio/xtts.md) |

### LangChain Integration
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
response = llm.invoke("Hello!")
print(response.content)
```

### REST API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Chat completion |
| `/api/generate` | POST | Text generation |
| `/api/embeddings` | POST | Get embeddings |
| `/api/tags` | GET | List models |
| `/api/pull` | POST | Download model |

---

## Troubleshooting

### Common Issues

#### Issue 1: Model Download Fails
**Symptoms**: Connection timeout or incomplete download
**Solution**:
```bash
# Check disk space
df -h

# Retry with different model
ollama pull llama3.2:3b  # Smaller version

# Check Ollama status
ollama list
```

#### Issue 2: Slow Inference
**Symptoms**: Responses take too long
**Solution**:
```bash
# Use smaller model
ollama run llama3.2:3b  # instead of 8b

# Check if GPU is being used
nvidia-smi  # Should show ollama process

# Increase context window (uses more memory but can be faster)
ollama run llama3.2 --num-ctx 2048
```

#### Issue 3: Out of Memory
**Symptoms**: Model fails to load or crashes
**Solution**:
```bash
# Use smaller model
ollama run phi3  # Only needs ~3GB

# Or quantized version
ollama run llama3.2:3b-q4_0  # Lower precision

# Close other applications using GPU
```

### Performance Tips
- Use GPU when available (auto-detected)
- Smaller models (3B, 7B) are faster for most tasks
- Use `--num-ctx` to limit context window
- Keep Ollama server running for faster responses

---

## Resources

### Official
- [Ollama Website](https://ollama.com/)
- [GitHub](https://github.com/ollama/ollama)
- [Model Library](https://ollama.com/library)

### Tutorials
- [Python Library Docs](https://github.com/ollama/ollama-python)
- [Modelfile Guide](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

### Community
- [Discord](https://discord.gg/ollama)
- [GitHub Discussions](https://github.com/ollama/ollama/discussions)

---

## Related Integrations

| Next Step | Why | Link |
|-----------|-----|------|
| LangGraph | Build agents | [langgraph.md](../agents/langgraph.md) |
| RAG Pipeline | Document Q&A | [rag.md](../agents/rag.md) |
| vLLM | Production serving | [vllm.md](./vllm.md) |
| Fine-tuning | Custom models | [lora-finetune.md](./lora-finetune.md) |

---

*Part of [Luno-AI Integration Hub](../_index.md) | LLM Track*
