# LLaVA Integration

> **Open-source vision-language model for visual chat**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Large Language and Vision Assistant |
| **Why** | Open alternative to GPT-4V, runs locally |
| **Versions** | LLaVA 1.5, LLaVA 1.6, LLaVA-NeXT |
| **Best For** | Image understanding, visual Q&A, descriptions |

### LLaVA vs Alternatives

| Model | Open | Quality | Speed | Local |
|-------|------|---------|-------|-------|
| **LLaVA 1.6** | Yes | ⭐⭐⭐⭐ | Medium | Yes |
| **GPT-4V** | No | ⭐⭐⭐⭐⭐ | Fast | No |
| **Qwen-VL** | Yes | ⭐⭐⭐⭐ | Fast | Yes |
| **Claude Vision** | No | ⭐⭐⭐⭐⭐ | Fast | No |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 8GB+ VRAM (16GB for larger models) |
| **Python** | 3.10+ |
| **RAM** | 16GB+ |

---

## Quick Start (15 min)

### With Ollama (Easiest)

```bash
ollama pull llava:13b
```

```python
import ollama
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

response = ollama.chat(
    model="llava:13b",
    messages=[{
        "role": "user",
        "content": "What's in this image?",
        "images": [encode_image("photo.jpg")]
    }]
)
print(response["message"]["content"])
```

### With Transformers

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("photo.jpg")
prompt = "[INST] <image>\nDescribe this image in detail [/INST]"

inputs = processor(prompt, image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=500)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Install via Ollama
- [ ] Describe images
- [ ] Ask questions about images
- [ ] Compare model sizes

### L1: Advanced Queries (2-3 hours)
- [ ] Multi-turn conversations
- [ ] Detailed analysis prompts
- [ ] Multiple images
- [ ] Structured output

### L2: Integration (4-6 hours)
- [ ] Build image analysis API
- [ ] Combine with object detection
- [ ] Document understanding
- [ ] Batch processing

---

## Code Examples

### Detailed Analysis

```python
response = ollama.chat(
    model="llava:13b",
    messages=[{
        "role": "user",
        "content": """Analyze this image and provide:
1. Main subject
2. Setting/background
3. Colors and lighting
4. Mood/atmosphere
5. Any text visible""",
        "images": [encode_image("photo.jpg")]
    }]
)
```

### Multi-turn Conversation

```python
messages = [
    {"role": "user", "content": "What's in this image?", "images": [img_b64]},
]
response = ollama.chat(model="llava:13b", messages=messages)
messages.append({"role": "assistant", "content": response["message"]["content"]})

# Follow-up question (no need to re-send image)
messages.append({"role": "user", "content": "How many people are there?"})
response = ollama.chat(model="llava:13b", messages=messages)
```

### JSON Output

```python
response = ollama.chat(
    model="llava:13b",
    messages=[{
        "role": "user",
        "content": """Analyze this image and return JSON:
{
  "objects": ["list of objects"],
  "scene": "description",
  "colors": ["dominant colors"]
}""",
        "images": [encode_image("photo.jpg")]
    }],
    format="json"
)
```

---

## Model Variants

| Model | Size | VRAM | Quality |
|-------|------|------|---------|
| `llava:7b` | 4GB | 6GB | ⭐⭐⭐ |
| `llava:13b` | 8GB | 10GB | ⭐⭐⭐⭐ |
| `llava:34b` | 20GB | 24GB | ⭐⭐⭐⭐⭐ |
| `llava-v1.6-mistral-7b` | 4GB | 8GB | ⭐⭐⭐⭐ |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Use smaller model, quantized version |
| Slow response | Use GPU, reduce max_tokens |
| Wrong descriptions | Improve prompt, be more specific |
| Image not loading | Check file path, format (jpg/png) |

---

## Resources

- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
- [HuggingFace Models](https://huggingface.co/llava-hf)
- [Ollama LLaVA](https://ollama.com/library/llava)

---

*Part of [Luno-AI](../../README.md) | Visual AI Track*
