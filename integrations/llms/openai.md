# OpenAI API Integration

> **Access GPT-4, embeddings, and more via API**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | API access to OpenAI's models (GPT-4, embeddings, DALL-E) |
| **Why** | State-of-the-art reasoning, wide capabilities |
| **Models** | GPT-4o, GPT-4o-mini, o1, text-embedding-3 |
| **Pricing** | Pay per token (varies by model) |

### Model Comparison

| Model | Speed | Quality | Cost (1M tokens) | Best For |
|-------|-------|---------|------------------|----------|
| `gpt-4o` | Fast | ⭐⭐⭐⭐⭐ | $2.50/$10 | General |
| `gpt-4o-mini` | Fastest | ⭐⭐⭐⭐ | $0.15/$0.60 | Cost-effective |
| `o1` | Slow | ⭐⭐⭐⭐⭐ | $15/$60 | Complex reasoning |
| `o1-mini` | Medium | ⭐⭐⭐⭐ | $3/$12 | Code, math |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **API Key** | From platform.openai.com |
| **Python** | 3.8+ |
| **Budget** | Pay-as-you-go |

---

## Quick Start (5 min)

```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in 3 sentences."}
    ]
)

print(response.choices[0].message.content)
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Get API key
- [ ] Make first API call
- [ ] Understand messages format
- [ ] Handle responses

### L1: Advanced Features (2-3 hours)
- [ ] Streaming responses
- [ ] Function calling / tools
- [ ] JSON mode
- [ ] Vision (images)

### L2: Production (4-6 hours)
- [ ] Error handling and retries
- [ ] Rate limiting
- [ ] Cost tracking
- [ ] Prompt caching

---

## Code Examples

### Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Function Calling

```python
import json

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Function: {tool_call.function.name}, Args: {args}")
```

### JSON Mode

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List 3 programming languages as JSON"}],
    response_format={"type": "json_object"}
)

data = json.loads(response.choices[0].message.content)
```

### Vision

```python
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image('photo.jpg')}"
            }}
        ]
    }]
)
```

### Embeddings

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world", "Machine learning is amazing"]
)

embeddings = [e.embedding for e in response.data]
```

---

## Error Handling

```python
from openai import RateLimitError, APIError
import time

def call_with_retry(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except RateLimitError:
            wait = 2 ** attempt
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APIError as e:
            print(f"API error: {e}")
            raise
    raise Exception("Max retries exceeded")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Rate limited | Add exponential backoff, use batch API |
| High costs | Use gpt-4o-mini, cache responses |
| Timeout | Increase timeout, use streaming |
| Invalid API key | Check key format, regenerate if needed |

---

## Resources

- [OpenAI Docs](https://platform.openai.com/docs)
- [API Reference](https://platform.openai.com/docs/api-reference)
- [Pricing](https://openai.com/pricing)
- [Cookbook](https://cookbook.openai.com/)

---

*Part of [Luno-AI](../../README.md) | LLM Track*
