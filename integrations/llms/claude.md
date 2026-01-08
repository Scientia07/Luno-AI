# Claude API Integration

> **Anthropic's Claude models for code, analysis, and long context**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | API access to Claude models |
| **Why** | Excellent for code, long documents, structured output |
| **Models** | Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku |
| **Context** | Up to 200K tokens |

### Model Comparison

| Model | Speed | Quality | Cost (1M tokens) | Best For |
|-------|-------|---------|------------------|----------|
| `claude-3-5-sonnet` | Fast | ⭐⭐⭐⭐⭐ | $3/$15 | Code, general |
| `claude-3-opus` | Slow | ⭐⭐⭐⭐⭐ | $15/$75 | Complex tasks |
| `claude-3-haiku` | Fastest | ⭐⭐⭐ | $0.25/$1.25 | Simple tasks |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **API Key** | From console.anthropic.com |
| **Python** | 3.8+ |

---

## Quick Start (5 min)

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain recursion with a simple example"}
    ]
)

print(message.content[0].text)
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Get API key
- [ ] Make first request
- [ ] Use system prompts
- [ ] Handle responses

### L1: Advanced Features (2-3 hours)
- [ ] Streaming
- [ ] Tool use
- [ ] Vision
- [ ] Extended thinking

### L2: Production (4-6 hours)
- [ ] Error handling
- [ ] Prompt caching
- [ ] Batch processing
- [ ] Cost optimization

---

## Code Examples

### System Prompt

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system="You are a Python expert. Provide concise, working code examples.",
    messages=[
        {"role": "user", "content": "How do I read a CSV file?"}
    ]
)
```

### Streaming

```python
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a short story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Tool Use

```python
import json

tools = [{
    "name": "get_weather",
    "description": "Get weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}]

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in London?"}]
)

for block in message.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}")
```

### Vision

```python
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encode_image("photo.jpg")
            }},
            {"type": "text", "text": "Describe this image"}
        ]
    }]
)
```

### Extended Thinking (Claude 3.5)

```python
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000
    },
    messages=[{"role": "user", "content": "Solve this complex math problem..."}]
)

for block in message.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking)
    elif block.type == "text":
        print("Answer:", block.text)
```

### Prompt Caching

```python
# Cache large context (saves cost on repeated calls)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "You are an expert on this 50,000 word document: ...",
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": "Summarize chapter 3"}]
)
```

---

## Error Handling

```python
from anthropic import RateLimitError, APIError

try:
    message = client.messages.create(...)
except RateLimitError:
    print("Rate limited, wait and retry")
except APIError as e:
    print(f"API error: {e}")
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Rate limited | Add backoff, use prompt caching |
| Context too long | Summarize, use 200K context model |
| High costs | Use Haiku for simple tasks |
| Tool not called | Improve tool description |

---

## Resources

- [Anthropic Docs](https://docs.anthropic.com/)
- [API Reference](https://docs.anthropic.com/en/api/messages)
- [Prompt Library](https://docs.anthropic.com/en/prompt-library)
- [Claude Code](https://claude.ai/claude-code)

---

*Part of [Luno-AI](../../README.md) | LLM Track*
