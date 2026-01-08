# LLMs: Projects & Comparisons

> **Hands-on projects and framework comparisons for Large Language Models**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Local Chatbot
**Goal**: Run a chatbot entirely on your machine

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 1-2 hours |
| Technologies | Ollama |
| Skills | API basics, prompting |

**Tasks**:
- [ ] Install Ollama
- [ ] Pull a model (llama3.2)
- [ ] Chat via CLI
- [ ] Use Python API
- [ ] Add conversation history

**Starter Code**:
```python
import ollama

messages = []

while True:
    user_input = input("You: ")
    messages.append({"role": "user", "content": user_input})

    response = ollama.chat(model="llama3.2", messages=messages)
    assistant_msg = response["message"]["content"]

    messages.append({"role": "assistant", "content": assistant_msg})
    print(f"Bot: {assistant_msg}")
```

---

#### Project 2: Code Explainer
**Goal**: Explain code in plain language

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | Ollama + codellama |
| Skills | Prompting, code understanding |

**Tasks**:
- [ ] Load code from file
- [ ] Send to code-specialized model
- [ ] Get explanation
- [ ] Format output (markdown)
- [ ] Add "explain like I'm 5" mode

---

### Intermediate Projects (L2)

#### Project 3: Document Q&A (Simple RAG)
**Goal**: Ask questions about your documents

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | Ollama + ChromaDB |
| Skills | Embeddings, retrieval |

**Tasks**:
- [ ] Load documents (PDF/text)
- [ ] Chunk into segments
- [ ] Generate embeddings
- [ ] Store in vector DB
- [ ] Query with questions
- [ ] Generate answers with context

---

#### Project 4: Writing Assistant
**Goal**: Help improve written content

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | LLM + prompting |
| Skills | Prompt engineering |

**Tasks**:
- [ ] Grammar correction
- [ ] Tone adjustment (formal/casual)
- [ ] Summarization
- [ ] Expansion
- [ ] Translation
- [ ] Build simple UI

---

#### Project 5: Structured Data Extractor
**Goal**: Extract structured data from text

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | LLM + JSON mode |
| Skills | Output parsing, schemas |

**Tasks**:
- [ ] Define output schema
- [ ] Extract entities from text
- [ ] Output as JSON
- [ ] Validate output
- [ ] Handle edge cases

```python
import ollama
import json

schema = {
    "name": "string",
    "email": "string",
    "company": "string",
    "role": "string"
}

text = "Hi, I'm John Smith from Acme Corp. I'm the CTO. Reach me at john@acme.com"

response = ollama.chat(
    model="llama3.2",
    messages=[{
        "role": "user",
        "content": f"Extract info as JSON: {text}\nSchema: {schema}"
    }],
    format="json"
)
```

---

### Advanced Projects (L3-L4)

#### Project 6: Multi-Model Router
**Goal**: Route queries to best model

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 8-12 hours |
| Technologies | Multiple LLMs |
| Skills | Model selection, routing |

**Tasks**:
- [ ] Define model capabilities
- [ ] Classify incoming queries
- [ ] Route to appropriate model
- [ ] Compare costs/speed
- [ ] Implement fallbacks

**Architecture**:
```
Query → Classifier → Router → Best Model → Response
                        ↓
            ┌───────────┼───────────┐
            ▼           ▼           ▼
         Code LLM   General LLM   Math LLM
```

---

#### Project 7: Fine-tuned Specialist
**Goal**: Train model on custom data

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | Unsloth/Axolotl + QLoRA |
| Skills | Fine-tuning, datasets |

**Tasks**:
- [ ] Prepare training dataset
- [ ] Format as conversations
- [ ] Configure QLoRA training
- [ ] Train on GPU
- [ ] Evaluate quality
- [ ] Merge and export

---

#### Project 8: LLM Evaluation Suite
**Goal**: Compare models systematically

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | Multiple LLMs |
| Skills | Evaluation, benchmarking |

**Tasks**:
- [ ] Design test cases
- [ ] Run same prompts on all models
- [ ] Measure accuracy, speed, cost
- [ ] Generate comparison report
- [ ] Visualize results

---

## Framework Comparisons

### Comparison 1: Local LLM Showdown

**Question**: Which local model for your hardware?

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| **Llama 3.2 1B** | 1.3 GB | 2 GB | ⚡⚡⚡⚡ | ⭐⭐ |
| **Llama 3.2 3B** | 2 GB | 4 GB | ⚡⚡⚡ | ⭐⭐⭐ |
| **Phi-3 mini** | 2 GB | 4 GB | ⚡⚡⚡ | ⭐⭐⭐ |
| **Mistral 7B** | 4 GB | 8 GB | ⚡⚡ | ⭐⭐⭐⭐ |
| **Llama 3.1 8B** | 4.7 GB | 8 GB | ⚡⚡ | ⭐⭐⭐⭐ |
| **Qwen 2.5 7B** | 4 GB | 8 GB | ⚡⚡ | ⭐⭐⭐⭐ |
| **DeepSeek-R1 8B** | 4.9 GB | 8 GB | ⚡⚡ | ⭐⭐⭐⭐ |

**Lab Exercise**: Run same prompts through all, compare quality and speed.

```bash
# Test all models
for model in llama3.2:3b phi3 mistral qwen2.5:7b; do
    echo "Testing $model..."
    time ollama run $model "Explain quantum computing in 3 sentences"
done
```

---

### Comparison 2: Cloud API Battle

**Question**: Which API for production?

| Provider | Model | Speed | Cost (1M tokens) | Best For |
|----------|-------|-------|------------------|----------|
| **OpenAI** | GPT-4o | Fast | $5/$15 | General |
| **OpenAI** | GPT-4o-mini | Fastest | $0.15/$0.60 | Cost |
| **Anthropic** | Claude 3.5 Sonnet | Fast | $3/$15 | Code, long context |
| **Google** | Gemini 2.0 Flash | Fastest | $0.075/$0.30 | Multimodal |
| **Mistral** | Large | Fast | $2/$6 | Europe, efficiency |

**Lab Exercise**: Compare response quality on same task across providers.

---

### Comparison 3: Serving Solutions

**Question**: How to serve LLMs in production?

| Solution | Throughput | Ease | Features | Best For |
|----------|------------|------|----------|----------|
| **Ollama** | Low | Easiest | Basic | Development |
| **vLLM** | Highest | Medium | PagedAttention | Production |
| **TGI** | High | Medium | HuggingFace | HF models |
| **llama.cpp** | Medium | Medium | CPU support | Edge |
| **Triton** | Highest | Hard | Multi-model | Enterprise |

**Lab Exercise**: Benchmark same model with different servers.

---

### Comparison 4: Quantization Methods

**Question**: How much quality do you lose?

| Method | Bits | Size Reduction | Quality Loss |
|--------|------|----------------|--------------|
| **FP16** | 16 | 2x | ~0% |
| **Q8_0** | 8 | 4x | ~1% |
| **Q5_K_M** | 5 | 6x | ~3% |
| **Q4_K_M** | 4 | 8x | ~5% |
| **Q3_K_M** | 3 | 10x | ~10% |
| **Q2_K** | 2 | 16x | ~15% |

**Lab Exercise**: Run same prompts with different quantizations, measure quality.

---

## Hands-On Labs

### Lab 1: Local Chatbot (2 hours)
```
Install Ollama → Pull Model → Python API → Conversation Loop
```

### Lab 2: Simple RAG (4 hours)
```
Documents → Chunk → Embed → Store → Query → Answer
```

### Lab 3: Prompt Engineering (3 hours)
```
Bad Prompt → Analyze → Improve → Test → Iterate
```

### Lab 4: Model Comparison (4 hours)
```
Define Tasks → Run All Models → Measure → Compare → Report
```

### Lab 5: Fine-tuning (1 day)
```
Prepare Data → Format → Train QLoRA → Evaluate → Deploy
```

---

## Prompt Engineering Patterns

### Pattern 1: Role Assignment
```
You are an expert [ROLE] with deep knowledge in [DOMAIN].
Your task is to [TASK].
```

### Pattern 2: Few-Shot Examples
```
Here are examples:
Input: X1 → Output: Y1
Input: X2 → Output: Y2

Now process:
Input: X3 → Output: ?
```

### Pattern 3: Chain of Thought
```
Let's solve this step by step:
1. First, ...
2. Then, ...
3. Finally, ...
```

### Pattern 4: Output Format
```
Respond in JSON format:
{
  "answer": "...",
  "confidence": 0.0-1.0,
  "reasoning": "..."
}
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Functionality** | 40 | Does it work correctly? |
| **Prompt Quality** | 20 | Effective prompting |
| **Error Handling** | 15 | Graceful failures |
| **Documentation** | 15 | Clear instructions |
| **Innovation** | 10 | Creative extensions |

---

## Resources

- [Ollama](https://ollama.com/)
- [vLLM](https://docs.vllm.ai/)
- [LangChain](https://python.langchain.com/)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [Anthropic Docs](https://docs.anthropic.com/)

---

*Part of [Luno-AI](../../../README.md) | LLM Track*
