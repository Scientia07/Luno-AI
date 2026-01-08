# AI Cognition: Projects & Comparisons

> **Hands-on projects and framework comparisons for Reasoning, Memory, and Cognitive Architectures**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Chain-of-Thought Prompting
**Goal**: Implement reasoning through prompting

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | OpenAI API or Ollama |
| Skills | Prompt engineering, reasoning |

**Tasks**:
- [ ] Create baseline prompt
- [ ] Add "Let's think step by step"
- [ ] Compare accuracy on math problems
- [ ] Test on logic puzzles
- [ ] Measure reasoning improvement

**Starter Code**:
```python
import ollama

def solve_with_cot(problem):
    response = ollama.chat(
        model="llama3.2",
        messages=[{
            "role": "user",
            "content": f"""Solve this problem step by step:

{problem}

Let's think through this carefully:
1. First, I'll identify what we know
2. Then, I'll work through the logic
3. Finally, I'll give the answer

Reasoning:"""
        }]
    )
    return response["message"]["content"]

# Test
problem = "If John has 3 apples and gives half to Mary, how many does John have?"
print(solve_with_cot(problem))
```

---

#### Project 2: Working Memory Buffer
**Goal**: Implement conversation memory

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | Python + LLM |
| Skills | Context management |

**Tasks**:
- [ ] Implement sliding window memory
- [ ] Add summary-based memory
- [ ] Test conversation coherence
- [ ] Compare memory strategies
- [ ] Handle context overflow

---

### Intermediate Projects (L2)

#### Project 3: Retrieval-Augmented Generation
**Goal**: Ground LLM responses in knowledge

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | LangChain + ChromaDB |
| Skills | Embeddings, retrieval |

**Tasks**:
- [ ] Load documents
- [ ] Chunk and embed
- [ ] Store in vector DB
- [ ] Retrieve relevant context
- [ ] Generate grounded responses

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Store
vectorstore = Chroma.from_documents(chunks, embeddings)

# Query
results = vectorstore.similarity_search("What is X?", k=3)
```

---

#### Project 4: Self-Reflection Agent
**Goal**: Agent that evaluates and improves its responses

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | LangGraph |
| Skills | Meta-cognition, feedback loops |

**Tasks**:
- [ ] Generate initial response
- [ ] Evaluate response quality
- [ ] Identify improvements
- [ ] Regenerate with feedback
- [ ] Loop until satisfactory

---

#### Project 5: Planning Agent
**Goal**: Agent that plans before acting

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | LangGraph |
| Skills | Task decomposition, planning |

**Tasks**:
- [ ] Decompose goal into subtasks
- [ ] Order tasks by dependencies
- [ ] Execute step by step
- [ ] Handle failures and replan
- [ ] Report progress

---

### Advanced Projects (L3-L4)

#### Project 6: Long-Term Memory System
**Goal**: Persistent memory across sessions

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | Vector DB + Graph DB |
| Skills | Memory architecture, retrieval |

**Tasks**:
- [ ] Design memory schema
- [ ] Implement episodic memory (events)
- [ ] Implement semantic memory (facts)
- [ ] Implement procedural memory (skills)
- [ ] Query and consolidate memories
- [ ] Forget irrelevant information

**Architecture**:
```
User Input → Short-term Memory → Working Memory
                    ↓                  ↑
              Consolidation        Retrieval
                    ↓                  ↑
              Long-term Memory (Vector + Graph)
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
Episodic       Semantic       Procedural
(Events)       (Facts)        (Skills)
```

---

#### Project 7: Multi-Step Reasoning Engine
**Goal**: Complex reasoning with verification

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | LangGraph + Tools |
| Skills | Reasoning chains, verification |

**Tasks**:
- [ ] Parse complex questions
- [ ] Break into reasoning steps
- [ ] Execute each step with verification
- [ ] Handle uncertainty
- [ ] Provide confidence scores

---

#### Project 8: Cognitive Architecture Implementation
**Goal**: Build a complete cognitive system

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 weeks |
| Technologies | LangGraph + Multi-system |
| Skills | Architecture design, integration |

**Tasks**:
- [ ] Implement perception module
- [ ] Implement working memory
- [ ] Implement long-term memory
- [ ] Implement reasoning engine
- [ ] Implement action selection
- [ ] Integrate all components

---

## Framework Comparisons

### Comparison 1: Reasoning Techniques

**Question**: Which reasoning approach?

| Technique | Accuracy | Complexity | Use Case |
|-----------|----------|------------|----------|
| **Zero-shot** | Baseline | None | Simple tasks |
| **Few-shot** | +10-20% | Examples | Pattern matching |
| **Chain-of-Thought** | +20-40% | Prompt | Math, logic |
| **Tree-of-Thought** | +30-50% | High | Complex problems |
| **ReAct** | Varies | Medium | Tool use |

**Lab Exercise**: Compare accuracy on GSM8K math problems.

```python
# Reasoning technique comparison
techniques = {
    "zero-shot": "What is 23 + 45?",
    "cot": "What is 23 + 45? Let's think step by step.",
    "few-shot": """
Q: What is 12 + 34? A: 12 + 34 = 46
Q: What is 23 + 45? A:"""
}

for name, prompt in techniques.items():
    response = llm(prompt)
    print(f"{name}: {response}")
```

---

### Comparison 2: Memory Systems

**Question**: Which memory architecture?

| Type | Duration | Retrieval | Best For |
|------|----------|-----------|----------|
| **Buffer** | Session | Sequential | Short conversations |
| **Summary** | Session | Compressed | Long conversations |
| **Vector** | Persistent | Semantic | Knowledge retrieval |
| **Graph** | Persistent | Relational | Entity relationships |
| **Hybrid** | Both | Multi-modal | Production systems |

**Lab Exercise**: Compare memory retention across sessions.

---

### Comparison 3: Knowledge Grounding

**Question**: How to ground LLM responses?

| Method | Accuracy | Latency | Complexity |
|--------|----------|---------|------------|
| **RAG** | High | Medium | Low |
| **Fine-tuning** | High | Low | High |
| **Tool Use** | Very High | High | Medium |
| **Knowledge Graphs** | High | Medium | High |

**Lab Exercise**: Compare factual accuracy with different grounding methods.

---

### Comparison 4: Planning Paradigms

**Question**: How should agents plan?

| Paradigm | Flexibility | Robustness | Complexity |
|----------|-------------|------------|------------|
| **Reactive** | High | Low | Low |
| **Plan-then-Execute** | Low | Medium | Medium |
| **Interleaved** | High | High | Medium |
| **Hierarchical** | Medium | High | High |

**Lab Exercise**: Compare task completion rates on multi-step tasks.

---

## Hands-On Labs

### Lab 1: Chain-of-Thought (2 hours)
```
Baseline → Add CoT → Compare Accuracy → Analyze Failures
```

### Lab 2: RAG Pipeline (4 hours)
```
Documents → Embed → Store → Retrieve → Generate → Evaluate
```

### Lab 3: Self-Reflection (4 hours)
```
Generate → Critique → Improve → Regenerate → Compare
```

### Lab 4: Memory Systems (6 hours)
```
Design Schema → Implement Storage → Add Retrieval → Test Retention
```

### Lab 5: Cognitive Architecture (2 days)
```
Perception → Memory → Reasoning → Action → Integration
```

---

## Cognitive Patterns

### Pattern 1: Think-Act-Observe
```
Think: What should I do?
Act: Execute action
Observe: What happened?
Loop until goal achieved
```

### Pattern 2: Plan-Execute-Verify
```
Plan: Decompose into steps
Execute: Run each step
Verify: Check correctness
Replan if needed
```

### Pattern 3: Generate-Critique-Refine
```
Generate: Initial response
Critique: Identify weaknesses
Refine: Improve response
Repeat until satisfactory
```

### Pattern 4: Remember-Recall-Apply
```
Remember: Store important information
Recall: Retrieve when relevant
Apply: Use in current context
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Reasoning Quality** | 35 | Logical, correct reasoning |
| **Memory Effectiveness** | 25 | Appropriate recall |
| **Architecture** | 20 | Clean component design |
| **Error Handling** | 10 | Graceful failures |
| **Innovation** | 10 | Novel approaches |

---

## Resources

- [Chain-of-Thought Paper](https://arxiv.org/abs/2201.11903)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [MemGPT](https://github.com/cpacker/MemGPT)
- [Cognitive Architectures Survey](https://arxiv.org/abs/2309.02427)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

---

*Part of [Luno-AI](../../../README.md) | AI Cognition Track*
