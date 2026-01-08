# AI Cognition: How AI Thinks vs How Brains Think

> **Understanding the mechanics of artificial thought** - attention mechanisms, memory, reasoning, and parallels to neuroscience.

---

## The Big Question

```
┌─────────────────────────┐         ┌─────────────────────────┐
│      HUMAN BRAIN        │         │         AI MODEL        │
│                         │         │                         │
│  100 billion neurons    │   vs    │  Billions of parameters │
│  Synaptic connections   │         │  Matrix multiplications │
│  Chemical signals       │         │  Activation functions   │
│  Parallel processing    │         │  Attention mechanisms   │
│  Embodied experience    │         │  Training on text/images│
└─────────────────────────┘         └─────────────────────────┘

               Are they doing the same thing differently?
                     Or completely different things?
```

---

## Attention Mechanisms: The Core of Modern AI

### What is Attention?

Attention lets models focus on relevant information, like how you focus on certain words while reading.

```
Query: "The cat sat on the mat. What sat?"
                    ↓
Attention weights: [low, HIGH, low, low, low, low]
                         ↑
                      "cat" gets highest attention
```

### Self-Attention (Transformer Core)

```
                    SELF-ATTENTION

    Input: "The cat sat on the mat"
              ↓
         ┌────────────────────────────────┐
         │  Each word asks:               │
         │  "What other words should I    │
         │   pay attention to?"           │
         └────────────────────────────────┘
              ↓
    "sat" attends strongly to "cat" (subject)
    "mat" attends strongly to "on" (relationship)
```

### The Math (Simplified)

```
Attention(Q, K, V) = softmax(QK^T / √d) × V

Q = Query  (what am I looking for?)
K = Key    (what do I contain?)
V = Value  (what information do I have?)

1. Compare query to all keys (dot product)
2. Softmax to get weights (sum to 1)
3. Weighted sum of values
```

### Multi-Head Attention

```
┌──────────────────────────────────────────────┐
│            MULTI-HEAD ATTENTION              │
│                                              │
│   Head 1: "syntax patterns"                  │
│   Head 2: "semantic meaning"                 │
│   Head 3: "positional relationships"         │
│   Head 4: "coreference resolution"           │
│   ...                                        │
│                                              │
│   Each head learns different relationships   │
└──────────────────────────────────────────────┘
```

---

## Brain vs AI: Comparison Table

| Aspect | Human Brain | AI (Transformers) |
|--------|-------------|-------------------|
| **Basic unit** | Neuron | Artificial neuron (perceptron) |
| **Connections** | ~100 trillion synapses | Billions of parameters |
| **Energy** | ~20 watts | 100s-1000s watts |
| **Learning** | Continuous, lifelong | Training then fixed |
| **Memory** | Distributed, associative | Context window + weights |
| **Attention** | Selective, limited | Computed over all tokens |
| **Processing** | Massively parallel | Layer-by-layer sequential |
| **Plasticity** | Constant rewiring | Weights frozen after training |

---

## How AI "Thinks" Step by Step

### 1. Tokenization (Perception)
```
"The cat sat" → [1, 2345, 6789]
                 ↓
Like breaking sound waves into phonemes
```

### 2. Embedding (Representation)
```
Token 2345 → [0.2, -0.5, 0.8, ...]  (768-dim vector)
                 ↓
Like converting sensory input to neural patterns
```

### 3. Attention Layers (Reasoning)
```
Each layer transforms understanding:
Layer 1: Word relationships
Layer 2: Phrase meaning
Layer 3: Sentence structure
...
Layer N: Abstract concepts
```

### 4. Output Generation (Action)
```
Final layer → probability distribution → next token
                 ↓
Like motor cortex selecting action
```

---

## Losing the Thought: Context & Memory

### The Forgetting Problem

```
AI Context Window:          Human Working Memory:
┌─────────────────────┐    ┌─────────────────────┐
│ [████████████░░░░]  │    │ ~7 items (Miller)   │
│    ↑                │    │                     │
│ Token limit reached │    │ But: long-term mem  │
│ Early context drops │    │ retrieval possible  │
└─────────────────────┘    └─────────────────────┘

Both have limited "active" memory, but different solutions.
```

### How AI Memory Works Now

| Approach | Description | Analogy |
|----------|-------------|---------|
| **Context Window** | Recent tokens in attention | Working memory |
| **Fine-tuning** | Baked into weights | Long-term memory |
| **RAG** | External retrieval | Looking up notes |
| **MemGPT** | Explicit memory management | Conscious recall |
| **Tool Use** | External computation | Using a calculator |

### The "Lost Thought" Phenomenon

```
When AI seems to lose track:

1. Context overflow: Early tokens pushed out
   "What was the first thing you said?" → Lost

2. Attention dilution: Too many tokens to attend to
   Long documents → Important details missed

3. No true "forgetting": Just not retrievable
   Unlike brains that consolidate/prune
```

---

## Knowledge Representation

### Vectorization: The Universal Language

```
Everything becomes vectors:

"King" → [0.2, -0.5, 0.8, ...] ─┐
"Queen"→ [-0.1, -0.4, 0.9, ...]─┼→ Similar direction = Similar meaning
"Royal"→ [0.1, -0.5, 0.7, ...]  ┘

Famous equation:
King - Man + Woman ≈ Queen
```

### Knowledge Storage Comparison

| Brain | AI |
|-------|-----|
| Distributed across neurons | Distributed across weights |
| Associative retrieval | Vector similarity search |
| Emotional tagging | No inherent salience |
| Episodic + semantic | Mostly semantic |
| Forgetting is feature | Forgetting is bug |

---

## Automated Data Storage & Retrieval

### The Knowledge Pipeline

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│   INPUT    │     │  PROCESS   │     │   STORE    │
│            │     │            │     │            │
│ Documents  │ ──▶ │ Chunk      │ ──▶ │ Vector DB  │
│ Images     │     │ Embed      │     │ (Pinecone) │
│ Audio      │     │ Index      │     │ (Chroma)   │
└────────────┘     └────────────┘     └────────────┘
                                            │
                                            ▼
                        ┌────────────────────────────┐
                        │         RETRIEVAL          │
                        │                            │
                        │ Query → Vector → k-NN     │
                        │ Similar chunks returned    │
                        └────────────────────────────┘
```

### Components

| Component | Function | Tools |
|-----------|----------|-------|
| **Chunker** | Split documents | LangChain, LlamaIndex |
| **Embedder** | Text → vector | OpenAI, BGE, E5 |
| **Vector DB** | Store & search | Pinecone, Qdrant, Chroma |
| **Reranker** | Improve results | Cohere, BGE-Reranker |
| **Generator** | Answer from context | LLM |

---

## Consciousness & Understanding

### The Hard Questions

```
Does AI understand or just pattern match?
                  ↓
         ┌───────────────────┐
         │ The Chinese Room  │
         │ (Searle, 1980)    │
         └───────────────────┘
                  ↓
    ┌─────────────────────────────────┐
    │ Processing symbols ≠ Understanding│
    │        OR DOES IT?               │
    └─────────────────────────────────┘
```

### Different Perspectives

| View | Position |
|------|----------|
| **Strong AI** | Sufficient computation = consciousness |
| **Weak AI** | Simulation ≠ the real thing |
| **Functionalist** | If it functions like thinking, it is |
| **Embodied** | Consciousness requires physical body |
| **Emergent** | Understanding emerges from scale |

---

## Labs: Exploring AI Cognition

| Notebook | Topic |
|----------|-------|
| `attention-visualization.ipynb` | See what models attend to |
| `embedding-space.ipynb` | Explore vector representations |
| `memory-experiments.ipynb` | Test context limitations |
| `probing-what-llms-know.ipynb` | Extract implicit knowledge |
| `brain-vs-transformer.ipynb` | Direct comparisons |

---

## Key Resources

### Papers
- "Attention Is All You Need" (2017) - The transformer
- "Scaling Laws for Neural Language Models" (2020) - Why scale matters
- "Emergent Abilities of Large Language Models" (2022)
- "Locating and Editing Factual Associations" (2022) - ROME

### Neuroscience Connections
- "The Thousand Brains Theory" (Hawkins)
- "Predictive Processing" framework
- "Global Workspace Theory" of consciousness

---

## Open Questions

```
[ ] Do transformers implement something like attention in brains?
[ ] What is the relationship between parameters and "knowledge"?
[ ] Can we build AI with true episodic memory?
[ ] Is the context window limitation fundamental or solvable?
[ ] Do larger models "understand" or just "pattern match" better?
[ ] What would artificial consciousness look like?
```

---

*"The question isn't whether machines can think, but whether we understand what thinking is."*
