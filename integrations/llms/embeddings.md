# Embeddings Integration

> **Convert text to semantic vectors for search and RAG**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Dense vector representations of text meaning |
| **Why** | Enable semantic search, clustering, similarity |
| **Key Use** | RAG retrieval, document search, recommendations |

### Model Comparison

| Model | Dimensions | Speed | Quality | Cost |
|-------|------------|-------|---------|------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ | Free |
| `nomic-embed-text` | 768 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Free |
| `bge-large-en-v1.5` | 1024 | ⚡⚡ | ⭐⭐⭐⭐⭐ | Free |
| `text-embedding-3-small` | 1536 | ⚡⚡⚡ | ⭐⭐⭐⭐ | $0.02/1M |
| `text-embedding-3-large` | 3072 | ⚡⚡ | ⭐⭐⭐⭐⭐ | $0.13/1M |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.9+ |
| **RAM** | 2-8GB depending on model |
| **GPU** | Optional, speeds up inference |

---

## Quick Start (10 min)

### Sentence Transformers (Local)

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single text
embedding = model.encode("What is machine learning?")
print(f"Shape: {embedding.shape}")  # (384,)

# Batch
texts = ["Hello world", "Machine learning is AI", "Python programming"]
embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)

# Similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.3f}")
```

### OpenAI Embeddings (API)

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["What is machine learning?", "AI and neural networks"]
)

embeddings = [e.embedding for e in response.data]
print(f"Dimensions: {len(embeddings[0])}")  # 1536
```

### Ollama Embeddings (Local)

```bash
ollama pull nomic-embed-text
```

```python
import ollama

response = ollama.embed(
    model="nomic-embed-text",
    input=["What is machine learning?"]
)
embedding = response["embeddings"][0]
print(f"Dimensions: {len(embedding)}")  # 768
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Install sentence-transformers
- [ ] Generate embeddings for texts
- [ ] Calculate cosine similarity
- [ ] Compare embedding models

### L1: RAG Integration (2-3 hours)
- [ ] Chunk documents for embedding
- [ ] Store in vector database
- [ ] Implement semantic search
- [ ] Add to retrieval pipeline

### L2: Production (4-6 hours)
- [ ] Batch processing optimization
- [ ] GPU acceleration
- [ ] Caching embeddings
- [ ] Dimensionality reduction

### L3: Advanced (1 day)
- [ ] Fine-tune embedding models
- [ ] Matryoshka embeddings
- [ ] Multi-vector representations
- [ ] Cross-encoder reranking

---

## Code Examples

### Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Document corpus
documents = [
    "Python is great for data science",
    "JavaScript powers the web",
    "Machine learning uses neural networks",
    "SQL is used for databases"
]
doc_embeddings = model.encode(documents)

# Query
query = "What language is best for AI?"
query_embedding = model.encode(query)

# Find most similar
similarities = np.dot(doc_embeddings, query_embedding)
top_idx = np.argmax(similarities)
print(f"Best match: {documents[top_idx]}")
print(f"Score: {similarities[top_idx]:.3f}")
```

### LangChain Integration

```python
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

# Local (Ollama)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector = embeddings.embed_query("What is AI?")

# API (OpenAI)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectors = embeddings.embed_documents(["doc1", "doc2", "doc3"])
```

### Batch Processing

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Large dataset
documents = ["doc " + str(i) for i in range(10000)]

# Batch with progress
embeddings = model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
```

### Caching Embeddings

```python
import pickle
from pathlib import Path

CACHE_FILE = Path("embeddings_cache.pkl")

def get_embeddings(texts, model):
    # Load cache
    cache = {}
    if CACHE_FILE.exists():
        cache = pickle.loads(CACHE_FILE.read_bytes())

    # Find uncached
    uncached = [t for t in texts if t not in cache]

    # Embed uncached
    if uncached:
        new_embeddings = model.encode(uncached)
        for text, emb in zip(uncached, new_embeddings):
            cache[text] = emb
        CACHE_FILE.write_bytes(pickle.dumps(cache))

    return [cache[t] for t in texts]
```

---

## Choosing the Right Model

| Use Case | Recommended Model |
|----------|-------------------|
| **Quick prototype** | `all-MiniLM-L6-v2` |
| **Production (local)** | `bge-large-en-v1.5` |
| **Production (API)** | `text-embedding-3-small` |
| **Multilingual** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Code** | `text-embedding-3-large` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow embedding | Use GPU, batch processing |
| OOM error | Reduce batch size, use smaller model |
| Poor retrieval | Try different model, add reranking |
| Dimension mismatch | Check model consistency in pipeline |

---

## Resources

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [BGE Models](https://huggingface.co/BAAI)

---

*Part of [Luno-AI](../../README.md) | LLM Track*
