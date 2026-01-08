# Vector Database Integration

> **Semantic search and retrieval for RAG systems**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Databases optimized for storing and querying embedding vectors |
| **Why** | Enable semantic search, power RAG pipelines |
| **Key Metric** | Recall@K, Query latency, Scalability |

### Quick Comparison

| Database | Scale | Ease | Cost | Best For |
|----------|-------|------|------|----------|
| **Chroma** | Small | ⭐⭐⭐⭐⭐ | Free | Prototyping |
| **Qdrant** | Large | ⭐⭐⭐⭐ | Free/Paid | Production |
| **Weaviate** | Large | ⭐⭐⭐ | Free/Paid | Hybrid search |
| **Pinecone** | Huge | ⭐⭐⭐⭐ | Paid | Managed |
| **pgvector** | Medium | ⭐⭐⭐⭐ | Free | PostgreSQL users |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.9+ |
| **Embeddings** | OpenAI, Sentence Transformers, or similar |
| **RAM** | 4GB+ for small datasets |

---

## Quick Start: Chroma (10 min)

```bash
pip install chromadb sentence-transformers
```

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize
client = chromadb.Client()
collection = client.create_collection("docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add documents
docs = [
    "Python is a programming language",
    "JavaScript runs in browsers",
    "Machine learning uses neural networks"
]
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2", "doc3"]
)

# Query
query = "What language is used for AI?"
query_embedding = model.encode([query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=2)
print(results["documents"])
```

## Quick Start: Qdrant (15 min)

```bash
pip install qdrant-client sentence-transformers
# Or run Qdrant server: docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# Initialize (in-memory for testing)
client = QdrantClient(":memory:")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create collection
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Add documents
docs = ["Python for AI", "JavaScript for web", "Rust for performance"]
embeddings = model.encode(docs)

points = [
    PointStruct(id=i, vector=emb.tolist(), payload={"text": doc})
    for i, (doc, emb) in enumerate(zip(docs, embeddings))
]
client.upsert(collection_name="docs", points=points)

# Query
query_embedding = model.encode("programming language for machine learning")
results = client.search(
    collection_name="docs",
    query_vector=query_embedding.tolist(),
    limit=2
)
for r in results:
    print(f"{r.score:.3f}: {r.payload['text']}")
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Install Chroma
- [ ] Add documents with embeddings
- [ ] Query with semantic search
- [ ] Filter by metadata

### L1: RAG Integration (2-3 hours)
- [ ] Chunk documents properly
- [ ] Use with LangChain
- [ ] Implement hybrid search
- [ ] Add reranking

### L2: Production Setup (4-6 hours)
- [ ] Deploy Qdrant/Weaviate server
- [ ] Configure persistence
- [ ] Optimize indexing
- [ ] Add authentication

### L3: Advanced (1 day)
- [ ] Multi-tenancy
- [ ] Sharding and replication
- [ ] Custom distance metrics
- [ ] Batch operations

---

## Code Examples

### LangChain Integration

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Create vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./db")

# Query
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("How does X work?")
```

### Hybrid Search (Qdrant)

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Combine vector search with filters
results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(key="category", match=MatchValue(value="tutorial"))
        ]
    ),
    limit=5
)
```

### Batch Upsert

```python
# Efficient batch insertion
BATCH_SIZE = 100
for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    embeddings = model.encode([d["text"] for d in batch])
    points = [
        PointStruct(id=d["id"], vector=emb.tolist(), payload=d)
        for d, emb in zip(batch, embeddings)
    ]
    client.upsert(collection_name="docs", points=points)
```

---

## Embedding Models

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good |
| `nomic-embed-text` | 768 | Fast | Better |
| `bge-large-en` | 1024 | Medium | Excellent |
| `text-embedding-3-small` | 1536 | API | Excellent |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Poor recall | Try different embedding model, increase k |
| Slow queries | Add HNSW index, reduce vector dimensions |
| OOM errors | Use disk-based storage, batch operations |
| Inconsistent results | Normalize vectors, check distance metric |

---

## Resources

- [Chroma Docs](https://docs.trychroma.com/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Weaviate Docs](https://weaviate.io/developers/weaviate)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

*Part of [Luno-AI](../../README.md) | Agentic AI Track*
