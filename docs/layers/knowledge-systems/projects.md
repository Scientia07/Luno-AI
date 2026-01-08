# Knowledge Systems: Projects & Comparisons

> **Hands-on projects and framework comparisons for Knowledge Graphs, RAG, and Information Systems**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Vector Search Engine
**Goal**: Build semantic search over documents

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | ChromaDB or Qdrant |
| Skills | Embeddings, similarity search |

**Tasks**:
- [ ] Load text documents
- [ ] Generate embeddings
- [ ] Store in vector DB
- [ ] Query with natural language
- [ ] Return ranked results

**Starter Code**:
```python
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize
client = chromadb.Client()
collection = client.create_collection("docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Add documents
docs = ["AI is transforming industries", "Machine learning enables automation"]
embeddings = model.encode(docs).tolist()
collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2"]
)

# Query
query_embedding = model.encode(["What is AI?"]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=2)
print(results)
```

---

#### Project 2: Simple Knowledge Graph
**Goal**: Build entity-relationship graph

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 3-4 hours |
| Technologies | NetworkX or Neo4j |
| Skills | Graph structures, queries |

**Tasks**:
- [ ] Define entities and relationships
- [ ] Build graph structure
- [ ] Add nodes and edges
- [ ] Query relationships
- [ ] Visualize graph

---

### Intermediate Projects (L2)

#### Project 3: RAG Pipeline
**Goal**: Retrieval-Augmented Generation system

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | LangChain + Vector DB |
| Skills | Chunking, retrieval, generation |

**Tasks**:
- [ ] Load and chunk documents
- [ ] Create embeddings
- [ ] Build retrieval pipeline
- [ ] Integrate with LLM
- [ ] Evaluate answer quality

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Build retriever
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Build QA chain
llm = Ollama(model="llama3.2")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Query
result = qa_chain({"query": "What is machine learning?"})
```

---

#### Project 4: Entity Extraction Pipeline
**Goal**: Extract entities and relationships from text

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | spaCy + LLM |
| Skills | NER, relation extraction |

**Tasks**:
- [ ] Extract named entities
- [ ] Identify relationships
- [ ] Build knowledge triples
- [ ] Store in graph database
- [ ] Query extracted knowledge

---

#### Project 5: Hybrid Search System
**Goal**: Combine keyword and semantic search

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | Elasticsearch + Vector DB |
| Skills | BM25, embeddings, fusion |

**Tasks**:
- [ ] Implement keyword search (BM25)
- [ ] Implement semantic search
- [ ] Combine with reciprocal rank fusion
- [ ] Tune weights
- [ ] Evaluate retrieval quality

---

### Advanced Projects (L3-L4)

#### Project 6: Knowledge Graph RAG
**Goal**: Combine KG and vector retrieval

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | Neo4j + LangChain |
| Skills | Graph queries, multi-hop reasoning |

**Tasks**:
- [ ] Build knowledge graph from documents
- [ ] Create vector index on nodes
- [ ] Query: vector → graph → expand
- [ ] Generate with graph context
- [ ] Compare to pure RAG

**Architecture**:
```
Query → Vector Search → Graph Traversal → Context → LLM → Answer
              ↓               ↓
         Find nodes      Expand relationships
```

---

#### Project 7: Multi-Modal Knowledge Base
**Goal**: Store and retrieve images + text

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | CLIP + Vector DB |
| Skills | Multi-modal embeddings |

**Tasks**:
- [ ] Embed images with CLIP
- [ ] Embed text descriptions
- [ ] Joint vector space
- [ ] Text-to-image retrieval
- [ ] Image-to-text retrieval

---

#### Project 8: Self-Updating Knowledge Base
**Goal**: Automatically maintain knowledge

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | Web scraping + RAG |
| Skills | Automation, change detection |

**Tasks**:
- [ ] Monitor web sources
- [ ] Detect content changes
- [ ] Update embeddings incrementally
- [ ] Handle conflicts
- [ ] Version knowledge over time

---

## Framework Comparisons

### Comparison 1: Vector Databases

**Question**: Which vector DB for your needs?

| Database | Speed | Scale | Ease | Features | Open Source |
|----------|-------|-------|------|----------|-------------|
| **Chroma** | Fast | Small | Easiest | Basic | Yes |
| **Qdrant** | Fast | Large | Easy | Rich | Yes |
| **Weaviate** | Fast | Large | Medium | Hybrid | Yes |
| **Pinecone** | Fastest | Huge | Easy | Managed | No |
| **Milvus** | Fast | Huge | Medium | Enterprise | Yes |
| **pgvector** | Medium | Medium | Easy | PostgreSQL | Yes |

**Lab Exercise**: Benchmark query latency and recall on same dataset.

```python
# Comparison setup
import chromadb
from qdrant_client import QdrantClient

# Test both on same data
# Measure: insert time, query time, recall@k
```

---

### Comparison 2: Embedding Models

**Question**: Which embeddings for your task?

| Model | Dimensions | Speed | Quality | Size |
|-------|------------|-------|---------|------|
| **all-MiniLM-L6-v2** | 384 | Fast | Good | 80MB |
| **nomic-embed-text** | 768 | Fast | Better | 250MB |
| **bge-large-en** | 1024 | Medium | Excellent | 1.3GB |
| **text-embedding-3-small** | 1536 | API | Excellent | - |
| **voyage-large-2** | 1024 | API | Best | - |

**Lab Exercise**: Compare retrieval accuracy on SQuAD.

---

### Comparison 3: Document Chunking

**Question**: How to split documents?

| Strategy | Coherence | Overlap | Best For |
|----------|-----------|---------|----------|
| **Fixed Size** | Low | Configurable | Simple docs |
| **Sentence** | Medium | None | Articles |
| **Paragraph** | High | None | Structured |
| **Recursive** | High | Configurable | Code/mixed |
| **Semantic** | Highest | Dynamic | Complex docs |

**Lab Exercise**: Compare retrieval quality with different chunking.

---

### Comparison 4: Graph Databases

**Question**: Which graph DB for knowledge?

| Database | Query Language | Scale | Ease | Best For |
|----------|---------------|-------|------|----------|
| **Neo4j** | Cypher | Large | Medium | General |
| **NetworkX** | Python | Small | Easy | Prototyping |
| **ArangoDB** | AQL | Large | Medium | Multi-model |
| **Amazon Neptune** | Gremlin | Huge | Medium | Cloud |

**Lab Exercise**: Build same KG in Neo4j and NetworkX.

---

## Hands-On Labs

### Lab 1: Vector Search (2 hours)
```
Documents → Embed → Store → Query → Rank Results
```

### Lab 2: RAG Pipeline (4 hours)
```
Load Docs → Chunk → Embed → Retrieve → Generate → Evaluate
```

### Lab 3: Knowledge Graph (4 hours)
```
Extract Entities → Find Relations → Build Graph → Query → Visualize
```

### Lab 4: Hybrid Search (4 hours)
```
BM25 Index → Vector Index → Fusion → Evaluate Recall
```

### Lab 5: GraphRAG (6 hours)
```
Build KG → Index Nodes → Query → Traverse → Generate
```

---

## Knowledge Engineering Patterns

### Pattern 1: Chunking Pipeline
```
Document → Split → Clean → Embed → Store → Index
```

### Pattern 2: Retrieval Pipeline
```
Query → Embed → Search → Rerank → Filter → Return
```

### Pattern 3: Knowledge Triple
```
(Subject, Predicate, Object)
("Einstein", "won", "Nobel Prize")
```

### Pattern 4: Query Expansion
```
Query → Synonyms → Related Terms → Multi-Query → Merge Results
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Retrieval Quality** | 35 | Relevant results returned |
| **System Design** | 25 | Clean architecture |
| **Scalability** | 20 | Handles growth |
| **Documentation** | 10 | Clear explanations |
| **Innovation** | 10 | Novel approaches |

---

## Resources

- [ChromaDB](https://www.trychroma.com/)
- [Qdrant](https://qdrant.tech/)
- [LangChain RAG](https://python.langchain.com/docs/tutorials/rag/)
- [Neo4j](https://neo4j.com/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding benchmarks

---

*Part of [Luno-AI](../../../README.md) | Knowledge Systems Track*
