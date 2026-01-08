# RAG Pipeline Integration

> **Category**: Agentic AI
> **Difficulty**: Intermediate
> **Setup Time**: 4-6 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
RAG (Retrieval Augmented Generation) enhances LLMs by giving them access to external knowledge. Instead of relying only on training data, the LLM retrieves relevant documents before generating answers.

### Why Use It
- **Up-to-date Info**: Access current information beyond training cutoff
- **Domain Knowledge**: Add specialized documents (docs, code, policies)
- **Reduced Hallucination**: Ground answers in real sources
- **Source Citations**: Know where answers come from
- **Cost Effective**: Smaller LLM + good retrieval > larger LLM

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Document Q&A | Answer questions from your documents |
| Code Search | Find relevant code snippets |
| Knowledge Base | Build searchable company knowledge |
| Hybrid Search | Combine semantic + keyword search |
| Multi-modal | Images, PDFs, tables |

### RAG Pipeline
```
Documents → Chunking → Embedding → Vector Store
                                        ↓
User Query → Embedding → Search → Retrieved Context
                                        ↓
                              LLM + Context → Answer
```

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU embeddings) | NVIDIA 4GB+ |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 20 GB |

### Software Dependencies
```bash
# Core
pip install langchain langchain-openai langchain-community

# Vector stores (pick one or more)
pip install chromadb         # Local, easy
pip install qdrant-client    # Production
pip install pinecone-client  # Cloud

# Document loaders
pip install pypdf            # PDFs
pip install unstructured     # Various formats
pip install docling          # Advanced PDF/DOCX

# Embeddings
pip install sentence-transformers  # Local embeddings
```

### Prior Knowledge
- [x] Python basics
- [x] LLM concepts
- [ ] Vector databases (helpful)

---

## Quick Start (20 minutes)

### 1. Install
```bash
pip install langchain langchain-openai chromadb pypdf
```

### 2. Basic RAG
```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load document
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)

# Create QA chain
llm = ChatOpenAI(model="gpt-4o-mini")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
)

# Ask questions
answer = qa.invoke("What is the main topic of this document?")
print(answer["result"])
```

### 3. Verify
```python
# Test retrieval
results = vectorstore.similarity_search("your query", k=3)
for doc in results:
    print(f"- {doc.page_content[:100]}...")
```

---

## Full Setup

### Document Loading

```python
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)

# Single PDF
loader = PyPDFLoader("file.pdf")

# Directory of PDFs
loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)

# Markdown files
loader = UnstructuredMarkdownLoader("README.md")

# Web pages
loader = WebBaseLoader(["https://example.com/page1", "https://example.com/page2"])

documents = loader.load()
```

### Chunking Strategies

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Recursive (recommended default)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

# Token-based (for LLM context limits)
splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# Markdown-aware
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
)

chunks = splitter.split_documents(documents)
```

### Embedding Models

```python
# OpenAI (recommended for simplicity)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Local embeddings (free, private)
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Ollama embeddings
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

### Vector Stores

```python
# Chroma (local, easy)
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

# Qdrant (production)
from langchain_qdrant import QdrantVectorStore
vectorstore = QdrantVectorStore.from_documents(
    chunks,
    embeddings,
    url="http://localhost:6333",
    collection_name="my_docs",
)

# Pinecone (cloud)
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore.from_documents(
    chunks,
    embeddings,
    index_name="my-index",
)
```

---

## Learning Path

### L0: Basic Q&A (1 hour)
**Goal**: Answer questions from documents

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Simple setup
loader = TextLoader("document.txt")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# Query
retriever = vectorstore.as_retriever()
docs = retriever.invoke("What is the main idea?")
print(docs[0].page_content)
```

### L1: Custom RAG Chain (2 hours)
**Goal**: Build customized retrieval pipeline

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Custom prompt
template = """Answer the question based only on the following context:

{context}

Question: {question}

If you cannot answer from the context, say "I don't have enough information."
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = chain.invoke("What are the key points?")
print(answer)
```

### L2: Hybrid Search & Reranking (3 hours)
**Goal**: Improve retrieval quality

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# BM25 (keyword) retriever
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

# Vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Hybrid: combine both
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7],  # Weight towards semantic
)

# Add reranking
reranker = CohereRerank(model="rerank-english-v3.0", top_n=3)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever,
)

results = retriever.invoke("query")
```

### L3: Production RAG (4+ hours)
**Goal**: Scalable, monitored system

```python
from langchain.callbacks import LangChainTracer
from langchain_core.runnables import RunnableConfig

# Add tracing
config = RunnableConfig(callbacks=[LangChainTracer()])

# Async for production
async def query_rag(question: str):
    docs = await retriever.ainvoke(question)
    answer = await chain.ainvoke(question, config=config)
    return {
        "answer": answer,
        "sources": [doc.metadata.get("source") for doc in docs],
    }

# With caching
from langchain.cache import SQLiteCache
import langchain
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
```

---

## Code Examples

### Example 1: Multi-Document RAG
```python
from langchain_community.document_loaders import DirectoryLoader

# Load multiple document types
pdf_loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
md_loader = DirectoryLoader("./docs/", glob="**/*.md", loader_cls=TextLoader)

all_docs = pdf_loader.load() + md_loader.load()
chunks = splitter.split_documents(all_docs)

# Add source metadata
for chunk in chunks:
    chunk.metadata["type"] = "internal_docs"

vectorstore = Chroma.from_documents(chunks, embeddings)
```

### Example 2: Conversational RAG
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

qa = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

# Multi-turn conversation
qa.invoke({"question": "What is the document about?"})
qa.invoke({"question": "Can you elaborate on that?"})  # Uses context
```

### Example 3: RAG with Citations
```python
from langchain_core.prompts import ChatPromptTemplate

template = """Answer based on the context. Include [Source: X] citations.

Context:
{context}

Question: {question}

Answer with citations:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs_with_sources(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", f"Doc {i}")
        formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

chain = (
    {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Example 4: Code Repository RAG
```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load code files
loader = DirectoryLoader(
    "./src/",
    glob="**/*.py",
    loader_cls=TextLoader,
)
code_docs = loader.load()

# Use code-aware splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
)

chunks = python_splitter.split_documents(code_docs)
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| LangGraph | Agentic RAG | [langgraph.md](./langgraph.md) |
| Ollama | Local LLM | [ollama.md](../llms/ollama.md) |
| Docling | PDF parsing | [docling.md](../specialized/docling.md) |
| Vector DBs | Storage | [vector-db.md](./vector-db.md) |

### Embedding Models Comparison
| Model | Dimensions | Quality | Speed |
|-------|------------|---------|-------|
| text-embedding-3-small | 1536 | Good | Fast |
| text-embedding-3-large | 3072 | Better | Medium |
| BGE-small-en | 384 | Good | Fast |
| BGE-large-en | 1024 | Better | Medium |
| nomic-embed-text | 768 | Good | Fast |

### Vector Store Comparison
| Store | Type | Best For |
|-------|------|----------|
| Chroma | Local | Development, small scale |
| Qdrant | Self-host/Cloud | Production |
| Pinecone | Cloud | Managed, scalable |
| Weaviate | Self-host/Cloud | Hybrid search |
| FAISS | Local | Research, speed |

---

## Troubleshooting

### Common Issues

#### Issue 1: Poor Retrieval Quality
**Symptoms**: Wrong documents retrieved
**Solution**:
```python
# Increase k
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Use MMR for diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20},
)

# Better chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks
    chunk_overlap=100,
)
```

#### Issue 2: Context Window Exceeded
**Symptoms**: LLM errors about max tokens
**Solution**:
```python
# Limit retrieved docs
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Compress documents
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
```

#### Issue 3: Slow Queries
**Symptoms**: High latency
**Solution**:
```python
# Use local embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Enable caching
import langchain
langchain.llm_cache = SQLiteCache()

# Async processing
docs = await retriever.ainvoke(query)
```

---

## Resources

### Official
- [LangChain RAG Docs](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB](https://docs.trychroma.com/)
- [Qdrant](https://qdrant.tech/documentation/)

### Tutorials
- [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch)
- [Advanced RAG](https://github.com/NirDiamant/RAG_Techniques)

---

*Part of [Luno-AI Integration Hub](../_index.md) | Agentic AI Track*
