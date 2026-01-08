# Docling Document Processing Integration

> **Convert documents to AI-ready formats**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | IBM's document processing library |
| **Why** | High-quality extraction from PDFs, Office docs |
| **Output** | Markdown, JSON, structured data |
| **Best For** | RAG pipelines, document understanding |

### Supported Formats

| Format | Support |
|--------|---------|
| PDF | ✓ Full |
| DOCX | ✓ Full |
| PPTX | ✓ Full |
| XLSX | ✓ Full |
| Images | ✓ OCR |
| HTML | ✓ Full |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ |
| **Memory** | 4GB+ RAM |

---

## Quick Start (10 min)

```bash
pip install docling
```

```python
from docling.document_converter import DocumentConverter

# Initialize converter
converter = DocumentConverter()

# Convert PDF to structured format
result = converter.convert("document.pdf")

# Get markdown output
markdown = result.document.export_to_markdown()
print(markdown)

# Get structured data
for element in result.document.iterate_items():
    print(f"Type: {element.type}, Text: {element.text[:100]}...")
```

---

## Learning Path

### L0: Basic Conversion (1 hour)
- [ ] Install Docling
- [ ] Convert PDF to markdown
- [ ] Extract tables
- [ ] Handle images

### L1: Advanced Processing (2-3 hours)
- [ ] Custom pipelines
- [ ] OCR configuration
- [ ] Metadata extraction
- [ ] Batch processing

### L2: RAG Integration (4-6 hours)
- [ ] Chunking strategies
- [ ] Embedding preparation
- [ ] Vector store loading
- [ ] Citation tracking

---

## Code Examples

### PDF Conversion

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat

converter = DocumentConverter()

# Convert with options
result = converter.convert(
    "research_paper.pdf",
    input_format=InputFormat.PDF
)

# Access document structure
doc = result.document

# Get full markdown
print(doc.export_to_markdown())

# Get JSON structure
import json
print(json.dumps(doc.export_to_dict(), indent=2))
```

### Extract Tables

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("report.pdf")

# Find tables
tables = []
for item in result.document.iterate_items():
    if item.type == "table":
        tables.append(item)

# Convert tables to pandas
import pandas as pd

for i, table in enumerate(tables):
    df = table.to_dataframe()
    print(f"\nTable {i+1}:")
    print(df)
    df.to_csv(f"table_{i+1}.csv", index=False)
```

### Batch Processing

```python
from docling.document_converter import DocumentConverter
from pathlib import Path
import concurrent.futures

def process_document(file_path: str):
    """Process single document"""
    converter = DocumentConverter()
    try:
        result = converter.convert(file_path)
        return {
            "file": file_path,
            "status": "success",
            "content": result.document.export_to_markdown()
        }
    except Exception as e:
        return {
            "file": file_path,
            "status": "error",
            "error": str(e)
        }

def batch_process(directory: str, pattern: str = "*.pdf"):
    """Process all documents in directory"""
    files = list(Path(directory).glob(pattern))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_document, str(f)): f for f in files}

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results

# Usage
results = batch_process("./documents/", "*.pdf")
for r in results:
    print(f"{r['file']}: {r['status']}")
```

### RAG Pipeline Integration

```python
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def prepare_for_rag(file_path: str):
    """Convert document and prepare for RAG"""
    # Convert document
    converter = DocumentConverter()
    result = converter.convert(file_path)

    # Get markdown with source tracking
    chunks = []
    for item in result.document.iterate_items():
        if item.text:
            chunks.append({
                "content": item.text,
                "type": item.type,
                "page": getattr(item, 'page_number', None),
                "source": file_path
            })

    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    documents = []
    for chunk in chunks:
        splits = splitter.split_text(chunk["content"])
        for split in splits:
            documents.append({
                "content": split,
                "metadata": {
                    "source": chunk["source"],
                    "type": chunk["type"],
                    "page": chunk["page"]
                }
            })

    return documents

def create_vector_store(documents: list):
    """Create vector store from documents"""
    texts = [d["content"] for d in documents]
    metadatas = [d["metadata"] for d in documents]

    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )

    return vectorstore

# Usage
docs = prepare_for_rag("report.pdf")
vectorstore = create_vector_store(docs)

# Query
results = vectorstore.similarity_search("What are the key findings?", k=3)
for r in results:
    print(f"[Page {r.metadata['page']}] {r.page_content[:200]}...")
```

### Custom Pipeline

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions

# Configure pipeline
options = PipelineOptions(
    do_ocr=True,
    ocr_lang="eng",
    do_table_structure=True,
    do_figure_classification=True
)

converter = DocumentConverter(pipeline_options=options)
result = converter.convert("scanned_document.pdf")
```

### Extract Images

```python
from docling.document_converter import DocumentConverter
from pathlib import Path

converter = DocumentConverter()
result = converter.convert("document_with_images.pdf")

# Create output directory
output_dir = Path("extracted_images")
output_dir.mkdir(exist_ok=True)

# Extract images
for i, item in enumerate(result.document.iterate_items()):
    if item.type == "figure":
        if hasattr(item, 'image'):
            image_path = output_dir / f"image_{i}.png"
            item.image.save(image_path)
            print(f"Saved: {image_path}")

            # Get caption if available
            if hasattr(item, 'caption'):
                print(f"Caption: {item.caption}")
```

### Export Formats

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")
doc = result.document

# Markdown
markdown = doc.export_to_markdown()
with open("output.md", "w") as f:
    f.write(markdown)

# JSON
import json
json_data = doc.export_to_dict()
with open("output.json", "w") as f:
    json.dump(json_data, f, indent=2)

# Plain text
text = doc.export_to_text()
with open("output.txt", "w") as f:
    f.write(text)
```

---

## Pipeline Options

| Option | Default | Description |
|--------|---------|-------------|
| `do_ocr` | False | Enable OCR |
| `ocr_lang` | "eng" | OCR language |
| `do_table_structure` | True | Parse tables |
| `do_figure_classification` | True | Classify figures |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OCR not working | Install tesseract-ocr |
| Tables wrong | Check PDF structure |
| Slow processing | Disable unused features |
| Memory issues | Process in batches |

---

## Resources

- [Docling GitHub](https://github.com/DS4SD/docling)
- [Documentation](https://ds4sd.github.io/docling/)
- [Blog Post](https://research.ibm.com/blog/docling-documents-to-ai)
- [Examples](https://github.com/DS4SD/docling/tree/main/examples)

---

*Part of [Luno-AI](../../README.md) | Specialized Track*
