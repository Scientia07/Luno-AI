# CLIP Integration

> **Connect images and text in shared embedding space**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Vision-language model mapping images and text to same space |
| **Why** | Zero-shot classification, image search, multimodal AI |
| **Creator** | OpenAI (2021), now widely available |
| **Key Use** | Image search by text, image classification without training |

### Capabilities
- Zero-shot image classification
- Text-to-image search
- Image-to-image similarity
- Image clustering

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **GPU** | Recommended (4GB+ VRAM) |
| **RAM** | 4GB+ |

---

## Quick Start (10 min)

### OpenAI CLIP

```bash
pip install torch torchvision transformers pillow
```

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image
image = Image.open("photo.jpg")

# Zero-shot classification
labels = ["a dog", "a cat", "a car", "a house"]
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.2%}")
```

### SigLIP (Improved CLIP)

```python
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Same usage pattern as CLIP
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Install CLIP
- [ ] Zero-shot image classification
- [ ] Text-to-image matching
- [ ] Compare CLIP variants

### L1: Image Search (2-3 hours)
- [ ] Embed image collection
- [ ] Search by text query
- [ ] Search by image similarity
- [ ] Build simple search UI

### L2: Advanced (4-6 hours)
- [ ] Fine-tune on custom data
- [ ] Combine with other models
- [ ] Multi-modal RAG
- [ ] Image clustering

---

## Code Examples

### Image Search Engine

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from pathlib import Path

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Index images
image_dir = Path("images/")
image_paths = list(image_dir.glob("*.jpg"))
image_embeddings = []

for path in image_paths:
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    image_embeddings.append(embedding.numpy().flatten())

image_embeddings = np.array(image_embeddings)

# Search by text
def search(query, top_k=5):
    inputs = processor(text=query, return_tensors="pt")
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).numpy().flatten()

    similarities = np.dot(image_embeddings, text_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(image_paths[i], similarities[i]) for i in top_indices]

results = search("a sunset over mountains")
for path, score in results:
    print(f"{score:.3f}: {path}")
```

### Image Similarity

```python
def find_similar_images(query_image_path, top_k=5):
    query_image = Image.open(query_image_path)
    inputs = processor(images=query_image, return_tensors="pt")

    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs).numpy().flatten()

    similarities = np.dot(image_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [(image_paths[i], similarities[i]) for i in top_indices]
```

### With Vector Database

```python
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

# Use CLIP with ChromaDB
embedding_fn = OpenCLIPEmbeddingFunction()
client = chromadb.Client()
collection = client.create_collection("images", embedding_function=embedding_fn)

# Add images (as base64 or URIs)
collection.add(
    ids=["img1", "img2"],
    images=["image1.jpg", "image2.jpg"]  # Paths or base64
)

# Search by text
results = collection.query(query_texts=["sunset"], n_results=5)
```

---

## Model Variants

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `clip-vit-base-patch32` | 400MB | Fast | Good |
| `clip-vit-large-patch14` | 1.7GB | Medium | Better |
| `siglip-base-patch16-224` | 400MB | Fast | Better |
| `siglip-large-patch16-384` | 1.1GB | Medium | Best |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Use smaller model, reduce batch size |
| Poor matching | Try different model, improve text prompts |
| Slow inference | Use GPU, batch processing |

---

## Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [HuggingFace CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224)

---

*Part of [Luno-AI](../../README.md) | Visual AI Track*
