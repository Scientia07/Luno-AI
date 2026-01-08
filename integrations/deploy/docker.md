# Docker for ML Integration

> **Containerize ML models for reproducible deployment**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Container platform for packaging ML applications |
| **Why** | Reproducibility, isolation, easy deployment |
| **Key Concepts** | Images, containers, volumes, GPU support |

### Why Docker for ML?

- **Reproducibility**: Same environment everywhere
- **Isolation**: No dependency conflicts
- **Portability**: Run anywhere with Docker
- **GPU Support**: nvidia-docker for CUDA

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Docker** | 20.10+ |
| **NVIDIA Container Toolkit** | For GPU support |
| **Disk Space** | 10GB+ for ML images |

---

## Quick Start (15 min)

### Basic ML Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t my-ml-app .

# Run container
docker run -p 8000:8000 my-ml-app

# Run with GPU
docker run --gpus all -p 8000:8000 my-ml-app
```

---

## Learning Path

### L0: Basics (1-2 hours)
- [ ] Create Dockerfile
- [ ] Build and run container
- [ ] Mount volumes
- [ ] View logs

### L1: GPU Support (2-3 hours)
- [ ] Install nvidia-docker
- [ ] Use CUDA base images
- [ ] Test GPU access
- [ ] Multi-stage builds

### L2: Production (4-6 hours)
- [ ] Docker Compose
- [ ] Image optimization
- [ ] Health checks
- [ ] Multi-container apps

---

## Code Examples

### PyTorch GPU Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install additional packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.pt .
COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
```

### Multi-stage Build (Smaller Image)

```dockerfile
# Build stage
FROM python:3.11 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

COPY . .

CMD ["python", "main.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  worker:
    build: .
    command: python worker.py
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Ollama in Docker

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  ollama_data:
```

### Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

---

## GPU Base Images

| Image | Size | CUDA | Use Case |
|-------|------|------|----------|
| `nvidia/cuda:12.1-runtime` | 2GB | 12.1 | Inference |
| `pytorch/pytorch:2.1.0-cuda12.1` | 6GB | 12.1 | PyTorch |
| `tensorflow/tensorflow:2.15.0-gpu` | 5GB | 11.8 | TensorFlow |
| `nvcr.io/nvidia/tritonserver` | 10GB | Various | Serving |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not found | Install nvidia-container-toolkit |
| Image too large | Use multi-stage build, slim base |
| Permission denied | Add user to docker group |
| Out of disk | `docker system prune` |

---

## Resources

- [Docker Docs](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

---

*Part of [Luno-AI](../../README.md) | Deploy Track*
