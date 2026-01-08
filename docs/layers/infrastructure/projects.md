# AI Infrastructure: Projects & Comparisons

> **Hands-on projects and framework comparisons for ML Ops, Deployment, and Scaling**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Model Serving with FastAPI
**Goal**: Serve ML model as REST API

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | FastAPI + PyTorch |
| Skills | API design, model loading |

**Tasks**:
- [ ] Load trained model
- [ ] Create FastAPI endpoint
- [ ] Handle input validation
- [ ] Return predictions
- [ ] Add health check endpoint

**Starter Code**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()

# Load model once at startup
model = torch.load("model.pt")
model.eval()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    with torch.no_grad():
        output = model(request.text)
    return PredictResponse(
        prediction=output.argmax().item(),
        confidence=output.max().item()
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

#### Project 2: Docker Container for ML
**Goal**: Containerize ML application

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | Docker |
| Skills | Containerization, dependencies |

**Tasks**:
- [ ] Write Dockerfile
- [ ] Handle CUDA support
- [ ] Optimize image size
- [ ] Set up docker-compose
- [ ] Test locally

---

### Intermediate Projects (L2)

#### Project 3: Experiment Tracking
**Goal**: Track ML experiments systematically

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | MLflow or Weights & Biases |
| Skills | Logging, metrics, versioning |

**Tasks**:
- [ ] Set up tracking server
- [ ] Log hyperparameters
- [ ] Track metrics per epoch
- [ ] Store artifacts (models, plots)
- [ ] Compare experiments

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("image-classifier")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    })

    # Training loop
    for epoch in range(10):
        train_loss = train_epoch()
        val_acc = validate()

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_acc
        }, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

---

#### Project 4: GPU Cluster with Ray
**Goal**: Distribute training across GPUs

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | Ray |
| Skills | Distributed computing, scheduling |

**Tasks**:
- [ ] Set up Ray cluster
- [ ] Distribute hyperparameter search
- [ ] Parallelize data processing
- [ ] Monitor cluster dashboard
- [ ] Handle failures

---

#### Project 5: Model Optimization Pipeline
**Goal**: Optimize model for deployment

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | ONNX + TensorRT |
| Skills | Quantization, optimization |

**Tasks**:
- [ ] Export to ONNX
- [ ] Apply quantization (INT8)
- [ ] Optimize with TensorRT
- [ ] Benchmark speedup
- [ ] Validate accuracy retention

---

### Advanced Projects (L3-L4)

#### Project 6: Kubernetes ML Platform
**Goal**: Deploy on Kubernetes

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | K8s + KServe |
| Skills | Orchestration, scaling |

**Tasks**:
- [ ] Create Kubernetes cluster
- [ ] Deploy model with KServe
- [ ] Configure autoscaling
- [ ] Set up monitoring (Prometheus)
- [ ] Handle rolling updates

**Architecture**:
```
Client → Ingress → KServe → Model Pod (GPU)
                      ↓
              HorizontalPodAutoscaler
```

---

#### Project 7: Feature Store
**Goal**: Centralized feature management

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | Feast |
| Skills | Feature engineering, serving |

**Tasks**:
- [ ] Define feature definitions
- [ ] Compute features from raw data
- [ ] Store in offline/online stores
- [ ] Serve features in real-time
- [ ] Track feature versions

---

#### Project 8: CI/CD for ML
**Goal**: Automated ML pipeline

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | GitHub Actions + DVC |
| Skills | Automation, testing, deployment |

**Tasks**:
- [ ] Version data with DVC
- [ ] Automate training on push
- [ ] Run model tests
- [ ] Deploy if tests pass
- [ ] Monitor in production

---

## Framework Comparisons

### Comparison 1: Model Serving

**Question**: How to serve models in production?

| Framework | Speed | Ease | Features | Best For |
|-----------|-------|------|----------|----------|
| **FastAPI** | Fast | Easy | Flexible | Prototypes |
| **TorchServe** | Fast | Medium | PyTorch native | PyTorch |
| **Triton** | Fastest | Hard | Multi-framework | Enterprise |
| **TFServing** | Fast | Medium | TensorFlow native | TensorFlow |
| **KServe** | Fast | Medium | K8s native | Cloud |
| **vLLM** | Fastest | Medium | LLM optimized | LLMs |

**Lab Exercise**: Deploy same model with different frameworks.

```python
# Benchmark serving latency
import requests
import time

def benchmark(url, payload, n=100):
    times = []
    for _ in range(n):
        start = time.time()
        requests.post(url, json=payload)
        times.append(time.time() - start)
    return sum(times) / n
```

---

### Comparison 2: Experiment Tracking

**Question**: Which tool for experiment tracking?

| Tool | Self-hosted | Ease | Features | Cost |
|------|-------------|------|----------|------|
| **MLflow** | Yes | Easy | Full | Free |
| **Weights & Biases** | Partial | Easiest | Rich | Freemium |
| **Neptune** | No | Easy | Rich | Paid |
| **ClearML** | Yes | Medium | Full | Free |
| **Comet** | No | Easy | Good | Freemium |

**Lab Exercise**: Track same experiment with 3 tools.

---

### Comparison 3: Orchestration Platforms

**Question**: Which platform for ML pipelines?

| Platform | Ease | Scale | Features | Open Source |
|----------|------|-------|----------|-------------|
| **Airflow** | Medium | Large | General | Yes |
| **Prefect** | Easy | Large | Modern | Yes |
| **Kubeflow** | Hard | Huge | ML-native | Yes |
| **Dagster** | Easy | Large | Data-aware | Yes |
| **SageMaker** | Easy | Huge | Managed | No |

**Lab Exercise**: Build same pipeline with Airflow and Prefect.

---

### Comparison 4: GPU Cloud Providers

**Question**: Where to train models?

| Provider | $/hr (A100) | Availability | Ease | Best For |
|----------|-------------|--------------|------|----------|
| **Lambda Labs** | $1.10 | Medium | Easy | Training |
| **RunPod** | $1.50 | High | Easy | Flexible |
| **Vast.ai** | $0.80 | Variable | Medium | Cost |
| **Modal** | $2.00 | High | Easiest | Serverless |
| **AWS** | $4.00+ | High | Hard | Enterprise |
| **Google Cloud** | $3.50+ | High | Medium | TPUs |

**Lab Exercise**: Compare cost for 1 hour training job.

---

## Hands-On Labs

### Lab 1: FastAPI Serving (2 hours)
```
Train Model → Save → Load in FastAPI → Test Endpoint → Docker
```

### Lab 2: Experiment Tracking (3 hours)
```
Setup MLflow → Log Runs → Compare → Visualize → Select Best
```

### Lab 3: Model Optimization (4 hours)
```
PyTorch → ONNX → Quantize → TensorRT → Benchmark
```

### Lab 4: Docker + GPU (3 hours)
```
Dockerfile → CUDA Base → Install Deps → Build → Run with GPU
```

### Lab 5: Kubernetes Deployment (1 day)
```
K8s Cluster → Deploy → Autoscale → Monitor → Update
```

---

## Infrastructure Patterns

### Pattern 1: Model Serving Pipeline
```
Request → Preprocess → Inference → Postprocess → Response
```

### Pattern 2: A/B Testing
```
Traffic → Router → Model A (50%) / Model B (50%) → Log → Analyze
```

### Pattern 3: Shadow Deployment
```
Production Model → Response
        ↓
Shadow Model → Log (compare) → Don't return
```

### Pattern 4: Blue-Green Deployment
```
Blue (current) ← Traffic
Green (new) ← Test
Switch DNS → Green (current)
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Reliability** | 35 | Uptime, error handling |
| **Performance** | 25 | Latency, throughput |
| **Scalability** | 20 | Handles load increase |
| **Monitoring** | 10 | Observability, alerts |
| **Documentation** | 10 | Runbooks, README |

---

## Resources

- [FastAPI](https://fastapi.tiangolo.com/)
- [MLflow](https://mlflow.org/)
- [Kubeflow](https://www.kubeflow.org/)
- [NVIDIA Triton](https://developer.nvidia.com/nvidia-triton-inference-server)
- [Ray](https://www.ray.io/)

---

*Part of [Luno-AI](../../../README.md) | AI Infrastructure Track*
