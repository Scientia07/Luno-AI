# Triton Inference Server Integration

> **High-performance model serving at scale**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | NVIDIA's production inference server |
| **Why** | Multi-model, multi-framework, high throughput |
| **Supports** | ONNX, TensorRT, PyTorch, TensorFlow |
| **Best For** | Production ML systems at scale |

### Triton vs Alternatives

| Feature | Triton | TorchServe | TF Serving |
|---------|--------|------------|------------|
| Multi-framework | ✓ | PyTorch | TensorFlow |
| Dynamic batching | ✓ | ✓ | ✓ |
| Model ensemble | ✓ | Limited | Limited |
| GPU optimization | Excellent | Good | Good |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA GPU recommended |
| **Docker** | Required |
| **Models** | ONNX, TensorRT, or framework format |

---

## Quick Start (30 min)

### Docker Setup

```bash
# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# Create model repository
mkdir -p model_repository/my_model/1/

# Run Triton
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

### Model Repository Structure

```
model_repository/
├── my_model/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
└── another_model/
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

### Config File (config.pbtxt)

```protobuf
name: "my_model"
backend: "onnxruntime"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]
```

---

## Learning Path

### L0: Basic Setup (2-3 hours)
- [ ] Install Triton via Docker
- [ ] Deploy ONNX model
- [ ] Test with HTTP client
- [ ] Understand config files

### L1: Optimization (4-6 hours)
- [ ] Dynamic batching
- [ ] TensorRT models
- [ ] Model versioning
- [ ] Performance tuning

### L2: Production (1-2 days)
- [ ] Model ensembles
- [ ] Kubernetes deployment
- [ ] Monitoring
- [ ] Auto-scaling

---

## Code Examples

### Python Client

```python
import tritonclient.http as httpclient
import numpy as np

# Create client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Check server health
print(f"Server live: {client.is_server_live()}")
print(f"Model ready: {client.is_model_ready('my_model')}")

# Prepare input
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

inputs = [
    httpclient.InferInput("input", input_data.shape, "FP32")
]
inputs[0].set_data_from_numpy(input_data)

outputs = [
    httpclient.InferRequestedOutput("output")
]

# Inference
result = client.infer(
    model_name="my_model",
    inputs=inputs,
    outputs=outputs
)

output = result.as_numpy("output")
print(f"Output shape: {output.shape}")
```

### gRPC Client (Faster)

```python
import tritonclient.grpc as grpcclient
import numpy as np

client = grpcclient.InferenceServerClient(url="localhost:8001")

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

inputs = [
    grpcclient.InferInput("input", input_data.shape, "FP32")
]
inputs[0].set_data_from_numpy(input_data)

outputs = [
    grpcclient.InferRequestedOutput("output")
]

result = client.infer(
    model_name="my_model",
    inputs=inputs,
    outputs=outputs
)

print(result.as_numpy("output"))
```

### Dynamic Batching Config

```protobuf
name: "batched_model"
backend: "onnxruntime"
max_batch_size: 32

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

# Enable dynamic batching
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}

instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
```

### Model Ensemble

```protobuf
# ensemble_model/config.pbtxt
name: "ensemble_model"
platform: "ensemble"
max_batch_size: 8

input [
  {
    name: "raw_image"
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3 ]
  }
]

output [
  {
    name: "classification"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "preprocessing"
      model_version: 1
      input_map {
        key: "raw_input"
        value: "raw_image"
      }
      output_map {
        key: "processed_output"
        value: "preprocessed"
      }
    },
    {
      model_name: "classifier"
      model_version: 1
      input_map {
        key: "input"
        value: "preprocessed"
      }
      output_map {
        key: "output"
        value: "classification"
      }
    }
  ]
}
```

### TensorRT Model Config

```protobuf
name: "tensorrt_model"
backend: "tensorrt"
max_batch_size: 16

input [
  {
    name: "input"
    data_type: TYPE_FP16
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP16
    dims: [ 1000 ]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# Enable TensorRT optimizations
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [
      { name : "tensorrt" }
    ]
  }
}
```

### Async Client

```python
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import numpy as np
import queue
import threading

class AsyncTritonClient:
    def __init__(self, url: str):
        self.client = httpclient.InferenceServerClient(url=url)
        self.results = queue.Queue()

    def callback(self, result, error):
        if error:
            self.results.put(("error", error))
        else:
            self.results.put(("success", result))

    def async_infer(self, model_name: str, inputs: list, outputs: list):
        self.client.async_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            callback=self.callback
        )

    def get_result(self, timeout: float = 10.0):
        return self.results.get(timeout=timeout)

# Usage
client = AsyncTritonClient("localhost:8000")

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)
outputs = [httpclient.InferRequestedOutput("output")]

# Send async request
client.async_infer("my_model", inputs, outputs)

# Get result
status, result = client.get_result()
if status == "success":
    print(result.as_numpy("output"))
```

---

## Performance Tuning

| Setting | Default | Recommended |
|---------|---------|-------------|
| `instance_group.count` | 1 | 2-4 per GPU |
| `max_batch_size` | 0 | 8-64 |
| `dynamic_batching` | Off | On |
| `max_queue_delay` | 0 | 100-1000μs |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not loading | Check config.pbtxt syntax |
| Low throughput | Enable dynamic batching |
| High latency | Use TensorRT, tune batch size |
| OOM | Reduce instance count, batch size |

---

## Resources

- [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md)
- [Client Libraries](https://github.com/triton-inference-server/client)
- [Performance Tuning](https://github.com/triton-inference-server/server/blob/main/docs/optimization.md)

---

*Part of [Luno-AI](../../README.md) | Deployment Track*
