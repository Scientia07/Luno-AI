# TensorRT Integration

> **NVIDIA's high-performance inference optimizer**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Deep learning inference optimizer and runtime |
| **Why** | 2-6x speedup on NVIDIA GPUs |
| **Works With** | PyTorch, TensorFlow, ONNX models |
| **Best For** | Production deployment, real-time inference |

### Speedup Examples

| Model | Without TRT | With TRT | Speedup |
|-------|-------------|----------|---------|
| ResNet-50 | 5ms | 1.2ms | 4x |
| BERT-base | 12ms | 3ms | 4x |
| YOLOv8 | 15ms | 4ms | 3.7x |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA (compute capability 7.0+) |
| **CUDA** | 12.0+ |
| **TensorRT** | 8.6+ |
| **Python** | 3.8+ |

---

## Quick Start (30 min)

### Installation

```bash
pip install tensorrt
pip install torch-tensorrt  # For PyTorch models
```

### Convert ONNX to TensorRT

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# Build engine
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

engine = builder.build_serialized_network(network, config)

# Save engine
with open("model.trt", "wb") as f:
    f.write(engine)
```

### Run Inference

```python
import tensorrt as trt
import numpy as np

# Load engine
with open("model.trt", "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Allocate buffers
import pycuda.driver as cuda
import pycuda.autoinit

input_shape = (1, 3, 224, 224)
output_shape = (1, 1000)

d_input = cuda.mem_alloc(np.prod(input_shape) * 4)
d_output = cuda.mem_alloc(np.prod(output_shape) * 4)

# Run inference
input_data = np.random.randn(*input_shape).astype(np.float32)
cuda.memcpy_htod(d_input, input_data)

context.execute_v2([int(d_input), int(d_output)])

output = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output, d_output)
```

---

## Learning Path

### L0: Basic Conversion (2 hours)
- [ ] Install TensorRT
- [ ] Convert ONNX model
- [ ] Run inference
- [ ] Benchmark speed

### L1: Optimization (4-6 hours)
- [ ] FP16 precision
- [ ] INT8 quantization
- [ ] Dynamic shapes
- [ ] Batch optimization

### L2: Production (1-2 days)
- [ ] Triton Inference Server
- [ ] Multi-stream inference
- [ ] Memory optimization
- [ ] Profiling

---

## Code Examples

### Torch-TensorRT (Easier)

```python
import torch
import torch_tensorrt

model = torch.load("model.pt").eval().cuda()

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(
        shape=[1, 3, 224, 224],
        dtype=torch.float32
    )],
    enabled_precisions={torch.float16}
)

# Run inference
input_tensor = torch.randn(1, 3, 224, 224).cuda()
output = trt_model(input_tensor)
```

### FP16 Optimization

```python
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
```

### INT8 Quantization

```python
class CalibrationDataset:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.batch_idx = 0

    def get_batch(self, names):
        if self.batch_idx < len(self.data_loader):
            batch = next(iter(self.data_loader))
            self.batch_idx += 1
            return [batch.numpy()]
        return None

config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = trt.IInt8EntropyCalibrator2(
    CalibrationDataset(calibration_loader),
    "calibration_cache"
)
```

### Dynamic Batch Size

```python
profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 3, 224, 224),
    opt=(8, 3, 224, 224),
    max=(32, 3, 224, 224)
)
config.add_optimization_profile(profile)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Build fails | Check ONNX opset version, unsupported ops |
| Slow build | Reduce workspace, simplify network |
| Wrong outputs | Check input preprocessing |
| OOM | Reduce batch size, use FP16 |

---

## Resources

- [TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Torch-TensorRT](https://pytorch.org/TensorRT/)
- [TensorRT OSS](https://github.com/NVIDIA/TensorRT)
- [Triton Inference Server](https://github.com/triton-inference-server/server)

---

*Part of [Luno-AI](../../README.md) | Deploy Track*
