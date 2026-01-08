# ONNX Runtime Integration

> **Category**: Edge & Deployment
> **Difficulty**: Intermediate
> **Setup Time**: 2-3 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
ONNX (Open Neural Network Exchange) is a universal format for ML models. ONNX Runtime is a high-performance inference engine that runs ONNX models on any platform.

### Why Use It
- **Universal Format**: One model, any framework
- **Cross-Platform**: Windows, Linux, macOS, mobile, edge
- **Optimized**: Hardware-specific optimizations
- **Ecosystem**: Works with PyTorch, TensorFlow, scikit-learn
- **Production Ready**: Used by Microsoft, Azure, many enterprises

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Cross-Framework | Export from PyTorch, TensorFlow, sklearn |
| Hardware Acceleration | CPU, GPU, NPU, FPGA |
| Quantization | INT8, FP16 optimization |
| Graph Optimization | Automatic model optimization |
| Mobile/Edge | iOS, Android, Raspberry Pi |

### Deployment Flow
```
PyTorch/TF/sklearn Model
         │
         ▼
    Export to ONNX
         │
         ▼
    Optimize (optional)
         │
         ▼
    ONNX Runtime
    ┌────┴────┐
    │         │
    ▼         ▼
  Cloud     Edge
  Server    Device
```

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None | NVIDIA (for GPU inference) |
| RAM | 4 GB | 8 GB |
| Storage | 500 MB | 2 GB |

### Software Dependencies
```bash
# Core
pip install onnx onnxruntime

# GPU support
pip install onnxruntime-gpu

# For PyTorch export
pip install torch

# For sklearn export
pip install skl2onnx

# For TensorFlow
pip install tf2onnx
```

### Prior Knowledge
- [x] Python basics
- [x] PyTorch or TensorFlow basics
- [ ] Model training experience

---

## Quick Start (15 minutes)

### 1. Install
```bash
pip install onnx onnxruntime torch
```

### 2. Export PyTorch Model
```python
import torch
import torch.nn as nn

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
print("Exported to model.onnx")
```

### 3. Run with ONNX Runtime
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Inference
input_data = np.random.randn(1, 10).astype(np.float32)
outputs = session.run(None, {"input": input_data})
print(f"Output: {outputs[0]}")
```

---

## Full Setup

### PyTorch to ONNX

```python
import torch
from torchvision import models

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)
```

### TensorFlow to ONNX

```bash
# Command line
python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx

# Or in Python
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())
```

### scikit-learn to ONNX

```python
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Convert
initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

---

## Learning Path

### L0: Basic Export & Inference (1 hour)
**Goal**: Convert and run a model

```python
import onnxruntime as ort
import numpy as np

# Create session
session = ort.InferenceSession("model.onnx")

# Get input/output info
print("Inputs:", [i.name for i in session.get_inputs()])
print("Outputs:", [o.name for o in session.get_outputs()])

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run(
    [output_name],
    {input_name: np.random.randn(1, 10).astype(np.float32)}
)
print(result)
```

### L1: GPU Acceleration (2 hours)
**Goal**: Use GPU for faster inference

```python
import onnxruntime as ort

# Check available providers
print(ort.get_available_providers())
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

# GPU session
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Or specific GPU
session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
        }),
        'CPUExecutionProvider'
    ]
)
```

### L2: Model Optimization (3 hours)
**Goal**: Optimize model for deployment

```python
import onnxruntime as ort
from onnxruntime.transformers import optimizer

# Optimize model
optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type='bert',  # or 'gpt2', 'vit', etc.
    opt_level=2,        # 0-2, higher = more aggressive
    use_gpu=True,
)
optimized_model.save_model_to_file("model_optimized.onnx")

# Or use graph optimization in session
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "model_optimized.onnx"

session = ort.InferenceSession("model.onnx", sess_options)
```

### L3: Quantization (4+ hours)
**Goal**: Reduce model size and increase speed

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static
from onnxruntime.quantization import QuantType, CalibrationDataReader

# Dynamic quantization (no calibration needed)
quantize_dynamic(
    "model.onnx",
    "model_int8.onnx",
    weight_type=QuantType.QUInt8,
)

# Static quantization (better accuracy, needs calibration)
class CalibrationData(CalibrationDataReader):
    def __init__(self, data):
        self.data = iter(data)

    def get_next(self):
        try:
            return {"input": next(self.data)}
        except StopIteration:
            return None

calibration_data = [np.random.randn(1, 10).astype(np.float32) for _ in range(100)]

quantize_static(
    "model.onnx",
    "model_int8_static.onnx",
    CalibrationData(calibration_data),
)
```

---

## Code Examples

### Example 1: Image Classification
```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image
image = Image.open("cat.jpg")
input_tensor = transform(image).unsqueeze(0).numpy()

# Inference
session = ort.InferenceSession("resnet18.onnx")
outputs = session.run(None, {"input": input_tensor})

# Get prediction
predicted_class = np.argmax(outputs[0])
print(f"Predicted class: {predicted_class}")
```

### Example 2: Batch Processing
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")

# Process batch
batch_size = 32
inputs = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

outputs = session.run(None, {"input": inputs})
print(f"Processed {batch_size} images")
```

### Example 3: Multiple Outputs
```python
# Model with multiple outputs
session = ort.InferenceSession("multi_output.onnx")

# Get all output names
output_names = [o.name for o in session.get_outputs()]
print(f"Outputs: {output_names}")

# Run and get all outputs
results = session.run(output_names, {"input": input_data})

for name, result in zip(output_names, results):
    print(f"{name}: {result.shape}")
```

### Example 4: YOLO Export
```python
from ultralytics import YOLO

# Load and export YOLO
model = YOLO("yolo11n.pt")
model.export(format="onnx", dynamic=True, simplify=True)

# Run with ONNX Runtime
import onnxruntime as ort
import cv2
import numpy as np

session = ort.InferenceSession("yolo11n.onnx")

# Preprocess
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (640, 640))
input_tensor = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
input_tensor = np.expand_dims(input_tensor, 0)

# Inference
outputs = session.run(None, {"images": input_tensor})
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| TensorRT | NVIDIA optimization | [tensorrt.md](./tensorrt.md) |
| Quantization | Size reduction | [quantization.md](./quantization.md) |
| Docker | Containerization | [docker.md](./docker.md) |
| Triton | Inference server | [triton.md](./triton.md) |

### Execution Providers
```python
# Priority order
providers = [
    'TensorrtExecutionProvider',  # NVIDIA TensorRT
    'CUDAExecutionProvider',      # NVIDIA CUDA
    'ROCMExecutionProvider',      # AMD ROCm
    'MIGraphXExecutionProvider',  # AMD MIGraphX
    'OpenVINOExecutionProvider',  # Intel
    'CoreMLExecutionProvider',    # Apple
    'DmlExecutionProvider',       # DirectML (Windows)
    'CPUExecutionProvider',       # CPU fallback
]
```

### Model Formats
| Source | Export Command |
|--------|----------------|
| PyTorch | `torch.onnx.export()` |
| TensorFlow | `tf2onnx.convert` |
| Keras | `tf2onnx.convert.from_keras()` |
| sklearn | `skl2onnx.convert_sklearn()` |
| Hugging Face | `optimum-cli export onnx` |

---

## Troubleshooting

### Common Issues

#### Issue 1: Opset Version Error
**Symptoms**: "Unsupported opset version"
**Solution**:
```python
# Export with specific opset
torch.onnx.export(model, dummy, "model.onnx", opset_version=14)

# Or upgrade onnxruntime
pip install --upgrade onnxruntime
```

#### Issue 2: Dynamic Shapes Fail
**Symptoms**: Error with variable batch size
**Solution**:
```python
# Enable dynamic axes
torch.onnx.export(
    model, dummy, "model.onnx",
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch"},
    }
)
```

#### Issue 3: GPU Not Used
**Symptoms**: Slow inference despite GPU
**Solution**:
```python
# Check providers
print(ort.get_available_providers())

# Ensure CUDA provider is available
pip install onnxruntime-gpu

# Explicitly set GPU
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider']
)
```

### Performance Tips
- Use dynamic quantization for 2-4x speedup
- Enable graph optimization (level 2)
- Use IO binding for GPU inference
- Batch inputs when possible

---

## Resources

### Official
- [ONNX Runtime Docs](https://onnxruntime.ai/docs/)
- [ONNX GitHub](https://github.com/onnx/onnx)
- [Model Zoo](https://github.com/onnx/models)

### Tutorials
- [PyTorch to ONNX](https://pytorch.org/docs/stable/onnx.html)
- [Optimization Guide](https://onnxruntime.ai/docs/performance/tune-performance.html)

---

*Part of [Luno-AI Integration Hub](../_index.md) | Edge & Deployment Track*
