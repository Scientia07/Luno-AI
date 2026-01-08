# Custom YOLO Training Integration

> **Train YOLO on your own dataset**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Fine-tune YOLO for custom object detection |
| **Why** | Detect objects not in pre-trained models |
| **Time** | 1-8 hours depending on dataset size |
| **Best For** | Domain-specific detection (products, defects, custom objects) |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA 8GB+ VRAM |
| **Dataset** | 100+ labeled images (more = better) |
| **Tools** | Label Studio, Roboflow, or CVAT |

---

## Quick Start (2-4 hours)

### 1. Prepare Dataset

```
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── labels/
│       ├── img001.txt
│       └── img002.txt
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

**Label format** (YOLO txt):
```
# class_id x_center y_center width height (normalized 0-1)
0 0.5 0.5 0.3 0.4
1 0.2 0.8 0.1 0.2
```

**data.yaml**:
```yaml
path: /path/to/dataset
train: train/images
val: valid/images

names:
  0: cat
  1: dog
  2: bird
```

### 2. Train Model

```python
from ultralytics import YOLO

# Load base model
model = YOLO("yolo11n.pt")  # nano for speed, or yolo11s/m/l/x

# Train
results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,  # Early stopping
    device=0      # GPU
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

### 3. Use Trained Model

```python
# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Inference
results = model("test_image.jpg")
results[0].show()
```

---

## Learning Path

### L0: Basic Training (2-3 hours)
- [ ] Prepare small dataset (50-100 images)
- [ ] Label with Roboflow/Label Studio
- [ ] Train with default settings
- [ ] Test on new images

### L1: Optimization (4-6 hours)
- [ ] Data augmentation tuning
- [ ] Hyperparameter optimization
- [ ] Learning rate scheduling
- [ ] Model selection (n/s/m/l)

### L2: Production (1-2 days)
- [ ] Large dataset (1000+ images)
- [ ] Cross-validation
- [ ] Export to ONNX/TensorRT
- [ ] Deploy as API

---

## Code Examples

### Using Roboflow Dataset

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("workspace").project("project")
dataset = project.version(1).download("yolov8")

model = YOLO("yolo11n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=100)
```

### Data Augmentation

```python
model.train(
    data="data.yaml",
    epochs=100,
    # Augmentation settings
    augment=True,
    hsv_h=0.015,    # Hue
    hsv_s=0.7,      # Saturation
    hsv_v=0.4,      # Value
    degrees=10,     # Rotation
    translate=0.1,  # Translation
    scale=0.5,      # Scale
    flipud=0.5,     # Vertical flip
    fliplr=0.5,     # Horizontal flip
    mosaic=1.0,     # Mosaic augmentation
    mixup=0.1       # Mixup
)
```

### Hyperparameter Tuning

```python
# Automatic hyperparameter tuning
model.tune(
    data="data.yaml",
    epochs=30,
    iterations=50,
    optimizer="AdamW",
    plots=True,
    save=True
)
```

### Resume Training

```python
# Resume from checkpoint
model = YOLO("runs/detect/train/weights/last.pt")
model.train(resume=True)
```

### Export for Deployment

```python
# Export to different formats
model.export(format="onnx")       # ONNX
model.export(format="tensorrt")   # TensorRT
model.export(format="coreml")     # iOS
model.export(format="tflite")     # Mobile
```

### Multi-GPU Training

```python
model.train(
    data="data.yaml",
    epochs=100,
    device=[0, 1]  # Use GPU 0 and 1
)
```

---

## Dataset Tips

| Aspect | Recommendation |
|--------|----------------|
| **Min images** | 100 per class |
| **Ideal images** | 1000+ per class |
| **Balance** | Similar count per class |
| **Variety** | Different angles, lighting, backgrounds |
| **Quality** | Clear, properly labeled |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Low mAP | More data, longer training, check labels |
| Overfitting | Add augmentation, early stopping |
| OOM error | Reduce batch size, use smaller model |
| NaN loss | Lower learning rate, check data |
| Slow training | Use larger batch, mixed precision |

---

## Resources

- [Ultralytics Training Docs](https://docs.ultralytics.com/modes/train/)
- [Label Studio](https://labelstud.io/)
- [Roboflow](https://roboflow.com/)
- [CVAT](https://www.cvat.ai/)

---

*Part of [Luno-AI](../../README.md) | Visual AI Track*
