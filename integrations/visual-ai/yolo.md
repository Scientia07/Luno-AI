# YOLO Object Detection Integration

> **Category**: Visual AI
> **Difficulty**: Beginner
> **Setup Time**: 2-4 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
YOLO (You Only Look Once) is a real-time object detection system that can identify and locate multiple objects in images and video with a single forward pass through the network.

### Why Use It
- **Real-time Speed**: 100+ FPS on modern GPUs
- **Accuracy**: State-of-the-art detection performance
- **Ease of Use**: Simple API, works out of the box
- **Versatile**: Detection, segmentation, pose, tracking, OBB

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Object Detection | Locate and classify objects with bounding boxes |
| Instance Segmentation | Pixel-level object masks |
| Pose Estimation | Human keypoint detection |
| Object Tracking | Track objects across video frames |
| Oriented Bounding Boxes | Rotated boxes for angled objects |

### Version Comparison
| Version | Developer | Best For | mAP | Speed |
|---------|-----------|----------|-----|-------|
| **YOLOv11** | Ultralytics | General use (recommended) | 54.7% | Fast |
| **YOLOv10** | Tsinghua | NMS-free deployment | 53.9% | Fastest |
| **YOLOv9** | Wang & Liao | Max accuracy | 55.6% | Medium |
| **YOLOv12** | Tian et al. | Cutting edge | 55.2% | Medium |
| **YOLO-World** | AILab-CVC | Zero-shot detection | 35.0% | Medium |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU works) | NVIDIA 4GB+ VRAM |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 5 GB |

### Software Dependencies
```bash
# Required
python >= 3.8
pip install ultralytics

# Optional for advanced features
pip install lap  # tracking
pip install onnx onnxruntime  # export
```

### Prior Knowledge
- [x] Python basics
- [ ] Basic understanding of images/video (helpful)

---

## Quick Start (15 minutes)

### 1. Install
```bash
pip install ultralytics
```

### 2. Basic Detection
```python
from ultralytics import YOLO

# Load model (auto-downloads)
model = YOLO("yolo11n.pt")  # nano model, fastest

# Run detection
results = model("path/to/image.jpg")

# Show results
results[0].show()

# Save results
results[0].save("output.jpg")
```

### 3. Verify Installation
```bash
yolo predict model=yolo11n.pt source="https://ultralytics.com/images/bus.jpg"
```

---

## Full Setup

### Model Sizes

| Model | Size | mAP | Speed (ms) | Use Case |
|-------|------|-----|------------|----------|
| `yolo11n.pt` | 6.5 MB | 39.5% | 1.5 | Edge, mobile |
| `yolo11s.pt` | 21.5 MB | 47.0% | 2.5 | Balanced |
| `yolo11m.pt` | 68.0 MB | 51.5% | 4.7 | Production |
| `yolo11l.pt` | 87.0 MB | 53.4% | 6.2 | High accuracy |
| `yolo11x.pt` | 194 MB | 54.7% | 11.3 | Max accuracy |

### Installation Options

#### Option A: pip (Recommended)
```bash
pip install ultralytics
```

#### Option B: conda
```bash
conda install -c conda-forge ultralytics
```

#### Option C: From Source
```bash
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
```

### Configuration
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Configure inference
results = model.predict(
    source="image.jpg",
    conf=0.25,        # confidence threshold
    iou=0.45,         # NMS IoU threshold
    max_det=300,      # max detections
    classes=[0, 1],   # filter classes (0=person, 1=bicycle)
    device="cuda",    # or "cpu", "mps"
    half=True,        # FP16 inference
    imgsz=640,        # input size
    save=True,        # save results
    show=False,       # display results
)
```

---

## Learning Path

### L0: Basic Detection (1 hour)
**Goal**: Detect objects in images

- [x] Install ultralytics
- [ ] Run detection on images
- [ ] Understand results format

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("image.jpg")

# Access detections
for box in results[0].boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}, Box: {box.xyxy}")
```

### L1: Video & Tracking (2 hours)
**Goal**: Process video with tracking

- [ ] Run on video files
- [ ] Enable object tracking
- [ ] Process webcam stream

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Track objects in video
results = model.track(
    source="video.mp4",
    tracker="bytetrack.yaml",  # or botsort.yaml
    persist=True,
    show=True,
)

# Webcam
results = model.track(source=0, show=True)
```

### L2: Segmentation & Pose (3 hours)
**Goal**: Use specialized models

- [ ] Instance segmentation
- [ ] Pose estimation
- [ ] Combine models

```python
# Segmentation
seg_model = YOLO("yolo11n-seg.pt")
results = seg_model("image.jpg")
masks = results[0].masks  # segmentation masks

# Pose estimation
pose_model = YOLO("yolo11n-pose.pt")
results = pose_model("image.jpg")
keypoints = results[0].keypoints  # 17 body keypoints

# Oriented bounding boxes
obb_model = YOLO("yolo11n-obb.pt")
results = obb_model("aerial.jpg")
```

### L3: Custom Training (1+ days)
**Goal**: Train on custom data

- [ ] Prepare dataset (YOLO format)
- [ ] Configure training
- [ ] Train and evaluate
- [ ] Export for deployment

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11n.pt")

# Train on custom data
results = model.train(
    data="custom_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    patience=50,
    device=0,
)

# Validate
metrics = model.val()

# Export
model.export(format="onnx")  # or tensorrt, coreml, etc.
```

**Dataset Format (custom_dataset.yaml):**
```yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: class1
  1: class2
  2: class3
```

---

## Code Examples

### Example 1: Batch Processing
```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt")

# Process folder of images
image_folder = Path("images/")
for img_path in image_folder.glob("*.jpg"):
    results = model(img_path)
    results[0].save(f"output/{img_path.name}")
```

### Example 2: Extract Detected Objects
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
results = model("image.jpg")

img = cv2.imread("image.jpg")
for i, box in enumerate(results[0].boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(f"crop_{i}.jpg", crop)
```

### Example 3: Real-time with Custom Callback
```python
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    # Count people
    people = sum(1 for box in results[0].boxes if box.cls == 0)
    cv2.putText(annotated, f"People: {people}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 4: YOLO-World Zero-Shot
```python
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8l-world.pt")

# Set custom classes (no training needed!)
model.set_classes(["coffee cup", "laptop", "phone", "notebook"])

results = model("desk.jpg")
results[0].show()
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| SAM | Refined segmentation from YOLO boxes | [sam.md](./sam.md) |
| CLIP | Classify detected objects | [clip.md](./clip.md) |
| Depth Anything | 3D localization | [depth-anything.md](./depth-anything.md) |
| DeepSORT | Advanced tracking | External |
| TensorRT | 2-3x faster inference | [tensorrt.md](../deploy/tensorrt.md) |

### Export Formats
```python
model.export(format="onnx")      # Cross-platform
model.export(format="tensorrt")  # NVIDIA optimized
model.export(format="coreml")    # Apple devices
model.export(format="tflite")    # Mobile/edge
model.export(format="openvino")  # Intel optimized
```

### Data Formats
| Format | Input | Output |
|--------|-------|--------|
| Images | jpg, png, bmp, webp | Annotated images |
| Video | mp4, avi, mov | Annotated video |
| Stream | RTSP, HTTP, webcam | Real-time display |
| Results | - | boxes, masks, keypoints |

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory
**Symptoms**: RuntimeError: CUDA out of memory
**Cause**: Model or batch too large for GPU
**Solution**:
```python
# Use smaller model
model = YOLO("yolo11n.pt")  # instead of yolo11x.pt

# Reduce image size
results = model(img, imgsz=320)

# Use CPU
results = model(img, device="cpu")
```

#### Issue 2: Slow Inference
**Symptoms**: Low FPS
**Cause**: Not using GPU or FP16
**Solution**:
```python
# Enable GPU + FP16
results = model(img, device="cuda", half=True)

# Export to TensorRT
model.export(format="tensorrt", half=True)
trt_model = YOLO("yolo11n.engine")
```

#### Issue 3: Poor Detection
**Symptoms**: Missing objects or false positives
**Cause**: Threshold too high/low or wrong model
**Solution**:
```python
# Adjust thresholds
results = model(img, conf=0.1, iou=0.3)  # lower conf = more detections

# Use larger model
model = YOLO("yolo11l.pt")
```

### Performance Tips
- Use FP16 (`half=True`) for 2x speedup
- Export to TensorRT for NVIDIA GPUs
- Use `yolo11n` for real-time, `yolo11x` for accuracy
- Batch process images when possible

---

## Resources

### Official
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [GitHub](https://github.com/ultralytics/ultralytics)
- [Model Hub](https://docs.ultralytics.com/models/)

### Tutorials
- [Training Custom Dataset](https://docs.ultralytics.com/modes/train/)
- [Export Guide](https://docs.ultralytics.com/modes/export/)
- [Tracking Guide](https://docs.ultralytics.com/modes/track/)

### Community
- [Discord](https://discord.gg/ultralytics)
- [GitHub Discussions](https://github.com/ultralytics/ultralytics/discussions)

---

## Related Integrations

| Next Step | Why | Link |
|-----------|-----|------|
| SAM Segmentation | Better masks from YOLO boxes | [sam.md](./sam.md) |
| Custom Training | Your own objects | [yolo-training.md](./yolo-training.md) |
| TensorRT | Production speed | [tensorrt.md](../deploy/tensorrt.md) |
| LLaVA | Visual understanding | [llava.md](./llava.md) |

---

*Part of [Luno-AI Integration Hub](../_index.md) | Visual AI Track*
