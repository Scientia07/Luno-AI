# Visual AI: Computer Vision Technologies

> **Teaching machines to see** - from object detection to scene understanding.

---

## Layer Navigation

| Layer | Content | Status |
|-------|---------|--------|
| L0 | [Overview](#overview) | This file |
| L1 | [Concepts](./concepts.md) | Pending |
| L2 | [Deep Dive](./deep-dive.md) | Pending |
| L3 | [Labs](../../labs/visual-ai/) | Pending |
| L4 | [Advanced](./advanced.md) | Pending |

---

## Overview

Visual AI enables machines to interpret and understand visual information from the world - images, videos, and real-time camera feeds.

```
                    VISUAL AI HIERARCHY

                    ┌─────────────────┐
                    │ Scene           │  ← "A kitchen with a person cooking"
                    │ Understanding   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────────┐  ┌─▼────────┐  ┌──▼──────────┐
    │ Object Detection  │  │ Semantic │  │ Instance    │
    │ "person at (x,y)" │  │ Segment. │  │ Segmentation│
    └───────────────────┘  │ "floor"  │  │ "person #1" │
                           └──────────┘  └─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
    ┌─────────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │ Classification│   │ Pose/Face   │   │    OCR      │
    │ "this is a cat"│  │ Estimation  │   │ "STOP" sign │
    └───────────────┘   └─────────────┘   └─────────────┘
```

---

## Core Technologies

### 1. Object Detection

**Find objects and their locations in images**

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| **YOLOv8/v9/v10** | Very Fast | High | Real-time apps |
| **RT-DETR** | Fast | Very High | Accuracy-critical |
| **DETR** | Slow | Very High | Research |
| **Faster R-CNN** | Slow | High | Legacy systems |

```python
# Quick YOLO example
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model("image.jpg")
```

### 2. Image Segmentation

**Understand images at pixel level**

| Type | What It Does | Models |
|------|--------------|--------|
| **Semantic** | Label every pixel | SAM, Mask2Former |
| **Instance** | Separate individual objects | Mask R-CNN, YOLACT |
| **Panoptic** | Semantic + Instance | Mask2Former |

```
Input Image:        Segmentation:
┌─────────────┐     ┌─────────────┐
│  🐱  🐕     │     │ ████  ▓▓▓  │  ← Cat vs Dog pixels
│   floor     │ ──▶ │   ░░░░░░░  │  ← Floor pixels
└─────────────┘     └─────────────┘
```

### 3. Image Classification

**Categorize entire images**

| Model | Params | Top-1 Acc | Speed |
|-------|--------|-----------|-------|
| **ViT-L** | 307M | 88.5% | Medium |
| **ConvNeXt-L** | 198M | 87.5% | Fast |
| **EfficientNet-B7** | 66M | 84.4% | Fast |
| **ResNet-50** | 25M | 80.4% | Very Fast |

### 4. SAM (Segment Anything Model)

**The universal segmenter**

```
Click/Box/Point ──▶ SAM ──▶ Perfect Mask
                    ↓
           Works on ANYTHING
```

Versions:
- **SAM** - Original (slow, accurate)
- **SAM 2** - Video support, faster
- **FastSAM** - Real-time
- **MobileSAM** - Mobile deployment

### 5. Pose & Face

| Task | Model | Speed |
|------|-------|-------|
| Body Pose | MediaPipe, RTMPose | Real-time |
| Hand Tracking | MediaPipe | Real-time |
| Face Mesh | MediaPipe | Real-time |
| Face Detection | RetinaFace, MTCNN | Fast |

### 6. OCR / Document AI

| Tool | Strength |
|------|----------|
| **Tesseract** | Open source, many languages |
| **PaddleOCR** | High accuracy, fast |
| **EasyOCR** | Simple API |
| **DocTR** | Document layout |
| **Surya** | Modern, accurate |

---

## Quick Comparison

| Task | Best Speed | Best Accuracy |
|------|------------|---------------|
| Detection | YOLOv8n | RT-DETR |
| Segmentation | FastSAM | SAM 2 |
| Classification | EfficientNet | ViT-L |
| OCR | PaddleOCR | DocTR |
| Pose | MediaPipe | RTMPose |

---

## Use Case Decision Tree

```
What do you need?
│
├── Find objects in image? ──────────────▶ YOLO / RT-DETR
│
├── Cut out objects precisely? ──────────▶ SAM
│
├── Label every pixel? ──────────────────▶ Mask2Former
│
├── Read text from images? ──────────────▶ PaddleOCR / Surya
│
├── Track body/hands/face? ──────────────▶ MediaPipe
│
├── Categorize images? ──────────────────▶ ViT / ConvNeXt
│
└── All of the above on video? ──────────▶ SAM 2 + YOLO + MediaPipe
```

---

## Getting Started

### Install Core Libraries

```bash
# Object Detection
pip install ultralytics

# Segmentation
pip install segment-anything

# General Vision
pip install transformers timm opencv-python

# Pose
pip install mediapipe

# OCR
pip install paddlepaddle paddleocr
```

### First Detection

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")  # nano model, fast

# Run detection
results = model("https://ultralytics.com/images/bus.jpg")

# Show results
results[0].show()
```

---

## Labs

| Notebook | Focus |
|----------|-------|
| `01-yolo-quickstart.ipynb` | Object detection basics |
| `02-sam-segmentation.ipynb` | Segment anything |
| `03-classification.ipynb` | Image classification |
| `04-ocr-documents.ipynb` | Text extraction |
| `05-pose-tracking.ipynb` | Body/hand tracking |
| `06-video-pipeline.ipynb` | Real-time video |

---

## Next Steps

- L1: [How Object Detection Works](./concepts.md)
- L2: [YOLO Architecture Deep Dive](./deep-dive.md)
- Related: [Spatial AI](../spatial-ai/README.md) (depth, 3D)

---

*"Vision is the most powerful sense - now machines have it too."*
