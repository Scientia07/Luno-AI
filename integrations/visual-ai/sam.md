# SAM Segmentation Integration

> **Category**: Visual AI
> **Difficulty**: Beginner
> **Setup Time**: 2-3 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
SAM (Segment Anything Model) from Meta can segment any object in any image with a single click or text prompt. It's the foundation for many vision applications.

### Why Use It
- **Zero-shot**: Works on any object without training
- **Flexible Input**: Points, boxes, or text prompts
- **High Quality**: Precise pixel-level masks
- **Foundation Model**: Build other tools on top
- **Open Source**: Free to use

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Point Prompt | Click to segment |
| Box Prompt | Draw box to segment |
| Auto Segment | Segment everything |
| Text Prompt | SAM 2 + Grounding DINO |
| Video | SAM 2 tracks through video |

### Model Variants
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| SAM ViT-B | 358 MB | Fast | Good |
| SAM ViT-L | 1.2 GB | Medium | Better |
| SAM ViT-H | 2.4 GB | Slow | Best |
| SAM 2 | Various | Fast | Best + Video |
| MobileSAM | 40 MB | Fastest | Good |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4 GB VRAM | 8 GB VRAM |
| RAM | 8 GB | 16 GB |
| Storage | 3 GB | 5 GB |

### Software Dependencies
```bash
pip install segment-anything opencv-python matplotlib
# or for SAM 2
pip install sam-2
```

---

## Quick Start (15 minutes)

### 1. Install
```bash
pip install segment-anything
```

### 2. Download Model
```bash
# ViT-H (best quality)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 3. Segment with Point
```python
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# Load model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda")
predictor = SamPredictor(sam)

# Load image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# Segment at point
input_point = np.array([[500, 375]])  # x, y coordinates
input_label = np.array([1])  # 1 = foreground

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Best mask
best_mask = masks[np.argmax(scores)]
```

---

## Full Setup

### Point Prompting
```python
# Single point
masks, scores, _ = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),  # 1=foreground, 0=background
)

# Multiple points
masks, scores, _ = predictor.predict(
    point_coords=np.array([[x1, y1], [x2, y2]]),
    point_labels=np.array([1, 1]),  # Both foreground
)
```

### Box Prompting
```python
# Box prompt [x1, y1, x2, y2]
input_box = np.array([100, 100, 400, 400])

masks, scores, _ = predictor.predict(
    box=input_box,
    multimask_output=False,
)
```

### Automatic Segmentation
```python
from segment_anything import SamAutomaticMaskGenerator

generator = SamAutomaticMaskGenerator(sam)
masks = generator.generate(image)

# masks is list of dicts with 'segmentation', 'area', 'bbox', etc.
for mask in masks:
    print(f"Area: {mask['area']}, Score: {mask['stability_score']}")
```

---

## Code Examples

### Example 1: YOLO + SAM
```python
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Detect with YOLO
yolo = YOLO("yolo11n.pt")
results = yolo("image.jpg")

# Segment each detection
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam.to("cuda"))
predictor.set_image(image)

for box in results[0].boxes.xyxy:
    masks, scores, _ = predictor.predict(box=box.cpu().numpy())
    # Process mask...
```

### Example 2: Save Masks
```python
import cv2

for i, mask in enumerate(masks):
    # Convert to image
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(f"mask_{i}.png", mask_image)

    # Apply to original
    colored = image.copy()
    colored[mask] = [255, 0, 0]  # Red overlay
    cv2.imwrite(f"overlay_{i}.png", colored)
```

---

## Integration Points

| Integration | Purpose | Link |
|-------------|---------|------|
| YOLO | Detection + Segmentation | [yolo.md](./yolo.md) |
| CLIP | Text prompts | [clip.md](./clip.md) |
| Grounding DINO | Text to box | External |

---

## Troubleshooting

### Out of Memory
```python
# Use smaller model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")

# Or MobileSAM
pip install mobile-sam
```

### Slow Processing
- Use GPU
- Use MobileSAM for speed
- Resize large images first

---

## Resources

- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM 2](https://github.com/facebookresearch/segment-anything-2)
- [Demo](https://segment-anything.com/demo)

---

*Part of [Luno-AI Integration Hub](../_index.md) | Visual AI Track*
