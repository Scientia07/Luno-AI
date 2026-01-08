# Visual AI: Projects & Comparisons

> **Hands-on projects and framework comparisons for Visual AI**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Object Counter
**Goal**: Count specific objects in images/video

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | YOLO |
| Skills | Basic detection, counting logic |

**Tasks**:
- [ ] Load YOLOv11 model
- [ ] Detect objects in image
- [ ] Filter by class (e.g., "person", "car")
- [ ] Display count overlay
- [ ] Extend to video stream

**Starter Code**:
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("image.jpg")

# Count people
people_count = sum(1 for box in results[0].boxes if box.cls == 0)
print(f"People detected: {people_count}")
```

---

#### Project 2: Image Similarity Search
**Goal**: Find similar images in a collection

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 3-4 hours |
| Technologies | CLIP |
| Skills | Embeddings, similarity |

**Tasks**:
- [ ] Generate CLIP embeddings for image collection
- [ ] Store embeddings (numpy/pickle)
- [ ] Query with new image
- [ ] Return top-K similar images
- [ ] Add text-based search

---

### Intermediate Projects (L2)

#### Project 3: Smart Security Camera
**Goal**: Detect and alert on specific events

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | YOLO + tracking |
| Skills | Video processing, tracking, alerts |

**Tasks**:
- [ ] Process webcam/RTSP stream
- [ ] Detect persons entering zone
- [ ] Track across frames (ByteTrack)
- [ ] Count unique individuals
- [ ] Send alerts (email/webhook)
- [ ] Log events with timestamps

---

#### Project 4: Document Scanner & OCR
**Goal**: Extract text from photos of documents

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | OpenCV + PaddleOCR/DocTR |
| Skills | Perspective transform, OCR |

**Tasks**:
- [ ] Detect document edges
- [ ] Apply perspective correction
- [ ] Enhance image quality
- [ ] Extract text with OCR
- [ ] Structure output (JSON)

---

#### Project 5: Pose-Controlled Game
**Goal**: Control a game using body movements

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | YOLO-pose or MediaPipe |
| Skills | Pose estimation, gesture recognition |

**Tasks**:
- [ ] Detect body keypoints in real-time
- [ ] Define gesture patterns (raise hand, jump, etc.)
- [ ] Map gestures to game controls
- [ ] Build simple game (Pygame)
- [ ] Add visual feedback

---

### Advanced Projects (L3-L4)

#### Project 6: Automated Product Photography
**Goal**: Remove backgrounds and standardize product images

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 8-12 hours |
| Technologies | SAM + rembg + enhancement |
| Skills | Segmentation, image processing |

**Tasks**:
- [ ] Segment product from background (SAM)
- [ ] Remove/replace background
- [ ] Auto-crop and center
- [ ] Standardize lighting
- [ ] Batch process folder
- [ ] Generate multiple views

---

#### Project 7: Visual Search Engine
**Goal**: Search products by image

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | CLIP + Vector DB + YOLO |
| Skills | Embeddings, indexing, retrieval |

**Tasks**:
- [ ] Build product image database
- [ ] Generate CLIP embeddings
- [ ] Store in vector DB (Chroma/Qdrant)
- [ ] Detect objects in query image
- [ ] Search by detected regions
- [ ] Build web interface

---

#### Project 8: Custom Object Detector
**Goal**: Train YOLO on your own dataset

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | YOLO + Label Studio |
| Skills | Annotation, training, evaluation |

**Tasks**:
- [ ] Collect training images (100+)
- [ ] Annotate with Label Studio
- [ ] Export in YOLO format
- [ ] Train YOLOv11
- [ ] Evaluate on test set
- [ ] Export for deployment (ONNX)

---

## Framework Comparisons

### Comparison 1: Object Detection Showdown

**Question**: Which detector to use for your project?

| Framework | Speed | Accuracy | Ease | Best For |
|-----------|-------|----------|------|----------|
| **YOLOv11** | ⚡⚡⚡ | ⭐⭐⭐ | Easy | General use |
| **YOLOv10** | ⚡⚡⚡⚡ | ⭐⭐⭐ | Easy | Edge/NMS-free |
| **RT-DETR** | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Transformers |
| **DETIC** | ⚡ | ⭐⭐⭐ | Medium | Open vocabulary |
| **YOLO-World** | ⚡⚡ | ⭐⭐⭐ | Easy | Zero-shot |

**Lab Exercise**: Run same image through all 5, compare results.

```python
# Compare detectors
from ultralytics import YOLO, YOLOWorld

models = {
    "YOLOv11": YOLO("yolo11n.pt"),
    "YOLOv10": YOLO("yolov10n.pt"),
    "YOLO-World": YOLOWorld("yolov8l-world.pt"),
}

for name, model in models.items():
    results = model("test.jpg")
    print(f"{name}: {len(results[0].boxes)} detections")
```

---

### Comparison 2: Segmentation Battle

**Question**: SAM vs YOLO-Seg vs Mask R-CNN?

| Framework | Interactive | Speed | Quality | Use Case |
|-----------|-------------|-------|---------|----------|
| **SAM** | Yes (clicks) | Slow | Best | Manual selection |
| **SAM 2** | Yes + Video | Medium | Best | Video tracking |
| **YOLO-Seg** | No | Fast | Good | Real-time |
| **Mask R-CNN** | No | Slow | Good | Classic approach |

**Lab Exercise**: Segment same object with all methods.

---

### Comparison 3: Depth Estimation

**Question**: Which depth model for 3D understanding?

| Model | Speed | Quality | Metric | Best For |
|-------|-------|---------|--------|----------|
| **Depth Anything V2** | Fast | Best | Relative | General |
| **MiDaS** | Medium | Good | Relative | Compatibility |
| **ZoeDepth** | Slow | Best | Metric | Real measurements |
| **Marigold** | Slow | Excellent | Relative | Quality |

**Lab Exercise**: Generate depth maps, compare on indoor/outdoor scenes.

---

### Comparison 4: Vision-Language Models

**Question**: Which VLM for image understanding?

| Model | Size | Speed | Capabilities | Open |
|-------|------|-------|--------------|------|
| **LLaVA 1.6** | 7-34B | Medium | Chat, VQA | Yes |
| **Qwen-VL** | 7B | Fast | Multi-image | Yes |
| **GPT-4V** | - | Fast | Best reasoning | No |
| **Claude Vision** | - | Fast | Best analysis | No |
| **Gemini Pro** | - | Fast | Native multimodal | No |

**Lab Exercise**: Ask same question about image to all models.

---

## Hands-On Labs

### Lab 1: Detection Pipeline (2 hours)
```
Input Image → YOLO Detection → Filter Classes → Draw Boxes → Save
```

### Lab 2: Segmentation Pipeline (2 hours)
```
Input Image → YOLO Detect → SAM Segment → Extract Masks → Composite
```

### Lab 3: Visual Search (4 hours)
```
Image DB → CLIP Embed → Vector Store → Query Image → Top-K Results
```

### Lab 4: Real-time Analysis (3 hours)
```
Webcam → YOLO → Track → Count → Display Stats
```

### Lab 5: Custom Training (1 day)
```
Collect Data → Annotate → Train YOLO → Evaluate → Deploy
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Functionality** | 40 | Does it work as specified? |
| **Code Quality** | 20 | Clean, documented, modular |
| **Innovation** | 20 | Creative extensions beyond basics |
| **Documentation** | 10 | README, comments, examples |
| **Performance** | 10 | Speed, accuracy metrics |

---

## Resources

- [Ultralytics Docs](https://docs.ultralytics.com/)
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [CLIP OpenAI](https://github.com/openai/CLIP)
- [Roboflow Universe](https://universe.roboflow.com/) - Free datasets

---

*Part of [Luno-AI](../../../README.md) | Visual AI Track*
