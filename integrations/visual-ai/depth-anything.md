# Depth Anything Integration

> **State-of-the-art monocular depth estimation**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Estimate depth from single RGB images |
| **Why** | 3D understanding, AR, robotics, image editing |
| **Version** | V2 (2024) - improved accuracy and speed |
| **Output** | Relative depth map (not absolute distances) |

### Use Cases
- 3D photo effects
- AR object placement
- Autonomous driving
- Robot navigation
- Image/video editing

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **GPU** | Recommended (2GB+ VRAM) |
| **RAM** | 4GB+ |

---

## Quick Start (10 min)

### Using Transformers

```bash
pip install transformers torch pillow
```

```python
from transformers import pipeline
from PIL import Image
import numpy as np

# Load model
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Estimate depth
image = Image.open("photo.jpg")
result = pipe(image)
depth_map = result["depth"]  # PIL Image

# Save depth visualization
depth_map.save("depth.png")
```

### Detailed Usage

```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image
import numpy as np
import cv2

# Load model
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

# Process image
image = Image.open("photo.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth

# Interpolate to original size
depth = torch.nn.functional.interpolate(
    depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
).squeeze()

# Normalize to 0-255
depth_np = depth.numpy()
depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255
depth_normalized = depth_normalized.astype(np.uint8)

# Apply colormap
depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
cv2.imwrite("depth_colored.png", depth_colored)
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Install and run depth estimation
- [ ] Visualize depth maps
- [ ] Try different images
- [ ] Compare model sizes

### L1: Applications (2-3 hours)
- [ ] 3D photo effect
- [ ] Background blur (portrait mode)
- [ ] Point cloud generation
- [ ] Video depth estimation

### L2: Advanced (4-6 hours)
- [ ] Integrate with 3DGS/NeRF
- [ ] Metric depth estimation
- [ ] Real-time depth
- [ ] Edge deployment

---

## Code Examples

### Depth-Based Background Blur

```python
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def apply_depth_blur(image_path, depth_map, blur_strength=10):
    image = np.array(Image.open(image_path))
    depth = np.array(depth_map)

    # Normalize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())

    # Create blurred version
    blurred = gaussian_filter(image, sigma=(blur_strength, blur_strength, 0))

    # Blend based on depth (far = blurred, close = sharp)
    alpha = depth_norm[:, :, np.newaxis]
    result = (image * (1 - alpha) + blurred * alpha).astype(np.uint8)

    return Image.fromarray(result)
```

### Point Cloud Generation

```python
import numpy as np
import open3d as o3d

def depth_to_pointcloud(image, depth_map, focal_length=500):
    """Convert RGB + depth to 3D point cloud."""
    h, w = depth_map.shape
    image_np = np.array(image)

    # Create mesh grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Convert to 3D
    z = np.array(depth_map)
    x = (u - w/2) * z / focal_length
    y = (v - h/2) * z / focal_length

    # Stack points and colors
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = image_np.reshape(-1, 3) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# Usage
pcd = depth_to_pointcloud(image, depth_map)
o3d.io.write_point_cloud("output.ply", pcd)
o3d.visualization.draw_geometries([pcd])
```

### Video Depth

```python
import cv2
from transformers import pipeline

pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

cap = cv2.VideoCapture("video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("depth_video.mp4", fourcc, 30.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Estimate depth
    result = pipe(pil_image)
    depth = np.array(result["depth"])

    # Colorize
    depth_colored = cv2.applyColorMap(
        (depth / depth.max() * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )

    out.write(depth_colored)

cap.release()
out.release()
```

---

## Model Variants

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `Depth-Anything-V2-Small` | 100MB | Fast | Good |
| `Depth-Anything-V2-Base` | 400MB | Medium | Better |
| `Depth-Anything-V2-Large` | 1.3GB | Slow | Best |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Use smaller model, reduce image size |
| Poor depth on reflections | Known limitation, preprocess image |
| Noisy depth | Post-process with bilateral filter |
| Wrong scale | Depth is relative, not metric |

---

## Resources

- [Depth Anything V2 Paper](https://arxiv.org/abs/2406.09414)
- [HuggingFace Models](https://huggingface.co/depth-anything)
- [GitHub](https://github.com/DepthAnything/Depth-Anything-V2)

---

*Part of [Luno-AI](../../README.md) | Visual AI Track*
