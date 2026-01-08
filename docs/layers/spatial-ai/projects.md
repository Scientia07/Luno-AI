# Spatial AI: Projects & Comparisons

> **Hands-on projects and framework comparisons for 3D Vision and Spatial Intelligence**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Depth Map Estimation
**Goal**: Estimate depth from single images

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | Depth Anything V2 |
| Skills | Monocular depth, visualization |

**Tasks**:
- [ ] Install Depth Anything V2
- [ ] Run on single images
- [ ] Visualize depth maps
- [ ] Export as 3D point cloud
- [ ] Batch process folder

**Starter Code**:
```python
import torch
from transformers import pipeline

# Load model
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Estimate depth
result = depth_estimator("image.jpg")
depth_map = result["depth"]

# Save depth visualization
import numpy as np
import cv2
depth_np = np.array(depth_map)
depth_colored = cv2.applyColorMap(
    (depth_np / depth_np.max() * 255).astype(np.uint8),
    cv2.COLORMAP_INFERNO
)
cv2.imwrite("depth.png", depth_colored)
```

---

#### Project 2: Point Cloud Viewer
**Goal**: Visualize and manipulate 3D point clouds

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 3-4 hours |
| Technologies | Open3D |
| Skills | 3D visualization, point cloud basics |

**Tasks**:
- [ ] Load point cloud (PLY/PCD)
- [ ] Visualize with Open3D
- [ ] Apply transformations
- [ ] Filter and downsample
- [ ] Save processed cloud

---

### Intermediate Projects (L2)

#### Project 3: 3D Gaussian Splatting
**Goal**: Create photorealistic 3D scenes

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | gaussian-splatting |
| Skills | Novel view synthesis, training |

**Tasks**:
- [ ] Capture multi-view images
- [ ] Run COLMAP for poses
- [ ] Train Gaussian Splatting
- [ ] Render novel views
- [ ] Export for viewer

---

#### Project 4: SLAM Pipeline
**Goal**: Build simultaneous localization and mapping

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 8-12 hours |
| Technologies | ORB-SLAM3 or RTAB-Map |
| Skills | Visual odometry, mapping |

**Tasks**:
- [ ] Set up stereo/RGB-D camera
- [ ] Run visual odometry
- [ ] Build 3D map
- [ ] Detect loop closures
- [ ] Export and visualize map

---

#### Project 5: Object 6DOF Pose Estimation
**Goal**: Estimate 3D pose of objects

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | FoundationPose or MegaPose |
| Skills | Pose estimation, 3D geometry |

**Tasks**:
- [ ] Load object CAD model
- [ ] Detect object in image
- [ ] Estimate 6DOF pose
- [ ] Visualize in 3D
- [ ] Track across frames

---

### Advanced Projects (L3-L4)

#### Project 6: NeRF Scene Reconstruction
**Goal**: Create neural radiance fields

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | nerfstudio |
| Skills | Neural rendering, optimization |

**Tasks**:
- [ ] Capture training images
- [ ] Process camera poses
- [ ] Train NeRF model
- [ ] Render novel views
- [ ] Export mesh/point cloud

**Architecture**:
```
Images + Poses → Neural Network → View Synthesis
                      ↓
              Volume Rendering
                      ↓
               Novel Views
```

---

#### Project 7: 3D Scene Understanding
**Goal**: Segment and label 3D scenes

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | OpenScene or LERF |
| Skills | 3D segmentation, language grounding |

**Tasks**:
- [ ] Reconstruct 3D scene
- [ ] Run 3D semantic segmentation
- [ ] Ground language queries
- [ ] Visualize labeled scene
- [ ] Query: "Find all chairs"

---

#### Project 8: AR Object Placement
**Goal**: Place virtual objects in real scenes

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | ARKit/ARCore + plane detection |
| Skills | AR, plane estimation, rendering |

**Tasks**:
- [ ] Detect planes in scene
- [ ] Estimate plane normals
- [ ] Place virtual object
- [ ] Handle occlusion
- [ ] Real-time rendering

---

## Framework Comparisons

### Comparison 1: Depth Estimation Models

**Question**: Which depth model for your needs?

| Model | Speed | Quality | Metric | Best For |
|-------|-------|---------|--------|----------|
| **Depth Anything V2** | Fast | ⭐⭐⭐⭐⭐ | Relative | General |
| **MiDaS** | Medium | ⭐⭐⭐⭐ | Relative | Compatibility |
| **ZoeDepth** | Slow | ⭐⭐⭐⭐⭐ | Metric | Real measurements |
| **Marigold** | Slow | ⭐⭐⭐⭐⭐ | Relative | Detail |
| **DepthPro** | Fast | ⭐⭐⭐⭐ | Metric | Apple ecosystem |

**Lab Exercise**: Compare depth maps on same scenes.

```python
# Quick comparison
from transformers import pipeline

models = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "Intel/dpt-large",
]

for model in models:
    pipe = pipeline("depth-estimation", model=model)
    result = pipe("scene.jpg")
    # Compare outputs
```

---

### Comparison 2: 3D Reconstruction Methods

**Question**: Which method for novel view synthesis?

| Method | Training | Quality | Speed (render) | Storage |
|--------|----------|---------|----------------|---------|
| **3D Gaussian Splatting** | Fast | ⭐⭐⭐⭐⭐ | Real-time | Large |
| **NeRF** | Slow | ⭐⭐⭐⭐ | Slow | Small |
| **Instant-NGP** | Fast | ⭐⭐⭐⭐ | Fast | Medium |
| **SfM + MVS** | N/A | ⭐⭐⭐ | N/A | Medium |

**Lab Exercise**: Reconstruct same scene with different methods.

---

### Comparison 3: SLAM Systems

**Question**: Which SLAM for your application?

| System | Sensor | Real-time | Map Quality | Ease |
|--------|--------|-----------|-------------|------|
| **ORB-SLAM3** | Mono/Stereo/RGB-D | Yes | ⭐⭐⭐⭐ | Hard |
| **RTAB-Map** | RGB-D/Stereo | Yes | ⭐⭐⭐⭐ | Medium |
| **DROID-SLAM** | RGB | No | ⭐⭐⭐⭐⭐ | Hard |
| **Gaussian-SLAM** | RGB-D | Yes | ⭐⭐⭐⭐⭐ | Hard |

**Lab Exercise**: Build maps with different SLAM systems.

---

### Comparison 4: Point Cloud Libraries

**Question**: Which library for 3D processing?

| Library | Speed | Features | Ease | Best For |
|---------|-------|----------|------|----------|
| **Open3D** | Fast | ⭐⭐⭐⭐⭐ | Easy | General |
| **PCL** | Fastest | ⭐⭐⭐⭐⭐ | Hard | Production |
| **PyTorch3D** | Fast | ⭐⭐⭐⭐ | Medium | Deep learning |
| **Trimesh** | Medium | ⭐⭐⭐ | Easy | Meshes |

**Lab Exercise**: Process same point cloud with each library.

---

## Hands-On Labs

### Lab 1: Depth Estimation (2 hours)
```
Image → Depth Model → Depth Map → Point Cloud → Visualize
```

### Lab 2: Point Cloud Processing (3 hours)
```
Load Cloud → Filter → Downsample → Segment → Transform → Save
```

### Lab 3: Gaussian Splatting (6 hours)
```
Capture Images → COLMAP → Train 3DGS → Render Views → Export
```

### Lab 4: Visual SLAM (8 hours)
```
Setup Camera → Run SLAM → Build Map → Loop Closure → Export
```

### Lab 5: NeRF Training (1 day)
```
Capture Data → Process Poses → Train NeRF → Render → Extract Mesh
```

---

## 3D Math Fundamentals

### Pattern 1: Homogeneous Coordinates
```
[x, y, z] → [x, y, z, 1]
Transform: [R|t] × [x, y, z, 1]^T
```

### Pattern 2: Camera Projection
```
[u, v, 1] = K × [R|t] × [X, Y, Z, 1]^T
K = intrinsic matrix (focal length, principal point)
```

### Pattern 3: Depth to Point Cloud
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth[v, u]
```

### Pattern 4: ICP Alignment
```
Repeat:
  1. Find closest points
  2. Compute optimal R, t
  3. Apply transform
Until converged
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **3D Accuracy** | 35 | Geometric correctness |
| **Visual Quality** | 25 | Rendering quality |
| **Performance** | 20 | Speed, efficiency |
| **Code Quality** | 10 | Clean, documented |
| **Innovation** | 10 | Creative applications |

---

## Resources

- [Open3D](http://www.open3d.org/)
- [nerfstudio](https://docs.nerf.studio/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything)
- [PyTorch3D](https://pytorch3d.org/)

---

*Part of [Luno-AI](../../../README.md) | Spatial AI Track*
