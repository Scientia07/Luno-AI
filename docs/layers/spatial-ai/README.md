# Spatial AI: 3D Understanding & Positioning

> **Making machines understand the physical world** - depth perception, 3D reconstruction, spatial reasoning, and positioning.

---

## Why Spatial AI Matters

```
            2D Image                    3D Understanding
         ┌───────────┐               ┌─────────────────────┐
         │           │   Spatial     │   • Object distances│
         │  📷 Flat  │ ───────────▶  │   • Room layout     │
         │   Image   │     AI        │   • Navigation paths│
         │           │               │   • Physical size   │
         └───────────┘               └─────────────────────┘
```

Spatial AI enables: robots that navigate, AR/VR experiences, autonomous vehicles, and 3D content creation.

---

## Core Technologies

### 1. Depth Estimation

**From 2D images to 3D depth**

| Model | Type | Accuracy | Speed |
|-------|------|----------|-------|
| **Depth Anything V2** | Monocular | Excellent | Fast |
| **MiDaS** | Monocular | Very Good | Medium |
| **ZoeDepth** | Metric depth | Excellent | Medium |
| **Marigold** | Diffusion-based | Excellent | Slow |
| **UniDepth** | Universal | Very Good | Fast |

```
Input Image → Depth Model → Depth Map (per-pixel distance)
     ↓
[RGB Photo] → [AI] → [Grayscale: white=close, black=far]
```

### 2. 3D Reconstruction

**Building 3D models from images**

| Technology | Input | Output | Use Case |
|------------|-------|--------|----------|
| **NeRF** | Multi-view images | Novel views | VFX, visualization |
| **3D Gaussian Splatting** | Multi-view images | Real-time 3D | Games, AR |
| **Photogrammetry** | Many photos | Mesh | Scanning, heritage |
| **Single-image 3D** | One image | 3D mesh | Quick prototyping |
| **LRM (Large Reconstruction Model)** | 1-few images | 3D | Generative 3D |

### 3. SLAM (Simultaneous Localization and Mapping)

**Know where you are while building a map**

| Type | Sensors | Use Case |
|------|---------|----------|
| **Visual SLAM** | Cameras | Drones, robots, AR |
| **LiDAR SLAM** | LiDAR | Autonomous vehicles |
| **Visual-Inertial** | Camera + IMU | Mobile AR, drones |
| **RGB-D SLAM** | Depth camera | Indoor robotics |

Key implementations:
- **ORB-SLAM3** - Classical visual SLAM
- **DROID-SLAM** - Deep learning SLAM
- **Gaussian-SLAM** - 3DGS-based SLAM
- **SplaTAM** - Real-time 3DGS SLAM

### 4. Spatial Understanding

**Making sense of 3D environments**

| Capability | Description | Models |
|------------|-------------|--------|
| **Scene Segmentation** | Identify regions in 3D | SAM3D, OpenMask3D |
| **Object Detection 3D** | Locate objects in 3D | PointPillars, VoxelNet |
| **Scene Graphs** | Relationship mapping | 3DSSG, SceneGraphFusion |
| **Occupancy Prediction** | What fills space | TPVFormer, SurroundOcc |

### 5. Positioning & Localization

**Know exactly where you are**

| Technology | Accuracy | Range |
|------------|----------|-------|
| **GPS/GNSS** | 2-10m | Global |
| **RTK GPS** | 1-2cm | Outdoor |
| **Visual Localization** | cm-level | Map-covered |
| **WiFi/BLE** | 1-5m | Indoor |
| **UWB** | 10-30cm | Indoor |
| **LiDAR Matching** | cm-level | Mapped areas |

---

## Depth Maps Deep Dive

### What is a Depth Map?

```
Original Image:           Depth Map:
┌─────────────────┐       ┌─────────────────┐
│    🏠           │       │ ██░░            │  Light = Close
│   Tree   Car    │  ───▶ │ ░░  ████        │  Dark = Far
│                 │       │ ░░░░░░░░        │
└─────────────────┘       └─────────────────┘
```

### Depth Map Types

| Type | Values | Use |
|------|--------|-----|
| **Relative/Inverse** | Arbitrary scale | Comparison, effects |
| **Metric** | Real-world units (m) | Navigation, measurement |
| **Disparity** | Pixel offset | Stereo matching |

### Applications

```
Depth Maps Power:
├── Bokeh/Portrait Mode (blur background)
├── 3D Photos (Facebook 3D)
├── AR Occlusion (objects behind real things)
├── Robot Navigation (avoid obstacles)
├── ControlNet (image generation guidance)
└── Video Effects (volumetric video)
```

---

## 3D Environment Understanding Pipeline

```
    ┌──────────────────────────────────────────────────────────────┐
    │                    SPATIAL AI PIPELINE                        │
    └──────────────────────────────────────────────────────────────┘
                                   │
    ┌───────────┐    ┌───────────┐ │ ┌───────────┐    ┌───────────┐
    │  Cameras  │    │   LiDAR   │ │ │    IMU    │    │    GPS    │
    └─────┬─────┘    └─────┬─────┘ │ └─────┬─────┘    └─────┬─────┘
          │                │       │       │                │
          ▼                ▼       ▼       ▼                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    SENSOR FUSION                              │
    └──────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
    ┌───────────┐            ┌───────────┐           ┌───────────┐
    │   Depth   │            │   SLAM    │           │  Object   │
    │Estimation │            │           │           │ Detection │
    └─────┬─────┘            └─────┬─────┘           └─────┬─────┘
          │                        │                       │
          ▼                        ▼                       ▼
    ┌──────────────────────────────────────────────────────────────┐
    │               SCENE UNDERSTANDING & REASONING                 │
    │  • 3D scene graph  • Semantic map  • Object relationships    │
    └──────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    DOWNSTREAM TASKS                           │
    │  Navigation │ Manipulation │ AR/VR │ Autonomous Driving      │
    └──────────────────────────────────────────────────────────────┘
```

---

## Key Projects & Tools

### Depth Estimation
- [Depth Anything](https://github.com/LiheYoung/Depth-Anything) - State of the art
- [MiDaS](https://github.com/isl-org/MiDaS) - Intel's robust depth
- [ZoeDepth](https://github.com/isl-org/ZoeDepth) - Metric depth

### 3D Reconstruction
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) - NeRF toolkit
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original 3DGS
- [InstantNGP](https://github.com/NVlabs/instant-ngp) - Fast NeRF

### SLAM
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - Visual SLAM
- [RTAB-Map](https://github.com/introlab/rtabmap) - RGB-D SLAM
- [OpenVSLAM](https://github.com/stella-cv/stella_vslam) - Modern visual SLAM

### Point Cloud Processing
- [Open3D](http://www.open3d.org/) - 3D data processing
- [PCL](https://pointclouds.org/) - Point Cloud Library
- [PyTorch3D](https://pytorch3d.org/) - 3D deep learning

---

## Labs & Experiments

| Lab | Focus | Prerequisites |
|-----|-------|---------------|
| `depth-estimation-basics.ipynb` | Run depth models | Python |
| `3d-gaussian-splatting.ipynb` | Create 3D scene | GPU, CUDA |
| `visual-slam-demo.ipynb` | Basic SLAM | OpenCV |
| `point-cloud-processing.ipynb` | Work with 3D points | Open3D |
| `ar-depth-occlusion.ipynb` | AR effects with depth | OpenCV |

---

## Connections to Other Domains

```
Spatial AI connects to:

Vision AI ──────────▶ Spatial AI ◀────────── Robotics
(perception)          (3D understanding)      (action)
                           │
                           ▼
                    ┌──────────────┐
                    │    LLMs +    │
                    │  Embodied AI │
                    │ (reasoning)  │
                    └──────────────┘
```

---

*"Spatial intelligence is what lets minds navigate, manipulate, and reason about the physical world."*
