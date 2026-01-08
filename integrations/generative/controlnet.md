# ControlNet Integration

> **Add precise control to image generation with poses, edges, depth**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Conditioning modules for spatial control of diffusion |
| **Why** | Control pose, composition, style while generating |
| **Types** | Canny, Pose, Depth, Scribble, Tile, IP-Adapter |

### Control Types

| Type | Input | Best For |
|------|-------|----------|
| **Canny** | Edge detection | Architecture, products |
| **OpenPose** | Body pose | Character art |
| **Depth** | Depth map | Scene composition |
| **Scribble** | Hand drawing | Sketch to image |
| **Tile** | Upscaling | Detail enhancement |
| **Softedge** | Soft edges | Artistic control |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 8GB+ VRAM |
| **Base Model** | SD 1.5 or SDXL |
| **Python** | 3.10+ |

---

## Quick Start (15 min)

```bash
pip install diffusers transformers accelerate controlnet_aux
```

### Canny Edge Control

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import cv2
import numpy as np

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Prepare canny image
image = load_image("input.jpg")
image_np = np.array(image)
canny = cv2.Canny(image_np, 100, 200)
canny_image = Image.fromarray(canny)

# Generate
output = pipe(
    "modern architecture building, professional photo",
    image=canny_image,
    num_inference_steps=30
).images[0]
output.save("controlled.png")
```

### OpenPose Control

```python
from controlnet_aux import OpenposeDetector

# Extract pose
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
pose_image = pose_detector(image)

# Load pose ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)

# Use same pipeline pattern...
```

---

## Learning Path

### L0: Basic Control (1-2 hours)
- [ ] Install ControlNet
- [ ] Use Canny edge control
- [ ] Experiment with conditioning scale
- [ ] Compare with/without control

### L1: Multiple Control Types (3-4 hours)
- [ ] OpenPose for characters
- [ ] Depth for composition
- [ ] Scribble for sketches
- [ ] Combine multiple ControlNets

### L2: Advanced (6-8 hours)
- [ ] Multi-ControlNet stacking
- [ ] SDXL ControlNets
- [ ] IP-Adapter for style
- [ ] Custom ControlNet training

---

## Code Examples

### Multi-ControlNet

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Load multiple ControlNets
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
controlnet_pose = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[controlnet_canny, controlnet_pose],
    torch_dtype=torch.float16
).to("cuda")

# Use both controls
output = pipe(
    "dancer in urban setting",
    image=[canny_image, pose_image],
    controlnet_conditioning_scale=[0.7, 1.0]
).images[0]
```

### SDXL ControlNet

```python
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")
```

### Depth Control

```python
from controlnet_aux import MidasDetector
from diffusers import ControlNetModel

# Extract depth
depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
depth_image = depth_detector(image)

# Load depth ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)
```

### Adjusting Control Strength

```python
# Lower = more creative freedom, Higher = stricter control
output = pipe(
    prompt="your prompt",
    image=control_image,
    controlnet_conditioning_scale=0.5,  # 0.0 to 1.0+
    guidance_scale=7.5
).images[0]
```

---

## Available ControlNets

### SD 1.5
| Type | Model ID |
|------|----------|
| Canny | `lllyasviel/sd-controlnet-canny` |
| OpenPose | `lllyasviel/sd-controlnet-openpose` |
| Depth | `lllyasviel/sd-controlnet-depth` |
| Scribble | `lllyasviel/sd-controlnet-scribble` |
| Tile | `lllyasviel/control_v11f1e_sd15_tile` |

### SDXL
| Type | Model ID |
|------|----------|
| Canny | `diffusers/controlnet-canny-sdxl-1.0` |
| Depth | `diffusers/controlnet-depth-sdxl-1.0` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Control too strong | Lower conditioning_scale |
| Control ignored | Increase conditioning_scale |
| OOM error | Enable CPU offload |
| Wrong output size | Resize control image to match |

---

## Resources

- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [Diffusers ControlNet Guide](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)
- [controlnet_aux](https://github.com/huggingface/controlnet_aux)

---

*Part of [Luno-AI](../../README.md) | Generative AI Track*
