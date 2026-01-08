# Flux Models Integration

> **Next-generation image generation with improved quality and speed**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Black Forest Labs' state-of-the-art text-to-image models |
| **Why** | Superior quality, better prompt following, faster generation |
| **Variants** | Flux.1 Dev, Flux.1 Schnell, Flux.1 Pro |
| **Best For** | High-quality image generation, commercial use |

### Flux vs SDXL

| Aspect | Flux.1 | SDXL |
|--------|--------|------|
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Prompt Following | Excellent | Good |
| Speed (Schnell) | Very Fast | Medium |
| Text Rendering | Better | Limited |
| VRAM | 12-24GB | 8-12GB |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 12GB+ VRAM (24GB for full precision) |
| **Python** | 3.10+ |
| **Disk** | 30GB+ for model weights |

---

## Quick Start (20 min)

### With Diffusers

```bash
pip install diffusers transformers accelerate torch
```

```python
import torch
from diffusers import FluxPipeline

# Load Flux.1 Schnell (fastest)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
).to("cuda")

# Generate image
image = pipe(
    prompt="A serene lake at sunset with mountains in the background, photorealistic",
    num_inference_steps=4,  # Schnell needs only 4 steps!
    guidance_scale=0.0,     # Schnell doesn't use guidance
    height=1024,
    width=1024
).images[0]

image.save("flux_output.png")
```

### With ComfyUI

1. Install ComfyUI
2. Download Flux models to `models/unet/`
3. Download CLIP and VAE to respective folders
4. Use Flux workflow nodes

---

## Learning Path

### L0: Basic Generation (1-2 hours)
- [ ] Install Flux.1 Schnell
- [ ] Generate first images
- [ ] Understand step counts
- [ ] Compare with SDXL

### L1: Quality Optimization (2-3 hours)
- [ ] Try Flux.1 Dev for quality
- [ ] Prompt engineering
- [ ] Resolution variations
- [ ] Batch generation

### L2: Advanced Usage (4-6 hours)
- [ ] ControlNet with Flux
- [ ] LoRA training for Flux
- [ ] Memory optimization
- [ ] Production deployment

---

## Code Examples

### Flux.1 Dev (Higher Quality)

```python
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Dev model uses more steps and guidance
image = pipe(
    prompt="Portrait of a cyberpunk character with neon lights, highly detailed",
    num_inference_steps=50,
    guidance_scale=3.5,
    height=1024,
    width=1024
).images[0]
```

### Memory Optimization

```python
from diffusers import FluxPipeline
import torch

# Enable memory optimizations
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)

# CPU offload for low VRAM
pipe.enable_model_cpu_offload()

# Or sequential CPU offload (slower but less VRAM)
pipe.enable_sequential_cpu_offload()

# VAE slicing for large images
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

### Batch Generation

```python
prompts = [
    "A futuristic city at night",
    "A peaceful forest clearing",
    "An abstract geometric pattern"
]

images = pipe(
    prompt=prompts,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=1024,
    width=1024
).images

for i, img in enumerate(images):
    img.save(f"batch_{i}.png")
```

### With ControlNet

```python
from diffusers import FluxControlNetPipeline, FluxControlNetModel

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-controlnet-canny",
    torch_dtype=torch.bfloat16
)

pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
).to("cuda")

# Use with canny edge image
image = pipe(
    prompt="A detailed architectural rendering",
    image=canny_image,
    num_inference_steps=50
).images[0]
```

---

## Model Variants

| Model | Steps | Quality | Speed | License |
|-------|-------|---------|-------|---------|
| **Flux.1 Schnell** | 4 | ⭐⭐⭐⭐ | ⚡⚡⚡ | Apache 2.0 |
| **Flux.1 Dev** | 50 | ⭐⭐⭐⭐⭐ | ⚡ | Non-commercial |
| **Flux.1 Pro** | 50 | ⭐⭐⭐⭐⭐ | ⚡ | Commercial API |

---

## Prompt Tips

| Technique | Example |
|-----------|---------|
| **Be specific** | "golden hour lighting" vs "nice lighting" |
| **Include style** | "oil painting style", "photorealistic" |
| **Add quality terms** | "highly detailed", "8k resolution" |
| **Describe composition** | "close-up portrait", "wide landscape" |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Enable CPU offload, reduce resolution |
| Slow generation | Use Schnell, enable attention optimizations |
| Poor quality | Use Dev model, increase steps |
| Black images | Check VRAM, try bfloat16 dtype |

---

## Resources

- [Black Forest Labs](https://blackforestlabs.ai/)
- [Flux on HuggingFace](https://huggingface.co/black-forest-labs)
- [Diffusers Flux Guide](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [ComfyUI Flux Nodes](https://github.com/comfyanonymous/ComfyUI)

---

*Part of [Luno-AI](../../README.md) | Generative AI Track*
