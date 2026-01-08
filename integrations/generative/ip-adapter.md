# IP-Adapter Integration

> **Use images as prompts for style and face transfer**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Image Prompt Adapter for Stable Diffusion |
| **Why** | Transfer style, face, or composition from reference images |
| **Types** | Style, Face, Plus, Full Face |
| **Works With** | SD 1.5, SDXL |

### Use Cases
- Style transfer from reference
- Consistent character faces
- Composition guidance
- Product variations

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
pip install diffusers transformers accelerate
```

### Style Transfer

```python
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

# Load pipeline
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load IP-Adapter
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

# Load style reference
style_image = load_image("style_reference.jpg")

# Generate with style
image = pipe(
    prompt="a cat sitting on a couch",
    ip_adapter_image=style_image,
    num_inference_steps=30
).images[0]

image.save("styled_output.png")
```

### Face Transfer

```python
# Load face-specific IP-Adapter
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus-face_sdxl_vit-h.bin"
)

# Load face reference
face_image = load_image("face_reference.jpg")

# Generate with face
image = pipe(
    prompt="a person in a business suit, professional photo",
    ip_adapter_image=face_image,
    num_inference_steps=30
).images[0]
```

---

## Learning Path

### L0: Basic Style Transfer (1-2 hours)
- [ ] Install IP-Adapter
- [ ] Transfer style from image
- [ ] Adjust adapter strength
- [ ] Compare results

### L1: Face Consistency (3-4 hours)
- [ ] Use face adapter
- [ ] Generate same face in different contexts
- [ ] Combine with ControlNet
- [ ] Multi-image reference

### L2: Advanced (6-8 hours)
- [ ] Multiple IP-Adapters
- [ ] Custom fine-tuning
- [ ] Production pipeline
- [ ] Batch generation

---

## Code Examples

### Adjusting Strength

```python
# Set adapter scale (0.0 to 1.0+)
pipe.set_ip_adapter_scale(0.6)  # Lower = more prompt influence

image = pipe(
    prompt="a cat in a garden",
    ip_adapter_image=style_image,
    num_inference_steps=30
).images[0]
```

### Multiple References

```python
# Use multiple style images
style_images = [
    load_image("style1.jpg"),
    load_image("style2.jpg")
]

pipe.set_ip_adapter_scale([0.5, 0.5])  # Weight for each

image = pipe(
    prompt="landscape painting",
    ip_adapter_image=style_images,
    num_inference_steps=30
).images[0]
```

### Combine with ControlNet

```python
from diffusers import ControlNetModel

# Load ControlNet for pose
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

# Use both
image = pipe(
    prompt="portrait photo",
    ip_adapter_image=style_image,
    image=canny_image,  # ControlNet input
    controlnet_conditioning_scale=0.8
).images[0]
```

### SD 1.5 Version

```python
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin"
)
```

---

## IP-Adapter Variants

| Variant | Best For | Model |
|---------|----------|-------|
| **ip-adapter** | General style | `ip-adapter_sdxl.bin` |
| **ip-adapter-plus** | Better style | `ip-adapter-plus_sdxl_vit-h.bin` |
| **ip-adapter-plus-face** | Face transfer | `ip-adapter-plus-face_sdxl_vit-h.bin` |
| **ip-adapter-full-face** | Full face control | `ip-adapter-full-face_sd15.bin` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Style too strong | Lower adapter scale |
| Style not applied | Increase scale, check model compatibility |
| Face distorted | Use face-specific adapter |
| OOM error | Use smaller base model, enable offload |

---

## Resources

- [IP-Adapter Paper](https://arxiv.org/abs/2308.06721)
- [IP-Adapter GitHub](https://github.com/tencent-ailab/IP-Adapter)
- [HuggingFace Models](https://huggingface.co/h94/IP-Adapter)
- [Diffusers Guide](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)

---

*Part of [Luno-AI](../../README.md) | Generative AI Track*
