# Stable Diffusion Integration

> **Open-source image generation with Diffusers**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Text-to-image generation using diffusion models |
| **Why** | Free, customizable, runs locally |
| **Versions** | SD 1.5, SDXL, SD 3, Flux |
| **License** | Open weights (various licenses) |

### Model Comparison

| Model | Resolution | Quality | Speed | VRAM |
|-------|------------|---------|-------|------|
| SD 1.5 | 512x512 | ⭐⭐⭐ | Fast | 4GB |
| SDXL | 1024x1024 | ⭐⭐⭐⭐ | Medium | 8GB |
| SD 3 Medium | 1024x1024 | ⭐⭐⭐⭐⭐ | Slow | 12GB |
| Flux.1 Schnell | 1024x1024 | ⭐⭐⭐⭐⭐ | Medium | 12GB |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA 8GB+ VRAM (16GB+ for SDXL) |
| **Python** | 3.10+ |
| **RAM** | 16GB+ |

---

## Quick Start (15 min)

```bash
pip install diffusers transformers accelerate torch
```

### SD 1.5

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe("a photo of an astronaut riding a horse on mars").images[0]
image.save("output.png")
```

### SDXL

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

image = pipe(
    "a majestic lion in a savanna at golden hour, photorealistic",
    num_inference_steps=30
).images[0]
image.save("lion.png")
```

---

## Learning Path

### L0: Basic Generation (1 hour)
- [ ] Install Diffusers
- [ ] Generate first image
- [ ] Experiment with prompts
- [ ] Adjust parameters

### L1: Advanced Prompting (2-3 hours)
- [ ] Negative prompts
- [ ] Prompt weighting
- [ ] Samplers and schedulers
- [ ] CFG scale tuning

### L2: Image-to-Image (4-6 hours)
- [ ] img2img transformation
- [ ] Inpainting
- [ ] Outpainting
- [ ] ControlNet integration

### L3: Custom Models (1-2 days)
- [ ] Load custom checkpoints
- [ ] Use LoRA adapters
- [ ] Fine-tune on custom data
- [ ] Merge models

---

## Code Examples

### Negative Prompts

```python
image = pipe(
    prompt="beautiful landscape, mountains, lake, sunset, photorealistic",
    negative_prompt="ugly, blurry, low quality, watermark, text, deformed",
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]
```

### Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("input.jpg").resize((512, 512))

image = pipe(
    prompt="transform to oil painting style",
    image=init_image,
    strength=0.75,  # 0.0 = no change, 1.0 = complete regeneration
    guidance_scale=7.5
).images[0]
```

### Inpainting

```python
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("photo.jpg")
mask = Image.open("mask.png")  # White = area to regenerate

result = pipe(
    prompt="a cat sitting on the couch",
    image=image,
    mask_image=mask
).images[0]
```

### Using LoRA

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("path/to/lora.safetensors")

image = pipe("trigger_word, your prompt here").images[0]

# Unload when done
pipe.unload_lora_weights()
```

### Memory Optimization

```python
# Enable attention slicing (less VRAM)
pipe.enable_attention_slicing()

# Enable VAE tiling (for large images)
pipe.enable_vae_tiling()

# Use sequential CPU offload
pipe.enable_sequential_cpu_offload()

# Or full model CPU offload
pipe.enable_model_cpu_offload()
```

---

## Samplers/Schedulers

| Scheduler | Quality | Speed | Best For |
|-----------|---------|-------|----------|
| Euler | Good | Fast | General |
| Euler a | Creative | Fast | Artistic |
| DPM++ 2M | Best | Medium | Quality |
| DPM++ 2M Karras | Best | Medium | Recommended |
| LCM | Good | Fastest | Speed |

```python
from diffusers import DPMSolverMultistepScheduler

pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True
)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Enable CPU offload, use fp16, reduce resolution |
| Black images | Check NSFW filter, adjust CFG |
| Blurry output | Increase steps, use better sampler |
| Wrong style | Adjust prompt, use negative prompts |

---

## Resources

- [Diffusers Docs](https://huggingface.co/docs/diffusers/)
- [Civitai](https://civitai.com/) - Models & LoRAs
- [Prompt Book](https://openart.ai/promptbook)
- [Stable Diffusion Art](https://stable-diffusion-art.com/)

---

*Part of [Luno-AI](../../README.md) | Generative AI Track*
