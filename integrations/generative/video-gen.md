# Video Generation Integration

> **AI-powered video creation from text and images**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Generate videos from text prompts or images |
| **Tools** | Wan2.1, CogVideoX, AnimateDiff, Stable Video Diffusion |
| **Best For** | Short clips, animations, video-to-video |
| **Output** | 2-16 seconds typically |

### Model Comparison

| Model | Quality | Length | VRAM | Local |
|-------|---------|--------|------|-------|
| **Wan2.1** | ⭐⭐⭐⭐⭐ | 5s | 24GB+ | Yes |
| **CogVideoX** | ⭐⭐⭐⭐ | 6s | 18GB+ | Yes |
| **AnimateDiff** | ⭐⭐⭐ | 2-4s | 8GB | Yes |
| **Stable Video** | ⭐⭐⭐⭐ | 4s | 16GB | Yes |
| **Runway Gen-3** | ⭐⭐⭐⭐⭐ | 10s | API | No |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 16GB+ VRAM (24GB+ recommended) |
| **Python** | 3.10+ |
| **Storage** | 50GB+ for models |

---

## Quick Start (30 min)

### AnimateDiff (Easiest Local)

```bash
pip install diffusers transformers accelerate torch
```

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# Load motion adapter
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2"
)

# Load pipeline with motion
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16
).to("cuda")

pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config,
    beta_schedule="linear"
)

# Generate video frames
output = pipe(
    prompt="A cat walking through a garden, cinematic",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25
)

# Export to GIF
export_to_gif(output.frames[0], "animation.gif")
```

### Stable Video Diffusion (Image-to-Video)

```python
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16
).to("cuda")

# Load reference image
image = load_image("reference.jpg")
image = image.resize((1024, 576))

# Generate video
frames = pipe(
    image,
    num_frames=25,
    decode_chunk_size=8
).frames[0]

export_to_video(frames, "output.mp4", fps=7)
```

---

## Learning Path

### L0: First Video (2-3 hours)
- [ ] Install AnimateDiff
- [ ] Generate first animation
- [ ] Export to GIF/MP4
- [ ] Understand frame counts

### L1: Quality Improvement (4-6 hours)
- [ ] Try Stable Video Diffusion
- [ ] Image-to-video workflows
- [ ] Motion LoRAs
- [ ] Prompt engineering for video

### L2: Advanced Generation (1-2 days)
- [ ] CogVideoX setup
- [ ] Wan2.1 installation
- [ ] Long video generation
- [ ] Video-to-video transformations

---

## Code Examples

### CogVideoX

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = "A drone shot flying over a mountain range at sunrise"

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6
).frames[0]

export_to_video(video, "cogvideo_output.mp4", fps=8)
```

### Motion LoRA with AnimateDiff

```python
from diffusers import AnimateDiffPipeline, MotionAdapter

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")

pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16
).to("cuda")

# Load motion LoRA for specific movements
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-in",
    adapter_name="zoom"
)

output = pipe(
    prompt="A flower blooming in timelapse",
    num_frames=16,
    guidance_scale=7.5
)
```

### Video-to-Video

```python
from diffusers import StableVideoDiffusionPipeline
import cv2

# Load source video frames
cap = cv2.VideoCapture("input.mp4")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Use first frame as reference
first_frame = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16
).to("cuda")

# Generate new video from reference
output_frames = pipe(first_frame, num_frames=len(frames)).frames[0]
```

### Batch Video Generation

```python
prompts = [
    "Ocean waves crashing on rocks",
    "Fire burning in a fireplace",
    "Rain falling on a window"
]

for i, prompt in enumerate(prompts):
    output = pipe(
        prompt=prompt,
        num_frames=16,
        guidance_scale=7.5
    )
    export_to_gif(output.frames[0], f"video_{i}.gif")
```

---

## Video Parameters

| Parameter | Recommended | Effect |
|-----------|-------------|--------|
| `num_frames` | 16-49 | Video length |
| `guidance_scale` | 6-8 | Prompt adherence |
| `num_inference_steps` | 25-50 | Quality vs speed |
| `fps` | 7-24 | Playback speed |
| `decode_chunk_size` | 4-8 | VRAM usage |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Enable CPU offload, reduce frames |
| Flickering | Increase steps, use motion LoRA |
| Slow generation | Use smaller model, reduce frames |
| Poor motion | Better prompt, motion adapter |

---

## Resources

- [AnimateDiff](https://github.com/guoyww/AnimateDiff)
- [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
- [CogVideoX](https://github.com/THUDM/CogVideo)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [Diffusers Video Guide](https://huggingface.co/docs/diffusers/using-diffusers/text-img2vid)

---

*Part of [Luno-AI](../../README.md) | Generative AI Track*
