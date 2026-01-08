# Generative AI: Projects & Comparisons

> **Hands-on projects and framework comparisons for Generative AI**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Text-to-Image Generator
**Goal**: Generate images from text prompts

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | ComfyUI or Automatic1111 |
| Skills | Prompt engineering, UI basics |

**Tasks**:
- [ ] Install ComfyUI/A1111
- [ ] Load Stable Diffusion model
- [ ] Generate images from prompts
- [ ] Experiment with different samplers
- [ ] Save and organize outputs

**Starter Code (Diffusers)**:
```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

image = pipe("a photo of an astronaut riding a horse on mars").images[0]
image.save("astronaut.png")
```

---

#### Project 2: Image Variation Generator
**Goal**: Create variations of existing images

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | img2img pipelines |
| Skills | Image manipulation, strength tuning |

**Tasks**:
- [ ] Load source image
- [ ] Apply img2img transformation
- [ ] Experiment with different strengths
- [ ] Create style transfers
- [ ] Batch process multiple images

---

### Intermediate Projects (L2)

#### Project 3: LoRA Character Creator
**Goal**: Train a LoRA for consistent character generation

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | kohya_ss or ComfyUI trainer |
| Skills | Dataset preparation, LoRA training |

**Tasks**:
- [ ] Collect 15-30 character images
- [ ] Prepare captions/tags
- [ ] Configure LoRA training
- [ ] Train on GPU (4-8 hours)
- [ ] Test with various prompts
- [ ] Optimize trigger words

---

#### Project 4: ControlNet Pose Transfer
**Goal**: Control generation with poses and sketches

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | ControlNet models |
| Skills | Conditioning, multi-model inference |

**Tasks**:
- [ ] Install ControlNet models
- [ ] Extract pose from reference image
- [ ] Generate with pose conditioning
- [ ] Combine multiple ControlNets
- [ ] Build pose-to-image workflow

---

#### Project 5: Inpainting Editor
**Goal**: Selectively edit parts of images

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | Inpainting models |
| Skills | Masking, prompt targeting |

**Tasks**:
- [ ] Create masks (manual or automatic)
- [ ] Configure inpainting pipeline
- [ ] Edit specific regions
- [ ] Blend seamlessly with original
- [ ] Build simple editing UI

---

### Advanced Projects (L3-L4)

#### Project 6: ComfyUI Custom Workflow
**Goal**: Build complex generation pipeline

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 8-12 hours |
| Technologies | ComfyUI + custom nodes |
| Skills | Node-based workflows |

**Tasks**:
- [ ] Design multi-stage workflow
- [ ] Chain: prompt → generate → upscale → face fix
- [ ] Add ControlNet conditioning
- [ ] Implement batch processing
- [ ] Export as API workflow

**Architecture**:
```
Prompt → CLIP Encode → KSampler → VAE Decode → Upscale → Face Restore → Save
           ↓
      ControlNet
      (optional)
```

---

#### Project 7: AI Avatar Generator
**Goal**: Generate consistent character across styles

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | IP-Adapter + LoRA |
| Skills | Face consistency, style transfer |

**Tasks**:
- [ ] Train character LoRA
- [ ] Use IP-Adapter for face reference
- [ ] Generate in multiple styles
- [ ] Ensure consistency across outputs
- [ ] Build avatar generation pipeline

---

#### Project 8: Video Generation Pipeline
**Goal**: Generate short video clips from text

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | AnimateDiff or SVD |
| Skills | Video diffusion, motion control |

**Tasks**:
- [ ] Set up AnimateDiff/SVD
- [ ] Generate video from text
- [ ] Add motion control (MotionCtrl)
- [ ] Interpolate frames for smoothness
- [ ] Export as MP4/GIF

---

## Framework Comparisons

### Comparison 1: Image Generation Backends

**Question**: Which tool for your workflow?

| Tool | Ease | Features | Speed | Customization |
|------|------|----------|-------|---------------|
| **ComfyUI** | Medium | ⭐⭐⭐⭐⭐ | Fast | Full control |
| **Automatic1111** | Easy | ⭐⭐⭐⭐ | Medium | Extensions |
| **Fooocus** | Easiest | ⭐⭐⭐ | Fast | Limited |
| **InvokeAI** | Easy | ⭐⭐⭐⭐ | Medium | Unified canvas |
| **Diffusers** | Code | ⭐⭐⭐⭐⭐ | Fastest | Python full |

**Lab Exercise**: Generate same prompt in all 5, compare quality and time.

```python
# Diffusers example
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe.to("cuda")

image = pipe("a majestic lion in a savanna at sunset").images[0]
```

---

### Comparison 2: Model Architectures

**Question**: Which base model for your needs?

| Model | Resolution | Quality | Speed | VRAM |
|-------|------------|---------|-------|------|
| **SD 1.5** | 512x512 | ⭐⭐⭐ | Fast | 4GB |
| **SD 2.1** | 768x768 | ⭐⭐⭐ | Medium | 6GB |
| **SDXL** | 1024x1024 | ⭐⭐⭐⭐ | Slow | 8GB |
| **SD 3** | 1024x1024 | ⭐⭐⭐⭐⭐ | Slow | 12GB |
| **Flux** | 1024x1024 | ⭐⭐⭐⭐⭐ | Slow | 16GB+ |

**Lab Exercise**: Generate same prompt, compare quality at different resolutions.

---

### Comparison 3: Fine-tuning Methods

**Question**: How to customize models?

| Method | Training Time | Data Needed | Quality | Flexibility |
|--------|---------------|-------------|---------|-------------|
| **LoRA** | 1-4 hours | 10-50 images | Good | Single concept |
| **LyCORIS** | 2-6 hours | 10-50 images | Better | More control |
| **DreamBooth** | 4-8 hours | 5-20 images | Excellent | Full model |
| **Textual Inversion** | 2-4 hours | 3-10 images | Limited | Embedding only |
| **Full Fine-tune** | 1-3 days | 1000+ images | Best | Complete |

**Lab Exercise**: Train LoRA vs DreamBooth on same dataset, compare.

---

### Comparison 4: Conditioning Methods

**Question**: How to control generation?

| Method | Control Type | Precision | Use Case |
|--------|--------------|-----------|----------|
| **ControlNet-Pose** | Body pose | High | Character posing |
| **ControlNet-Canny** | Edge detection | Medium | Sketch-to-image |
| **ControlNet-Depth** | 3D structure | High | Scene composition |
| **IP-Adapter** | Face/style | Medium | Style transfer |
| **T2I-Adapter** | Lightweight | Low | Fast conditioning |

**Lab Exercise**: Use same reference image with different ControlNets.

---

## Hands-On Labs

### Lab 1: Basic Generation (2 hours)
```
Install → Load Model → Generate → Iterate Prompts → Save Best
```

### Lab 2: ControlNet Pipeline (3 hours)
```
Reference Image → Extract Control → Generate → Compare Methods
```

### Lab 3: LoRA Training (6 hours)
```
Collect Images → Caption → Train → Test → Optimize
```

### Lab 4: ComfyUI Workflow (4 hours)
```
Design Nodes → Connect → Test → Add Conditions → Export API
```

### Lab 5: Video Generation (4 hours)
```
Setup AnimateDiff → Text-to-Video → Add Motion → Export
```

---

## Prompt Engineering Patterns

### Pattern 1: Quality Boosters
```
masterpiece, best quality, highly detailed, sharp focus, 8k uhd
```

### Pattern 2: Negative Prompts
```
ugly, blurry, low quality, watermark, text, deformed, bad anatomy
```

### Pattern 3: Style Transfer
```
[subject], in the style of [artist], [medium], [lighting]
```

### Pattern 4: Composition Control
```
[subject], [camera angle], [lighting], [background], [mood]
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Image Quality** | 35 | Visual appeal, coherence |
| **Prompt Control** | 25 | Subject follows prompt |
| **Technical Setup** | 20 | Pipeline efficiency |
| **Creativity** | 10 | Novel applications |
| **Documentation** | 10 | Workflow documented |

---

## Resources

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [Diffusers Docs](https://huggingface.co/docs/diffusers/)
- [Civitai](https://civitai.com/) - Models & LoRAs
- [ControlNet Guide](https://github.com/lllyasviel/ControlNet)
- [kohya_ss](https://github.com/kohya-ss/sd-scripts) - LoRA training

---

*Part of [Luno-AI](../../../README.md) | Generative AI Track*
