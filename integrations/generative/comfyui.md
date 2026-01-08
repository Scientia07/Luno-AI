# ComfyUI Integration

> **Category**: Generative AI
> **Difficulty**: Beginner
> **Setup Time**: 1-2 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
ComfyUI is a powerful, modular Stable Diffusion interface that lets you build image generation workflows by connecting nodes. Think of it as "visual programming" for AI art.

### Why Use It
- **Visual Workflows**: Drag-and-drop node-based interface
- **Full Control**: Every parameter accessible
- **Efficient**: Optimized for speed and VRAM
- **Extensible**: Massive ecosystem of custom nodes
- **Reproducible**: Save and share exact workflows

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Text-to-Image | Generate from prompts |
| Image-to-Image | Transform existing images |
| ControlNet | Structural guidance |
| IP-Adapter | Style from reference images |
| Inpainting | Edit specific regions |
| Upscaling | High-resolution output |
| Video | AnimateDiff, frame interpolation |

### Interface Comparison
| Interface | Learning Curve | Control | Speed |
|-----------|---------------|---------|-------|
| **ComfyUI** | Steep | Maximum | Fastest |
| AUTOMATIC1111 | Moderate | High | Medium |
| Forge | Moderate | High | Fast |
| Fooocus | Easy | Limited | Medium |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4 GB VRAM | 8-12 GB VRAM |
| RAM | 8 GB | 16-32 GB |
| Storage | 20 GB | 100 GB (models) |

### Software Dependencies
```bash
# Git
# Python 3.10+
# NVIDIA GPU drivers (CUDA 11.8+)
```

### Prior Knowledge
- [x] Basic understanding of Stable Diffusion
- [ ] Image generation concepts (helpful)

---

## Quick Start (30 minutes)

### 1. Install ComfyUI

```bash
# Clone repository
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install PyTorch (NVIDIA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 2. Download a Model

```bash
# Download SDXL (recommended starting model)
# Place in: ComfyUI/models/checkpoints/

# Option A: Hugging Face CLI
pip install huggingface_hub
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir models/checkpoints/

# Option B: Manual download from CivitAI or Hugging Face
```

### 3. Run ComfyUI

```bash
python main.py

# Open browser: http://127.0.0.1:8188
```

### 4. Generate First Image

1. Load default workflow (Queue Prompt → Load Default)
2. Select your checkpoint model
3. Enter a prompt: "a beautiful sunset over mountains, masterpiece"
4. Click "Queue Prompt"

---

## Full Setup

### Directory Structure

```
ComfyUI/
├── models/
│   ├── checkpoints/     # Main models (SD, SDXL, Flux)
│   ├── vae/             # VAE models
│   ├── loras/           # LoRA models
│   ├── controlnet/      # ControlNet models
│   ├── upscale_models/  # Upscalers
│   ├── embeddings/      # Textual inversions
│   └── clip/            # CLIP models
├── input/               # Input images
├── output/              # Generated images
└── custom_nodes/        # Extensions
```

### Recommended Models

| Type | Model | Size | Link |
|------|-------|------|------|
| **SDXL Base** | sd_xl_base_1.0 | 6.5 GB | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| **SDXL Refiner** | sd_xl_refiner_1.0 | 6.2 GB | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0) |
| **Flux.1-dev** | flux1-dev | 12 GB | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| **RealVisXL** | RealVisXL_V4.0 | 6.5 GB | [CivitAI](https://civitai.com/models/139562) |

### Essential Custom Nodes

```bash
cd custom_nodes

# ComfyUI Manager (must have!)
git clone https://github.com/ltdrdata/ComfyUI-Manager

# Restart ComfyUI, then use Manager to install more nodes
```

**Recommended nodes via Manager:**
- ComfyUI-Impact-Pack (face detection, inpainting)
- ComfyUI-AnimateDiff-Evolved (video)
- ComfyUI-Controlnet-Aux (preprocessors)
- ComfyUI-IPAdapter-plus (style transfer)
- rgthree-comfy (workflow utilities)

---

## Learning Path

### L0: Basic Generation (1 hour)
**Goal**: Generate images with text prompts

Basic workflow nodes:
```
[Load Checkpoint] → [CLIP Text Encode (Prompt)] → [KSampler] → [VAE Decode] → [Save Image]
                 → [CLIP Text Encode (Negative)] ↗
                 → [Empty Latent Image] ↗
```

Key settings:
- **Steps**: 20-30 (more = quality, slower)
- **CFG**: 7-8 (prompt adherence)
- **Sampler**: euler_ancestral, dpmpp_2m
- **Scheduler**: normal, karras

### L1: ControlNet & Conditioning (2 hours)
**Goal**: Guide generation with images

```
[Load Image] → [Canny Edge] → [ControlNet Apply] → [KSampler]
```

ControlNet types:
- **Canny**: Edge detection
- **Depth**: 3D structure
- **OpenPose**: Body pose
- **Tile**: Upscaling/detail

### L2: LoRA & Styles (2 hours)
**Goal**: Apply custom styles

```
[Load Checkpoint] → [Load LoRA] → [CLIP Text Encode]
```

LoRA usage:
- Download from CivitAI
- Place in `models/loras/`
- Stack multiple LoRAs

### L3: Advanced Workflows (4+ hours)
**Goal**: Complex multi-stage pipelines

- IP-Adapter for style reference
- Face restoration (FaceDetailer)
- Upscaling workflows
- AnimateDiff for video

---

## Code Examples

### Example 1: API Usage
```python
import requests
import json
import random

def generate_image(prompt, negative_prompt="", seed=-1):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 7,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]}
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}
        }
    }

    response = requests.post(
        "http://127.0.0.1:8188/prompt",
        json={"prompt": workflow}
    )
    return response.json()

# Generate
result = generate_image("a majestic lion in the savanna, golden hour")
print(f"Queued: {result}")
```

### Example 2: Queue and Wait
```python
import websocket
import uuid
import json

def queue_and_wait(workflow):
    client_id = str(uuid.uuid4())
    ws = websocket.create_connection(f"ws://127.0.0.1:8188/ws?clientId={client_id}")

    # Queue prompt
    response = requests.post(
        "http://127.0.0.1:8188/prompt",
        json={"prompt": workflow, "client_id": client_id}
    )
    prompt_id = response.json()["prompt_id"]

    # Wait for completion
    while True:
        msg = ws.recv()
        data = json.loads(msg)
        if data["type"] == "executing" and data["data"]["node"] is None:
            break

    ws.close()
    return prompt_id
```

### Example 3: Get Generated Image
```python
def get_image(prompt_id, filename_prefix="ComfyUI"):
    # Get history
    response = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
    history = response.json()

    # Find output image
    outputs = history[prompt_id]["outputs"]
    for node_id, output in outputs.items():
        if "images" in output:
            for image in output["images"]:
                # Get image data
                img_response = requests.get(
                    f"http://127.0.0.1:8188/view",
                    params={"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image["type"]}
                )
                return img_response.content
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| ControlNet | Structural control | [controlnet.md](./controlnet.md) |
| IP-Adapter | Style transfer | [ip-adapter.md](./ip-adapter.md) |
| LoRA | Custom styles | [lora-training.md](./lora-training.md) |
| Flux | Latest models | [flux.md](./flux.md) |

### Model Sources
| Source | Type | URL |
|--------|------|-----|
| CivitAI | Community models | [civitai.com](https://civitai.com) |
| Hugging Face | Official models | [huggingface.co](https://huggingface.co) |
| OpenArt | Workflows | [openart.ai](https://openart.ai) |

### Workflow Sharing
- Save: Right-click → Save (API Format)
- Share: Export as JSON
- Import: Drag JSON onto canvas

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of VRAM
**Symptoms**: CUDA out of memory
**Solution**:
```bash
# Enable memory optimizations
python main.py --lowvram
# or
python main.py --cpu  # Very slow but works
```

Or in workflow:
- Reduce image size
- Use FP16 models
- Disable preview images

#### Issue 2: Model Not Loading
**Symptoms**: "Model not found" error
**Solution**:
- Check model is in correct folder
- Verify filename matches exactly
- Try refreshing in ComfyUI

#### Issue 3: Slow Generation
**Symptoms**: Long generation times
**Solution**:
- Use FP16 instead of FP32
- Reduce steps (20 is often enough)
- Use faster samplers (euler_ancestral)
- Enable xformers: `pip install xformers`

### Performance Tips
- Install xformers for 20-30% speedup
- Use SDXL turbo for fast previews
- Enable tiled VAE for large images
- Queue multiple prompts for batch

---

## Resources

### Official
- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

### Tutorials
- [ComfyUI Academy](https://www.comfyworkflows.com/)
- [OpenArt Workflows](https://openart.ai/workflows)

### Community
- [Reddit r/comfyui](https://reddit.com/r/comfyui)
- [Discord](https://discord.gg/comfyui)

---

## Related Integrations

| Next Step | Why | Link |
|-----------|-----|------|
| ControlNet | Structural control | [controlnet.md](./controlnet.md) |
| IP-Adapter | Style from images | [ip-adapter.md](./ip-adapter.md) |
| LoRA Training | Custom models | [lora-training.md](./lora-training.md) |
| Flux | Latest models | [flux.md](./flux.md) |

---

*Part of [Luno-AI Integration Hub](../_index.md) | Generative AI Track*
