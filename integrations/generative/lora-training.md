# LoRA Training Integration

> **Fine-tune Stable Diffusion with minimal resources**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Low-Rank Adaptation for image generation models |
| **Why** | Train custom styles/characters with 10-50 images |
| **Output** | Small adapter file (10-200MB) |
| **Time** | 1-4 hours on consumer GPU |

### LoRA vs Full Fine-tuning

| Aspect | LoRA | Full Fine-tune |
|--------|------|----------------|
| VRAM | 8-12GB | 24GB+ |
| Training time | 1-4 hours | 8-24 hours |
| File size | 10-200MB | 2-6GB |
| Quality | Very good | Best |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 8GB+ VRAM (12GB recommended) |
| **Dataset** | 10-50 high-quality images |
| **Tool** | kohya_ss or diffusers |

---

## Quick Start (2-3 hours)

### Using Diffusers

```bash
pip install diffusers peft accelerate transformers
```

```python
from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch

# After training, load LoRA
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("path/to/lora.safetensors")

# Generate with trigger word
image = pipe("my_trigger_word, a portrait photo").images[0]

# Unload when done
pipe.unload_lora_weights()
```

### Using kohya_ss (Recommended)

```bash
# Clone kohya_ss
git clone https://github.com/kohya-ss/sd-scripts
cd sd-scripts
pip install -r requirements.txt
```

**Dataset structure**:
```
dataset/
├── 10_my_trigger/
│   ├── image1.jpg
│   ├── image1.txt  # Caption: "my_trigger, description"
│   ├── image2.jpg
│   └── image2.txt
```

**Training command**:
```bash
accelerate launch train_network.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --train_data_dir="dataset" \
    --output_dir="output" \
    --output_name="my_lora" \
    --save_model_as=safetensors \
    --network_module=networks.lora \
    --network_dim=32 \
    --network_alpha=16 \
    --max_train_epochs=10 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="fp16" \
    --cache_latents \
    --resolution=1024
```

---

## Learning Path

### L0: First LoRA (2-3 hours)
- [ ] Prepare 15-20 images
- [ ] Create captions
- [ ] Train with default settings
- [ ] Test results

### L1: Optimization (4-6 hours)
- [ ] Tune network rank (dim)
- [ ] Adjust learning rate
- [ ] Regularization images
- [ ] Multiple concepts

### L2: Advanced (1-2 days)
- [ ] LoRA merging
- [ ] LyCORIS/LoHa
- [ ] Textual Inversion combo
- [ ] Production workflow

---

## Code Examples

### Training with Diffusers

```python
from diffusers import AutoPipelineForText2Image
from peft import LoraConfig, get_peft_model

# This is simplified - full training requires more setup
lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1
)

# See diffusers train_dreambooth_lora.py for full example
```

### Caption Generation

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to("cuda")

def caption_image(image_path, trigger_word):
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return f"{trigger_word}, {caption}"

# Generate captions for dataset
for img in image_files:
    caption = caption_image(img, "my_trigger")
    with open(img.replace(".jpg", ".txt"), "w") as f:
        f.write(caption)
```

### Multiple LoRAs

```python
# Load multiple LoRAs with different weights
pipe.load_lora_weights("style_lora.safetensors", adapter_name="style")
pipe.load_lora_weights("character_lora.safetensors", adapter_name="character")

# Set weights
pipe.set_adapters(["style", "character"], adapter_weights=[0.7, 1.0])

image = pipe("style_trigger, character_trigger, portrait").images[0]
```

### LoRA Merging

```python
from safetensors.torch import load_file, save_file

lora1 = load_file("lora1.safetensors")
lora2 = load_file("lora2.safetensors")

merged = {}
for key in lora1:
    if key in lora2:
        merged[key] = 0.5 * lora1[key] + 0.5 * lora2[key]
    else:
        merged[key] = lora1[key]

save_file(merged, "merged_lora.safetensors")
```

---

## Training Parameters

| Parameter | Recommended | Effect |
|-----------|-------------|--------|
| `network_dim` | 32-128 | Higher = more capacity |
| `network_alpha` | dim/2 | Scaling factor |
| `learning_rate` | 1e-4 to 5e-5 | Lower = stable, slower |
| `max_train_epochs` | 10-20 | More for complex concepts |
| `resolution` | 512 (SD1.5) / 1024 (SDXL) | Match base model |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Overfitting | Lower epochs, add regularization images |
| Underfitting | Increase epochs, higher rank |
| Style not learned | Better captions, more images |
| OOM error | Lower batch size, use 8bit optimizer |

---

## Resources

- [kohya_ss](https://github.com/kohya-ss/sd-scripts)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Civitai](https://civitai.com/) - LoRA examples
- [Diffusers LoRA Training](https://huggingface.co/docs/diffusers/training/lora)

---

*Part of [Luno-AI](../../README.md) | Generative AI Track*
