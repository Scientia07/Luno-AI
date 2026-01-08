# QLoRA Integration

> **Fine-tune large models on consumer GPUs**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Quantized LoRA - 4-bit fine-tuning |
| **Why** | Fine-tune 65B+ models on 24GB GPU |
| **VRAM** | 4-12GB (vs 80GB+ normally) |
| **Best For** | Large model fine-tuning on limited hardware |

### QLoRA vs LoRA vs Full

| Method | 7B Model VRAM | 70B Model VRAM | Quality |
|--------|---------------|----------------|---------|
| Full Fine-tune | 80GB+ | 500GB+ | ⭐⭐⭐⭐⭐ |
| LoRA | 16GB | 80GB+ | ⭐⭐⭐⭐ |
| QLoRA | 6GB | 24GB | ⭐⭐⭐⭐ |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 8GB+ VRAM |
| **Python** | 3.10+ |
| **CUDA** | 11.8+ |

---

## Quick Start (1 hour)

### Installation

```bash
pip install transformers peft datasets accelerate bitsandbytes trl
```

### Basic QLoRA Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model in 4-bit
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files="training_data.jsonl")

# Training config
training_args = SFTConfig(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    max_seq_length=512
)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text"
)

trainer.train()
model.save_pretrained("./my-qlora")
```

---

## Learning Path

### L0: First QLoRA (2-3 hours)
- [ ] Install dependencies
- [ ] Load 4-bit model
- [ ] Train on small dataset
- [ ] Test inference

### L1: Optimization (4-6 hours)
- [ ] Tune quantization settings
- [ ] Gradient checkpointing
- [ ] Memory optimization
- [ ] Evaluation setup

### L2: Large Models (1-2 days)
- [ ] 70B model training
- [ ] Multi-GPU setup
- [ ] Merge and quantize
- [ ] Production deployment

---

## Code Examples

### Training 70B Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 4-bit config optimized for 70B
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load 70B model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# More aggressive LoRA for large models
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Memory Optimization

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,  # Save memory
    fp16=True,
    optim="paged_adamw_8bit",     # 8-bit optimizer
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)
```

### Inference with QLoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load base in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load QLoRA adapter
model = PeftModel.from_pretrained(base_model, "./my-qlora")

# Generate
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Merge QLoRA to Full Precision

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base in full precision
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and merge adapter
model = PeftModel.from_pretrained(base_model, "./my-qlora")
merged = model.merge_and_unload()

# Save merged model
merged.save_pretrained("./merged-model")

# Optionally quantize for deployment
# Use llama.cpp or GPTQ for quantization
```

### Chat Format Training

```python
from datasets import Dataset

def format_chat(example):
    """Format for chat fine-tuning"""
    chat = f"""<s>[INST] {example['instruction']}

{example['input']} [/INST]

{example['output']}</s>"""
    return {"text": chat}

# Prepare dataset
raw_data = [
    {
        "instruction": "Summarize this article",
        "input": "Long article...",
        "output": "Summary..."
    }
]

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_chat)
```

---

## Quantization Settings

| Setting | Options | Effect |
|---------|---------|--------|
| `load_in_4bit` | True/False | 4-bit quantization |
| `bnb_4bit_quant_type` | nf4, fp4 | Quantization type |
| `bnb_4bit_compute_dtype` | float16, bfloat16 | Compute precision |
| `bnb_4bit_use_double_quant` | True/False | Nested quantization |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM | Lower batch size, enable gradient checkpointing |
| NaN loss | Lower learning rate, check data |
| Slow training | Use bfloat16, optimize batch size |
| CUDA errors | Update bitsandbytes, check CUDA version |

---

## Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [PEFT QLoRA Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [TRL Documentation](https://huggingface.co/docs/trl)

---

*Part of [Luno-AI](../../README.md) | LLM Track*
