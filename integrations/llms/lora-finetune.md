# LoRA Fine-tuning Integration

> **Efficiently fine-tune LLMs on custom data**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Low-Rank Adaptation for language models |
| **Why** | Fine-tune with minimal resources |
| **VRAM** | 8-16GB (vs 80GB+ for full fine-tuning) |
| **Best For** | Custom tasks, domain adaptation, style |

### LoRA vs Full Fine-tuning

| Aspect | LoRA | Full Fine-tune |
|--------|------|----------------|
| VRAM | 8-16GB | 80GB+ |
| Time | Hours | Days |
| Storage | 10-100MB | Full model |
| Quality | Very Good | Best |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 8GB+ VRAM (16GB recommended) |
| **Python** | 3.10+ |
| **Dataset** | 1000+ examples |

---

## Quick Start (1 hour)

### Installation

```bash
pip install transformers peft datasets accelerate bitsandbytes
```

### Basic LoRA Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from trl import SFTTrainer

# Load base model
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=16,                       # Rank
    lora_alpha=32,              # Scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
dataset = load_dataset("json", data_files="training_data.jsonl")

# Training
training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=512
)

trainer.train()
model.save_pretrained("./my-lora")
```

---

## Learning Path

### L0: First Fine-tune (2-3 hours)
- [ ] Prepare dataset
- [ ] Train with defaults
- [ ] Test results
- [ ] Save adapter

### L1: Optimization (4-6 hours)
- [ ] Tune rank and alpha
- [ ] Target module selection
- [ ] Learning rate scheduling
- [ ] Evaluation metrics

### L2: Production (1-2 days)
- [ ] Multi-task training
- [ ] Merge adapters
- [ ] Quantized training
- [ ] Deployment

---

## Code Examples

### Prepare Dataset

```python
import json

# Format: instruction-response pairs
data = [
    {
        "instruction": "Summarize the following text",
        "input": "Long article text here...",
        "output": "Brief summary here..."
    },
    # More examples...
]

def format_example(example):
    """Format for training"""
    if example.get("input"):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

# Save as JSONL
with open("training_data.jsonl", "w") as f:
    for item in data:
        formatted = {"text": format_example(item)}
        f.write(json.dumps(formatted) + "\n")
```

### Advanced LoRA Config

```python
from peft import LoraConfig, TaskType

# For different model architectures
lora_configs = {
    "llama": LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    ),
    "mistral": LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
}
```

### Load and Use Trained LoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./my-lora")

# Inference
inputs = tokenizer("### Instruction:\nSummarize:", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### Merge LoRA into Base Model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base and adapter
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "./my-lora")

# Merge and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

### Training with Validation

```python
from transformers import TrainingArguments, EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=training_args,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

---

## Parameters Guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `r` (rank) | 8-256 | Higher = more capacity |
| `lora_alpha` | r to 2*r | Scaling factor |
| `lora_dropout` | 0.0-0.1 | Regularization |
| `learning_rate` | 1e-5 to 5e-4 | Training speed |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Lower batch size, use gradient checkpointing |
| Poor results | More data, higher rank, longer training |
| Overfitting | Lower epochs, add dropout, more data |
| Loss not decreasing | Lower learning rate, check data |

---

## Resources

- [PEFT Library](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Fine-tuning Guide](https://huggingface.co/blog/lora)

---

*Part of [Luno-AI](../../README.md) | LLM Track*
