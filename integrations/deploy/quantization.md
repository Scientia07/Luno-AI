# Model Quantization Integration

> **Category**: Edge & Deployment
> **Difficulty**: Intermediate
> **Setup Time**: 3-4 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
Quantization reduces model size and increases inference speed by using lower precision numbers (INT8/INT4 instead of FP32). Critical for deploying LLMs and running models on edge devices.

### Why Use It
- **4-8x Smaller Models**: Fit larger models in VRAM
- **2-4x Faster**: Reduced compute requirements
- **Edge Deployment**: Run on consumer hardware
- **Cost Savings**: Less GPU memory needed
- **Minimal Quality Loss**: Often <1% accuracy drop

### Key Capabilities
| Method | Bits | Size Reduction | Quality | Best For |
|--------|------|----------------|---------|----------|
| FP16 | 16 | 2x | ~100% | GPU inference |
| INT8 | 8 | 4x | 99%+ | Production |
| INT4 | 4 | 8x | 95-99% | Edge/mobile |
| GPTQ | 4 | 8x | 97%+ | LLM inference |
| AWQ | 4 | 8x | 98%+ | LLM deployment |
| GGUF | Various | Variable | Variable | llama.cpp |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4 GB VRAM | 8 GB+ VRAM |
| RAM | 16 GB | 32 GB |
| Storage | 10 GB | 50 GB |

### Software Dependencies
```bash
# For GPTQ
pip install auto-gptq

# For AWQ
pip install autoawq

# For bitsandbytes (QLoRA)
pip install bitsandbytes

# For llama.cpp (GGUF)
pip install llama-cpp-python
```

---

## Quick Start (15 minutes)

### GPTQ Quantization
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_id = "meta-llama/Llama-2-7b-hf"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

# Quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.1,
    desc_act=False,
)

# Quantize
model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)
model.quantize(examples)  # Calibration data

# Save
model.save_quantized("llama-2-7b-gptq-4bit")
```

### Load Quantized Model
```python
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "llama-2-7b-gptq-4bit",
    device="cuda:0",
)
```

---

## Full Setup

### AWQ Quantization (Recommended)
```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama-2-7b-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}
model.quantize(tokenizer, quant_config=quant_config)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

### bitsandbytes (4-bit Loading)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

# Load quantized
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

### GGUF for llama.cpp
```bash
# Convert to GGUF
python convert.py model_path --outtype f16 --outfile model.gguf

# Quantize
./quantize model.gguf model-q4_k_m.gguf q4_k_m
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="model-q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=35,  # Layers on GPU
)

output = llm("Hello, my name is", max_tokens=50)
print(output["choices"][0]["text"])
```

---

## Method Comparison

### GPTQ vs AWQ vs GGUF

| Aspect | GPTQ | AWQ | GGUF |
|--------|------|-----|------|
| Speed | Fast | Fastest | Fast |
| Quality | Good | Best | Good |
| GPU Support | Yes | Yes | Partial |
| CPU Support | No | No | Yes |
| vLLM Support | Yes | Yes | No |
| Ease of Use | Medium | Easy | Easy |

### Quantization Levels

| Quantization | Size (7B) | VRAM | Quality |
|--------------|-----------|------|---------|
| FP16 | 14 GB | 16 GB | 100% |
| INT8 | 7 GB | 8 GB | 99.5% |
| Q8_0 (GGUF) | 7 GB | 8 GB | 99% |
| Q5_K_M (GGUF) | 5 GB | 6 GB | 97% |
| Q4_K_M (GGUF) | 4 GB | 5 GB | 95% |
| Q3_K_M (GGUF) | 3 GB | 4 GB | 90% |
| Q2_K (GGUF) | 2.5 GB | 3 GB | 85% |

---

## Code Examples

### Example 1: Download Pre-quantized
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Many models on HuggingFace are pre-quantized
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",
)
```

### Example 2: Quantize YOLO
```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Export with INT8
model.export(format="onnx", int8=True)

# Or TensorRT with FP16
model.export(format="tensorrt", half=True)
```

### Example 3: Ollama GGUF
```bash
# Create Modelfile
echo "FROM ./model-q4_k_m.gguf" > Modelfile

# Create Ollama model
ollama create my-model -f Modelfile

# Run
ollama run my-model
```

---

## Integration Points

| Integration | Purpose | Link |
|-------------|---------|------|
| ONNX | Export quantized | [onnx.md](./onnx.md) |
| TensorRT | NVIDIA optimization | [tensorrt.md](./tensorrt.md) |
| Ollama | Run GGUF models | [ollama.md](../llms/ollama.md) |
| vLLM | Serve quantized | [vllm.md](../llms/vllm.md) |

---

## Troubleshooting

### Quality Degradation
- Use AWQ instead of GPTQ
- Try higher bit quantization (Q5 vs Q4)
- Increase group_size

### Out of Memory During Quantization
- Use gradient checkpointing
- Quantize in smaller batches
- Use machine with more RAM

---

## Resources

- [Auto-GPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [TheBloke Models](https://huggingface.co/TheBloke)

---

*Part of [Luno-AI Integration Hub](../_index.md) | Edge & Deployment Track*
