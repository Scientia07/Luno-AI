# Luno-AI Integration Hub

> **Your roadmap to AI mastery** - Setup guides, learning paths, and integration blueprints.

---

## Quick Navigation

| Domain | Integrations | Difficulty | Status |
|--------|--------------|------------|--------|
| [Visual AI](#visual-ai) | YOLO, SAM, CLIP, Depth Anything | Beginner-Advanced | Ready |
| [Generative AI](#generative-ai) | Stable Diffusion, ComfyUI, ControlNet | Intermediate-Advanced | Ready |
| [Audio AI](#audio-ai) | Whisper, XTTS, Deepgram, RVC | Beginner-Advanced | Ready |
| [LLMs](#llms) | Ollama, vLLM, OpenAI, Claude, Fine-tuning | Beginner-Advanced | Ready |
| [Agentic AI](#agentic-ai) | LangGraph, CrewAI, MCP, RAG | Intermediate-Advanced | Ready |
| [Classical ML](#classical-ml) | XGBoost, AutoML, Polars, scikit-learn | Beginner-Intermediate | Ready |
| [Edge & Deploy](#edge--deployment) | TensorRT, ONNX, Quantization, Cloud | Intermediate-Advanced | Ready |
| [Robotics](#robotics) | LeRobot, SO-101, ROS 2, Simulation | Advanced | Ready |
| [Specialized](#specialized-domains) | Time Series, Anomaly, Translation, XAI | Intermediate-Advanced | Ready |

---

## Learning Paths

### Path 1: AI Beginner (2-4 weeks)
```
Start Here
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Python    │────▶│   Ollama    │────▶│   Whisper   │
│   Basics    │     │  Local LLM  │     │     STT     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐     ┌─────────────┐
                    │    YOLO     │────▶│   Simple    │
                    │  Detection  │     │    Agent    │
                    └─────────────┘     └─────────────┘
```

### Path 2: ML Engineer (4-8 weeks)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  XGBoost/   │────▶│   AutoML    │────▶│   Polars    │
│  LightGBM   │     │  AutoGluon  │     │   + Dask    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   MLflow    │────▶│  TensorRT   │────▶│   Deploy    │
│  Tracking   │     │    ONNX     │     │   to Prod   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Path 3: LLM Application Developer (4-6 weeks)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Ollama    │────▶│  LangChain  │────▶│  LangGraph  │
│   + APIs    │     │   Basics    │     │   Agents    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    RAG      │────▶│     MCP     │────▶│  Production │
│  Pipeline   │     │    Tools    │     │   System    │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Path 4: Computer Vision Specialist (6-8 weeks)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    YOLO     │────▶│     SAM     │────▶│    CLIP     │
│  Detection  │     │ Segmentation│     │  Multimodal │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Depth     │────▶│  3D Gauss   │────▶│   Custom    │
│  Anything   │     │  Splatting  │     │  Training   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Path 5: Generative AI Artist (4-6 weeks)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   ComfyUI   │────▶│ ControlNet  │────▶│ IP-Adapter  │
│   Basics    │     │   Control   │     │   Styles    │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    LoRA     │────▶│    Flux     │────▶│   Video     │
│  Training   │     │   Models    │     │   (Wan2.1)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Path 6: Audio/Voice Engineer (3-5 weeks)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Whisper   │────▶│    XTTS     │────▶│    RVC      │
│     STT     │     │     TTS     │     │   Cloning   │
└─────────────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Deepgram   │────▶│  Real-time  │────▶│  MusicGen   │
│     API     │     │  Pipeline   │     │   Audio     │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Visual AI

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| YOLO Object Detection | [yolo.md](./visual-ai/yolo.md) | Beginner | 2-4 hrs |
| SAM Segmentation | [sam.md](./visual-ai/sam.md) | Beginner | 2-3 hrs |
| CLIP Multimodal | [clip.md](./visual-ai/clip.md) | Intermediate | 3-4 hrs |
| Depth Anything | [depth-anything.md](./visual-ai/depth-anything.md) | Intermediate | 2-3 hrs |
| LLaVA Vision LLM | [llava.md](./visual-ai/llava.md) | Intermediate | 3-4 hrs |
| Custom YOLO Training | [yolo-training.md](./visual-ai/yolo-training.md) | Advanced | 1-2 days |

---

## Generative AI

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| ComfyUI Setup | [comfyui.md](./generative/comfyui.md) | Beginner | 1-2 hrs |
| Stable Diffusion | [stable-diffusion.md](./generative/stable-diffusion.md) | Intermediate | 3-4 hrs |
| ControlNet | [controlnet.md](./generative/controlnet.md) | Intermediate | 2-3 hrs |
| IP-Adapter | [ip-adapter.md](./generative/ip-adapter.md) | Intermediate | 2-3 hrs |
| LoRA Training | [lora-training.md](./generative/lora-training.md) | Advanced | 4-8 hrs |
| Flux Models | [flux.md](./generative/flux.md) | Intermediate | 2-3 hrs |
| Video Generation | [video-gen.md](./generative/video-gen.md) | Advanced | 4-6 hrs |

---

## Audio AI

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| Whisper STT | [whisper.md](./audio/whisper.md) | Beginner | 1-2 hrs |
| faster-whisper | [faster-whisper.md](./audio/faster-whisper.md) | Beginner | 1-2 hrs |
| Deepgram API | [deepgram.md](./audio/deepgram.md) | Beginner | 1 hr |
| XTTS Voice Clone | [xtts.md](./audio/xtts.md) | Intermediate | 2-3 hrs |
| RVC Voice Convert | [rvc.md](./audio/rvc.md) | Intermediate | 3-4 hrs |
| Real-time Pipeline | [realtime-audio.md](./audio/realtime-audio.md) | Advanced | 4-6 hrs |
| MusicGen | [musicgen.md](./audio/musicgen.md) | Intermediate | 2-3 hrs |

---

## LLMs

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| Ollama Local | [ollama.md](./llms/ollama.md) | Beginner | 30 min |
| OpenAI API | [openai.md](./llms/openai.md) | Beginner | 1 hr |
| Claude API | [claude.md](./llms/claude.md) | Beginner | 1 hr |
| vLLM Serving | [vllm.md](./llms/vllm.md) | Intermediate | 2-3 hrs |
| llama.cpp | [llamacpp.md](./llms/llamacpp.md) | Intermediate | 2-3 hrs |
| LoRA Fine-tuning | [lora-finetune.md](./llms/lora-finetune.md) | Advanced | 4-8 hrs |
| QLoRA Training | [qlora.md](./llms/qlora.md) | Advanced | 4-8 hrs |
| Embeddings | [embeddings.md](./llms/embeddings.md) | Intermediate | 2-3 hrs |

---

## Agentic AI

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| LangGraph Basics | [langgraph.md](./agents/langgraph.md) | Intermediate | 3-4 hrs |
| CrewAI Teams | [crewai.md](./agents/crewai.md) | Intermediate | 2-3 hrs |
| MCP Tools | [mcp.md](./agents/mcp.md) | Intermediate | 2-3 hrs |
| RAG Pipeline | [rag.md](./agents/rag.md) | Intermediate | 4-6 hrs |
| Vector Databases | [vector-db.md](./agents/vector-db.md) | Intermediate | 2-3 hrs |
| Multi-Agent Systems | [multi-agent.md](./agents/multi-agent.md) | Advanced | 6-8 hrs |
| Production Agents | [production-agents.md](./agents/production-agents.md) | Advanced | 1-2 days |

---

## Classical ML

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| scikit-learn | [sklearn.md](./ml/sklearn.md) | Beginner | 2-3 hrs |
| XGBoost/LightGBM | [gradient-boosting.md](./ml/gradient-boosting.md) | Beginner | 2-3 hrs |
| AutoGluon AutoML | [autogluon.md](./ml/autogluon.md) | Beginner | 1-2 hrs |
| Polars Data | [polars.md](./ml/polars.md) | Beginner | 2-3 hrs |
| MLflow Tracking | [mlflow.md](./ml/mlflow.md) | Intermediate | 2-3 hrs |
| Feature Engineering | [feature-eng.md](./ml/feature-eng.md) | Intermediate | 3-4 hrs |
| Time Series | [time-series.md](./ml/time-series.md) | Intermediate | 4-6 hrs |

---

## Edge & Deployment

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| ONNX Export | [onnx.md](./deploy/onnx.md) | Intermediate | 2-3 hrs |
| TensorRT Optimize | [tensorrt.md](./deploy/tensorrt.md) | Advanced | 4-6 hrs |
| Quantization (GPTQ/AWQ) | [quantization.md](./deploy/quantization.md) | Intermediate | 3-4 hrs |
| Docker Containers | [docker.md](./deploy/docker.md) | Intermediate | 2-3 hrs |
| GPU Cloud Setup | [gpu-cloud.md](./deploy/gpu-cloud.md) | Intermediate | 2-3 hrs |
| Triton Inference | [triton.md](./deploy/triton.md) | Advanced | 4-6 hrs |
| Mobile Deploy | [mobile.md](./deploy/mobile.md) | Advanced | 1-2 days |

---

## Robotics

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| LeRobot Setup | [lerobot.md](./robotics/lerobot.md) | Advanced | 4-6 hrs |
| SO-101 Arm | [so101.md](./robotics/so101.md) | Advanced | 1-2 days |
| ROS 2 Basics | [ros2.md](./robotics/ros2.md) | Advanced | 1-2 days |
| MuJoCo Simulation | [mujoco.md](./robotics/mujoco.md) | Advanced | 4-6 hrs |
| Imitation Learning | [imitation.md](./robotics/imitation.md) | Advanced | 1-2 days |

---

## Specialized Domains

| Integration | File | Difficulty | Time |
|-------------|------|------------|------|
| Chronos Time Series | [chronos.md](./specialized/chronos.md) | Intermediate | 2-3 hrs |
| PyOD Anomaly | [pyod.md](./specialized/pyod.md) | Intermediate | 2-3 hrs |
| SeamlessM4T Translate | [seamless.md](./specialized/seamless.md) | Intermediate | 2-3 hrs |
| SHAP Explainability | [shap.md](./specialized/shap.md) | Intermediate | 2-3 hrs |
| Docling Documents | [docling.md](./specialized/docling.md) | Beginner | 1-2 hrs |
| Label Studio | [label-studio.md](./specialized/label-studio.md) | Beginner | 1-2 hrs |

---

## PRD Template

Each integration file follows this structure:

```markdown
# [Technology Name] Integration

## Overview
- What it does
- Why use it
- Key capabilities

## Prerequisites
- Hardware requirements
- Software dependencies
- Prior knowledge needed

## Quick Start (15 min)
- Minimal setup steps
- First working example

## Full Setup
- Complete installation
- Configuration options
- Environment setup

## Learning Path
- L0: Basic usage
- L1: Intermediate features
- L2: Advanced techniques
- L3: Custom development

## Code Examples
- Common use cases
- Copy-paste snippets

## Integration Points
- How to combine with other tools
- API interfaces
- Data formats

## Troubleshooting
- Common issues
- Solutions

## Resources
- Official docs
- Tutorials
- Community
```

---

## Dashboard Integration

This integration hub is designed to power a dashboard with:

1. **Progress Tracking** - Mark completed integrations
2. **Skill Trees** - Visual learning paths
3. **Quick Actions** - One-click setup scripts
4. **Search** - Find integrations by capability
5. **Recommendations** - "If you liked X, try Y"

---

*Last Updated: 2026-01-05*
