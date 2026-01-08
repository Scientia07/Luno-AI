# Luno-AI - Claude Context File

> **Purpose**: Persistent context for AI assistants working on this project.
> **Last Updated**: 2026-01-05

---

## Project Overview

**Luno-AI** is a unified AI technology exploration platform with layered learning (Layer 0-4 depth system). Users can explore from high-level concepts down to implementation code.

### Core Vision
See: [CONCEPT.md](./CONCEPT.md)

---

## Key Directories

| Path | Purpose |
|------|---------|
| `/research/` | Research vault with dated sessions |
| `/research/_index.md` | Research session index (UPDATE THIS) |
| `/docs/layers/` | Layered knowledge base |
| `/labs/` | Interactive experiments & notebooks |
| `/src/` | Source code |

---

## Research Vault

### How to Log Research

When doing web research, ALWAYS create a research session:

```bash
# Create new session folder
research/YYYY-MM-DD_topic-name/
├── README.md      # Session overview & summary
├── sources.md     # ALL URLs, papers, repos referenced
├── findings.md    # Detailed notes & code
└── artifacts/     # Downloads, screenshots
```

### After Research

1. Update `research/_index.md` with the new session
2. Cross-reference in `research/topics/` if applicable
3. Link from relevant `docs/layers/` content

---

## Technology Domains

### 1. Foundations (NEW)
- **Math**: Vectors, matrices, linear algebra
- **Calculus**: Gradients, optimization
- **Statistics**: Probability, distributions
- **Programming**: Python, NumPy, PyTorch basics

### 2. Visual AI
- Object Detection (YOLO, DETR)
- Image Classification (ResNet, ViT)
- Segmentation (SAM, U-Net)

### 3. Generative AI
- Image Generation (Stable Diffusion)
- Audio (Bark, XTTS)
- Video (Runway, Pika)

### 4. LLMs
- Foundation Models (GPT, Claude, Llama)
- Fine-tuning (LoRA, QLoRA)
- Local Models (Ollama, llama.cpp)

### 5. Agentic AI
- Frameworks (LangGraph, CrewAI)
- Tool Use (MCP, Function Calling)
- Multi-Agent Systems

---

## Current State

### Completed
- [x] CONCEPT.md - Core vision document
- [x] Research folder structure with templates
- [x] CLAUDE.md - This file
- [x] MASTER_INDEX.md - Complete technology catalog
- [x] Domain READMEs (Visual AI, LLMs, Agents, Audio, Robotics, Spatial AI, Cognition, Knowledge Systems, Foundations)
- [x] First research session: AI Technology Stack Overview

### In Progress
- [ ] Interactive notebooks (Jupyter labs)
- [ ] Dashboard prototype

### Planned
- [ ] Vector DB integration for knowledge search
- [ ] More deep-dive research sessions

---

## Completed Research Sessions

| Date | Topic | Key Findings |
|------|-------|--------------|
| 2026-01-05 | LangGraph Multi-Agent Framework | LangGraph 1.0, StateGraph, checkpointers, human-in-the-loop, streaming, supervisor vs swarm |
| 2026-01-05 | MCP Implementation Patterns | MCP spec, servers/clients, OAuth 2.1, security, LangGraph integration |
| 2026-01-03 | Autonomous AI Agents 2025 | Production frameworks, deployment patterns, agent architectures |
| 2026-01-02 | Edge AI & Model Deployment | TensorRT, ONNX, quantization, cloud GPU providers |
| 2026-01-02 | Specialized AI Domains | Time series, anomaly detection, translation, XAI, document AI |
| 2026-01-02 | Speech-to-Text Alternatives | Deepgram, AssemblyAI, NVIDIA Canary, Vosk |
| 2026-01-02 | Classical ML & AI Frameworks | scikit-learn, XGBoost, AutoML, NLP libraries |
| 2026-01-01 | AI Technology Stack Overview | YOLO, Stable Diffusion, LangGraph, Whisper, SO-101, MoE models |

See: [research/_index.md](./research/_index.md) for full index

---

## Conventions

### Naming
- Research folders: `YYYY-MM-DD_topic-name/`
- Labs: `domain/experiment-name/`
- Always use kebab-case for folders/files

### Documentation
- Layer 0: `overview.md` (what is it?)
- Layer 1: `concepts.md` (how does it work?)
- Layer 2: `deep-dive.md` (technical details)
- Layer 3: `lab.ipynb` (hands-on code)
- Layer 4: `advanced.md` (optimization, custom)

---

## Quick Commands

```bash
# Start new research session
cp -r research/templates/session-template research/$(date +%Y-%m-%d)_topic-name

# Run Jupyter for labs
jupyter lab labs/

# Check project structure
tree -L 2
```

---

## References

- [CONCEPT.md](./CONCEPT.md) - Full project vision
- [research/_index.md](./research/_index.md) - Research sessions
- [Research Template](./research/templates/session-template/) - New session starter

---

*This file should be updated as the project evolves.*
