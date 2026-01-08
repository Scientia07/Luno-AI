# Luno-AI

> **Unified AI Technology Exploration Platform** - Learn, experiment, and build with AI technologies from beginner to expert level.

---

## Vision

Luno-AI is a comprehensive platform for understanding and working with modern AI technologies. From visual analysis (YOLO, SAM) to generative AI (Stable Diffusion) to language models (LLMs) to agentic systems (LangGraph) - all in one organized, layered knowledge system.

### Key Features

- **Layered Learning** (L0-L4): Navigate from simple overviews to advanced implementations
- **Research Vault**: Never lose context - all research sessions are logged and searchable
- **Interactive Labs**: Jupyter notebooks for hands-on experimentation
- **Technology Comparisons**: Side-by-side analysis of competing tools
- **Dashboard**: Visual interface for exploring the AI landscape

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Luno-AI.git
cd Luno-AI

# Set up Python environment (recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies (when available)
pip install -r requirements.txt

# Start exploring
jupyter lab labs/
```

---

## Project Structure

```
Luno-AI/
├── CONCEPT.md          # Core vision document
├── CLAUDE.md           # AI assistant context
├── docs/
│   ├── MASTER_INDEX.md # Complete technology catalog
│   └── layers/         # Layered knowledge base
│       ├── foundations/    # Math, programming basics
│       ├── visual-ai/      # Object detection, segmentation
│       ├── generative/     # Image/video generation
│       ├── llms/           # Language models
│       ├── agents/         # Agentic frameworks
│       ├── audio/          # Speech, music AI
│       ├── spatial-ai/     # 3D, depth, positioning
│       ├── robotics/       # Robot learning
│       ├── cognition/      # How AI thinks
│       └── knowledge-systems/ # Data organization
├── research/           # Research vault
│   ├── _index.md       # Research session index
│   ├── templates/      # Session templates
│   └── YYYY-MM-DD_*/   # Dated research sessions
├── labs/               # Jupyter notebooks
├── src/                # Source code
└── tools/              # CLI utilities
```

---

## Layer System

| Layer | Content | Example |
|-------|---------|---------|
| **L0 - Overview** | What is it? | "YOLO is an object detector" |
| **L1 - Concepts** | How does it work? | Architecture diagrams |
| **L2 - Deep Dive** | Technical details | Math, algorithms |
| **L3 - Labs** | Hands-on code | Jupyter notebooks |
| **L4 - Advanced** | Optimization, edge cases | Production tips |

---

## Technology Domains

| Domain | Technologies |
|--------|--------------|
| **Foundations** | Linear algebra, calculus, Python, PyTorch |
| **Visual AI** | YOLO, SAM, DETR, ViT, MediaPipe |
| **Generative** | Stable Diffusion, DALL-E, ControlNet, ComfyUI |
| **LLMs** | GPT-4, Claude, Llama, Mistral, MoE models |
| **Agents** | LangGraph, CrewAI, AutoGen, MCP |
| **Audio** | Whisper, XTTS, Bark, MusicGen, RVC |
| **Spatial** | Depth Anything, NeRF, 3D Gaussian Splatting, SLAM |
| **Robotics** | SO-101, LeRobot, ALOHA, ROS 2 |
| **Tools** | Scrapers, vector DBs, orchestration |

---

## Research Vault

All research is preserved in dated sessions:

```bash
# Start new research session
cp -r research/templates/session-template research/$(date +%Y-%m-%d)_topic-name

# Session contains:
# - README.md    (overview)
# - sources.md   (all URLs/papers)
# - findings.md  (detailed notes)
# - artifacts/   (downloads)
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the layer structure for new content
4. Update the research index for new findings
5. Submit a pull request

---

## Documentation

- [CONCEPT.md](./CONCEPT.md) - Full project vision
- [CLAUDE.md](./CLAUDE.md) - AI assistant context
- [docs/MASTER_INDEX.md](./docs/MASTER_INDEX.md) - Technology catalog

---

## License

MIT License - See LICENSE file for details.

---

*Built with curiosity and a desire to understand AI deeply.*
