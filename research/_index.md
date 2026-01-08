# Research Vault Index

> Central hub for all research sessions. Never lose context again.

---

## Quick Links

| Date | Topic | Domain | Status |
|------|-------|--------|--------|
| 2026-01-05 | [Agentic Framework Comparison](./2026-01-05_agentic-framework-comparison/) | Multi-Agent, CrewAI, LangGraph, AutoGen | Complete |
| 2026-01-05 | [CrewAI Multi-Agent Framework](./2026-01-05_crewai-multi-agent-framework/) | Agentic AI, Multi-Agent, Role-Based Agents | Complete |
| 2026-01-05 | [AutoGen Multi-Agent Framework](./2026-01-05_autogen-multi-agent-framework/) | Agentic AI, Multi-Agent Systems, Frameworks | Complete |
| 2026-01-05 | [LangGraph Multi-Agent Framework](./2026-01-05_langgraph-multi-agent-framework/) | Agentic AI, Multi-Agent, Orchestration | Complete |
| 2026-01-05 | [MCP Implementation Patterns](./2026-01-05_mcp-implementation-patterns/) | Agentic AI, Tool Integration, Protocols | Complete |
| 2026-01-03 | [Autonomous AI Agents 2025](./2026-01-03_autonomous-ai-agents-2025/) | Agentic AI, Frameworks, Production | Complete |
| 2026-01-02 | [Edge AI & Model Deployment](./2026-01-02_edge-ai-model-deployment/) | Edge Inference, Quantization, Cloud AI, GPU Providers | Complete |
| 2026-01-02 | [Specialized AI Domains](./2026-01-02_specialized-ai-domains/) | Time Series, Anomaly, Translation, XAI, DocAI, Annotation | Complete |
| 2026-01-02 | [Speech-to-Text Alternatives](./2026-01-02_speech-to-text-alternatives/) | Audio AI | Complete |
| 2026-01-02 | [Classical ML & AI Frameworks](./2026-01-02_classical-ml-ai-frameworks/) | ML/Frameworks | Complete |
| 2026-01-01 | [AI Technology Stack Overview](./2026-01-01_ai-technology-stack-overview/) | Multi-domain | Complete |

---

## How to Use This Vault

### Creating a New Research Session

1. Create a folder: `YYYY-MM-DD_topic-name/`
2. Copy template from `templates/session-template/`
3. Fill in README.md with session goals
4. Log sources in sources.md as you research
5. Summarize findings in findings.md
6. Update this index when complete

### Folder Naming Convention

```
YYYY-MM-DD_descriptive-topic-name/
```

Examples:
- `2026-01-01_yolo-object-detection-overview/`
- `2026-01-05_stable-diffusion-architecture/`
- `2026-01-10_langraph-vs-crewai-comparison/`

### Session Status

- `in-progress` - Currently researching
- `complete` - Research finished, documented
- `needs-update` - Information may be outdated
- `archived` - Historical, superseded by newer research

---

## Research by Domain

### Visual AI
- [2026-01-01] YOLO v8/v9/v10/v11/v12, YOLO-World (in AI Tech Stack Overview)

### Generative AI
- [2026-01-01] Stable Diffusion, SDXL, SD3, Flux, ControlNet, LoRA (in AI Tech Stack Overview)

### Large Language Models
- [2026-01-01] MoE Models: Mixtral, DeepSeek-V3, DBRX, Grok, Llama 4 (in AI Tech Stack Overview)

### Agentic Frameworks
- [2026-01-05] **Agentic Framework Comparison** - Deep comparison of CrewAI vs LangGraph vs AutoGen: architecture differences (role-based vs graph-based vs conversational), feature matrix, code examples, production readiness, decision flowchart, use case recommendations
- [2026-01-05] **CrewAI Multi-Agent Framework** - CrewAI 1.7.2 architecture, dual-mode (Crews/Flows), Agents/Tasks/Crews/Tools, role-based design, hierarchical/sequential processes, memory system (short-term/long-term/entity), MCP integration, Flows event-driven orchestration, production patterns, 5.76x faster than LangGraph in benchmarks
- [2026-01-05] **AutoGen Multi-Agent Framework** - AutoGen 0.4 architecture (Core/AgentChat/Extensions layers), AssistantAgent, Teams (RoundRobinGroupChat, SelectorGroupChat, Swarm), handoff patterns, tool integration, MCP support, Microsoft Agent Framework transition
- [2026-01-05] **LangGraph Multi-Agent Framework** - LangGraph 1.0 architecture, StateGraph, nodes/edges, checkpointing (SQLite/PostgreSQL/Redis), human-in-the-loop, streaming, subgraphs, supervisor vs swarm patterns, LangSmith deployment
- [2026-01-05] **MCP Implementation Patterns** - Model Context Protocol specification, server/client implementation, OAuth 2.1, security patterns, LangGraph integration
- [2026-01-03] **Autonomous AI Agents 2025** - Production frameworks, deployment patterns, agent architectures
- [2026-01-01] LangGraph, CrewAI, AutoGen, MCP integration (in AI Tech Stack Overview)

### Audio AI
- [2026-01-02] **Speech-to-Text Alternatives** - Deepgram Nova-3, AssemblyAI Universal-2, Google Chirp 3, NVIDIA Canary, Vosk, SpeechBrain, streaming options
- [2026-01-01] Whisper, faster-whisper, XTTS, Fish Speech, RVC, MusicGen, Suno (in AI Tech Stack Overview)

### Robotics
- [2026-01-01] SO-101, LeRobot, ALOHA, simulation tools (in AI Tech Stack Overview)

### Time Series / Forecasting
- [2026-01-02] **Specialized AI Domains** - Chronos-2, NeuralProphet, TimeGPT, Lag-Llama, Prophet

### Anomaly Detection
- [2026-01-02] **Specialized AI Domains** - PyOD 2.0, Isolation Forest, Autoencoders, Deep Isolation Forest

### Translation / Multilingual
- [2026-01-02] **Specialized AI Domains** - SeamlessM4T-v2, NLLB-200, mBART

### Explainable AI (XAI)
- [2026-01-02] **Specialized AI Domains** - SHAP, LIME, Captum, InterpretML EBM

### Document AI
- [2026-01-02] **Specialized AI Domains** - Docling, LlamaParse, Unstructured, Marker

### Annotation Tools
- [2026-01-02] **Specialized AI Domains** - CVAT, Label Studio, Roboflow, Labelbox

---

### Edge AI & Deployment
- [2026-01-02] **Edge AI & Model Deployment** - TensorRT, ONNX Runtime, OpenVINO, CoreML, TFLite, GPTQ, AWQ, GGUF, Cloud AI Platforms, GPU Providers

---

## Recent Sessions

1. **[2026-01-05] Agentic Framework Comparison** - Deep research comparing CrewAI, LangGraph, and AutoGen: architecture diagrams (role-based vs graph-based vs conversational), comprehensive feature matrix, code examples implementing same pipeline in all 3 frameworks, production readiness assessment, decision flowchart for framework selection, use case recommendations, migration paths, pros/cons analysis, version information (CrewAI 0.86+, LangGraph 1.0.5, AutoGen 0.4.x)

2. **[2026-01-05] CrewAI Multi-Agent Framework** - Complete deep dive into CrewAI 1.7.2: Dual-mode architecture (Crews for autonomous collaboration, Flows for production orchestration), core concepts (Agents, Tasks, Crews, Tools), role-based design with role/goal/backstory attributes, process types (sequential, hierarchical with manager agents), comprehensive memory system (short-term via ChromaDB, long-term via SQLite, entity memory via RAG), MCP integration via crewai-tools, Flows event-driven workflows with @start/@listen decorators, state persistence, async execution, 5.76x faster than LangGraph in benchmarks, 12M+ daily Flows executions, production use cases (IBM, PwC, Gelato), comparison with LangGraph/AutoGen, known limitations and workarounds

2. **[2026-01-05] AutoGen Multi-Agent Framework** - Microsoft AutoGen 0.4+ deep dive: Three-layer architecture (Core/AgentChat/Extensions), actor model with async messaging, AssistantAgent/UserProxyAgent/CodeExecutorAgent, team patterns (RoundRobinGroupChat, SelectorGroupChat, Swarm with handoffs), tool integration and MCP support, 30% message latency reduction, OpenTelemetry observability, comparison with LangGraph/CrewAI, Microsoft Agent Framework transition (October 2025), production readiness assessment

3. **[2026-01-05] LangGraph Multi-Agent Framework** - Deep dive into LangGraph 1.0 (October 2025 release): DAG-based architecture, StateGraph API, nodes/edges, persistence with checkpointers (SQLite, PostgreSQL, Redis), human-in-the-loop via interrupt(), streaming modes, subgraph composition, supervisor vs swarm multi-agent patterns, performance benchmarks, LangSmith deployment options (Cloud/Hybrid/Self-Hosted), production use cases (Klarna, Uber, LinkedIn, Replit)

4. **[2026-01-05] MCP Implementation Patterns** - Comprehensive guide to Model Context Protocol: specification overview, building MCP servers (Python/TypeScript), building MCP clients, transport mechanisms (stdio, Streamable HTTP), popular servers (filesystem, GitHub, Slack, databases), Claude Desktop integration, LangGraph/LangChain adapters, security vulnerabilities (prompt injection, tool poisoning), OAuth 2.1 authentication, debugging/logging best practices, November 2025 spec updates (Tasks, Extensions), MCP vs Google A2A comparison, Linux Foundation governance

2. **[2026-01-02] Edge AI & Model Deployment** - Edge inference frameworks (TensorRT, ONNX Runtime, OpenVINO, CoreML, TFLite), quantization methods (GPTQ, AWQ, bitsandbytes, GGUF), cloud AI platforms (SageMaker, Vertex AI, Azure ML), GPU cloud providers (Lambda Labs, RunPod, Vast.ai, Modal)

2. **[2026-01-02] Specialized AI Domains** - Time series (Chronos-2, NeuralProphet), anomaly detection (PyOD 2.0), translation (SeamlessM4T-v2), XAI (SHAP, LIME), document AI (Docling, LlamaParse), annotation tools (CVAT, Labelbox)

3. **[2026-01-02] Speech-to-Text Alternatives** - Commercial APIs (Deepgram, AssemblyAI, Google, AWS, Azure), open-source models (NVIDIA Canary, Vosk, SpeechBrain), real-time streaming, WER benchmarks, and pricing comparison

4. **[2026-01-02] Classical ML & AI Frameworks** - Comparison of classical ML (scikit-learn, XGBoost, LightGBM, CatBoost), AutoML (AutoGluon, H2O, PyCaret), NLP (spaCy, NLTK, Transformers), data processing (Pandas, Polars, Dask), and CV libraries (OpenCV, Albumentations, Kornia)

5. **[2026-01-01] AI Technology Stack Overview** - Comprehensive research on YOLO, Stable Diffusion, LangGraph/Agents, Whisper/Audio, SO-101/Robotics, and MoE models

---

## Tips for Effective Research

1. **Always log sources** - URLs, papers, repos
2. **Date everything** - AI field moves fast
3. **Include code snippets** - Practical > theoretical
4. **Cross-reference** - Link related sessions
5. **Summarize key takeaways** - Future you will thank you

---

*Last Updated: 2026-01-05*
