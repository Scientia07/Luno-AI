# Luno-AI: Master Technology Index

> **Complete catalog of AI technologies, tools, and concepts covered in this platform.**

---

## Domain Overview

```
                              LUNO-AI TECHNOLOGY MAP

    ┌─────────────────────────────────────────────────────────────────┐
    │                        FOUNDATIONS                               │
    │  Math │ Vectors │ Matrices │ Calculus │ Statistics │ Python     │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
│ VISION  │    │GENERATIVE│   │  LLMs   │    │ AGENTS  │    │  AUDIO   │
│   AI    │    │   AI    │    │         │    │         │    │    AI    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └──────────┘
    │               │              │              │               │
    └───────────────┴──────────────┴──────────────┴───────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
   ┌──────────┐            ┌─────────────┐            ┌──────────────┐
   │ ROBOTICS │            │ CLASSICAL   │            │  SPECIALIZED │
   │          │            │     ML      │            │   DOMAINS    │
   └──────────┘            └─────────────┘            └──────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────┐
               │         EDGE AI & DEPLOYMENT          │
               │  TensorRT │ ONNX │ Quantization │ GPU │
               └───────────────────────────────────────┘
                                   │
                                   ▼
               ┌───────────────────────────────────────┐
               │          TOOLS & ECOSYSTEM            │
               │    MCP │ Scrapers │ Pipelines │ Infra │
               └───────────────────────────────────────┘
```

---

## 1. FOUNDATIONS

### 1.1 Mathematics for AI

| Topic | Subtopics | Difficulty |
|-------|-----------|------------|
| **Linear Algebra** | Vectors, Matrices, Eigenvalues, SVD | Beginner-Intermediate |
| **Matrix Operations** | Multiplication, Transpose, Inverse | Beginner |
| **Calculus** | Derivatives, Gradients, Chain Rule | Intermediate |
| **Optimization** | Gradient Descent, Adam, Learning Rates | Intermediate |
| **Probability** | Distributions, Bayes, Sampling | Intermediate |
| **Information Theory** | Entropy, KL Divergence, Cross-Entropy | Advanced |

### 1.2 Programming Stack

| Tool | Purpose | Layer |
|------|---------|-------|
| **Python** | Core language | L1 |
| **NumPy** | Numerical computing | L2 |
| **PyTorch** | Deep learning framework | L2-L3 |
| **TensorFlow** | Alternative DL framework | L2-L3 |
| **JAX** | High-performance computing | L4 |
| **Jupyter** | Interactive notebooks | L1-L4 |

---

## 2. VISION AI

### 2.1 Visual Understanding

| Technology | Purpose | Key Models |
|------------|---------|------------|
| **Object Detection** | Locate objects | YOLOv8/v9, DETR, RT-DETR |
| **Image Classification** | Categorize images | ResNet, ViT, EfficientNet |
| **Semantic Segmentation** | Pixel-level labels | SAM, Mask2Former, U-Net |
| **Instance Segmentation** | Individual object masks | Mask R-CNN, YOLACT |
| **Pose Estimation** | Human keypoints | MediaPipe, OpenPose, RTMPose |
| **OCR/Document AI** | Text extraction | PaddleOCR, DocTR, Tesseract |
| **Video Understanding** | Temporal analysis | VideoMAE, TimeSformer |
| **Depth Estimation** | 3D from 2D | MiDaS, Depth Anything |
| **Scene Understanding** | Holistic scene parsing | OpenScene, SAM + CLIP |

### 2.2 Vision-Language Models

| Model | Capability |
|-------|------------|
| **CLIP** | Image-text matching |
| **BLIP-2** | Image captioning, VQA |
| **LLaVA** | Visual chat |
| **GPT-4V** | Multimodal reasoning |
| **Gemini** | Multimodal understanding |

---

## 3. GENERATIVE AI

### 3.1 Image Generation

| Technology | Approach | Examples |
|------------|----------|----------|
| **Diffusion Models** | Denoising | Stable Diffusion, DALL-E 3, Midjourney |
| **GANs** | Adversarial | StyleGAN, BigGAN |
| **Flow Models** | Normalizing flows | Stable Diffusion 3 (rectified flow) |
| **Autoregressive** | Token prediction | Parti, Imagen |

### 3.2 Image Editing & Control

| Technology | Purpose |
|------------|---------|
| **ControlNet** | Structural control (pose, edges, depth) |
| **IP-Adapter** | Image prompt adaptation |
| **InstructPix2Pix** | Text-based editing |
| **Inpainting** | Region replacement |
| **Outpainting** | Image extension |
| **Upscaling** | Super-resolution (ESRGAN, Real-ESRGAN) |

### 3.3 Video Generation

| Technology | Status |
|------------|--------|
| **Runway Gen-2/3** | Production ready |
| **Pika** | Production ready |
| **Kling** | Production ready |
| **Sora** | Limited access |
| **Open-Sora** | Open source alternative |
| **AnimateDiff** | Animation from images |

### 3.4 3D Generation

| Technology | Purpose |
|------------|---------|
| **NeRF** | Neural radiance fields |
| **3D Gaussian Splatting** | Real-time 3D |
| **Point-E / Shap-E** | 3D from text |
| **DreamGaussian** | Text/image to 3D |
| **Meshy** | Commercial 3D generation |

---

## 4. AUDIO AI

### 4.1 Speech Recognition (ASR/STT)

#### Whisper & Variants

| Implementation | Speed | Best For |
|----------------|-------|----------|
| **Whisper** (OpenAI) | Baseline | Standard use |
| **faster-whisper** | 4x faster | Production Python |
| **whisper.cpp** | CPU optimized | Edge devices |
| **WhisperX** | Word timestamps | Subtitles, diarization |
| **insanely-fast-whisper** | Max throughput | GPU batch processing |
| **Distil-Whisper** | 6x faster, 49% smaller | Mobile/edge |

#### Commercial STT APIs

| Service | Model | WER | Latency | Pricing |
|---------|-------|-----|---------|---------|
| **Deepgram** | Nova-3 | 7.8% | ~100ms | $0.0043/min |
| **AssemblyAI** | Universal-2 | 8.2% | ~200ms | $0.0062/min |
| **Google Cloud** | Chirp 3 | 8.5% | ~150ms | $0.009/min |
| **AWS Transcribe** | Latest | 9.1% | ~500ms | $0.024/min |
| **Azure Speech** | Latest | 9.0% | ~200ms | $0.016/min |

#### Open-Source Alternatives

| Model | Developer | Languages | Best For |
|-------|-----------|-----------|----------|
| **NVIDIA Canary** | NVIDIA | 8 | Multilingual, commercial |
| **Vosk** | Alpha Cephei | 20+ | Offline, lightweight |
| **SpeechBrain** | Mila | Multi | Research, customization |
| **NeMo ASR** | NVIDIA | Multi | Enterprise, GPU |

### 4.2 Audio Analysis

| Technology | Purpose | Examples |
|------------|---------|----------|
| **Speaker Diarization** | Who spoke when | pyannote, NeMo |
| **Audio Classification** | Sound identification | AudioSet models |
| **Music Analysis** | Tempo, key, structure | Librosa, Essentia |
| **Emotion Recognition** | Sentiment from voice | SpeechBrain |

### 4.3 Audio Generation

| Technology | Purpose | Examples |
|------------|---------|----------|
| **Text-to-Speech (TTS)** | Voice synthesis | XTTS, Bark, Coqui, ElevenLabs |
| **Voice Cloning** | Replicate voices | RVC, XTTS, OpenVoice |
| **Music Generation** | Create music | MusicGen, Suno, Udio |
| **Sound Effects** | Create sounds | AudioLDM, AudioGen |
| **Voice Conversion** | Change voice | RVC, So-VITS-SVC |

### 4.4 Whisper Model Sizes

| Variant | Size | Languages | Speed |
|---------|------|-----------|-------|
| tiny | 39M | Multi | Fastest |
| base | 74M | Multi | Fast |
| small | 244M | Multi | Medium |
| medium | 769M | Multi | Slow |
| large-v3 | 1.5B | 99 | Slowest |
| turbo | 809M | Multi | Fast + accurate |

---

## 5. LARGE LANGUAGE MODELS (LLMs)

### 5.1 Foundation Models

| Model Family | Provider | Open/Closed | Notes |
|--------------|----------|-------------|-------|
| **GPT-4/4o** | OpenAI | Closed | Best general reasoning |
| **Claude 3.5** | Anthropic | Closed | Best for code, long context |
| **Gemini** | Google | Closed | Multimodal native |
| **Llama 3.x** | Meta | Open weights | Best open model |
| **Mistral/Mixtral** | Mistral AI | Open weights | Efficient, strong |
| **Qwen 2.5** | Alibaba | Open weights | Excellent multilingual |
| **DeepSeek** | DeepSeek | Open weights | Strong reasoning |
| **Phi-3/4** | Microsoft | Open weights | Small but capable |
| **Gemma 2** | Google | Open weights | Efficient |

### 5.2 Model Architectures

| Architecture | Description | Examples |
|--------------|-------------|----------|
| **Transformer** | Attention-based | GPT, BERT, T5 |
| **MoE (Mixture of Experts)** | Sparse routing | Mixtral, GPT-4, DeepSeek |
| **SSM (State Space)** | Linear complexity | Mamba, Jamba |
| **Hybrid** | Combined approaches | Jamba (Mamba + Attention) |

### 5.3 MoE Models (Mixture of Experts)

| Model | Experts | Active | Total Params |
|-------|---------|--------|--------------|
| **Mixtral 8x7B** | 8 | 2 | 46.7B |
| **Mixtral 8x22B** | 8 | 2 | 176B |
| **DeepSeek-V2** | 160 | 6 | 236B |
| **DBRX** | 16 | 4 | 132B |
| **Grok-1** | 8 | 2 | 314B |

### 5.4 Specialized Models

| Type | Purpose | Examples |
|------|---------|----------|
| **Code** | Programming | Codestral, DeepSeek-Coder, StarCoder2 |
| **Math** | Reasoning | DeepSeek-Math, Llemma |
| **Embeddings** | Semantic vectors | BGE, E5, OpenAI Ada-002 |
| **Rerankers** | Search ranking | Cohere, BGE-Reranker |

### 5.5 Local LLM Tools

| Tool | Purpose |
|------|---------|
| **Ollama** | Easy local model running |
| **llama.cpp** | Efficient CPU inference |
| **vLLM** | High-throughput serving |
| **Text Generation Inference** | HuggingFace serving |
| **LM Studio** | GUI for local LLMs |

### 5.6 Fine-tuning

| Method | VRAM | Speed | Quality |
|--------|------|-------|---------|
| **Full Fine-tune** | Very High | Slow | Best |
| **LoRA** | Medium | Fast | Good |
| **QLoRA** | Low | Fast | Good |
| **PEFT** | Low | Fast | Good |
| **Adapters** | Low | Fast | Moderate |

---

## 6. AGENTIC AI

### 6.1 Agent Frameworks

| Framework | Approach | Best For |
|-----------|----------|----------|
| **LangGraph** | Graph-based workflows | Complex stateful agents |
| **CrewAI** | Role-based agents | Multi-agent collaboration |
| **AutoGen** | Conversational agents | Research, experimentation |
| **Agency Swarm** | Custom agents | Production systems |
| **OpenAI Assistants** | Hosted agents | Quick prototyping |
| **Claude Tools** | Native tool use | Integrated solutions |

### 6.2 Tool Use & MCP

| Technology | Purpose |
|------------|---------|
| **Model Context Protocol (MCP)** | Standardized tool interface |
| **Function Calling** | LLM-triggered functions |
| **Tool Use** | External capability access |
| **Code Interpreters** | Runtime code execution |

### 6.3 Memory Systems

| Type | Purpose | Examples |
|------|---------|----------|
| **Short-term** | Conversation context | Sliding window |
| **Long-term** | Persistent knowledge | Vector stores |
| **Episodic** | Event sequences | MemGPT |
| **Semantic** | Concept relationships | Knowledge graphs |

### 6.4 RAG (Retrieval Augmented Generation)

| Component | Options |
|-----------|---------|
| **Vector DBs** | Pinecone, Weaviate, Qdrant, Chroma, Milvus |
| **Chunking** | Semantic, recursive, sentence |
| **Embeddings** | BGE, E5, OpenAI, Cohere |
| **Reranking** | Cohere, BGE-Reranker |
| **Hybrid Search** | BM25 + Vector |

---

## 7. ROBOTICS

### 7.1 Hardware Platforms

| Platform | Type | Openness |
|----------|------|----------|
| **SO-100/SO-101** | Robotic arm | Open source |
| **ALOHA** | Bimanual manipulation | Open source |
| **Stretch** | Mobile manipulator | Commercial |
| **Unitree** | Quadruped/humanoid | Commercial |
| **Boston Dynamics** | Advanced robots | Commercial |

### 7.2 Robot Learning

| Approach | Description |
|----------|-------------|
| **Imitation Learning** | Learn from demonstrations |
| **Reinforcement Learning** | Learn from rewards |
| **Sim-to-Real** | Train in simulation |
| **Foundation Models for Robotics** | RT-1, RT-2, PaLM-E |

### 7.3 Key Projects

| Project | Focus |
|---------|-------|
| **LeRobot (HuggingFace)** | Open robot learning |
| **OpenVLA** | Vision-language-action |
| **RoboAgent** | Generalizable manipulation |
| **Mobile ALOHA** | Mobile bimanual |

---

## 8. TOOLS & ECOSYSTEM

### 8.1 Scrapers & Data Collection

| Tool | Purpose | Language |
|------|---------|----------|
| **Scrapy** | Web scraping framework | Python |
| **BeautifulSoup** | HTML parsing | Python |
| **Playwright** | Browser automation | Multi |
| **Selenium** | Browser automation | Multi |
| **Crawl4AI** | LLM-optimized crawling | Python |
| **Firecrawl** | AI-ready web data | API |
| **Apify** | Scraping platform | Multi |

### 8.2 MCP (Model Context Protocol)

| Category | Examples |
|----------|----------|
| **File Systems** | filesystem, memory |
| **Databases** | sqlite, postgres |
| **APIs** | github, slack, google-drive |
| **Search** | brave-search, tavily |
| **Development** | sequential-thinking, sentry |

### 8.3 Infrastructure

| Category | Tools |
|----------|-------|
| **Model Serving** | vLLM, TGI, Triton, Ollama |
| **Orchestration** | Prefect, Airflow, Dagster |
| **Monitoring** | LangSmith, W&B, Arize |
| **Vector Search** | Pinecone, Weaviate, Qdrant |
| **Experiment Tracking** | MLflow, W&B, CometML |

### 8.4 Development Tools

| Tool | Purpose |
|------|---------|
| **LangChain** | LLM application framework |
| **LlamaIndex** | Data framework for LLMs |
| **Semantic Kernel** | Microsoft's AI framework |
| **Haystack** | NLP/LLM pipelines |
| **Instructor** | Structured outputs |

---

## 9. MULTIMODAL AI

### 9.1 Vision-Language

| Model | Capabilities |
|-------|--------------|
| **GPT-4V/4o** | Full multimodal |
| **Claude 3.5** | Vision + reasoning |
| **Gemini Pro** | Native multimodal |
| **LLaVA** | Open visual LLM |
| **Qwen-VL** | Open multimodal |

### 9.2 Audio-Language

| Model | Capabilities |
|-------|--------------|
| **GPT-4o** | Voice mode |
| **Gemini** | Audio understanding |
| **Whisper + LLM** | Speech pipeline |

### 9.3 Any-to-Any

| Model | Modalities |
|-------|------------|
| **Gemini** | Text, image, audio, video |
| **GPT-4o** | Text, image, audio |
| **CoDi** | Any-to-any generation |

---

## Quick Reference: By Use Case

| I want to... | Technology |
|--------------|------------|
| Detect objects in images | YOLO, DETR |
| Generate images from text | Stable Diffusion, DALL-E |
| Transcribe audio | Whisper |
| Build a chatbot | OpenAI API, Claude, Ollama |
| Create an AI agent | LangGraph, CrewAI |
| Search semantic content | Vector DB + Embeddings |
| Fine-tune a model | LoRA, QLoRA |
| Run LLMs locally | Ollama, llama.cpp |
| Scrape web data | Scrapy, Playwright |
| Control a robot | LeRobot, ROS |
| Generate music | MusicGen, Suno |
| Clone a voice | XTTS, RVC |

---

## Layer System Quick Reference

| Layer | Content | Example |
|-------|---------|---------|
| **L0** | What is it? | "YOLO is an object detector" |
| **L1** | How does it work? | Architecture overview |
| **L2** | Technical deep-dive | Mathematical foundations |
| **L3** | Hands-on code | Jupyter notebooks |
| **L4** | Advanced mastery | Custom training, optimization |

---

## 10. CLASSICAL ML & DATA SCIENCE

### 10.1 Machine Learning Libraries

| Library | Strength | Best For |
|---------|----------|----------|
| **scikit-learn** | Complete, well-documented | General ML, prototyping |
| **XGBoost** | Speed, accuracy | Tabular data, competitions |
| **LightGBM** | Memory efficient, fast | Large datasets |
| **CatBoost** | Categorical features | Production, minimal tuning |

### 10.2 AutoML Frameworks

| Framework | Developer | Strength |
|-----------|-----------|----------|
| **AutoGluon** | AWS | Best accuracy, easy to use |
| **H2O AutoML** | H2O.ai | Enterprise, interpretability |
| **PyCaret** | Open Source | Low-code, fast prototyping |
| **Auto-sklearn** | AutoML.org | Sklearn integration |
| **FLAML** | Microsoft | Fast, resource efficient |
| **TPOT** | Penn | Genetic optimization |

### 10.3 NLP Libraries

| Library | Focus | Best For |
|---------|-------|----------|
| **spaCy** | Industrial NLP | Production pipelines |
| **NLTK** | Educational | Learning, prototyping |
| **Transformers** | Pretrained models | State-of-the-art NLP |
| **Stanza** | Stanford CoreNLP | Multilingual, research |
| **Flair** | Embeddings | NER, POS tagging |

### 10.4 Data Processing

| Library | Strength | Best For |
|---------|----------|----------|
| **Pandas** | Versatile, ecosystem | General data analysis |
| **Polars** | 10-100x faster | Large datasets, production |
| **Dask** | Parallel computing | Out-of-memory data |
| **Vaex** | Billion-row datasets | Exploration, visualization |
| **cuDF** | GPU acceleration | NVIDIA environments |

### 10.5 Computer Vision Libraries

| Library | Focus | Best For |
|---------|-------|----------|
| **OpenCV** | Classical CV | Image processing, video |
| **Albumentations** | Augmentation | Training pipelines |
| **Kornia** | Differentiable CV | PyTorch integration |
| **scikit-image** | Image algorithms | Scientific imaging |
| **Pillow** | Basic operations | Simple image tasks |

---

## 11. EDGE AI & DEPLOYMENT

### 11.1 Inference Frameworks

| Framework | Platform | Optimization |
|-----------|----------|--------------|
| **TensorRT** | NVIDIA GPU | 2-6x speedup |
| **ONNX Runtime** | Cross-platform | Universal format |
| **OpenVINO** | Intel CPU/GPU | Intel hardware |
| **CoreML** | Apple | iOS/macOS native |
| **TFLite** | Mobile/Edge | Android, Raspberry Pi |
| **NCNN** | Mobile | Lightweight, ARM |

### 11.2 Quantization Methods

| Method | Bits | Quality Loss | Best For |
|--------|------|--------------|----------|
| **FP16** | 16 | Minimal | GPU inference |
| **INT8** | 8 | Low | Production |
| **INT4** | 4 | Moderate | Edge devices |
| **GPTQ** | 4 | Low | LLM inference |
| **AWQ** | 4 | Very Low | LLM deployment |
| **GGUF** | Various | Low | llama.cpp, CPU |
| **bitsandbytes** | 4/8 | Low | Training, QLoRA |

### 11.3 Cloud AI Platforms

| Platform | Provider | Strength |
|----------|----------|----------|
| **SageMaker** | AWS | Full MLOps |
| **Vertex AI** | Google | AutoML, BigQuery |
| **Azure ML** | Microsoft | Enterprise integration |
| **Databricks** | Databricks | Unified analytics |
| **Hugging Face** | HF | Model Hub, Spaces |

### 11.4 GPU Cloud Providers

| Provider | GPUs | Best For | Pricing |
|----------|------|----------|---------|
| **Lambda Labs** | H100, A100 | ML training | ~$2-3/hr |
| **RunPod** | Various | Flexible, community | ~$0.30/hr+ |
| **Vast.ai** | Consumer GPUs | Budget | ~$0.15/hr+ |
| **Modal** | A10G, A100 | Serverless | Pay-per-use |
| **Together AI** | A100, H100 | Inference | API-based |
| **Replicate** | Various | Model hosting | API-based |

---

## 12. SPECIALIZED AI DOMAINS

### 12.1 Time Series & Forecasting

| Model | Type | Best For |
|-------|------|----------|
| **Chronos-2** | Foundation model | Zero-shot, general |
| **NeuralProphet** | Neural + classical | Business forecasting |
| **TimeGPT** | Foundation model | Quick deployment |
| **Lag-Llama** | LLM-based | Multi-domain |
| **Prophet** | Classical | Trend, seasonality |
| **N-BEATS** | Deep learning | Point forecasting |

### 12.2 Anomaly Detection

| Tool | Method | Best For |
|------|--------|----------|
| **PyOD 2.0** | Unified interface | Comprehensive toolkit |
| **Isolation Forest** | Tree-based | High-dimensional |
| **Autoencoders** | Reconstruction | Complex patterns |
| **Deep Isolation Forest** | Deep learning | Large datasets |
| **HBOS** | Histogram | Fast, streaming |

### 12.3 Translation & Multilingual

| Model | Languages | Features |
|-------|-----------|----------|
| **SeamlessM4T-v2** | 200+ | Speech + text, streaming |
| **NLLB-200** | 200 | Text only, open source |
| **mBART** | 50 | Fine-tunable |
| **M2M-100** | 100 | Direct translation |
| **Google Translate API** | 130+ | Production, reliable |

### 12.4 Explainable AI (XAI)

| Tool | Method | Best For |
|------|--------|----------|
| **SHAP** | Shapley values | Model-agnostic, accurate |
| **LIME** | Local perturbation | Quick explanations |
| **Captum** | PyTorch native | Deep learning |
| **InterpretML EBM** | Glassbox models | Interpretable by design |
| **Alibi** | Multiple methods | Production monitoring |

### 12.5 Document AI

| Tool | Function | Best For |
|------|----------|----------|
| **Docling** | Universal conversion | PDF, DOCX, HTML |
| **LlamaParse** | AI parsing | Complex documents |
| **Unstructured** | Pipeline | RAG preparation |
| **Marker** | PDF to markdown | Fast, accurate |
| **PaddleOCR** | OCR | Multi-language |
| **DocTR** | Deep learning OCR | Modern documents |

### 12.6 Annotation & Labeling

| Tool | Type | Best For |
|------|------|----------|
| **CVAT** | Open source | Computer vision |
| **Label Studio** | Open source | Multi-modal |
| **Roboflow** | Platform | CV workflows |
| **Labelbox** | Enterprise | Team collaboration |
| **Prodigy** | Scriptable | NLP, active learning |
| **Scale AI** | Managed | Large-scale labeling |

---

## Quick Reference: Extended Use Cases

| I want to... | Technology |
|--------------|------------|
| **Speech-to-Text (fast)** | Deepgram, faster-whisper |
| **Speech-to-Text (offline)** | Vosk, whisper.cpp |
| **Tabular ML** | XGBoost, LightGBM, CatBoost |
| **AutoML** | AutoGluon, H2O, PyCaret |
| **Time series forecast** | Chronos-2, NeuralProphet |
| **Anomaly detection** | PyOD 2.0, Isolation Forest |
| **Translate languages** | SeamlessM4T-v2, NLLB-200 |
| **Explain model predictions** | SHAP, LIME |
| **Process documents** | Docling, LlamaParse |
| **Label training data** | CVAT, Label Studio |
| **Optimize for mobile** | TFLite, CoreML, NCNN |
| **Quantize LLMs** | GPTQ, AWQ, GGUF |
| **Fast data processing** | Polars, Dask |
| **GPU cloud training** | Lambda Labs, RunPod |

---

*Last Updated: 2026-01-02*
*This index will grow as we add more technologies and research.*
