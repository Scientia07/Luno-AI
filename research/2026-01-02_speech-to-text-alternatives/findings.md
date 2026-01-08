# Detailed Findings - Speech-to-Text Alternatives 2025

---

## 1. Commercial APIs

### Deepgram Nova-2/Nova-3

**Overview**: Deepgram offers purpose-built speech recognition models optimized for speed and accuracy.

**Nova-2 Specifications**:
- Median WER: 8.4% (30% below industry average)
- 36% more accurate than OpenAI Whisper Large
- Languages: 7+ (English, Spanish, French, German, etc.)

**Nova-3 Specifications (February 2025)**:
- 54.3% WER reduction for streaming
- 47.4% WER reduction for batch processing
- First real-time multilingual transcription model
- Sub-300ms latency

**Pricing**:
- Streaming: $0.0077/min ($0.46/hour)
- Pre-recorded: Similar rates
- Pay-per-second billing (no rounding overhead)

**Specialized Models**:
- Nova-2 Medical: 42.8% WER improvement over alternatives

**Best For**: Real-time applications, call centers, medical transcription

---

### AssemblyAI Universal-2

**Overview**: Production-focused API with emphasis on reliability and streaming performance.

**Specifications**:
- Streaming WER: 14.5% (best among streaming commercial models)
- Latency: 300ms (P50)
- Languages: 99+ with automatic detection
- Uptime SLA: 99.95%

**Key Features**:
- Immutable transcripts (no mid-conversation changes)
- 24% improvement in rare word recognition (names, brands)
- 15% improvement in transcript structure (punctuation, casing)
- Speaker diarization for 95 languages

**Pricing**:
- Base: $0.15/hour ($0.0025/min)
- Speaker ID: +$0.02/hour
- Sentiment analysis: +$0.02/hour
- PII redaction: +$0.08/hour
- Summarization: +$0.03/hour

**Note**: Billing by session duration, not audio length (~65% overhead on short calls)

**Best For**: Customer-facing voice agents, production systems requiring SLA

---

### Google Cloud Speech-to-Text (Chirp 3)

**Overview**: Google's multilingual ASR using self-supervised training.

**Specifications**:
- Batch WER: 11.6% (best batch accuracy)
- Languages: 100+
- Training: Millions of hours audio + 28B text sentences

**Chirp 3 Features**:
- Speaker diarization
- Automatic language detection
- Speech adaptation (custom vocabularies)
- Built-in denoiser
- Streaming, batch, and real-time modes

**Pricing**:
- Standard: $0.016/min
- Batch: $0.004/min
- Free tier: 60 min/month + $300 new user credit

**Hidden Costs**:
- Storage: $0.020/GB/month
- Cloud Functions: $0.40/million invocations
- Egress: $0.08-0.23/GB
- Production pipelines can 2-3x headline rate

**Limitations**: Chirp processes in large chunks, not ideal for true real-time

**Best For**: Batch processing, multilingual applications, GCP ecosystem

---

### AWS Transcribe

**Specifications**:
- Languages: 54 languages/variants
- Speaker diarization: Up to 10 speakers
- Features: PII redaction, medical transcription, call analytics

**Pricing**:
- Batch: $0.024/min
- Streaming: $0.030/min
- Free tier: 60 min/month for first year

**Strengths**:
- Native AWS integration (S3, Lambda)
- Specialized medical model
- Good for general business use

**Weaknesses**:
- Struggles with background noise
- Reduced accuracy with regional accents
- Block rounding and concurrency caps add overhead

**Best For**: AWS-native applications, medical transcription

---

### Azure Speech Services

**Specifications**:
- Languages: 100+
- Rating: 9.5/10 (highest among cloud providers)
- Market share: 19.6%

**Features**:
- Customizable speech models
- Noise suppression
- Power BI integration
- Azure Cognitive Services ecosystem

**Pricing**:
- Standard: $1/audio hour (~$0.017/min)
- Custom speech: $1.40/audio hour
- Conversation transcription: $2.10/audio hour

**Strengths**:
- Best accent/dialect handling
- Enterprise integration
- Custom model training

**Best For**: Enterprise applications, custom vocabularies, Microsoft ecosystem

---

## 2. Open-Source Alternatives

### NVIDIA Canary-Qwen-2.5B (SOTA July 2025)

**Overview**: Hybrid ASR-LLM model combining FastConformer encoder with Qwen3-1.7B decoder.

**Specifications**:
- WER: 5.63% (tops Hugging Face Open ASR Leaderboard)
- LibriSpeech Clean: 1.6% WER
- LibriSpeech Other: 3.1% WER
- Parameters: 2.5B
- RTFx: 418 (418x faster than real-time)
- License: CC-BY (commercial use allowed)

**Architecture**:
- FastConformer encoder optimized for speech
- Unmodified Qwen3-1.7B LLM decoder
- Dual mode: Pure transcription + intelligent analysis (summarization, Q&A)

**Training**:
- 90k steps on 32 NVIDIA A100 80GB GPUs
- LLM parameters frozen; encoder, projection, LoRA trainable

**Best For**: Highest accuracy requirements, enterprise deployments

---

### NVIDIA Canary-1B Family

**Canary-1B**:
- WER: 6.67% average
- Languages: 4 (English, German, French, Spanish)
- Parameters: 1B
- Translation support

**Canary-1B-v2**:
- Languages: 25
- Quality comparable to 3x larger models
- Inference up to 10x faster

**Canary-1B-Flash**:
- Parameters: 883M
- RTFx: >1000 (fastest variant)
- Same language support as original

**Best For**: Multilingual transcription with good speed/accuracy balance

---

### NVIDIA Parakeet TDT

**Overview**: Streaming-optimized model using RNN-Transducer architecture.

**Specifications**:
- RTFx: >2000 (among fastest on Open ASR)
- Parameters: 1.1B
- Training data: 65,000 hours diverse English audio
- Architecture: RNN-Transducer for streaming

**Best For**: Low-latency streaming, real-time voice agents

---

### OpenAI Whisper

**Whisper Large v3**:
- WER: 7.88%
- Parameters: 1.55B
- Languages: 99+
- VRAM: ~10GB

**Whisper Large v3 Turbo**:
- 6x faster than Large v3
- Decoder layers: 4 (vs 32 in v3)
- Parameters: 809M
- VRAM: ~6GB
- Accuracy: Within 1-2% of full model
- RTFx: 216

**Self-Hosting Costs**:
- Hidden costs: GPU provisioning, auto-scaling, maintenance
- Real cost often exceeds $1/hour (3x managed alternatives)

**Best For**: Multilingual workloads, batch processing, prototyping

---

### Vosk

**Overview**: Lightweight offline speech recognition for edge devices.

**Specifications**:
- Languages: 20+
- Model sizes: 50MB to 1.8GB
- Platforms: Android, iOS, Raspberry Pi, servers

**Strengths**:
- True offline operation
- CPU-only (no GPU required)
- Minimal memory footprint
- Low latency

**Weaknesses**:
- Lower accuracy than Whisper/Canary
- Struggles with background noise
- Limited documentation

**Best For**: Offline applications, embedded systems, privacy-sensitive deployments

---

### SpeechBrain

**Overview**: PyTorch-based conversational AI toolkit.

**Specifications**:
- 200+ recipes for speech tasks
- 100+ models on Hugging Face
- LibriSpeech WER: 2.46% Clean, 5.77% Other
- Common Voice WER: 15.58%

**Capabilities**:
- ASR, TTS, speaker recognition
- Speech enhancement/separation
- Spoken language understanding
- Custom training support

**Best For**: Research, domain-specific training, custom vocabularies

---

## 3. Real-Time/Streaming Options

### Latency Comparison

| Solution | Latency | Streaming WER | RTFx |
|----------|---------|---------------|------|
| AssemblyAI Universal-Streaming | 300ms (P50) | 14.5% | N/A |
| Deepgram Nova-3 | <300ms | ~18% | High |
| Speechmatics | <200ms | Variable | High |
| NVIDIA Parakeet TDT | Very low | Good | >2000 |
| Kyutai 1B | 1s initial | 6.4% | Fast |
| Kyutai 2.6B | 2.5s initial | 6.4% | Fast |

### Recommendations by Use Case

**Voice Agents (Production)**:
- AssemblyAI Universal-Streaming (99.95% SLA, immutable transcripts)
- Deepgram Nova-3 (multilingual, low latency)

**Self-Hosted Streaming**:
- NVIDIA Parakeet TDT (2000+ RTFx)
- Speechmatics (on-premises option)

**Privacy-Sensitive**:
- Vosk (completely offline)
- SpeechBrain with custom training

---

## 4. Accuracy Benchmarks (WER Comparison)

### Hugging Face Open ASR Leaderboard (2025)

| Rank | Model | WER | Type |
|------|-------|-----|------|
| 1 | NVIDIA Canary-Qwen-2.5B | 5.63% | Open Source |
| 2 | Google Chirp 2 | ~6% | Commercial |
| 3 | NVIDIA Canary-1B | 6.67% | Open Source |
| 4 | Kyutai 2.6B | 6.4% | Open Source |
| 5 | AssemblyAI Universal-2 | 6.68% | Commercial |
| 6 | OpenAI Whisper Large v3 | 7.88% | Open Source |
| 7 | Deepgram Nova-2 | 8.4% | Commercial |

### LibriSpeech Benchmarks

| Model | Clean | Other |
|-------|-------|-------|
| Canary-Qwen-2.5B | 1.6% | 3.1% |
| SpeechBrain | 2.46% | 5.77% |
| Whisper Large v3 | 2.7% | 5.2% |

### Real-World Performance Notes

- Lab benchmarks (LibriSpeech) show 2-5% WER
- Real-world audio (noise, accents) typically 15-25% WER
- Streaming models show higher WER than batch processing
- Domain-specific models (medical, legal) significantly outperform general models

---

## 5. Pricing Summary

### Per-Minute Costs (Lowest to Highest)

| Provider | $/min | Notes |
|----------|-------|-------|
| OpenAI Whisper API | $0.006 | No streaming, limited features |
| AssemblyAI | $0.0025 | Base rate; add-ons extra |
| Deepgram | $0.0077 | Per-second billing |
| Google Cloud | $0.016 | Standard; $0.004 batch |
| Azure | $0.017 | Standard rate |
| AWS Transcribe | $0.024 | Batch; $0.030 streaming |

### Free Tiers

- **Google**: 60 min/month + $300 new user credit
- **AWS**: 60 min/month (first year)
- **AssemblyAI**: $50 credit (~300 hours)

### Hidden Cost Factors

1. **Billing granularity**: Per-second vs 15-second rounding (45% difference)
2. **Infrastructure**: Cloud storage, functions, egress fees
3. **Self-hosting**: GPU provisioning, scaling, maintenance
4. **Add-on features**: Diarization, PII redaction, sentiment

---

## 6. Use Case Recommendations

### Call Centers / Customer Service
- **Primary**: Deepgram Nova-3 or AssemblyAI Universal-2
- **Why**: Real-time streaming, speaker diarization, sentiment analysis

### Medical Transcription
- **Primary**: Deepgram Nova-2 Medical or AWS Transcribe Medical
- **Why**: HIPAA compliance, medical vocabulary

### Multilingual Applications
- **Primary**: Google Chirp 3 or Whisper Large v3
- **Why**: 100+ languages, automatic detection

### Offline / Edge Devices
- **Primary**: Vosk
- **Why**: 50MB models, no internet required

### Highest Accuracy (English)
- **Primary**: NVIDIA Canary-Qwen-2.5B
- **Why**: 5.63% WER, open source, LLM integration

### Budget-Conscious
- **Primary**: OpenAI Whisper API or self-hosted Whisper
- **Why**: Lowest per-minute cost

### Enterprise / Custom Vocabularies
- **Primary**: Azure Speech or SpeechBrain
- **Why**: Custom model training, enterprise features
