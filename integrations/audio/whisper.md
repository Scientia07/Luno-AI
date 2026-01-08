# Whisper Speech-to-Text Integration

> **Category**: Audio AI
> **Difficulty**: Beginner
> **Setup Time**: 1-2 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
Whisper is OpenAI's automatic speech recognition (ASR) system that transcribes audio to text with high accuracy across 99 languages, including translation to English.

### Why Use It
- **Multilingual**: 99 languages supported
- **Accurate**: State-of-the-art transcription quality
- **Robust**: Handles accents, background noise, technical jargon
- **Free**: Open source, no API costs
- **Versatile**: Transcription, translation, timestamps

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Transcription | Speech to text in original language |
| Translation | Translate any language to English |
| Timestamps | Word-level and segment timing |
| Language Detection | Auto-detect spoken language |
| Voice Activity | Detect speech vs silence |

### Implementation Comparison
| Implementation | Speed | Best For | VRAM |
|----------------|-------|----------|------|
| **Whisper** (OpenAI) | Baseline | Reference | High |
| **faster-whisper** | 4x faster | Production | Medium |
| **whisper.cpp** | CPU optimized | Edge/no GPU | None |
| **WhisperX** | Word timestamps | Subtitles | Medium |
| **insanely-fast-whisper** | Batched | Bulk processing | High |
| **Distil-Whisper** | 6x faster | Mobile | Low |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU works) | NVIDIA 4GB+ VRAM |
| RAM | 8 GB | 16 GB |
| Storage | 3 GB | 10 GB (all models) |

### Software Dependencies
```bash
# Required
python >= 3.8
pip install openai-whisper

# For faster-whisper (recommended)
pip install faster-whisper

# For GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Prior Knowledge
- [x] Python basics
- [ ] Audio file formats (helpful)

---

## Quick Start (15 minutes)

### 1. Install
```bash
# OpenAI Whisper
pip install openai-whisper

# OR faster-whisper (recommended)
pip install faster-whisper
```

### 2. Basic Transcription
```python
# Using faster-whisper (recommended)
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")
# For CPU: model = WhisperModel("base", device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3")

print(f"Detected language: {info.language} ({info.language_probability:.0%})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### 3. Verify Installation
```bash
# OpenAI Whisper CLI
whisper audio.mp3 --model base

# Or Python test
python -c "from faster_whisper import WhisperModel; print('OK')"
```

---

## Full Setup

### Model Sizes

| Model | Size | VRAM | English WER | Speed |
|-------|------|------|-------------|-------|
| `tiny` | 39 MB | ~1 GB | 7.7% | Fastest |
| `base` | 74 MB | ~1 GB | 5.3% | Fast |
| `small` | 244 MB | ~2 GB | 4.0% | Medium |
| `medium` | 769 MB | ~5 GB | 3.2% | Slow |
| `large-v3` | 1.5 GB | ~10 GB | 2.7% | Slowest |
| `turbo` | 809 MB | ~6 GB | 3.0% | Fast |

### Recommended Setup (faster-whisper)

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

# GPU setup
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",  # or int8_float16 for less VRAM
    download_root="./models",
)

# CPU setup
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
)
```

### Configuration Options

```python
segments, info = model.transcribe(
    "audio.mp3",

    # Language
    language="en",           # or None for auto-detect
    task="transcribe",       # or "translate" to English

    # Quality
    beam_size=5,             # higher = better but slower
    best_of=5,               # candidates per beam
    temperature=0.0,         # 0 = deterministic

    # Timestamps
    word_timestamps=True,    # word-level timing

    # VAD (voice activity detection)
    vad_filter=True,         # skip silence
    vad_parameters={
        "min_silence_duration_ms": 500,
    },

    # Output
    without_timestamps=False,
    max_new_tokens=448,
)
```

---

## Learning Path

### L0: Basic Transcription (1 hour)
**Goal**: Transcribe audio files

- [x] Install faster-whisper
- [ ] Transcribe an audio file
- [ ] Understand output format

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3")

# Full text
full_text = " ".join(segment.text for segment in segments)
print(full_text)
```

### L1: Timestamps & Translation (2 hours)
**Goal**: Generate subtitles and translate

- [ ] Word-level timestamps
- [ ] Generate SRT subtitles
- [ ] Translate to English

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Word timestamps
segments, _ = model.transcribe("audio.mp3", word_timestamps=True)

for segment in segments:
    for word in segment.words:
        print(f"[{word.start:.2f}s] {word.word}")

# Translation (any language -> English)
segments, _ = model.transcribe("german_audio.mp3", task="translate")
```

**Generate SRT:**
```python
def segments_to_srt(segments):
    srt = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp(seg.start)
        end = format_timestamp(seg.end)
        srt.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(srt)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

### L2: Real-time & Streaming (3 hours)
**Goal**: Live transcription

- [ ] Microphone input
- [ ] Streaming transcription
- [ ] Low-latency setup

```python
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")

def record_and_transcribe(duration=5, sample_rate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate),
                   samplerate=sample_rate,
                   channels=1,
                   dtype=np.float32)
    sd.wait()

    # Save temp file (faster-whisper needs file path)
    import tempfile
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, sample_rate)
        segments, _ = model.transcribe(f.name)
        return " ".join(s.text for s in segments)

print(record_and_transcribe())
```

### L3: WhisperX & Diarization (4+ hours)
**Goal**: Speaker identification + precise alignment

- [ ] Install WhisperX
- [ ] Speaker diarization
- [ ] Forced alignment

```bash
pip install whisperx
```

```python
import whisperx

# Load model
model = whisperx.load_model("large-v3", device="cuda", compute_type="float16")

# Transcribe
audio = whisperx.load_audio("audio.mp3")
result = model.transcribe(audio, batch_size=16)

# Align timestamps
model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda")
result = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda")

# Diarization (who said what)
diarize_model = whisperx.DiarizationPipeline(use_auth_token="HF_TOKEN", device="cuda")
diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

for segment in result["segments"]:
    print(f"[{segment['speaker']}] {segment['text']}")
```

---

## Code Examples

### Example 1: Batch Process Multiple Files
```python
from faster_whisper import WhisperModel
from pathlib import Path
import json

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe_folder(folder_path, output_path):
    results = {}
    for audio_file in Path(folder_path).glob("*.mp3"):
        segments, info = model.transcribe(str(audio_file))
        results[audio_file.name] = {
            "language": info.language,
            "text": " ".join(s.text for s in segments),
        }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

transcribe_folder("./audio_files", "transcripts.json")
```

### Example 2: With Progress Bar
```python
from faster_whisper import WhisperModel
from tqdm import tqdm

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

segments, info = model.transcribe("long_audio.mp3")

# Convert generator to list with progress
segment_list = []
for segment in tqdm(segments, desc="Transcribing"):
    segment_list.append(segment)

print(f"Total segments: {len(segment_list)}")
```

### Example 3: Language Detection Only
```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")

# Quick language detection (first 30 seconds)
_, info = model.transcribe("audio.mp3", language=None)

print(f"Language: {info.language}")
print(f"Confidence: {info.language_probability:.1%}")
```

### Example 4: YouTube Transcription
```python
from faster_whisper import WhisperModel
import yt_dlp

def transcribe_youtube(url):
    # Download audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Transcribe
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _ = model.transcribe("temp_audio.mp3")

    return " ".join(s.text for s in segments)

text = transcribe_youtube("https://youtube.com/watch?v=...")
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| Deepgram API | Cloud alternative, faster | [deepgram.md](./deepgram.md) |
| XTTS | Text-to-speech from transcript | [xtts.md](./xtts.md) |
| LangGraph | Voice-powered agents | [langgraph.md](../agents/langgraph.md) |
| RAG | Index audio content | [rag.md](../agents/rag.md) |
| pyannote | Speaker diarization | External |

### Output Formats
```python
# Get different formats
segments, info = model.transcribe("audio.mp3", word_timestamps=True)

# Plain text
text = " ".join(s.text for s in segments)

# JSON with timestamps
data = [{
    "start": s.start,
    "end": s.end,
    "text": s.text,
    "words": [{"word": w.word, "start": w.start, "end": w.end} for w in s.words]
} for s in segments]

# SRT subtitles (see L1 example)
```

---

## Troubleshooting

### Common Issues

#### Issue 1: CUDA Out of Memory
**Symptoms**: RuntimeError: CUDA out of memory
**Solution**:
```python
# Use smaller model
model = WhisperModel("base", ...)

# Or use INT8 quantization
model = WhisperModel("large-v3", compute_type="int8_float16")

# Or use CPU
model = WhisperModel("base", device="cpu", compute_type="int8")
```

#### Issue 2: Slow Transcription
**Symptoms**: Takes much longer than audio duration
**Solution**:
```python
# Enable VAD to skip silence
segments, _ = model.transcribe("audio.mp3", vad_filter=True)

# Use faster-whisper instead of openai-whisper
# Use GPU with FP16
model = WhisperModel("base", device="cuda", compute_type="float16")
```

#### Issue 3: Poor Accuracy
**Symptoms**: Wrong words, missing content
**Solution**:
```python
# Use larger model
model = WhisperModel("large-v3", ...)

# Specify language if known
segments, _ = model.transcribe("audio.mp3", language="en")

# Increase beam size
segments, _ = model.transcribe("audio.mp3", beam_size=10)
```

### Performance Tips
- Use `faster-whisper` (4x faster than OpenAI's implementation)
- Enable VAD filter to skip silence
- Use `turbo` model for best speed/accuracy balance
- Batch process with `insanely-fast-whisper` for bulk files

---

## Resources

### Official
- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [WhisperX](https://github.com/m-bain/whisperX)

### Tutorials
- [faster-whisper Guide](https://github.com/SYSTRAN/faster-whisper#usage)
- [WhisperX Diarization](https://github.com/m-bain/whisperX#speaker-diarization)

### Community
- [Whisper Discussions](https://github.com/openai/whisper/discussions)
- [HuggingFace Whisper](https://huggingface.co/openai/whisper-large-v3)

---

## Related Integrations

| Next Step | Why | Link |
|-----------|-----|------|
| Deepgram API | Cloud alternative | [deepgram.md](./deepgram.md) |
| XTTS | Generate speech | [xtts.md](./xtts.md) |
| Real-time Pipeline | Live transcription | [realtime-audio.md](./realtime-audio.md) |

---

*Part of [Luno-AI Integration Hub](../_index.md) | Audio AI Track*
