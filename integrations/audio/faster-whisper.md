# faster-whisper Integration

> **4x faster Whisper transcription with CTranslate2**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Optimized Whisper implementation using CTranslate2 |
| **Why** | 4x faster than OpenAI Whisper, lower VRAM |
| **Accuracy** | Same as original Whisper |
| **Best For** | Production transcription, batch processing |

### Speed Comparison

| Model | OpenAI Whisper | faster-whisper |
|-------|----------------|----------------|
| large-v3 | 1x (baseline) | 4x faster |
| VRAM usage | 10GB | 4GB |
| Batch support | Limited | Efficient |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **GPU** | NVIDIA recommended (CPU works) |
| **VRAM** | 2-8GB depending on model |

---

## Quick Start (10 min)

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

# Load model (auto-downloads)
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Transcribe
segments, info = model.transcribe("audio.mp3")

print(f"Detected language: {info.language} ({info.language_probability:.0%})")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

---

## Learning Path

### L0: Basic Transcription (1 hour)
- [ ] Install faster-whisper
- [ ] Transcribe audio file
- [ ] Export to text file
- [ ] Try different model sizes

### L1: Advanced Features (2-3 hours)
- [ ] Word-level timestamps
- [ ] VAD filtering
- [ ] Language detection
- [ ] SRT/VTT export

### L2: Production (4-6 hours)
- [ ] Batch processing
- [ ] Real-time streaming
- [ ] API wrapper
- [ ] Error handling

---

## Code Examples

### Word-Level Timestamps

```python
segments, _ = model.transcribe(
    "audio.mp3",
    word_timestamps=True
)

for segment in segments:
    for word in segment.words:
        print(f"{word.start:.2f} - {word.end:.2f}: {word.word}")
```

### Export to SRT

```python
def to_srt(segments):
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        srt_content.append(f"{i}\n{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(srt_content)

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

segments, _ = model.transcribe("audio.mp3")
srt = to_srt(segments)

with open("output.srt", "w") as f:
    f.write(srt)
```

### VAD Filtering

```python
# Use Silero VAD to skip silence
segments, _ = model.transcribe(
    "audio.mp3",
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500,
        speech_pad_ms=400
    )
)
```

### Batch Processing

```python
from pathlib import Path
import json

def batch_transcribe(audio_dir, output_dir):
    audio_files = list(Path(audio_dir).glob("*.mp3"))
    results = {}

    for audio_file in audio_files:
        print(f"Processing: {audio_file.name}")
        segments, info = model.transcribe(str(audio_file))

        results[audio_file.name] = {
            "language": info.language,
            "text": " ".join(s.text for s in segments),
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ]
        }

    Path(output_dir).mkdir(exist_ok=True)
    with open(f"{output_dir}/transcripts.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
```

### CPU Inference

```python
# For machines without GPU
model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"  # Quantized for CPU
)
```

### Streaming (Chunked)

```python
import sounddevice as sd
import numpy as np
from queue import Queue

audio_queue = Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

# Record in chunks
with sd.InputStream(samplerate=16000, channels=1, callback=audio_callback):
    while True:
        audio_chunk = audio_queue.get()
        # Process chunk (accumulate ~30s before transcribing)
```

---

## Model Sizes

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| tiny | 1GB | ⚡⚡⚡⚡ | ⭐⭐ |
| base | 1GB | ⚡⚡⚡ | ⭐⭐⭐ |
| small | 2GB | ⚡⚡⚡ | ⭐⭐⭐ |
| medium | 5GB | ⚡⚡ | ⭐⭐⭐⭐ |
| large-v3 | 8GB | ⚡ | ⭐⭐⭐⭐⭐ |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Use smaller model, or `compute_type="int8"` |
| Slow on CPU | Use `int8` quantization |
| Hallucinations | Enable VAD filter, check audio quality |
| Wrong language | Specify `language="en"` parameter |

---

## Resources

- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [Model Comparison](https://github.com/openai/whisper#available-models-and-languages)

---

*Part of [Luno-AI](../../README.md) | Audio AI Track*
