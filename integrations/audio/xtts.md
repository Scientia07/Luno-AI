# XTTS Voice Cloning Integration

> **Category**: Audio AI
> **Difficulty**: Intermediate
> **Setup Time**: 2-3 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
XTTS (Cross-lingual Text-to-Speech) generates natural speech and can clone any voice from just 3-10 seconds of audio. Supports 17+ languages.

### Why Use It
- **Voice Cloning**: Clone any voice with seconds of audio
- **Multilingual**: 17+ languages supported
- **High Quality**: Near-human speech quality
- **Open Source**: Free, MIT license (Coqui TTS)
- **Fast**: Real-time capable on GPU

### Key Capabilities
| Capability | Description |
|------------|-------------|
| TTS | Text to natural speech |
| Voice Cloning | Clone voice from sample |
| Cross-lingual | Speak any language in any voice |
| Emotion Control | Some style control |
| Streaming | Real-time generation |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 4 GB VRAM | 8 GB VRAM |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |

### Software Dependencies
```bash
pip install TTS
# or for latest
pip install git+https://github.com/coqui-ai/TTS.git
```

---

## Quick Start (15 minutes)

### 1. Install
```bash
pip install TTS
```

### 2. Basic TTS
```python
from TTS.api import TTS

# Load XTTS v2
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")  # GPU

# Generate speech
tts.tts_to_file(
    text="Hello, this is a test of the XTTS system.",
    file_path="output.wav",
    language="en",
)
```

### 3. Voice Cloning
```python
# Clone voice from sample
tts.tts_to_file(
    text="This is my cloned voice speaking!",
    speaker_wav="voice_sample.wav",  # 3-10 seconds
    language="en",
    file_path="cloned_output.wav",
)
```

---

## Full Setup

### Voice Cloning
```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Best results: 6-10 second clear audio sample
tts.tts_to_file(
    text="Your text here",
    speaker_wav="reference.wav",
    language="en",
    file_path="output.wav",
)
```

### Streaming
```python
import sounddevice as sd

# Stream to speakers
chunks = tts.tts_stream(
    text="This is streaming speech.",
    speaker_wav="voice.wav",
    language="en",
)

for chunk in chunks:
    sd.play(chunk, samplerate=24000)
    sd.wait()
```

### Multiple Languages
```python
# Same voice, different language
for lang in ["en", "es", "fr", "de", "ja"]:
    tts.tts_to_file(
        text="Hello world" if lang == "en" else f"Hello in {lang}",
        speaker_wav="voice.wav",
        language=lang,
        file_path=f"output_{lang}.wav",
    )
```

---

## Code Examples

### Example 1: Voice from YouTube
```python
import yt_dlp
from TTS.api import TTS

# Download audio
ydl_opts = {'format': 'bestaudio', 'outtmpl': 'sample.%(ext)s'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(["youtube_url"])

# Clone and generate
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(
    text="Now I'm speaking with this voice!",
    speaker_wav="sample.webm",  # Works with various formats
    language="en",
    file_path="cloned.wav",
)
```

### Example 2: Batch Generation
```python
texts = [
    "First sentence.",
    "Second sentence.",
    "Third sentence.",
]

for i, text in enumerate(texts):
    tts.tts_to_file(
        text=text,
        speaker_wav="voice.wav",
        language="en",
        file_path=f"output_{i}.wav",
    )
```

---

## Integration Points

| Integration | Purpose | Link |
|-------------|---------|------|
| Whisper | Transcribe first | [whisper.md](./whisper.md) |
| RVC | Voice conversion | [rvc.md](./rvc.md) |
| LangGraph | Voice agents | [langgraph.md](../agents/langgraph.md) |

---

## Troubleshooting

### Out of Memory
```python
# Use CPU (slower)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
```

### Poor Voice Quality
- Use 6-10 second clear sample
- Remove background noise
- Use consistent audio quality

---

## Resources

- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [XTTS Paper](https://arxiv.org/abs/2406.04904)

---

*Part of [Luno-AI Integration Hub](../_index.md) | Audio AI Track*
