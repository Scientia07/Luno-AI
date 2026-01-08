# Audio AI: Speech, Music & Sound

> **Machines that hear and speak** - from transcription to voice cloning to music generation.

---

## Layer Navigation

| Layer | Content | Status |
|-------|---------|--------|
| L0 | [Overview](#overview) | This file |
| L1 | [Concepts](./concepts.md) | Pending |
| L2 | [Deep Dive](./deep-dive.md) | Pending |
| L3 | [Labs](../../labs/audio/) | Pending |
| L4 | [Advanced](./advanced.md) | Pending |

---

## Overview

Audio AI has exploded with capabilities: real-time transcription, voice cloning in seconds, music generation from text, and more. This domain is where AI gets truly creative.

```
                    AUDIO AI LANDSCAPE

    ┌─────────────────────────────────────────────────────┐
    │                  UNDERSTANDING                       │
    │  Speech-to-Text │ Speaker ID │ Emotion │ Music      │
    └─────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │                   GENERATION                         │
    │  Text-to-Speech │ Voice Clone │ Music │ Sound FX    │
    └─────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │                  TRANSFORMATION                      │
    │  Voice Conversion │ Enhancement │ Separation        │
    └─────────────────────────────────────────────────────┘
```

---

## Speech Recognition (ASR)

### Whisper AI

OpenAI's game-changing transcription model.

| Model | Size | Speed | Accuracy | Languages |
|-------|------|-------|----------|-----------|
| `tiny` | 39M | Fastest | Good | Multi |
| `base` | 74M | Fast | Better | Multi |
| `small` | 244M | Medium | Good | Multi |
| `medium` | 769M | Slow | Very Good | Multi |
| `large-v3` | 1.5B | Slowest | Best | 99 |
| `turbo` | 809M | Fast | Excellent | Multi |

```python
# Basic Whisper usage
import whisper
model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

### Faster Implementations

| Tool | Speed | Use Case |
|------|-------|----------|
| **faster-whisper** | 4x faster | Python, production |
| **whisper.cpp** | CPU optimized | Edge devices |
| **WhisperX** | Word-level timestamps | Subtitles |
| **Insanely-Fast-Whisper** | Batched GPU | Bulk processing |

```python
# faster-whisper example
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cuda")
segments, info = model.transcribe("audio.mp3")
```

---

## Text-to-Speech (TTS)

### Top Models

| Model | Quality | Speed | Voice Cloning | Open |
|-------|---------|-------|---------------|------|
| **XTTS v2** | Excellent | Medium | Yes (3s sample) | Yes |
| **Bark** | Very Good | Slow | Limited | Yes |
| **StyleTTS2** | Excellent | Fast | Fine-tune | Yes |
| **ElevenLabs** | Best | Fast | Yes | No |
| **OpenAI TTS** | Very Good | Fast | No | No |
| **Piper** | Good | Very Fast | No | Yes |

### XTTS (Recommended Open Source)

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# Clone voice from sample
tts.tts_to_file(
    text="Hello, this is my cloned voice!",
    speaker_wav="voice_sample.wav",
    language="en",
    file_path="output.wav"
)
```

### Quick Comparison

```
Need real-time? ──────────▶ Piper, Coqui
Need quality? ────────────▶ ElevenLabs, XTTS
Need voice clone? ────────▶ XTTS, ElevenLabs
Need many languages? ─────▶ XTTS, Bark
```

---

## Voice Cloning & Conversion

### Cloning Approaches

| Approach | Data Needed | Quality | Training |
|----------|-------------|---------|----------|
| **Zero-shot** | 3-10s sample | Good | None |
| **Few-shot** | 1-5 mins | Better | None |
| **Fine-tuned** | 30+ mins | Best | Hours |

### RVC (Retrieval-based Voice Conversion)

Convert any voice to any other voice.

```
Your Voice ──▶ RVC Model ──▶ Target Voice
                   ↑
          Trained on target
```

Uses:
- Singing voice conversion
- Voice acting
- Accessibility

### OpenVoice

Instant voice cloning with style control.

```python
# Clone voice with emotion control
from openvoice import se_extractor, voice_conversion

# Extract speaker embedding
speaker_embedding = se_extractor.extract("reference.wav")

# Convert with style
voice_conversion.convert(
    audio="input.wav",
    speaker=speaker_embedding,
    style="cheerful"
)
```

---

## Music Generation

### Models

| Model | Output | Quality | Open |
|-------|--------|---------|------|
| **Suno v4** | Full songs | Best | No |
| **Udio** | Full songs | Excellent | No |
| **MusicGen** | Instrumental | Very Good | Yes |
| **Stable Audio** | Instrumental | Good | Partial |
| **AudioCraft** | Various | Very Good | Yes |

### MusicGen (Open Source)

```python
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-large")
model.set_generation_params(duration=30)

# Generate from text
wav = model.generate(["upbeat electronic dance music"])

# Generate continuation
wav = model.generate_continuation(existing_audio, prompt)
```

---

## Sound Effects & Audio

### AudioLDM / AudioGen

Text to sound effects:

```python
# "A dog barking in the distance"
# "Rain on a tin roof"
# "Footsteps on gravel"
```

### Audio Separation

Split audio into components:

| Tool | Separates |
|------|-----------|
| **Demucs** | Vocals, drums, bass, other |
| **Spleeter** | Vocals, accompaniment |
| **UVR** | Various stems |

```python
from demucs import separate
tracks = separate.main(["--mp3", "song.mp3"])
# Creates: vocals.mp3, drums.mp3, bass.mp3, other.mp3
```

---

## Real-time Processing

### Live Transcription

```python
from faster_whisper import WhisperModel
import sounddevice as sd

model = WhisperModel("base")

def callback(indata, frames, time, status):
    # Process audio chunk
    segments, _ = model.transcribe(indata)
    for segment in segments:
        print(segment.text)

with sd.InputStream(callback=callback):
    input("Recording... Press Enter to stop")
```

### Live Voice Conversion

```
Mic Input ──▶ RVC ──▶ Speaker Output
              ↓
         Real-time
         (~50ms latency)
```

---

## Audio Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    AUDIO PIPELINE                            │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   PREPROCESSING │     │   ENHANCEMENT   │
│  - Normalize    │ ──▶ │  - Denoise      │
│  - Resample     │     │  - Dereverberate│
└─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   TRANSCRIBE    │     │    ANALYZE      │
│   (Whisper)     │ ──▶ │  - Diarization  │
│                 │     │  - Emotion      │
└─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   GENERATE      │     │    OUTPUT       │
│   (TTS/Music)   │ ──▶ │  - File         │
│                 │     │  - Stream       │
└─────────────────┘     └─────────────────┘
```

---

## Labs

| Notebook | Focus |
|----------|-------|
| `01-whisper-transcription.ipynb` | Speech to text |
| `02-tts-comparison.ipynb` | Text to speech |
| `03-voice-cloning.ipynb` | Clone voices |
| `04-music-generation.ipynb` | Create music |
| `05-audio-separation.ipynb` | Split tracks |
| `06-realtime-audio.ipynb` | Live processing |

---

## Next Steps

- L1: [How Audio AI Works](./concepts.md)
- L2: [Whisper Architecture](./deep-dive.md)
- Related: [LLMs](../llms/README.md) (voice assistants)

---

*"Audio is half the experience - now AI masters both."*
