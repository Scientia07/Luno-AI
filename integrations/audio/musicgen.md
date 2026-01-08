# MusicGen Integration

> **Generate music from text descriptions**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Meta's text-to-music generation model |
| **Why** | Create royalty-free music from descriptions |
| **Output** | 30 seconds of audio |
| **Best For** | Background music, soundtracks, audio content |

### Model Variants

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `small` | 300M | ⭐⭐⭐ | Fast |
| `medium` | 1.5B | ⭐⭐⭐⭐ | Medium |
| `large` | 3.3B | ⭐⭐⭐⭐⭐ | Slow |
| `melody` | 1.5B | ⭐⭐⭐⭐ | Medium |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 8GB+ VRAM (16GB for large) |
| **Python** | 3.9+ |
| **Storage** | 5-15GB for models |

---

## Quick Start (15 min)

```bash
pip install audiocraft torch torchaudio
```

```python
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load model
model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=15)

# Generate music
descriptions = ["upbeat electronic dance music with synths"]
wav = model.generate(descriptions)

# Save output
audio_write("generated_music", wav[0].cpu(), model.sample_rate, strategy="loudness")
```

---

## Learning Path

### L0: Basic Generation (1-2 hours)
- [ ] Install MusicGen
- [ ] Generate first track
- [ ] Try different prompts
- [ ] Compare model sizes

### L1: Advanced Control (2-3 hours)
- [ ] Melody conditioning
- [ ] Duration control
- [ ] Batch generation
- [ ] Prompt engineering

### L2: Production Usage (4-6 hours)
- [ ] Long-form generation
- [ ] Audio continuation
- [ ] API deployment
- [ ] Post-processing

---

## Code Examples

### Melody Conditioning

```python
from audiocraft.models import MusicGen
import torchaudio

# Load melody model
model = MusicGen.get_pretrained("facebook/musicgen-melody")
model.set_generation_params(duration=15)

# Load melody reference
melody, sr = torchaudio.load("reference_melody.wav")
melody = torchaudio.transforms.Resample(sr, model.sample_rate)(melody)

# Generate with melody
wav = model.generate_with_chroma(
    descriptions=["orchestral arrangement with strings"],
    melody_wavs=melody.unsqueeze(0),
    melody_sample_rate=model.sample_rate,
    progress=True
)

audio_write("melody_conditioned", wav[0].cpu(), model.sample_rate)
```

### Extended Generation

```python
from audiocraft.models import MusicGen
import torch

model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=30)

def generate_long_audio(description, total_duration=120, segment_duration=30):
    """Generate audio longer than 30 seconds by continuation"""
    segments = []
    prompt = None

    for i in range(0, total_duration, segment_duration):
        if prompt is None:
            # First segment
            wav = model.generate([description])
        else:
            # Continue from previous
            wav = model.generate_continuation(
                prompt,
                model.sample_rate,
                [description],
                progress=True
            )

        segments.append(wav[0])
        # Use last 3 seconds as prompt for next
        prompt = wav[0][:, -3*model.sample_rate:]

    # Concatenate all segments
    full_audio = torch.cat(segments, dim=1)
    return full_audio

long_track = generate_long_audio(
    "ambient electronic music with pads",
    total_duration=120
)
audio_write("long_track", long_track.cpu(), model.sample_rate)
```

### Batch Generation

```python
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)

# Multiple descriptions at once
descriptions = [
    "upbeat pop music with guitar",
    "calm piano jazz",
    "epic orchestral trailer music",
    "lo-fi hip hop beat"
]

# Generate batch
wavs = model.generate(descriptions, progress=True)

# Save all
for i, wav in enumerate(wavs):
    audio_write(f"track_{i}", wav.cpu(), model.sample_rate)
```

### Genre-Specific Prompts

```python
genre_prompts = {
    "electronic": "electronic dance music with heavy bass and synth leads, 128 bpm",
    "classical": "classical orchestral piece with strings and woodwinds, peaceful",
    "rock": "energetic rock music with electric guitar and drums",
    "jazz": "smooth jazz with saxophone, piano, and walking bass",
    "ambient": "atmospheric ambient music with pads and subtle textures",
    "hiphop": "lo-fi hip hop beat with vinyl crackle and mellow keys",
    "cinematic": "epic cinematic trailer music with brass and percussion",
}

model = MusicGen.get_pretrained("facebook/musicgen-medium")
model.set_generation_params(duration=15)

for genre, prompt in genre_prompts.items():
    wav = model.generate([prompt])
    audio_write(f"genre_{genre}", wav[0].cpu(), model.sample_rate)
```

### API Server

```python
from fastapi import FastAPI
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import io
import base64

app = FastAPI()
model = MusicGen.get_pretrained("facebook/musicgen-small")

@app.post("/generate")
async def generate_music(description: str, duration: int = 15):
    model.set_generation_params(duration=min(duration, 30))
    wav = model.generate([description])

    # Save to bytes
    buffer = io.BytesIO()
    audio_write(buffer, wav[0].cpu(), model.sample_rate, format="wav")

    return {
        "audio": base64.b64encode(buffer.getvalue()).decode(),
        "sample_rate": model.sample_rate,
        "duration": duration
    }

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Prompt Engineering

| Element | Examples |
|---------|----------|
| **Genre** | electronic, classical, rock, jazz, ambient |
| **Instruments** | guitar, piano, synth, drums, strings |
| **Mood** | upbeat, calm, energetic, melancholic, epic |
| **Tempo** | slow, medium tempo, fast, 120 bpm |
| **Style** | lo-fi, cinematic, minimal, complex |

### Effective Prompts

```python
good_prompts = [
    "upbeat electronic dance music with synthesizer leads and heavy bass drops",
    "peaceful acoustic guitar melody with soft percussion and nature sounds",
    "epic orchestral trailer music with brass fanfares and timpani drums",
    "smooth jazz piano trio with walking bass and brushed drums",
]

# Less effective
poor_prompts = [
    "good music",  # Too vague
    "song",        # No description
    "beat",        # Minimal info
]
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Use smaller model, reduce duration |
| Poor quality | Use medium/large model, better prompt |
| Repetitive | Add more variation to prompt |
| Wrong style | Be more specific with genre/instruments |

---

## Resources

- [AudioCraft GitHub](https://github.com/facebookresearch/audiocraft)
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [HuggingFace Models](https://huggingface.co/facebook/musicgen-small)
- [AudioCraft Demo](https://huggingface.co/spaces/facebook/MusicGen)

---

*Part of [Luno-AI](../../README.md) | Audio AI Track*
