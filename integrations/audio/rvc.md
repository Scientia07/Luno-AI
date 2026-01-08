# RVC Voice Conversion Integration

> **Transform any voice to sound like another**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Retrieval-based Voice Conversion |
| **Why** | High-quality voice cloning and conversion |
| **Training** | 10-20 minutes of audio |
| **Best For** | Voice cloning, singing conversion, character voices |

### RVC vs Alternatives

| Tool | Quality | Training Time | Real-time | Singing |
|------|---------|---------------|-----------|---------|
| **RVC** | ⭐⭐⭐⭐⭐ | 20-60 min | Yes | Excellent |
| **So-VITS-SVC** | ⭐⭐⭐⭐ | 2-4 hours | Limited | Excellent |
| **XTTS** | ⭐⭐⭐⭐ | Minutes | No | No |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 4GB+ VRAM (8GB recommended) |
| **Audio** | 10-20 minutes clean audio |
| **Python** | 3.10 |

---

## Quick Start (1 hour)

### Installation

```bash
# Clone RVC WebUI
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
cd Retrieval-based-Voice-Conversion-WebUI

# Install dependencies
pip install -r requirements.txt

# Download models
python tools/download_models.py
```

### Using Pre-trained Voice

```python
import sys
sys.path.append("Retrieval-based-Voice-Conversion-WebUI")

from infer.modules.vc.modules import VC

vc = VC()
vc.get_vc("pretrained_voice.pth")

# Convert audio
result = vc.vc_single(
    sid=0,
    input_audio_path="input.wav",
    f0_up_key=0,  # Pitch shift
    f0_method="rmvpe",
    index_path="pretrained_voice.index",
    index_rate=0.75
)

# Save output
import soundfile as sf
sf.write("output.wav", result[1], result[0])
```

### Web Interface

```bash
python web.py
# Open http://localhost:7865
```

---

## Learning Path

### L0: Basic Conversion (1-2 hours)
- [ ] Install RVC WebUI
- [ ] Use pre-trained voice
- [ ] Convert speech audio
- [ ] Adjust pitch settings

### L1: Training Custom Voice (3-4 hours)
- [ ] Prepare audio dataset
- [ ] Train voice model
- [ ] Create index file
- [ ] Test and refine

### L2: Advanced Usage (1-2 days)
- [ ] Real-time conversion
- [ ] Singing voice conversion
- [ ] Batch processing
- [ ] Integration with TTS

---

## Code Examples

### Prepare Training Data

```python
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

def prepare_dataset(input_file, output_dir, target_name):
    """Split audio into training chunks"""
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_file(input_file)

    # Split on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=500,
        silence_thresh=-40,
        keep_silence=200
    )

    # Export chunks
    for i, chunk in enumerate(chunks):
        if len(chunk) > 1000:  # Skip very short clips
            chunk.export(
                f"{output_dir}/{target_name}_{i:04d}.wav",
                format="wav"
            )

    print(f"Created {len(chunks)} training clips")

prepare_dataset("raw_audio.mp3", "dataset/my_voice", "voice")
```

### Training Script

```python
# Training is typically done via WebUI, but CLI is available:
# python train.py --exp_dir my_voice --sr 40000 --n_threads 4

# After training, files are in:
# - logs/my_voice/my_voice.pth (model)
# - logs/my_voice/my_voice.index (index)
```

### Batch Conversion

```python
import os
from pathlib import Path

def batch_convert(input_dir, output_dir, model_path, index_path):
    """Convert all audio files in directory"""
    os.makedirs(output_dir, exist_ok=True)

    vc = VC()
    vc.get_vc(model_path)

    for audio_file in Path(input_dir).glob("*.wav"):
        result = vc.vc_single(
            sid=0,
            input_audio_path=str(audio_file),
            f0_up_key=0,
            f0_method="rmvpe",
            index_path=index_path,
            index_rate=0.75
        )

        output_path = Path(output_dir) / audio_file.name
        sf.write(str(output_path), result[1], result[0])
        print(f"Converted: {audio_file.name}")

batch_convert("inputs/", "outputs/", "model.pth", "model.index")
```

### Real-time Conversion

```python
import pyaudio
import numpy as np

# Real-time requires specific setup
# Use RVC WebUI's real-time mode or:

class RealtimeVC:
    def __init__(self, model_path, index_path):
        self.vc = VC()
        self.vc.get_vc(model_path)
        self.index_path = index_path

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            output=True,
            frames_per_buffer=1024
        )

    def process_chunk(self, audio_chunk):
        # Convert chunk (simplified)
        result = self.vc.vc_single(
            sid=0,
            input_audio_path=audio_chunk,
            f0_up_key=0,
            f0_method="rmvpe",
            index_path=self.index_path,
            index_rate=0.75
        )
        return result[1]

    def run(self):
        while True:
            data = self.stream.read(1024)
            converted = self.process_chunk(data)
            self.stream.write(converted)
```

### Integration with TTS

```python
from TTS.api import TTS
import soundfile as sf

# Generate speech with TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
tts.tts_to_file(
    text="Hello, this is a test.",
    file_path="tts_output.wav"
)

# Convert to target voice with RVC
vc = VC()
vc.get_vc("target_voice.pth")

result = vc.vc_single(
    sid=0,
    input_audio_path="tts_output.wav",
    f0_up_key=0,
    f0_method="rmvpe",
    index_path="target_voice.index",
    index_rate=0.75
)

sf.write("final_output.wav", result[1], result[0])
```

---

## Training Parameters

| Parameter | Recommended | Effect |
|-----------|-------------|--------|
| Sample rate | 40000/48000 | Audio quality |
| Epochs | 200-500 | Training depth |
| Batch size | 8-16 | GPU usage |
| Index rate | 0.5-0.8 | Voice similarity |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Robotic output | Lower index rate, more training data |
| Wrong pitch | Adjust f0_up_key parameter |
| Artifacts | Use rmvpe f0 method |
| Training fails | Check audio quality, reduce batch size |

---

## Resources

- [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [RVC Guide](https://docs.google.com/document/d/13ebnzmeEBc6uzYCMt-QVFQk-whVrK4bw)
- [Voice Models](https://voice-models.com/)
- [Audio Preprocessing](https://github.com/Anjok07/ultimatevocalremovergui)

---

*Part of [Luno-AI](../../README.md) | Audio AI Track*
