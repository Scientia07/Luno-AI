# SeamlessM4T Translation Integration

> **Universal speech and text translation**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Meta's massively multilingual translation |
| **Why** | 100+ languages, speech-to-speech |
| **Capabilities** | S2S, S2T, T2S, T2T |
| **Best For** | Cross-language communication |

### Capabilities

| Mode | Description |
|------|-------------|
| **S2S** | Speech to Speech |
| **S2T** | Speech to Text |
| **T2S** | Text to Speech |
| **T2T** | Text to Text |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | 8GB+ VRAM |
| **Python** | 3.10+ |
| **Storage** | 10GB+ for models |

---

## Quick Start (20 min)

```bash
pip install transformers torch torchaudio
```

### Text Translation

```python
from transformers import SeamlessM4TModel, SeamlessM4TProcessor
import torch

# Load model
model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-medium")
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-medium")

# Text to text translation
text = "Hello, how are you today?"
inputs = processor(text=text, src_lang="eng", return_tensors="pt")

output_tokens = model.generate(
    **inputs,
    tgt_lang="spa",
    generate_speech=False
)

translated = processor.decode(output_tokens[0], skip_special_tokens=True)
print(f"Spanish: {translated}")
```

### Speech Translation

```python
import torchaudio

# Load audio
audio, sample_rate = torchaudio.load("speech.wav")

# Resample to 16kHz if needed
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    audio = resampler(audio)

# Speech to text
inputs = processor(audios=audio.squeeze(), return_tensors="pt")

output_tokens = model.generate(
    **inputs,
    tgt_lang="fra",
    generate_speech=False
)

translation = processor.decode(output_tokens[0], skip_special_tokens=True)
print(f"French translation: {translation}")
```

---

## Learning Path

### L0: Text Translation (1-2 hours)
- [ ] Install SeamlessM4T
- [ ] Translate text
- [ ] Try multiple languages
- [ ] Understand language codes

### L1: Speech Translation (2-3 hours)
- [ ] Speech to text
- [ ] Text to speech
- [ ] Speech to speech
- [ ] Audio preprocessing

### L2: Production (4-6 hours)
- [ ] Batch processing
- [ ] Streaming translation
- [ ] API deployment
- [ ] Quality evaluation

---

## Code Examples

### Speech to Speech

```python
from transformers import SeamlessM4TModel, SeamlessM4TProcessor
import torchaudio
import soundfile as sf

model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-medium")
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-medium")

# Load audio
audio, sr = torchaudio.load("input.wav")
if sr != 16000:
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)

# Speech to speech translation
inputs = processor(audios=audio.squeeze(), return_tensors="pt")

output = model.generate(
    **inputs,
    tgt_lang="deu",  # German
    generate_speech=True,
    return_intermediate_token_ids=True
)

# Get audio output
audio_array = output.audio_array.squeeze().cpu().numpy()

# Save
sf.write("translated_speech.wav", audio_array, 16000)

# Also get text
text_tokens = output.sequences
text = processor.decode(text_tokens[0], skip_special_tokens=True)
print(f"German text: {text}")
```

### Batch Translation

```python
def batch_translate(texts: list, src_lang: str, tgt_lang: str):
    """Translate multiple texts efficiently"""
    results = []

    for text in texts:
        inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")

        output_tokens = model.generate(
            **inputs,
            tgt_lang=tgt_lang,
            generate_speech=False
        )

        translated = processor.decode(output_tokens[0], skip_special_tokens=True)
        results.append(translated)

    return results

# Usage
texts = [
    "The weather is nice today.",
    "I love learning new languages.",
    "Technology connects the world."
]

translations = batch_translate(texts, "eng", "jpn")
for orig, trans in zip(texts, translations):
    print(f"{orig} -> {trans}")
```

### Text to Speech

```python
def text_to_speech(text: str, src_lang: str, tgt_lang: str):
    """Generate speech in target language from text"""
    inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")

    output = model.generate(
        **inputs,
        tgt_lang=tgt_lang,
        generate_speech=True
    )

    audio = output.audio_array.squeeze().cpu().numpy()
    return audio

# Generate Chinese speech from English text
audio = text_to_speech(
    "Welcome to our presentation.",
    src_lang="eng",
    tgt_lang="cmn"  # Mandarin Chinese
)

sf.write("chinese_speech.wav", audio, 16000)
```

### Real-time Translation

```python
import pyaudio
import numpy as np
import threading
import queue

class RealtimeTranslator:
    def __init__(self, src_lang: str, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.audio_queue = queue.Queue()
        self.model = SeamlessM4TModel.from_pretrained("facebook/seamless-m4t-medium")
        self.processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-medium")

    def record_audio(self, duration: float = 3.0):
        """Record audio chunk"""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True)

        frames = []
        for _ in range(int(16000 * duration / 1024)):
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))

        stream.stop_stream()
        stream.close()
        p.terminate()

        return np.concatenate(frames)

    def translate_chunk(self, audio):
        """Translate audio chunk"""
        import torch
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        inputs = self.processor(audios=audio_tensor, return_tensors="pt")

        output = self.model.generate(
            **inputs,
            tgt_lang=self.tgt_lang,
            generate_speech=False
        )

        return self.processor.decode(output[0], skip_special_tokens=True)

    def run(self):
        """Main loop"""
        print(f"Translating {self.src_lang} -> {self.tgt_lang}")
        print("Speak now...")

        while True:
            audio = self.record_audio()
            translation = self.translate_chunk(audio)
            if translation.strip():
                print(f"Translation: {translation}")

# Usage
translator = RealtimeTranslator("eng", "fra")
translator.run()
```

### Language Detection

```python
def detect_and_translate(audio_or_text, tgt_lang: str, is_audio: bool = False):
    """Auto-detect source language and translate"""
    if is_audio:
        inputs = processor(audios=audio_or_text, return_tensors="pt")
    else:
        # For text, we need to specify a source language
        # SeamlessM4T doesn't have built-in language detection
        inputs = processor(text=audio_or_text, src_lang="eng", return_tensors="pt")

    output = model.generate(
        **inputs,
        tgt_lang=tgt_lang,
        generate_speech=False
    )

    return processor.decode(output[0], skip_special_tokens=True)
```

---

## Supported Languages

Common language codes:
- `eng` - English
- `spa` - Spanish
- `fra` - French
- `deu` - German
- `cmn` - Mandarin Chinese
- `jpn` - Japanese
- `kor` - Korean
- `arb` - Arabic
- `hin` - Hindi
- `por` - Portuguese

See full list in documentation.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM error | Use smaller model, reduce batch size |
| Poor audio quality | Ensure 16kHz sample rate |
| Wrong language | Verify language codes |
| Slow inference | Use GPU, batch requests |

---

## Resources

- [SeamlessM4T Paper](https://arxiv.org/abs/2308.11596)
- [HuggingFace Model](https://huggingface.co/facebook/seamless-m4t-medium)
- [Seamless GitHub](https://github.com/facebookresearch/seamless_communication)
- [Language Codes](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md)

---

*Part of [Luno-AI](../../README.md) | Specialized Track*
