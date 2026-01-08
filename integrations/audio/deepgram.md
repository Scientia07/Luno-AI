# Deepgram Integration

> **Real-time speech-to-text API with Nova-3**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Cloud STT API optimized for speed and accuracy |
| **Why** | Real-time streaming, low latency, enterprise ready |
| **Model** | Nova-3 (latest), Nova-2 |
| **Pricing** | $0.0043/min (pay-as-you-go) |

### Why Deepgram?

| Feature | Deepgram | Whisper API |
|---------|----------|-------------|
| Latency | ~100ms | ~1-2s |
| Streaming | Yes | No |
| Diarization | Built-in | No |
| Custom vocab | Yes | No |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **API Key** | From console.deepgram.com |
| **Python** | 3.8+ |

---

## Quick Start (10 min)

```bash
pip install deepgram-sdk
```

### Pre-recorded Audio

```python
from deepgram import DeepgramClient, PrerecordedOptions

client = DeepgramClient("YOUR_API_KEY")

with open("audio.mp3", "rb") as audio:
    source = {"buffer": audio.read()}

options = PrerecordedOptions(
    model="nova-2",
    smart_format=True,
    punctuate=True
)

response = client.listen.prerecorded.v("1").transcribe_file(source, options)

print(response.results.channels[0].alternatives[0].transcript)
```

### Real-time Streaming

```python
import asyncio
from deepgram import DeepgramClient, LiveOptions

async def main():
    client = DeepgramClient("YOUR_API_KEY")

    connection = client.listen.live.v("1")

    async def on_message(self, result, **kwargs):
        transcript = result.channel.alternatives[0].transcript
        if transcript:
            print(f"Transcript: {transcript}")

    connection.on("transcript", on_message)

    options = LiveOptions(
        model="nova-2",
        language="en",
        smart_format=True
    )

    await connection.start(options)

    # Send audio bytes to connection.send(audio_bytes)
    # ...

    await connection.finish()

asyncio.run(main())
```

---

## Learning Path

### L0: Basic Usage (1 hour)
- [ ] Get API key
- [ ] Transcribe audio file
- [ ] Explore response format
- [ ] Try different models

### L1: Real-time (2-3 hours)
- [ ] Set up live streaming
- [ ] Handle interim results
- [ ] Microphone integration
- [ ] Error handling

### L2: Advanced (4-6 hours)
- [ ] Speaker diarization
- [ ] Custom vocabulary
- [ ] Sentiment analysis
- [ ] Webhook callbacks

---

## Code Examples

### With Speaker Diarization

```python
options = PrerecordedOptions(
    model="nova-2",
    diarize=True,
    punctuate=True
)

response = client.listen.prerecorded.v("1").transcribe_file(source, options)

for word in response.results.channels[0].alternatives[0].words:
    print(f"Speaker {word.speaker}: {word.word}")
```

### Custom Vocabulary

```python
options = PrerecordedOptions(
    model="nova-2",
    keywords=["Kubernetes:2", "PostgreSQL:2"],  # Boost keywords
    punctuate=True
)
```

### From URL

```python
source = {"url": "https://example.com/audio.mp3"}
response = client.listen.prerecorded.v("1").transcribe_url(source, options)
```

### Microphone Streaming

```python
import pyaudio
import asyncio
from deepgram import DeepgramClient, LiveOptions

async def main():
    client = DeepgramClient("YOUR_API_KEY")
    connection = client.listen.live.v("1")

    async def on_transcript(self, result, **kwargs):
        transcript = result.channel.alternatives[0].transcript
        if transcript:
            print(transcript, end=" ", flush=True)

    connection.on("transcript", on_transcript)

    options = LiveOptions(model="nova-2", language="en")
    await connection.start(options)

    # Set up microphone
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=8000
    )

    try:
        while True:
            data = stream.read(8000)
            await connection.send(data)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        await connection.finish()

asyncio.run(main())
```

---

## Models

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| nova-2 | Fast | ⭐⭐⭐⭐⭐ | General |
| nova | Fast | ⭐⭐⭐⭐ | Legacy |
| enhanced | Medium | ⭐⭐⭐⭐ | Noisy audio |
| base | Fastest | ⭐⭐⭐ | Speed priority |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection drops | Implement reconnection logic |
| High latency | Use closer region, check network |
| Poor accuracy | Try enhanced model, add keywords |
| Rate limited | Implement backoff |

---

## Resources

- [Deepgram Docs](https://developers.deepgram.com/)
- [API Reference](https://developers.deepgram.com/reference/)
- [Python SDK](https://github.com/deepgram/deepgram-python-sdk)
- [Pricing](https://deepgram.com/pricing)

---

*Part of [Luno-AI](../../README.md) | Audio AI Track*
