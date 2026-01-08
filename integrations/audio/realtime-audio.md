# Real-time Audio Pipeline Integration

> **Build streaming audio AI applications**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Low-latency audio processing pipelines |
| **Components** | STT → LLM → TTS in real-time |
| **Latency** | <500ms achievable |
| **Use Cases** | Voice assistants, live translation, call centers |

### Pipeline Architecture

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│   Mic   │────▶│   STT   │────▶│   LLM   │────▶│   TTS   │────▶ Speaker
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     │               │               │               │
     │          Streaming       Streaming       Streaming
     │          <100ms          <200ms          <200ms
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **CPU/GPU** | GPU recommended for low latency |
| **Python** | 3.10+ |
| **Audio** | Working microphone and speakers |

---

## Quick Start (30 min)

### Basic Pipeline

```bash
pip install faster-whisper pyaudio numpy openai pyttsx3
```

```python
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue

class RealtimeSTT:
    def __init__(self):
        self.model = WhisperModel("tiny", device="cuda", compute_type="float16")
        self.audio_queue = queue.Queue()
        self.running = False

        # Audio settings
        self.RATE = 16000
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def start(self):
        self.running = True
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=1,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

    def transcribe_chunk(self, audio_buffer):
        segments, _ = self.model.transcribe(
            audio_buffer,
            beam_size=1,
            vad_filter=True
        )
        return " ".join([s.text for s in segments])

    def run(self):
        self.start()
        buffer = []

        try:
            while self.running:
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    buffer.append(chunk)

                    # Process every 2 seconds
                    if len(buffer) * self.CHUNK / self.RATE >= 2:
                        audio = np.concatenate(buffer)
                        text = self.transcribe_chunk(audio)
                        if text.strip():
                            print(f"Transcribed: {text}")
                        buffer = []
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

# Run
stt = RealtimeSTT()
stt.run()
```

---

## Learning Path

### L0: Basic STT Streaming (1-2 hours)
- [ ] Set up audio capture
- [ ] Stream to Whisper
- [ ] Display transcriptions
- [ ] Handle silence detection

### L1: Full Pipeline (3-4 hours)
- [ ] Add LLM processing
- [ ] Integrate TTS output
- [ ] Optimize latency
- [ ] Handle interruptions

### L2: Production Pipeline (1-2 days)
- [ ] WebSocket streaming
- [ ] Multiple speakers
- [ ] Error recovery
- [ ] Load balancing

---

## Code Examples

### Complete Voice Assistant

```python
import asyncio
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
import edge_tts
import io
from pydub import AudioSegment
from pydub.playback import play

class VoiceAssistant:
    def __init__(self):
        self.whisper = WhisperModel("small", device="cuda")
        self.openai = AsyncOpenAI()
        self.conversation = []

    async def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.whisper.transcribe(audio, beam_size=1)
        return " ".join([s.text for s in segments])

    async def get_response(self, text: str) -> str:
        self.conversation.append({"role": "user", "content": text})

        response = await self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.conversation,
            stream=True
        )

        full_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content

        self.conversation.append({"role": "assistant", "content": full_response})
        return full_response

    async def speak(self, text: str):
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        audio_data = b""

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]

        # Play audio
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        play(audio)

    async def process_turn(self, audio: np.ndarray):
        # STT
        text = await self.transcribe(audio)
        print(f"You: {text}")

        # LLM
        response = await self.get_response(text)
        print(f"Assistant: {response}")

        # TTS
        await self.speak(response)

# Usage
assistant = VoiceAssistant()
# Record audio and call: await assistant.process_turn(audio_array)
```

### WebSocket Streaming Server

```python
import asyncio
import websockets
import json
from faster_whisper import WhisperModel
import numpy as np

class StreamingServer:
    def __init__(self):
        self.model = WhisperModel("tiny", device="cuda")

    async def handle_audio(self, websocket):
        buffer = []

        async for message in websocket:
            # Receive audio chunks
            audio_chunk = np.frombuffer(message, dtype=np.float32)
            buffer.append(audio_chunk)

            # Process every 1 second
            if len(buffer) * 1024 / 16000 >= 1:
                audio = np.concatenate(buffer)

                segments, _ = self.model.transcribe(
                    audio,
                    beam_size=1,
                    vad_filter=True
                )

                for segment in segments:
                    await websocket.send(json.dumps({
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end
                    }))

                buffer = []

    async def run(self, host="localhost", port=8765):
        async with websockets.serve(self.handle_audio, host, port):
            print(f"Server running on ws://{host}:{port}")
            await asyncio.Future()

# Run server
server = StreamingServer()
asyncio.run(server.run())
```

### VAD-based Segmentation

```python
import webrtcvad
import numpy as np

class VADProcessor:
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.vad = webrtcvad.Vad(3)  # Aggressiveness 0-3
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)

    def is_speech(self, audio_frame: bytes) -> bool:
        return self.vad.is_speech(audio_frame, self.sample_rate)

    def segment_audio(self, audio: np.ndarray) -> list:
        """Split audio into speech segments"""
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()

        segments = []
        current_segment = []
        in_speech = False
        silence_frames = 0

        for i in range(0, len(audio_bytes), self.frame_size * 2):
            frame = audio_bytes[i:i + self.frame_size * 2]
            if len(frame) < self.frame_size * 2:
                break

            speech = self.is_speech(frame)

            if speech:
                current_segment.append(frame)
                in_speech = True
                silence_frames = 0
            elif in_speech:
                silence_frames += 1
                current_segment.append(frame)

                # End segment after 300ms silence
                if silence_frames > 10:
                    segments.append(b"".join(current_segment))
                    current_segment = []
                    in_speech = False
                    silence_frames = 0

        return segments
```

### Streaming TTS with Chunked Output

```python
import asyncio
import edge_tts
import pyaudio

async def streaming_tts(text: str):
    """Stream TTS output as it's generated"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=24000,
        output=True
    )

    communicate = edge_tts.Communicate(text, "en-US-GuyNeural")

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            stream.write(chunk["data"])

    stream.stop_stream()
    stream.close()
    p.terminate()

# Usage
asyncio.run(streaming_tts("Hello, this is streaming text to speech!"))
```

---

## Latency Optimization

| Component | Technique | Improvement |
|-----------|-----------|-------------|
| STT | Use tiny/base model | 2-5x faster |
| STT | GPU inference | 3-10x faster |
| LLM | Streaming responses | First token <100ms |
| TTS | Edge TTS / local | <100ms latency |
| Audio | Small chunk sizes | Lower latency |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| High latency | Use smaller models, GPU |
| Audio crackling | Increase buffer size |
| Missed speech | Tune VAD sensitivity |
| Echo | Add echo cancellation |

---

## Resources

- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [edge-tts](https://github.com/rany2/edge-tts)
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)

---

*Part of [Luno-AI](../../README.md) | Audio AI Track*
