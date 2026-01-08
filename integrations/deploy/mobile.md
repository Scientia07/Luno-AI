# Mobile AI Deployment Integration

> **Run AI models on iOS and Android devices**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Deploy ML models to mobile devices |
| **Why** | Offline inference, privacy, low latency |
| **Frameworks** | TensorFlow Lite, Core ML, ONNX Mobile |
| **Best For** | On-device AI, edge applications |

### Framework Comparison

| Framework | iOS | Android | Performance |
|-----------|-----|---------|-------------|
| **Core ML** | ✓ | ✗ | Excellent |
| **TF Lite** | ✓ | ✓ | Good |
| **ONNX Runtime** | ✓ | ✓ | Good |
| **PyTorch Mobile** | ✓ | ✓ | Good |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Development** | Xcode (iOS) / Android Studio |
| **Model** | Trained model to convert |
| **Python** | For model conversion |

---

## Quick Start (1 hour)

### TensorFlow Lite Conversion

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("my_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

### Core ML Conversion

```python
import coremltools as ct

# From PyTorch
import torch

model = torch.load("model.pt")
model.eval()

traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, 3, 224, 224), name="input")],
    convert_to="mlprogram"
)

mlmodel.save("Model.mlpackage")
```

---

## Learning Path

### L0: Model Conversion (2-3 hours)
- [ ] Convert to TFLite
- [ ] Convert to Core ML
- [ ] Test on simulator
- [ ] Basic app integration

### L1: Optimization (4-6 hours)
- [ ] Quantization
- [ ] Model pruning
- [ ] Benchmark performance
- [ ] Battery optimization

### L2: Production (1-2 days)
- [ ] Custom operators
- [ ] Model updates
- [ ] A/B testing
- [ ] Analytics

---

## Code Examples

### TensorFlow Lite - Android

```kotlin
// build.gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
}
```

```kotlin
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer

class ImageClassifier(context: Context) {
    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context, "model.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            // Enable GPU
            addDelegate(GpuDelegate())
        }
        interpreter = Interpreter(model, options)
    }

    fun classify(bitmap: Bitmap): FloatArray {
        val inputBuffer = preprocessImage(bitmap)
        val outputBuffer = Array(1) { FloatArray(1000) }

        interpreter.run(inputBuffer, outputBuffer)

        return outputBuffer[0]
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(224 * 224)
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16 and 0xFF) - 127.5f) / 127.5f)
            buffer.putFloat(((pixel shr 8 and 0xFF) - 127.5f) / 127.5f)
            buffer.putFloat(((pixel and 0xFF) - 127.5f) / 127.5f)
        }

        return buffer
    }

    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val assetManager = context.assets
        val inputStream = assetManager.open(filename)
        val bytes = inputStream.readBytes()
        val buffer = ByteBuffer.allocateDirect(bytes.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        return buffer
    }
}
```

### Core ML - iOS (Swift)

```swift
import CoreML
import Vision

class ImageClassifier {
    private let model: VNCoreMLModel

    init() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // CPU + GPU + Neural Engine

        let mlModel = try MyModel(configuration: config)
        model = try VNCoreMLModel(for: mlModel.model)
    }

    func classify(image: CGImage) async throws -> [(String, Float)] {
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) { request, error in
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let results = request.results as? [VNClassificationObservation] else {
                    continuation.resume(returning: [])
                    return
                }

                let classifications = results.map { ($0.identifier, $0.confidence) }
                continuation.resume(returning: classifications)
            }

            request.imageCropAndScaleOption = .centerCrop

            let handler = VNImageRequestHandler(cgImage: image)
            try? handler.perform([request])
        }
    }
}

// Usage
let classifier = try ImageClassifier()
let results = try await classifier.classify(image: cgImage)
print(results.prefix(5))
```

### Quantization for Mobile

```python
import tensorflow as tf

# Full integer quantization (smallest, fastest)
def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

quantized_model = converter.convert()

# Save
with open("model_int8.tflite", "wb") as f:
    f.write(quantized_model)

print(f"Original: {len(original_model)} bytes")
print(f"Quantized: {len(quantized_model)} bytes")
```

### ONNX Runtime Mobile

```python
# Convert to ONNX
import torch
import torch.onnx

model = torch.load("model.pt")
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
)

# Optimize for mobile
from onnxruntime.transformers import optimizer
from onnxruntime.quantization import quantize_dynamic

optimized = optimizer.optimize_model("model.onnx")
optimized.save_model_to_file("model_optimized.onnx")

quantize_dynamic(
    "model_optimized.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
```

### React Native Integration

```javascript
// Using TensorFlow.js
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';

async function loadModel() {
  await tf.ready();

  const modelJSON = require('./model/model.json');
  const modelWeights = require('./model/weights.bin');

  const model = await tf.loadLayersModel(
    bundleResourceIO(modelJSON, modelWeights)
  );

  return model;
}

async function classify(imageTensor) {
  const model = await loadModel();

  // Preprocess
  const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
  const normalized = resized.div(255).expandDims(0);

  // Inference
  const predictions = model.predict(normalized);
  const results = await predictions.data();

  return results;
}
```

### Model Benchmarking

```python
import time
import numpy as np

def benchmark_tflite(model_path: str, input_shape: tuple, num_runs: int = 100):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        times.append(time.perf_counter() - start)

    print(f"Average: {np.mean(times)*1000:.2f}ms")
    print(f"Min: {np.min(times)*1000:.2f}ms")
    print(f"Max: {np.max(times)*1000:.2f}ms")

benchmark_tflite("model.tflite", (1, 224, 224, 3))
```

---

## Size Optimization

| Technique | Size Reduction | Quality Impact |
|-----------|----------------|----------------|
| Float16 | 50% | Minimal |
| INT8 Quantization | 75% | Low |
| Pruning | 50-90% | Medium |
| Knowledge Distillation | 80%+ | Medium |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model too large | Quantize, prune, or distill |
| Slow inference | Enable GPU/NPU, quantize |
| Unsupported ops | Use custom operators or different model |
| High battery usage | Reduce inference frequency |

---

## Resources

- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [Core ML](https://developer.apple.com/documentation/coreml)
- [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
- [PyTorch Mobile](https://pytorch.org/mobile/)

---

*Part of [Luno-AI](../../README.md) | Deployment Track*
