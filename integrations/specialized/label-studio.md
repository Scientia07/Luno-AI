# Label Studio Integration

> **Open-source data labeling platform**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Multi-type data annotation tool |
| **Why** | Flexible, self-hosted, ML integration |
| **Data Types** | Images, text, audio, video, time series |
| **Best For** | Creating training datasets |

### Label Studio vs Alternatives

| Feature | Label Studio | Labelbox | Scale AI |
|---------|--------------|----------|----------|
| Self-hosted | ✓ | Limited | ✗ |
| Free tier | ✓ | Limited | ✗ |
| ML backend | ✓ | ✓ | ✓ |
| Customization | High | Medium | Low |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **Docker** | Optional |
| **Storage** | For data files |

---

## Quick Start (15 min)

### Installation

```bash
pip install label-studio
```

### Start Server

```bash
label-studio start
# Opens at http://localhost:8080
```

### Python SDK

```python
from label_studio_sdk import Client

# Connect to Label Studio
ls = Client(url='http://localhost:8080', api_key='YOUR_API_KEY')

# Create project
project = ls.start_project(
    title='Image Classification',
    label_config='''
    <View>
      <Image name="image" value="$image"/>
      <Choices name="choice" toName="image">
        <Choice value="Cat"/>
        <Choice value="Dog"/>
        <Choice value="Other"/>
      </Choices>
    </View>
    '''
)

# Import data
project.import_tasks([
    {'image': 'https://example.com/image1.jpg'},
    {'image': 'https://example.com/image2.jpg'}
])
```

---

## Learning Path

### L0: Basic Labeling (1-2 hours)
- [ ] Install Label Studio
- [ ] Create first project
- [ ] Label data
- [ ] Export annotations

### L1: Custom Templates (2-3 hours)
- [ ] Custom label configs
- [ ] Multiple annotation types
- [ ] Validation rules
- [ ] Keyboard shortcuts

### L2: ML Integration (4-6 hours)
- [ ] ML backend setup
- [ ] Pre-annotation
- [ ] Active learning
- [ ] Model training loop

---

## Code Examples

### Label Configurations

```xml
<!-- Image Classification -->
<View>
  <Image name="image" value="$image"/>
  <Choices name="label" toName="image">
    <Choice value="Positive"/>
    <Choice value="Negative"/>
    <Choice value="Neutral"/>
  </Choices>
</View>
```

```xml
<!-- Object Detection -->
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Car" background="red"/>
    <Label value="Person" background="blue"/>
    <Label value="Traffic Sign" background="green"/>
  </RectangleLabels>
</View>
```

```xml
<!-- Text Classification -->
<View>
  <Text name="text" value="$text"/>
  <Choices name="sentiment" toName="text">
    <Choice value="Positive"/>
    <Choice value="Negative"/>
    <Choice value="Neutral"/>
  </Choices>
</View>
```

```xml
<!-- Named Entity Recognition -->
<View>
  <Labels name="label" toName="text">
    <Label value="Person" background="red"/>
    <Label value="Organization" background="blue"/>
    <Label value="Location" background="green"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>
```

```xml
<!-- Audio Transcription -->
<View>
  <Audio name="audio" value="$audio"/>
  <TextArea name="transcription" toName="audio" rows="4"/>
</View>
```

### Import Data

```python
from label_studio_sdk import Client

ls = Client(url='http://localhost:8080', api_key='YOUR_KEY')
project = ls.get_project(1)

# Import from local files
tasks = []
for img_path in Path('images/').glob('*.jpg'):
    tasks.append({
        'image': f'/data/local-files/?d=images/{img_path.name}'
    })
project.import_tasks(tasks)

# Import with predictions (pre-labels)
tasks_with_predictions = [
    {
        'image': 'https://example.com/img.jpg',
        'predictions': [{
            'model_version': 'v1.0',
            'result': [{
                'from_name': 'label',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': ['Cat']}
            }],
            'score': 0.95
        }]
    }
]
project.import_tasks(tasks_with_predictions)
```

### Export Annotations

```python
from label_studio_sdk import Client
import json

ls = Client(url='http://localhost:8080', api_key='YOUR_KEY')
project = ls.get_project(1)

# Export all annotations
annotations = project.export_tasks()

# Save to file
with open('annotations.json', 'w') as f:
    json.dump(annotations, f, indent=2)

# Export specific format
yolo_export = project.export_tasks(export_type='YOLO')
coco_export = project.export_tasks(export_type='COCO')
```

### ML Backend

```python
from label_studio_ml import LabelStudioMLBase
import torch
from transformers import pipeline

class TextClassifierBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = pipeline("text-classification")

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            text = task['data']['text']
            result = self.classifier(text)[0]

            predictions.append({
                'result': [{
                    'from_name': 'sentiment',
                    'to_name': 'text',
                    'type': 'choices',
                    'value': {'choices': [result['label']]}
                }],
                'score': result['score']
            })

        return predictions

    def fit(self, annotations, **kwargs):
        # Train model on new annotations
        pass

# Run backend
if __name__ == '__main__':
    from label_studio_ml import serve
    serve(TextClassifierBackend, port=9090)
```

### Active Learning Loop

```python
from label_studio_sdk import Client
import numpy as np

def active_learning_loop(project_id: int, model, n_samples: int = 10):
    """Select most uncertain samples for labeling"""
    ls = Client(url='http://localhost:8080', api_key='YOUR_KEY')
    project = ls.get_project(project_id)

    # Get unlabeled tasks
    tasks = project.get_tasks(filters={'annotation': 'empty'})

    # Get model predictions
    uncertainties = []
    for task in tasks:
        pred = model.predict(task['data'])
        # Calculate entropy as uncertainty
        entropy = -np.sum(pred * np.log(pred + 1e-10))
        uncertainties.append((task['id'], entropy))

    # Sort by uncertainty
    uncertainties.sort(key=lambda x: x[1], reverse=True)

    # Return most uncertain task IDs
    return [task_id for task_id, _ in uncertainties[:n_samples]]

# Usage
uncertain_ids = active_learning_loop(project_id=1, model=my_model)
print(f"Label these tasks first: {uncertain_ids}")
```

### Convert to Training Format

```python
import json
from pathlib import Path

def convert_to_yolo(annotations_file: str, output_dir: str):
    """Convert Label Studio annotations to YOLO format"""
    with open(annotations_file) as f:
        data = json.load(f)

    output_path = Path(output_dir)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)

    for item in data:
        image_name = Path(item['data']['image']).stem
        labels = []

        for annotation in item.get('annotations', []):
            for result in annotation['result']:
                if result['type'] == 'rectanglelabels':
                    # Convert to YOLO format
                    x = result['value']['x'] / 100
                    y = result['value']['y'] / 100
                    w = result['value']['width'] / 100
                    h = result['value']['height'] / 100

                    # YOLO: center_x, center_y, width, height
                    cx = x + w/2
                    cy = y + h/2

                    label = result['value']['rectanglelabels'][0]
                    class_id = CLASS_MAP.get(label, 0)

                    labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save label file
        label_file = output_path / 'labels' / f"{image_name}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(labels))

CLASS_MAP = {'Car': 0, 'Person': 1, 'Traffic Sign': 2}
convert_to_yolo('annotations.json', './yolo_dataset')
```

---

## Docker Deployment

```bash
docker run -it -p 8080:8080 \
    -v $(pwd)/mydata:/label-studio/data \
    heartexlabs/label-studio:latest
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't import images | Check file permissions, use local-files |
| Slow with many tasks | Use pagination, optimize queries |
| ML backend not connecting | Check port, logs |
| Export missing data | Ensure annotations are completed |

---

## Resources

- [Label Studio Docs](https://labelstud.io/guide/)
- [Label Studio GitHub](https://github.com/HumanSignal/label-studio)
- [Templates](https://labelstud.io/templates/)
- [ML Backend](https://github.com/HumanSignal/label-studio-ml-backend)

---

*Part of [Luno-AI](../../README.md) | Specialized Track*
