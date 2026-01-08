# Imitation Learning Integration

> **Train robots by watching demonstrations**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Learn policies from expert demonstrations |
| **Why** | No reward engineering, natural teaching |
| **Methods** | BC, GAIL, DAGGER, ACT, Diffusion Policy |
| **Best For** | Manipulation, complex tasks, human-like behavior |

### Methods Comparison

| Method | Data Needed | Online | Quality |
|--------|-------------|--------|---------|
| **Behavior Cloning (BC)** | Demos only | No | ⭐⭐⭐ |
| **DAgger** | Demos + Online | Yes | ⭐⭐⭐⭐ |
| **GAIL** | Demos + RL | Yes | ⭐⭐⭐⭐ |
| **ACT** | Demos only | No | ⭐⭐⭐⭐⭐ |
| **Diffusion Policy** | Demos only | No | ⭐⭐⭐⭐⭐ |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Hardware** | Robot or simulator |
| **Data** | 50-500 demonstrations |
| **GPU** | Recommended for training |

---

## Quick Start (1-2 hours)

### Behavior Cloning with LeRobot

```bash
pip install lerobot torch
```

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig
import torch

# Load demonstration dataset
dataset = LeRobotDataset("lerobot/pusht")

# Configure policy
config = ACTConfig(
    input_shapes={
        "observation.state": dataset.features["observation.state"].shape,
    },
    output_shapes={
        "action": dataset.features["action"].shape,
    }
)

# Create policy
policy = ACTPolicy(config)

# Training loop
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataset:
        loss = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Save
policy.save_pretrained("./trained_policy")
```

---

## Learning Path

### L0: Behavior Cloning (2-3 hours)
- [ ] Collect demonstrations
- [ ] Train BC policy
- [ ] Evaluate on robot
- [ ] Understand failure modes

### L1: Advanced Methods (4-6 hours)
- [ ] ACT (Action Chunking Transformer)
- [ ] Diffusion Policy
- [ ] Multi-modal inputs
- [ ] Data augmentation

### L2: Production (1-2 days)
- [ ] DAgger for improvement
- [ ] Sim-to-real transfer
- [ ] Multi-task learning
- [ ] Deployment

---

## Code Examples

### Data Collection

```python
import numpy as np
from dataclasses import dataclass
from typing import List
import json

@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    timestamp: float

class DemoCollector:
    def __init__(self):
        self.episodes = []
        self.current_episode = []

    def add_transition(self, obs: np.ndarray, action: np.ndarray, timestamp: float):
        self.current_episode.append(Transition(obs, action, timestamp))

    def end_episode(self, success: bool = True):
        if success and len(self.current_episode) > 10:
            self.episodes.append(self.current_episode)
        self.current_episode = []

    def save(self, path: str):
        data = []
        for episode in self.episodes:
            ep_data = {
                "observations": [t.observation.tolist() for t in episode],
                "actions": [t.action.tolist() for t in episode],
                "timestamps": [t.timestamp for t in episode]
            }
            data.append(ep_data)

        with open(path, "w") as f:
            json.dump(data, f)

    @property
    def num_episodes(self):
        return len(self.episodes)

    @property
    def num_transitions(self):
        return sum(len(ep) for ep in self.episodes)

# Usage
collector = DemoCollector()

# During teleoperation
for obs, action in teleop_stream:
    collector.add_transition(obs, action, time.time())

# Mark end of episode
collector.end_episode(success=True)

# Save
collector.save("demonstrations.json")
```

### Simple Behavior Cloning

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DemoDataset(Dataset):
    def __init__(self, demos_path: str):
        with open(demos_path) as f:
            self.data = json.load(f)

        self.observations = []
        self.actions = []

        for episode in self.data:
            self.observations.extend(episode["observations"])
            self.actions.extend(episode["actions"])

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return {
            "observation": torch.tensor(self.observations[idx], dtype=torch.float32),
            "action": torch.tensor(self.actions[idx], dtype=torch.float32)
        }

class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs):
        with torch.no_grad():
            return self.forward(obs)

# Train
dataset = DemoDataset("demonstrations.json")
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

policy = BCPolicy(obs_dim=10, action_dim=6)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        pred = policy(batch["observation"])
        loss = criterion(pred, batch["action"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

### ACT Policy

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig
import torch

# ACT configuration for manipulation
config = ACTConfig(
    input_shapes={
        "observation.state": [14],  # Joint positions + gripper
        "observation.image": [3, 480, 640]  # Camera input
    },
    output_shapes={
        "action": [14]
    },
    # Transformer settings
    hidden_dim=512,
    dim_feedforward=3200,
    n_heads=8,
    n_encoder_layers=4,
    n_decoder_layers=7,
    # Action chunking
    chunk_size=100,
    n_action_steps=100,
    # Training
    dropout=0.1
)

policy = ACTPolicy(config)

# ACT uses action chunking - predicts sequence of future actions
def train_step(batch):
    # batch contains sequences of (obs, action) pairs
    loss_dict = policy.forward(batch)
    return loss_dict["loss"]

# During inference - execute action chunks
action_queue = []

def get_action(observation):
    global action_queue

    if len(action_queue) == 0:
        # Predict new action chunk
        with torch.no_grad():
            action_chunk = policy.select_action(observation)
            action_queue = list(action_chunk)

    return action_queue.pop(0)
```

### Diffusion Policy

```python
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig

# Diffusion Policy configuration
config = DiffusionConfig(
    input_shapes={
        "observation.state": [14],
        "observation.image": [3, 480, 640]
    },
    output_shapes={
        "action": [14]
    },
    # Diffusion settings
    n_action_steps=8,
    horizon=16,
    n_obs_steps=2,
    # UNet architecture
    down_dims=[256, 512, 1024],
    # Diffusion process
    num_inference_steps=100,
    noise_scheduler_type="DDPM"
)

policy = DiffusionPolicy(config)

# Training
def train_diffusion(policy, dataset, epochs=100):
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataset:
            loss_dict = policy.forward(batch)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### DAgger (Dataset Aggregation)

```python
class DAgger:
    def __init__(self, policy, expert, env):
        self.policy = policy
        self.expert = expert
        self.env = env
        self.dataset = []

    def collect_with_policy(self, n_episodes=10):
        """Collect trajectories using learned policy"""
        for _ in range(n_episodes):
            obs = self.env.reset()
            episode = []

            while True:
                # Get policy action (for execution)
                policy_action = self.policy.get_action(obs)

                # Get expert action (for labeling)
                expert_action = self.expert.get_action(obs)

                episode.append((obs, expert_action))

                # Execute policy action
                obs, _, done, _ = self.env.step(policy_action)

                if done:
                    break

            self.dataset.extend(episode)

    def aggregate_and_train(self, iterations=10):
        for i in range(iterations):
            # Collect data with current policy
            self.collect_with_policy()

            # Train on aggregated dataset
            self.train_policy()

            print(f"Iteration {i}, Dataset size: {len(self.dataset)}")

    def train_policy(self):
        # Standard BC training on self.dataset
        pass
```

---

## Data Augmentation

```python
import torch
import torchvision.transforms as T

class DemoAugmentation:
    def __init__(self):
        self.image_aug = T.Compose([
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        ])

    def augment_batch(self, batch):
        # Image augmentation
        if "observation.image" in batch:
            batch["observation.image"] = self.image_aug(batch["observation.image"])

        # State noise
        if "observation.state" in batch:
            noise = torch.randn_like(batch["observation.state"]) * 0.01
            batch["observation.state"] += noise

        return batch
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Drifting behavior | Use DAgger, more diverse demos |
| Not generalizing | Add augmentation, more demos |
| Jerky motion | Use action chunking, smoothing |
| Poor accuracy | Increase model capacity, more data |

---

## Resources

- [LeRobot](https://github.com/huggingface/lerobot)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)
- [Imitation Learning Tutorial](https://imitation.readthedocs.io/)

---

*Part of [Luno-AI](../../README.md) | Robotics Track*
