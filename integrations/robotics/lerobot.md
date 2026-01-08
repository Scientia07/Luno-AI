# LeRobot Integration

> **Hugging Face's open robot learning platform**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Framework for robot learning with imitation/RL |
| **Why** | State-of-the-art policies, easy data collection |
| **Creator** | Hugging Face |
| **Best For** | Manipulation, imitation learning |

### Key Features
- Pre-trained policies (ACT, Diffusion)
- Dataset hub for robot data
- Simulation environments
- Real robot support (ALOHA, SO-101)

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ |
| **GPU** | NVIDIA 8GB+ for training |
| **Hardware** | Robot arm (optional for sim) |

---

## Quick Start (30 min)

```bash
pip install lerobot
```

### Run Pre-trained Policy

```python
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load policy
policy = ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")

# Load dataset for environment info
dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")

# Get action from observation
observation = {
    "observation.images.top": image_top,
    "observation.state": robot_state
}
action = policy.select_action(observation)
```

### Visualize Dataset

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames_torchvision

dataset = LeRobotDataset("lerobot/pusht")

# View episode
episode = dataset[0]
print(f"Keys: {episode.keys()}")
print(f"State shape: {episode['observation.state'].shape}")
```

---

## Learning Path

### L0: Explore (2-3 hours)
- [ ] Install LeRobot
- [ ] Browse dataset hub
- [ ] Visualize trajectories
- [ ] Run pre-trained policy in sim

### L1: Train Policies (4-6 hours)
- [ ] Collect your own data
- [ ] Train ACT policy
- [ ] Evaluate in simulation
- [ ] Iterate on dataset

### L2: Real Robot (1-2 weeks)
- [ ] Set up hardware (SO-101/ALOHA)
- [ ] Teleoperate and record
- [ ] Train on real data
- [ ] Deploy policy

---

## Code Examples

### Train ACT Policy

```bash
python lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    training.num_epochs=100 \
    training.batch_size=8
```

### Evaluate Policy

```bash
python lerobot/scripts/eval.py \
    policy=act \
    env=aloha \
    policy.pretrained_model_path=outputs/train/act_aloha/checkpoints/last.ckpt
```

### Collect Data with Teleoperation

```python
from lerobot.common.robot_devices.robots.factory import make_robot

# Configure robot
robot = make_robot("so100")
robot.connect()

# Record episode
robot.start_recording()
# ... teleoperate ...
robot.stop_recording()

# Save dataset
robot.save_dataset("my_dataset")
```

### Custom Environment

```python
import gymnasium as gym
from lerobot.common.envs.factory import make_env

env = make_env(
    cfg={
        "name": "pusht",
        "task": "PushT-v0"
    }
)

obs, info = env.reset()
for _ in range(100):
    action = policy.select_action(obs)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

---

## Available Policies

| Policy | Type | Best For |
|--------|------|----------|
| **ACT** | Imitation | Short horizon tasks |
| **Diffusion** | Imitation | Complex manipulation |
| **VQ-BeT** | Imitation | Multi-modal behaviors |
| **TDMPC** | RL | Continuous control |

---

## Datasets on Hub

| Dataset | Task | Episodes |
|---------|------|----------|
| `lerobot/pusht` | Push T-shape | 206 |
| `lerobot/aloha_sim_transfer_cube` | Cube transfer | 50 |
| `lerobot/xarm_lift_medium` | Object lifting | 1000 |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Policy diverges | Check data quality, reduce LR |
| Slow training | Use smaller batch, mixed precision |
| Robot not found | Check USB connection, permissions |
| Simulation crash | Check MuJoCo installation |

---

## Resources

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [LeRobot Docs](https://huggingface.co/docs/lerobot)
- [Dataset Hub](https://huggingface.co/datasets?search=lerobot)
- [ACT Paper](https://arxiv.org/abs/2304.13705)

---

*Part of [Luno-AI](../../README.md) | Robotics Track*
