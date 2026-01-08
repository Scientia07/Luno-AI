# Robotics: AI Meets the Physical World

> **Embodied intelligence** - robots that learn, perceive, and act.

---

## Layer Navigation

| Layer | Content | Status |
|-------|---------|--------|
| L0 | [Overview](#overview) | This file |
| L1 | [Concepts](./concepts.md) | Pending |
| L2 | [Deep Dive](./deep-dive.md) | Pending |
| L3 | [Labs](../../labs/robotics/) | Pending |
| L4 | [Advanced](./advanced.md) | Pending |

---

## Overview

The frontier of AI: building machines that interact with the physical world. Modern robotics combines vision, language, and learning to create systems that can manipulate objects, navigate spaces, and learn from demonstrations.

```
                    ROBOTICS STACK

    ┌──────────────────────────────────────────────┐
    │              FOUNDATION MODELS                │
    │   Vision │ Language │ Action (VLA models)    │
    └──────────────────────────────────────────────┘
                          │
    ┌──────────────────────────────────────────────┐
    │              LEARNING METHODS                 │
    │  Imitation │ Reinforcement │ Sim-to-Real     │
    └──────────────────────────────────────────────┘
                          │
    ┌──────────────────────────────────────────────┐
    │              PERCEPTION                       │
    │  Vision │ Depth │ Tactile │ Proprioception   │
    └──────────────────────────────────────────────┘
                          │
    ┌──────────────────────────────────────────────┐
    │              HARDWARE                         │
    │  Arms │ Hands │ Mobile │ Humanoid            │
    └──────────────────────────────────────────────┘
```

---

## Open-Source Robot Platforms

### SO-100 / SO-101

Low-cost robotic arms for learning and research.

| Spec | SO-100 | SO-101 |
|------|--------|--------|
| DOF | 5-6 | 6 |
| Payload | ~500g | ~500g |
| Cost | ~$200-300 | ~$300-400 |
| Control | Servo | Servo |
| Interface | Python | Python |

```python
# Basic SO-101 control
from so101 import Robot

robot = Robot()
robot.move_to(x=0.3, y=0.1, z=0.2)
robot.gripper.close()
```

### ALOHA / Mobile ALOHA

Bimanual manipulation robots from Stanford/Google.

```
         ALOHA Setup
    ┌─────────────────────┐
    │   Leader Robot      │  ← Human controls this
    │   (Teleoperation)   │
    └─────────┬───────────┘
              │ Demonstration
              ▼
    ┌─────────────────────┐
    │   Follower Robot    │  ← Learns to imitate
    │   (Learning)        │
    └─────────────────────┘
```

**Mobile ALOHA** adds mobility - a rolling base for the bimanual system.

### Other Platforms

| Platform | Type | Cost | Open |
|----------|------|------|------|
| **Stretch (Hello Robot)** | Mobile manipulator | $25K | Partial |
| **Unitree G1** | Humanoid | $16K | No |
| **LoCoBot** | Mobile base | ~$3K | Yes |
| **xArm** | Industrial arm | $5-10K | SDK |

---

## Robot Learning

### Imitation Learning

**Learn from human demonstrations**

```
1. Human demonstrates task (teleoperation)
              ↓
2. Record observations + actions
              ↓
3. Train policy: observation → action
              ↓
4. Robot executes learned policy
```

Key approaches:
- **Behavior Cloning** - Direct supervised learning
- **DAgger** - Iterative with corrections
- **Diffusion Policy** - Diffusion models for actions

### Reinforcement Learning

**Learn from trial and error**

```
┌─────────┐    action    ┌─────────────┐
│  Robot  │─────────────▶│ Environment │
│  Agent  │              │             │
│         │◀─────────────│             │
└─────────┘   state,     └─────────────┘
              reward
```

Challenges:
- Sample efficiency (millions of trials)
- Sim-to-real gap
- Reward engineering

### Sim-to-Real Transfer

**Train in simulation, deploy on real robot**

```
┌───────────────────┐          ┌───────────────────┐
│    SIMULATION     │          │   REAL WORLD      │
│                   │          │                   │
│  - Fast           │  ──────▶ │  - Slow           │
│  - Safe           │          │  - Risky          │
│  - Cheap          │          │  - Expensive      │
│  - Parallelizable │  Gap!    │  - Single         │
└───────────────────┘          └───────────────────┘
```

Techniques to bridge the gap:
- **Domain Randomization** - Vary simulation parameters
- **System Identification** - Match sim to real
- **Fine-tuning** - Adapt with real data

---

## Foundation Models for Robotics

### Vision-Language-Action (VLA) Models

```
Image + Language ──▶ VLA Model ──▶ Robot Actions
    ↓                     ↓
"Pick up the        [joint angles,
 red cup"            gripper cmd]
```

| Model | Developer | Key Feature |
|-------|-----------|-------------|
| **RT-1** | Google | Real robot training |
| **RT-2** | Google | VLM for robotics |
| **OpenVLA** | Berkeley | Open source VLA |
| **Octo** | Berkeley | Generalist robot |
| **RoboAgent** | CMU | Multi-task |

### LeRobot (HuggingFace)

Open-source robot learning library.

```python
from lerobot import Robot, Policy

# Load pretrained policy
policy = Policy.from_pretrained("lerobot/aloha-act")

# Run on robot
robot = Robot("so101")
while True:
    obs = robot.get_observation()
    action = policy.predict(obs)
    robot.step(action)
```

Features:
- Pretrained policies for various robots
- Training scripts for imitation learning
- Support for common robot platforms
- Simulation environments

---

## Perception for Robotics

### Vision Pipeline

```
Camera ──▶ Detection ──▶ Segmentation ──▶ Pose Estimation
              │              │                  │
              ▼              ▼                  ▼
         "Objects"     "Object masks"    "6D object pose"
```

### Key Technologies

| Task | Model | Use |
|------|-------|-----|
| Object Detection | YOLO, DETIC | Find graspable objects |
| Segmentation | SAM | Precise object boundaries |
| Depth | Depth Anything | 3D understanding |
| Pose Estimation | FoundationPose | Object manipulation |
| Hand/Body | MediaPipe | Human demonstration |

### Tactile Sensing

```
Touch Sensor ──▶ Contact Detection ──▶ Grasp Adjustment
                       │
                       ▼
              "Is object slipping?"
```

---

## Control & Motion Planning

### Motion Planning

```
Start Pose ──▶ Planner ──▶ Waypoints ──▶ Controller ──▶ Motor Cmds
                  │
                  ▼
         Collision-free path
```

Tools:
- **MoveIt** - ROS motion planning
- **OMPL** - Open motion planning library
- **cuRobo** - GPU-accelerated planning

### Control Methods

| Method | Use Case |
|--------|----------|
| **Position Control** | Simple, repeatable tasks |
| **Velocity Control** | Smooth motions |
| **Force/Torque Control** | Contact tasks |
| **Impedance Control** | Compliant interaction |

---

## ROS 2 (Robot Operating System)

The standard middleware for robotics.

```
┌─────────────────────────────────────────────────────────┐
│                       ROS 2                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Sensor  │  │ Planner │  │ Control │  │   UI    │    │
│  │  Node   │  │  Node   │  │  Node   │  │  Node   │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│       └────────────┴────────────┴────────────┘          │
│                    Topics / Services                     │
└─────────────────────────────────────────────────────────┘
```

Key concepts:
- **Nodes** - Independent processes
- **Topics** - Pub/sub message passing
- **Services** - Request/response
- **Actions** - Long-running tasks with feedback

---

## Simulation Environments

| Simulator | Strength | Physics |
|-----------|----------|---------|
| **Isaac Sim** | GPU, realistic | PhysX |
| **MuJoCo** | Fast, accurate | Custom |
| **PyBullet** | Easy, free | Bullet |
| **Gazebo** | ROS integration | ODE/Bullet |

```python
# MuJoCo example
import mujoco

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

for _ in range(1000):
    data.ctrl[:] = policy(data.qpos, data.qvel)
    mujoco.mj_step(model, data)
```

---

## Labs

| Notebook | Focus |
|----------|-------|
| `01-so101-basics.ipynb` | Control SO-101 arm |
| `02-lerobot-intro.ipynb` | HuggingFace LeRobot |
| `03-imitation-learning.ipynb` | Learn from demos |
| `04-sim-to-real.ipynb` | Simulation transfer |
| `05-vision-grasping.ipynb` | Object manipulation |
| `06-ros2-basics.ipynb` | ROS 2 fundamentals |

---

## Resources

### Hardware
- [SO-101 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [ALOHA Project](https://tonyzhaozh.github.io/aloha/)
- [Hello Robot Stretch](https://hello-robot.com/)

### Software
- [LeRobot](https://github.com/huggingface/lerobot)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [MuJoCo](https://mujoco.org/)

---

*"The future of AI is embodied - thinking machines that act in the world."*
