# Robotics AI: Projects & Comparisons

> **Hands-on projects and framework comparisons for AI Robotics**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Simulated Robot Arm Control
**Goal**: Control a robot arm in simulation

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 3-4 hours |
| Technologies | PyBullet or MuJoCo |
| Skills | Simulation basics, joint control |

**Tasks**:
- [ ] Install PyBullet
- [ ] Load robot arm URDF
- [ ] Move joints to positions
- [ ] Implement inverse kinematics
- [ ] Record trajectories

**Starter Code**:
```python
import pybullet as p
import pybullet_data
import time

# Connect to physics simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load environment and robot
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

# Get joint info
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: {info[1].decode()}")

# Control loop
while True:
    p.stepSimulation()
    time.sleep(1./240.)
```

---

#### Project 2: Basic Pick and Place
**Goal**: Robot picks up object and moves it

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 4-5 hours |
| Technologies | PyBullet + gripper |
| Skills | Gripper control, motion planning |

**Tasks**:
- [ ] Load robot with gripper
- [ ] Add object to scene
- [ ] Move to object position
- [ ] Close gripper (grasp)
- [ ] Move to target location
- [ ] Open gripper (release)

---

### Intermediate Projects (L2)

#### Project 3: Vision-Guided Manipulation
**Goal**: Robot uses camera to find and grasp objects

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 8-12 hours |
| Technologies | PyBullet + OpenCV + YOLO |
| Skills | Visual servoing, detection |

**Tasks**:
- [ ] Add camera to simulation
- [ ] Detect objects with YOLO
- [ ] Transform pixel to world coordinates
- [ ] Plan grasp approach
- [ ] Execute pick and place
- [ ] Handle detection failures

---

#### Project 4: LeRobot Imitation Learning
**Goal**: Teach robot by demonstration

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | LeRobot (Hugging Face) |
| Skills | Imitation learning, trajectory collection |

**Tasks**:
- [ ] Install LeRobot
- [ ] Collect demonstration trajectories
- [ ] Train ACT/Diffusion policy
- [ ] Evaluate in simulation
- [ ] Iterate on dataset

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Load dataset
dataset = LeRobotDataset("lerobot/pusht")

# Create policy
policy = ACTPolicy.from_pretrained("lerobot/act_pusht")

# Run inference
action = policy.select_action(observation)
```

---

#### Project 5: Multi-Robot Coordination
**Goal**: Multiple robots work together

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 8-12 hours |
| Technologies | PyBullet + multi-agent |
| Skills | Coordination, collision avoidance |

**Tasks**:
- [ ] Spawn multiple robots
- [ ] Define shared workspace
- [ ] Implement collision avoidance
- [ ] Coordinate task handoffs
- [ ] Optimize throughput

---

### Advanced Projects (L3-L4)

#### Project 6: SO-101 Low-Cost Arm Build
**Goal**: Build and program real robot arm

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-4 weeks |
| Technologies | SO-101 hardware + LeRobot |
| Skills | Hardware assembly, calibration |

**Tasks**:
- [ ] 3D print parts
- [ ] Assemble servos and frame
- [ ] Wire electronics
- [ ] Calibrate joint limits
- [ ] Teleoperate with leader arm
- [ ] Collect and train policy

**Hardware Bill of Materials**:
```
- 6x Feetech STS3215 servos ($10 each)
- 1x Waveshare servo driver
- 1x Raspberry Pi 4/5
- 3D printed parts (from STL files)
- Total: ~$300
```

---

#### Project 7: Sim-to-Real Transfer
**Goal**: Train in sim, deploy on real robot

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 weeks |
| Technologies | Isaac Sim + real hardware |
| Skills | Domain randomization, transfer |

**Tasks**:
- [ ] Create accurate sim model
- [ ] Train policy in simulation
- [ ] Apply domain randomization
- [ ] Deploy on real hardware
- [ ] Fine-tune with real data
- [ ] Evaluate sim-to-real gap

---

#### Project 8: Language-Conditioned Manipulation
**Goal**: Robot follows natural language commands

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 weeks |
| Technologies | VLM + Robot Policy |
| Skills | Multimodal learning, grounding |

**Tasks**:
- [ ] Integrate vision-language model
- [ ] Ground language to objects
- [ ] Plan actions from commands
- [ ] Execute manipulation tasks
- [ ] Handle ambiguous commands

**Architecture**:
```
"Pick up the red cup" → VLM → Object Detection → Grasp Planning → Execute
                         ↓
                    Scene Understanding
```

---

## Framework Comparisons

### Comparison 1: Robot Simulators

**Question**: Which simulator for your needs?

| Simulator | Physics | Rendering | Speed | Ease | Best For |
|-----------|---------|-----------|-------|------|----------|
| **PyBullet** | Good | Basic | Fast | Easy | Learning |
| **MuJoCo** | Excellent | Good | Fastest | Medium | Research |
| **Isaac Sim** | Excellent | Photorealistic | GPU | Hard | Production |
| **Gazebo** | Good | Good | Medium | Medium | ROS integration |
| **CoppeliaSim** | Good | Good | Medium | Easy | Education |

**Lab Exercise**: Implement same task in PyBullet and MuJoCo.

```python
# PyBullet vs MuJoCo comparison
# PyBullet
import pybullet as p
p.connect(p.GUI)
robot = p.loadURDF("robot.urdf")

# MuJoCo
import mujoco
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
```

---

### Comparison 2: Learning Frameworks

**Question**: Which framework for robot learning?

| Framework | Approach | Ease | Features | Community |
|-----------|----------|------|----------|-----------|
| **LeRobot** | Imitation | Easy | ⭐⭐⭐⭐ | Growing |
| **Stable-Baselines3** | RL | Easy | ⭐⭐⭐ | Large |
| **RLlib** | RL (distributed) | Hard | ⭐⭐⭐⭐⭐ | Large |
| **robomimic** | Imitation | Medium | ⭐⭐⭐⭐ | Research |
| **NVIDIA Isaac Lab** | Both | Hard | ⭐⭐⭐⭐⭐ | Enterprise |

**Lab Exercise**: Train same task with imitation vs RL.

---

### Comparison 3: Policy Architectures

**Question**: Which policy for your task?

| Architecture | Training | Generalization | Multi-modal | Use Case |
|--------------|----------|----------------|-------------|----------|
| **ACT** | Fast | Good | Vision | Short tasks |
| **Diffusion Policy** | Slow | Excellent | Vision | Complex tasks |
| **VQ-BeT** | Medium | Good | Vision | Diverse behaviors |
| **RT-1/RT-2** | Very slow | Excellent | Language+Vision | General purpose |

**Lab Exercise**: Compare ACT vs Diffusion Policy on same dataset.

---

### Comparison 4: Low-Cost Robot Arms

**Question**: Which arm for learning robotics?

| Arm | Cost | DOF | Payload | Repeatability | Open Source |
|-----|------|-----|---------|---------------|-------------|
| **SO-101** | $300 | 6 | 250g | ±2mm | Yes |
| **ALOHA (DIY)** | $2000 | 2x6 | 300g | ±1mm | Yes |
| **UFactory Lite 6** | $1200 | 6 | 1kg | ±0.5mm | Partial |
| **myCobot 280** | $600 | 6 | 250g | ±0.5mm | Partial |
| **Koch v1.1** | $500 | 6 | 200g | ±1mm | Yes |

**Lab Exercise**: Evaluate accuracy and repeatability of chosen arm.

---

## Hands-On Labs

### Lab 1: Simulation Basics (3 hours)
```
Install PyBullet → Load Robot → Joint Control → Record Video
```

### Lab 2: Vision-Guided Grasping (6 hours)
```
Add Camera → Detect Object → Plan Grasp → Execute → Evaluate
```

### Lab 3: Imitation Learning (8 hours)
```
Collect Demos → Format Dataset → Train Policy → Evaluate → Iterate
```

### Lab 4: Hardware Integration (1 day)
```
Assemble Arm → Calibrate → Teleoperate → Collect Data → Train
```

### Lab 5: Sim-to-Real (2 days)
```
Model Robot → Domain Randomization → Train → Deploy → Fine-tune
```

---

## Robotics Design Patterns

### Pattern 1: Sense-Plan-Act
```
Perceive → Model World → Plan Actions → Execute → Loop
```

### Pattern 2: Behavior Trees
```
Root → Selector → Sequence → Conditions/Actions
```

### Pattern 3: Imitation Learning Pipeline
```
Demonstrations → Dataset → Policy Training → Evaluation → Deployment
```

### Pattern 4: Hierarchical Control
```
High-Level (language) → Mid-Level (skills) → Low-Level (motor)
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Task Success** | 40 | Robot completes objective |
| **Reliability** | 20 | Consistent performance |
| **Safety** | 15 | No collisions, graceful failures |
| **Code Quality** | 15 | Clean, modular, documented |
| **Innovation** | 10 | Creative approaches |

---

## Resources

- [LeRobot](https://github.com/huggingface/lerobot) - Hugging Face robotics
- [PyBullet](https://pybullet.org/) - Physics simulation
- [MuJoCo](https://mujoco.org/) - Fast physics
- [SO-101](https://github.com/TheRobotStudio/SO-ARM100) - Low-cost arm
- [ALOHA](https://github.com/tonyzhaozh/aloha) - Bimanual setup

---

*Part of [Luno-AI](../../../README.md) | Robotics AI Track*
