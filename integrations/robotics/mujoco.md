# MuJoCo Simulation Integration

> **Physics simulation for robotics and RL research**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Multi-Joint dynamics with Contact (physics engine) |
| **Why** | Fast, accurate simulation for robotics/RL |
| **License** | Free and open source (Apache 2.0) |
| **Best For** | Robot training, policy learning, testing |

### MuJoCo vs Alternatives

| Feature | MuJoCo | PyBullet | Isaac Sim |
|---------|--------|----------|-----------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Ease of use | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| GPU | Limited | No | Yes |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.8+ |
| **OS** | Linux, macOS, Windows |
| **Optional** | GPU for rendering |

---

## Quick Start (30 min)

### Installation

```bash
pip install mujoco
pip install mujoco-python-viewer  # For visualization
```

### Basic Simulation

```python
import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# Simulation loop
for _ in range(1000):
    # Set control
    data.ctrl[:] = np.random.randn(model.nu) * 0.1

    # Step simulation
    mujoco.mj_step(model, data)

    # Read state
    print(f"Position: {data.qpos}")
    print(f"Velocity: {data.qvel}")
```

### Simple Model XML

```xml
<!-- robot.xml -->
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="10 10 0.1" rgba=".9 .9 .9 1"/>

    <body name="box" pos="0 0 1">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="slide" gear="100"/>
  </actuator>
</mujoco>
```

---

## Learning Path

### L0: Basic Simulation (2-3 hours)
- [ ] Install MuJoCo
- [ ] Load example models
- [ ] Step simulation
- [ ] Visualize with viewer

### L1: Custom Models (4-6 hours)
- [ ] Create MJCF models
- [ ] Add joints and actuators
- [ ] Contact physics
- [ ] Sensors

### L2: RL Integration (1-2 days)
- [ ] Gymnasium environments
- [ ] Train policies
- [ ] Domain randomization
- [ ] Sim-to-real transfer

---

## Code Examples

### Interactive Viewer

```python
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("humanoid.xml")
data = mujoco.MjData(model)

# Launch interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

### Robot Arm Model

```xml
<!-- arm.xml -->
<mujoco model="simple_arm">
  <option gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 2"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>

    <!-- Base -->
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.1 0.05" rgba=".5 .5 .5 1"/>

      <!-- Link 1 -->
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-180 180"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.3" rgba="1 0 0 1"/>

        <!-- Link 2 -->
        <body name="link2" pos="0 0 0.3">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-90 90"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0" rgba="0 1 0 1"/>

          <!-- End effector -->
          <body name="end_effector" pos="0.3 0 0">
            <geom type="sphere" size="0.03" rgba="0 0 1 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor joint="joint1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor joint="joint2" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>

  <sensor>
    <jointpos joint="joint1"/>
    <jointpos joint="joint2"/>
  </sensor>
</mujoco>
```

### Gymnasium Environment

```python
import gymnasium as gym
import numpy as np
import mujoco

class RobotArmEnv(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("arm.xml")
        self.data = mujoco.MjData(self.model)

        # Define spaces
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.target = np.array([0.2, 0.0, 0.3])

    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Randomize target
        self.target = np.array([
            np.random.uniform(0.1, 0.4),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.1, 0.4)
        ])

        return self._get_obs(), {}

    def step(self, action):
        # Apply action
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation
        obs = self._get_obs()

        # Compute reward
        ee_pos = self.data.body("end_effector").xpos
        distance = np.linalg.norm(ee_pos - self.target)
        reward = -distance

        # Check done
        done = distance < 0.05

        return obs, reward, done, False, {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel
        ]).astype(np.float32)

    def render(self):
        pass

# Register and use
gym.register(id='RobotArm-v0', entry_point=RobotArmEnv)
env = gym.make('RobotArm-v0')
```

### Domain Randomization

```python
import mujoco
import numpy as np

def randomize_dynamics(model):
    """Randomize physics parameters for sim-to-real"""
    # Randomize mass
    for i in range(model.nbody):
        model.body_mass[i] *= np.random.uniform(0.8, 1.2)

    # Randomize friction
    for i in range(model.ngeom):
        model.geom_friction[i] *= np.random.uniform(0.5, 2.0)

    # Randomize damping
    for i in range(model.nv):
        model.dof_damping[i] *= np.random.uniform(0.8, 1.2)

    return model

def train_with_randomization(base_model_path, episodes=1000):
    for episode in range(episodes):
        # Load fresh model
        model = mujoco.MjModel.from_xml_path(base_model_path)

        # Randomize
        model = randomize_dynamics(model)
        data = mujoco.MjData(model)

        # Train episode with randomized dynamics
        for step in range(500):
            action = policy(data.qpos, data.qvel)
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
```

### Sensor Reading

```python
import mujoco

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)

# Simulate
mujoco.mj_step(model, data)

# Read sensors
for i in range(model.nsensor):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
    value = data.sensordata[i]
    print(f"Sensor {name}: {value}")

# Get specific body position
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
position = data.xpos[body_id]
orientation = data.xquat[body_id]
```

### Rendering to Video

```python
import mujoco
import cv2
import numpy as np

model = mujoco.MjModel.from_xml_path("robot.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('simulation.mp4', fourcc, 30.0, (640, 480))

for _ in range(300):
    mujoco.mj_step(model, data)

    # Render frame
    renderer.update_scene(data)
    frame = renderer.render()

    # Write to video (convert RGB to BGR)
    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

video.release()
```

---

## Model Building Tips

| Element | Description |
|---------|-------------|
| `<body>` | Rigid body with mass/inertia |
| `<geom>` | Collision/visual geometry |
| `<joint>` | Connects bodies (hinge, slide, ball) |
| `<actuator>` | Motors/forces |
| `<sensor>` | Measurements |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not loading | Check XML syntax, paths |
| Unstable simulation | Reduce timestep, check masses |
| Slow rendering | Use offscreen rendering |
| Joints exploding | Add damping, limits |

---

## Resources

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo GitHub](https://github.com/google-deepmind/mujoco)
- [Model Gallery](https://github.com/google-deepmind/mujoco_menagerie)
- [Gymnasium MuJoCo](https://gymnasium.farama.org/environments/mujoco/)

---

*Part of [Luno-AI](../../README.md) | Robotics Track*
