# ROS 2 Integration

> **Robot Operating System for production robotics**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Middleware framework for robotics |
| **Why** | Standard for robot software development |
| **Version** | ROS 2 Humble/Iron (recommended) |
| **Best For** | Multi-robot systems, production deployment |

### ROS 2 vs ROS 1

| Feature | ROS 2 | ROS 1 |
|---------|-------|-------|
| Real-time | ✓ | Limited |
| Multi-robot | Native | Complex |
| Security | Built-in | None |
| Python | 3.x | 2.7/3.x |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **OS** | Ubuntu 22.04 (Humble) / 24.04 (Jazzy) |
| **Python** | 3.10+ |
| **Disk** | 5GB+ |

---

## Quick Start (1-2 hours)

### Installation (Ubuntu)

```bash
# Setup sources
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install ros-humble-desktop python3-colcon-common-extensions

# Source
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### First Node

```python
# my_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main():
    rclpy.init()
    node = MinimalPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```bash
# Run
python3 my_node.py
```

---

## Learning Path

### L0: ROS 2 Basics (2-4 hours)
- [ ] Install ROS 2
- [ ] Create publisher/subscriber
- [ ] Understand topics
- [ ] Use command-line tools

### L1: Intermediate (1-2 days)
- [ ] Services and actions
- [ ] Launch files
- [ ] Parameters
- [ ] Create package

### L2: Advanced (1 week)
- [ ] Navigation stack
- [ ] MoveIt for manipulation
- [ ] Custom messages
- [ ] Multi-robot systems

---

## Code Examples

### Publisher and Subscriber

```python
# publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Publisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.pub = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(1.0, self.publish)

    def publish(self):
        msg = String()
        msg.data = 'Hello ROS 2!'
        self.pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(Publisher())
    rclpy.shutdown()
```

```python
# subscriber.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.sub = self.create_subscription(String, 'chatter', self.callback, 10)

    def callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

def main():
    rclpy.init()
    rclpy.spin(Subscriber())
    rclpy.shutdown()
```

### Service

```python
# service_server.py
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddService(Node):
    def __init__(self):
        super().__init__('add_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.callback)

    def callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'{request.a} + {request.b} = {response.sum}')
        return response

def main():
    rclpy.init()
    rclpy.spin(AddService())
    rclpy.shutdown()
```

```python
# service_client.py
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class AddClient(Node):
    def __init__(self):
        super().__init__('add_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')

    def call(self, a: int, b: int):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main():
    rclpy.init()
    client = AddClient()
    result = client.call(2, 3)
    print(f'Result: {result.sum}')
    rclpy.shutdown()
```

### Launch File

```python
# launch/robot_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_package',
            executable='publisher',
            name='publisher',
            output='screen'
        ),
        Node(
            package='my_package',
            executable='subscriber',
            name='subscriber',
            output='screen'
        )
    ])
```

```bash
# Run launch file
ros2 launch my_package robot_launch.py
```

### Camera Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.033, self.publish_frame)  # 30 FPS
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(CameraNode())
    rclpy.shutdown()
```

### AI Integration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from cv_bridge import CvBridge
from ultralytics import YOLO

class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        self.sub = self.create_subscription(
            Image, 'camera/image_raw', self.callback, 10
        )
        self.pub = self.create_publisher(
            Detection2DArray, 'detections', 10
        )

    def callback(self, msg):
        # Convert to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Run YOLO
        results = self.model(frame)[0]

        # Publish detections
        det_array = Detection2DArray()
        for box in results.boxes:
            det = Detection2D()
            det.bbox.center.position.x = float(box.xywh[0][0])
            det.bbox.center.position.y = float(box.xywh[0][1])
            det.bbox.size_x = float(box.xywh[0][2])
            det.bbox.size_y = float(box.xywh[0][3])
            det_array.detections.append(det)

        self.pub.publish(det_array)

def main():
    rclpy.init()
    rclpy.spin(YOLONode())
    rclpy.shutdown()
```

### Package Structure

```
my_robot_pkg/
├── package.xml
├── setup.py
├── setup.cfg
├── my_robot_pkg/
│   ├── __init__.py
│   ├── publisher.py
│   └── subscriber.py
├── launch/
│   └── robot_launch.py
├── config/
│   └── params.yaml
└── msg/
    └── CustomMsg.msg
```

```python
# setup.py
from setuptools import setup

package_name = 'my_robot_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'publisher = my_robot_pkg.publisher:main',
            'subscriber = my_robot_pkg.subscriber:main',
        ],
    },
)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `ros2 topic list` | List all topics |
| `ros2 topic echo /topic` | Print topic messages |
| `ros2 node list` | List running nodes |
| `ros2 service list` | List available services |
| `ros2 run pkg node` | Run a node |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Node not found | Source setup.bash, rebuild |
| Topic not visible | Check namespace, DDS config |
| Performance issues | Tune QoS, use callbacks |
| Multi-machine | Configure ROS_DOMAIN_ID |

---

## Resources

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [MoveIt 2](https://moveit.ros.org/)
- [Nav2](https://navigation.ros.org/)

---

*Part of [Luno-AI](../../README.md) | Robotics Track*
