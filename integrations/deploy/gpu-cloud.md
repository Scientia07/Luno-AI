# GPU Cloud Setup Integration

> **Deploy AI workloads on cloud GPUs**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Cloud GPU infrastructure for AI |
| **Why** | Scale compute, avoid hardware costs |
| **Providers** | RunPod, Lambda, AWS, GCP, Azure |
| **Best For** | Training, inference, development |

### Provider Comparison

| Provider | Price/hr (A100) | Setup | Best For |
|----------|-----------------|-------|----------|
| **RunPod** | $1.89 | Easy | Development |
| **Lambda** | $1.29 | Easy | Training |
| **Vast.ai** | $0.80+ | Medium | Budget |
| **AWS** | $3.50+ | Complex | Enterprise |
| **GCP** | $3.00+ | Complex | Enterprise |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Account** | Cloud provider account |
| **SSH** | SSH key pair |
| **Budget** | Pay-per-use billing |

---

## Quick Start (30 min)

### RunPod Setup

```bash
# Install CLI
pip install runpod

# Set API key
export RUNPOD_API_KEY="your-api-key"
```

```python
import runpod

# Create pod
pod = runpod.create_pod(
    name="ml-training",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
    gpu_type_id="NVIDIA RTX A6000",
    gpu_count=1,
    volume_in_gb=50,
    container_disk_in_gb=20,
    ports="8888/http,22/tcp"
)

print(f"Pod ID: {pod['id']}")
print(f"SSH: ssh root@{pod['machine']['podHostId']}.runpod.io")
```

### Lambda Labs

```bash
# SSH to instance
ssh ubuntu@<instance-ip> -i ~/.ssh/lambda_key

# Setup environment
sudo apt update && sudo apt install -y python3-pip
pip install torch torchvision
```

---

## Learning Path

### L0: First GPU Instance (1-2 hours)
- [ ] Create cloud account
- [ ] Launch GPU instance
- [ ] SSH connection
- [ ] Run first training

### L1: Optimization (3-4 hours)
- [ ] Spot instances
- [ ] Storage setup
- [ ] Docker workflows
- [ ] Cost monitoring

### L2: Production (1-2 days)
- [ ] Auto-scaling
- [ ] CI/CD integration
- [ ] Multi-region
- [ ] Kubernetes

---

## Code Examples

### RunPod Serverless

```python
import runpod

def handler(event):
    """Serverless inference handler"""
    prompt = event["input"]["prompt"]

    # Your inference code
    from transformers import pipeline
    generator = pipeline("text-generation", model="gpt2")
    result = generator(prompt, max_length=100)

    return {"output": result[0]["generated_text"]}

runpod.serverless.start({"handler": handler})
```

### AWS EC2 GPU Setup

```python
import boto3

ec2 = boto3.resource('ec2')

# Launch GPU instance
instances = ec2.create_instances(
    ImageId='ami-0123456789abcdef0',  # Deep Learning AMI
    InstanceType='g4dn.xlarge',        # T4 GPU
    MinCount=1,
    MaxCount=1,
    KeyName='your-key-pair',
    SecurityGroupIds=['sg-xxxxxxxx'],
    BlockDeviceMappings=[
        {
            'DeviceName': '/dev/sda1',
            'Ebs': {'VolumeSize': 100, 'VolumeType': 'gp3'}
        }
    ],
    TagSpecifications=[
        {
            'ResourceType': 'instance',
            'Tags': [{'Key': 'Name', 'Value': 'ML-Training'}]
        }
    ]
)

instance = instances[0]
instance.wait_until_running()
instance.reload()

print(f"Instance ID: {instance.id}")
print(f"Public IP: {instance.public_ip_address}")
```

### Cost Optimization Script

```python
import datetime

class CostTracker:
    def __init__(self, hourly_rate: float):
        self.hourly_rate = hourly_rate
        self.start_time = None
        self.total_cost = 0

    def start(self):
        self.start_time = datetime.datetime.now()

    def stop(self) -> float:
        if self.start_time:
            duration = (datetime.datetime.now() - self.start_time).total_seconds() / 3600
            cost = duration * self.hourly_rate
            self.total_cost += cost
            self.start_time = None
            return cost
        return 0

    def estimate_daily(self, hours_per_day: float) -> float:
        return hours_per_day * self.hourly_rate

# Usage
tracker = CostTracker(hourly_rate=1.89)  # RunPod A6000
tracker.start()
# ... run training ...
cost = tracker.stop()
print(f"Session cost: ${cost:.2f}")
```

### Docker for GPU Cloud

```dockerfile
# Dockerfile for GPU workloads
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Set environment
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "train.py"]
```

```bash
# Build and push
docker build -t your-registry/ml-training:latest .
docker push your-registry/ml-training:latest

# Run on GPU cloud
docker run --gpus all your-registry/ml-training:latest
```

### Spot Instance Management

```python
import boto3
import time

class SpotManager:
    def __init__(self):
        self.ec2 = boto3.client('ec2')

    def request_spot(self, instance_type: str, max_price: str):
        """Request spot instance with interruption handling"""
        response = self.ec2.request_spot_instances(
            InstanceCount=1,
            Type='one-time',
            SpotPrice=max_price,
            LaunchSpecification={
                'ImageId': 'ami-xxx',
                'InstanceType': instance_type,
                'KeyName': 'your-key',
                'SecurityGroupIds': ['sg-xxx'],
            }
        )
        return response['SpotInstanceRequests'][0]['SpotInstanceRequestId']

    def wait_for_fulfillment(self, request_id: str, timeout: int = 300):
        """Wait for spot request to be fulfilled"""
        start = time.time()
        while time.time() - start < timeout:
            response = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[request_id]
            )
            status = response['SpotInstanceRequests'][0]['Status']['Code']
            if status == 'fulfilled':
                return response['SpotInstanceRequests'][0]['InstanceId']
            time.sleep(10)
        raise TimeoutError("Spot request not fulfilled")

    def setup_interruption_notice(self):
        """Check for spot interruption warning"""
        # Instance metadata check
        import requests
        try:
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/spot/termination-time",
                timeout=1
            )
            if response.status_code == 200:
                return response.text  # Termination time
        except:
            pass
        return None
```

### Multi-GPU Training Setup

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def train_distributed(rank, world_size, model, dataset):
    setup_distributed(rank, world_size)

    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=32
    )

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            # Training step
            pass

    dist.destroy_process_group()

# Launch
import torch.multiprocessing as mp
world_size = torch.cuda.device_count()
mp.spawn(train_distributed, args=(world_size, model, dataset), nprocs=world_size)
```

---

## Provider Quick Reference

| Task | RunPod | Lambda | AWS |
|------|--------|--------|-----|
| Web UI | ✓ | ✓ | Console |
| CLI | runpod | - | aws |
| Spot | Community | - | ✓ |
| Kubernetes | ✓ | - | EKS |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Instance not starting | Check quotas, region availability |
| SSH timeout | Check security groups, VPC |
| CUDA errors | Verify driver installation |
| High costs | Use spot instances, auto-shutdown |

---

## Resources

- [RunPod Docs](https://docs.runpod.io/)
- [Lambda Labs](https://lambdalabs.com/service/gpu-cloud)
- [AWS EC2 GPU](https://aws.amazon.com/ec2/instance-types/g4/)
- [Vast.ai](https://vast.ai/)

---

*Part of [Luno-AI](../../README.md) | Deployment Track*
