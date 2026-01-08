# Production Agents Integration

> **Deploy reliable, scalable AI agents in production**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Production-grade agent infrastructure |
| **Why** | Reliability, observability, scalability |
| **Components** | Error handling, monitoring, rate limiting |
| **Best For** | Customer-facing applications, critical workflows |

### Production vs Development

| Aspect | Development | Production |
|--------|-------------|------------|
| Error Handling | Basic | Comprehensive |
| Monitoring | Logs | Metrics + Traces |
| Rate Limiting | None | Required |
| Fallbacks | Optional | Required |
| Testing | Manual | Automated |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Infrastructure** | Cloud or on-prem servers |
| **Monitoring** | Logging and metrics system |
| **Database** | For state persistence |

---

## Quick Start (1 hour)

### Robust Agent Structure

```python
import asyncio
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.7
    max_retries: int = 3
    timeout: float = 30.0

class ProductionAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = AsyncOpenAI()
        self.request_count = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def call_llm(self, messages: list) -> str:
        """Call LLM with retry logic"""
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                ),
                timeout=self.config.timeout
            )
            self.request_count += 1
            return response.choices[0].message.content

        except asyncio.TimeoutError:
            logger.error("LLM call timed out")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def run(self, task: str) -> dict:
        """Run agent with full error handling"""
        start_time = datetime.now()

        try:
            result = await self.call_llm([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": task}
            ])

            return {
                "status": "success",
                "result": result,
                "duration": (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }

# Usage
async def main():
    agent = ProductionAgent(AgentConfig())
    result = await agent.run("Explain quantum computing")
    print(result)

asyncio.run(main())
```

---

## Learning Path

### L0: Basic Production Setup (2-3 hours)
- [ ] Error handling
- [ ] Retry logic
- [ ] Timeout management
- [ ] Basic logging

### L1: Observability (4-6 hours)
- [ ] Structured logging
- [ ] Metrics collection
- [ ] Distributed tracing
- [ ] Alerting

### L2: Scalability (1-2 days)
- [ ] Rate limiting
- [ ] Queue management
- [ ] Horizontal scaling
- [ ] Cost optimization

---

## Code Examples

### Comprehensive Error Handling

```python
from enum import Enum
from typing import Optional
import traceback

class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    UNKNOWN = "unknown"

class AgentError(Exception):
    def __init__(self, error_type: ErrorType, message: str, recoverable: bool = True):
        self.error_type = error_type
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)

class ErrorHandler:
    def __init__(self):
        self.error_counts = {}

    def handle(self, error: Exception) -> tuple[ErrorType, bool]:
        """Categorize error and determine if recoverable"""
        error_str = str(error).lower()

        if "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT, True
        elif "timeout" in error_str:
            return ErrorType.TIMEOUT, True
        elif "api" in error_str or "500" in error_str:
            return ErrorType.API_ERROR, True
        elif "validation" in error_str:
            return ErrorType.VALIDATION, False
        else:
            return ErrorType.UNKNOWN, False

    def log_error(self, error: Exception, context: dict):
        """Log error with context"""
        error_type, recoverable = self.handle(error)
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        logger.error({
            "error_type": error_type.value,
            "message": str(error),
            "recoverable": recoverable,
            "context": context,
            "traceback": traceback.format_exc()
        })

        return error_type, recoverable
```

### Observability with OpenTelemetry

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Setup metrics
meter = metrics.get_meter(__name__)
request_counter = meter.create_counter("agent_requests")
latency_histogram = meter.create_histogram("agent_latency")

class ObservableAgent:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def run(self, task: str) -> dict:
        with tracer.start_as_current_span("agent_run") as span:
            span.set_attribute("task", task[:100])
            start = datetime.now()

            try:
                result = await self._execute(task)

                # Record metrics
                latency = (datetime.now() - start).total_seconds()
                request_counter.add(1, {"status": "success"})
                latency_histogram.record(latency)

                span.set_attribute("status", "success")
                return {"status": "success", "result": result}

            except Exception as e:
                request_counter.add(1, {"status": "error"})
                span.set_attribute("status", "error")
                span.record_exception(e)
                raise

    async def _execute(self, task: str) -> str:
        with tracer.start_as_current_span("llm_call"):
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": task}]
            )
            return response.choices[0].message.content
```

### Rate Limiting

```python
import asyncio
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until rate limit allows request"""
        async with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)

            # Remove old requests
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            if len(self.requests) >= self.requests_per_minute:
                # Wait until oldest request expires
                wait_time = (self.requests[0] + timedelta(minutes=1) - now).total_seconds()
                await asyncio.sleep(max(0, wait_time))
                return await self.acquire()

            self.requests.append(now)

class RateLimitedAgent:
    def __init__(self, rpm: int = 60):
        self.limiter = RateLimiter(rpm)
        self.client = AsyncOpenAI()

    async def run(self, task: str) -> str:
        await self.limiter.acquire()
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task}]
        )
        return response.choices[0].message.content
```

### State Persistence

```python
import json
from pathlib import Path
import redis.asyncio as redis

class StateManager:
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = redis.from_url(redis_url)

    async def save_state(self, session_id: str, state: dict):
        """Save agent state to Redis"""
        await self.redis.set(
            f"agent:state:{session_id}",
            json.dumps(state),
            ex=3600  # 1 hour TTL
        )

    async def load_state(self, session_id: str) -> dict:
        """Load agent state from Redis"""
        data = await self.redis.get(f"agent:state:{session_id}")
        if data:
            return json.loads(data)
        return {}

    async def save_checkpoint(self, session_id: str, checkpoint: dict):
        """Save checkpoint for recovery"""
        await self.redis.lpush(
            f"agent:checkpoints:{session_id}",
            json.dumps(checkpoint)
        )

class StatefulAgent:
    def __init__(self):
        self.state_manager = StateManager()
        self.client = AsyncOpenAI()

    async def run(self, session_id: str, task: str) -> dict:
        # Load previous state
        state = await self.state_manager.load_state(session_id)
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": task})

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            result = response.choices[0].message.content
            messages.append({"role": "assistant", "content": result})

            # Save state
            await self.state_manager.save_state(session_id, {"messages": messages})
            await self.state_manager.save_checkpoint(session_id, {
                "task": task,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })

            return {"status": "success", "result": result}

        except Exception as e:
            # State preserved, can retry
            await self.state_manager.save_state(session_id, {"messages": messages})
            raise
```

### FastAPI Deployment

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uuid

app = FastAPI()

class TaskRequest(BaseModel):
    task: str
    session_id: str = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: str = None

agent = ProductionAgent(AgentConfig())
tasks = {}

@app.post("/agent/run", response_model=TaskResponse)
async def run_agent(request: TaskRequest):
    task_id = str(uuid.uuid4())
    session_id = request.session_id or task_id

    try:
        result = await agent.run(request.task)
        return TaskResponse(
            task_id=task_id,
            status=result["status"],
            result=result.get("result")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/run-async")
async def run_agent_async(request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "pending"}

    background_tasks.add_task(execute_task, task_id, request.task)

    return {"task_id": task_id, "status": "pending"}

@app.get("/agent/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

async def execute_task(task_id: str, task: str):
    tasks[task_id] = {"status": "running"}
    result = await agent.run(task)
    tasks[task_id] = result
```

---

## Production Checklist

| Category | Item |
|----------|------|
| **Reliability** | Retry logic, timeouts, circuit breakers |
| **Observability** | Logging, metrics, tracing |
| **Security** | Input validation, secrets management |
| **Scalability** | Rate limiting, queueing, caching |
| **Recovery** | State persistence, checkpointing |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| High latency | Add caching, optimize prompts |
| Rate limits | Implement queuing, add retries |
| Memory leaks | Clear conversation history |
| Inconsistent results | Add validation, deterministic settings |

---

## Resources

- [LangSmith](https://smith.langchain.com/) - Agent observability
- [OpenTelemetry](https://opentelemetry.io/) - Distributed tracing
- [Redis](https://redis.io/) - State management
- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

---

*Part of [Luno-AI](../../README.md) | Agentic AI Track*
