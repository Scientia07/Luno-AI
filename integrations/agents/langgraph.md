# LangGraph Agent Framework Integration

> **Category**: Agentic AI
> **Difficulty**: Intermediate
> **Setup Time**: 3-4 hours
> **Last Updated**: 2026-01-03

---

## Overview

### What It Does
LangGraph is a framework for building stateful, multi-actor applications with LLMs. It models agent workflows as graphs where nodes are actions and edges are transitions, enabling complex reasoning and tool use.

### Why Use It
- **State Management**: Built-in persistence and checkpointing
- **Control Flow**: Conditional routing, loops, human-in-the-loop
- **Production Ready**: Used by Klarna, Uber, LinkedIn, Replit
- **Debuggable**: LangSmith integration for tracing
- **Flexible**: Custom nodes, edges, and state schemas

### Key Capabilities
| Capability | Description |
|------------|-------------|
| Stateful Agents | Persist conversation and tool state |
| Multi-Agent | Coordinate multiple specialized agents |
| Human-in-the-Loop | Pause for human approval/input |
| Tool Use | Integrate any Python function or API |
| Streaming | Real-time token and event streaming |
| Checkpointing | Save and resume agent state |

### When to Use What
| Use Case | Framework |
|----------|-----------|
| Complex workflows, production | **LangGraph** |
| Quick prototypes, simple agents | CrewAI |
| Research, experimentation | AutoGen |
| Minimal abstraction | OpenAI Agents SDK |

---

## Prerequisites

### Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None | None (LLM can be remote) |
| RAM | 4 GB | 8 GB |
| Storage | 1 GB | 2 GB |

### Software Dependencies
```bash
# Core
pip install langgraph langchain langchain-openai

# For local LLMs
pip install langchain-ollama

# For MCP tools
pip install langchain-mcp-adapters

# For persistence
pip install langgraph-checkpoint-postgres  # or sqlite
```

### Prior Knowledge
- [x] Python basics
- [x] LLM concepts (prompts, completions)
- [ ] LangChain basics (helpful but not required)

---

## Quick Start (20 minutes)

### 1. Install
```bash
pip install langgraph langchain langchain-openai
```

### 2. Basic Agent
```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

# Define state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Define node
def chatbot(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile
agent = graph.compile()

# Run
result = agent.invoke({"messages": [("user", "Hello!")]})
print(result["messages"][-1].content)
```

### 3. Verify
```python
# Visualize the graph
print(agent.get_graph().draw_mermaid())
```

---

## Full Setup

### Core Concepts

```
                    LANGGRAPH ARCHITECTURE

    ┌─────────────────────────────────────────────────────┐
    │                      STATE                           │
    │   Shared data structure passed between nodes         │
    │   (TypedDict, Pydantic, or dataclass)               │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                      GRAPH                           │
    │                                                      │
    │   START ──▶ Node A ──┬──▶ Node B ──▶ END           │
    │                      │                              │
    │                      └──▶ Node C ──┘               │
    │                                                      │
    │   Nodes: Functions that transform state             │
    │   Edges: Transitions (can be conditional)           │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────┐
    │                  CHECKPOINTER                        │
    │   Saves state for persistence, time-travel, replay   │
    └─────────────────────────────────────────────────────┘
```

### Agent with Tools

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

tools = [search_web, calculate]

# State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# Nodes
def agent(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    tools_condition,  # Routes to "tools" if tool call, else END
)
graph.add_edge("tools", "agent")

agent = graph.compile()

# Run
result = agent.invoke({
    "messages": [("user", "What is 25 * 4 + 10?")]
})
print(result["messages"][-1].content)
```

---

## Learning Path

### L0: Simple Graph (1 hour)
**Goal**: Build a basic stateful agent

- [x] Install LangGraph
- [ ] Create state schema
- [ ] Add nodes and edges
- [ ] Run the agent

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str

def process(state: State):
    return {"output": f"Processed: {state['input']}"}

graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
result = app.invoke({"input": "Hello"})
print(result["output"])
```

### L1: Tools & Routing (2 hours)
**Goal**: Add tool use and conditional logic

- [ ] Define custom tools
- [ ] Add conditional edges
- [ ] Handle tool responses

```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Literal

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 72°F in {city}"

@tool
def get_time(timezone: str) -> str:
    """Get current time in timezone."""
    return f"3:45 PM in {timezone}"

tools = [get_weather, get_time]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def should_continue(state) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

# ... (rest of graph setup)
```

### L2: Persistence & Human-in-the-Loop (3 hours)
**Goal**: Save state and add human approval

- [ ] Add checkpointer
- [ ] Implement interrupt
- [ ] Resume execution

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Checkpointer for persistence
memory = MemorySaver()

def human_approval(state):
    # Pause and wait for human
    response = interrupt({
        "question": "Do you approve this action?",
        "action": state["pending_action"]
    })
    return {"approved": response == "yes"}

graph = StateGraph(State)
# ... add nodes ...

agent = graph.compile(checkpointer=memory)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "user-123"}}

# First run - will pause at interrupt
result = agent.invoke({"messages": [...]}, config)

# Resume with human response
result = agent.invoke(
    Command(resume="yes"),  # Human approves
    config
)
```

### L3: Multi-Agent Systems (4+ hours)
**Goal**: Coordinate multiple agents

- [ ] Create specialized agents
- [ ] Build supervisor pattern
- [ ] Implement handoffs

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

# Specialized agents
researcher = create_react_agent(llm, [search_web], state_modifier="You are a researcher.")
coder = create_react_agent(llm, [run_code], state_modifier="You are a coder.")

def supervisor(state):
    """Route to appropriate specialist."""
    # Analyze task and decide routing
    response = llm.invoke(f"Route this task: {state['task']}")
    return {"next": "researcher" if "research" in response else "coder"}

# Supervisor graph
graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next"])
graph.add_edge("researcher", END)
graph.add_edge("coder", END)
```

---

## Code Examples

### Example 1: ReAct Agent (Recommended Pattern)
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}: ..."

@tool
def calculate(expr: str) -> float:
    """Calculate math expression."""
    return eval(expr)

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [search, calculate])

result = agent.invoke({
    "messages": [("user", "What is the population of France divided by 1000?")]
})
```

### Example 2: Streaming Responses
```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

async def stream_agent():
    async for event in agent.astream_events(
        {"messages": [("user", "Write a haiku")]},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)

import asyncio
asyncio.run(stream_agent())
```

### Example 3: With Ollama (Local LLM)
```python
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
agent = create_react_agent(llm, tools)

result = agent.invoke({"messages": [("user", "Hello!")]})
```

### Example 4: MCP Tool Integration
```python
from langchain_mcp_adapters import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Connect to MCP servers
async def create_mcp_agent():
    async with MultiServerMCPClient({
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@anthropic-ai/mcp-filesystem", "/tmp"]
        }
    }) as client:
        tools = client.get_tools()
        agent = create_react_agent(llm, tools)
        return agent
```

---

## Integration Points

### Works Well With
| Integration | Purpose | Link |
|-------------|---------|------|
| Ollama | Local LLMs | [ollama.md](../llms/ollama.md) |
| OpenAI | Cloud LLMs | [openai.md](../llms/openai.md) |
| MCP | Universal tools | [mcp.md](./mcp.md) |
| RAG | Document retrieval | [rag.md](./rag.md) |
| Whisper | Voice input | [whisper.md](../audio/whisper.md) |

### Checkpointers
```python
# Memory (development)
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SQLite (local persistence)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL (production)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

### LangSmith Tracing
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-key
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Infinite Loop
**Symptoms**: Agent keeps calling tools forever
**Solution**:
```python
# Add recursion limit
agent = graph.compile()
result = agent.invoke(
    {"messages": [...]},
    {"recursion_limit": 10}
)

# Or add explicit stop condition
def should_continue(state):
    if len(state["messages"]) > 20:
        return "end"
    # ... rest of logic
```

#### Issue 2: Tool Not Being Called
**Symptoms**: LLM ignores available tools
**Solution**:
```python
# Make tool descriptions clearer
@tool
def search_database(query: str) -> str:
    """
    Search the internal database for information.
    Use this when you need to find specific data.

    Args:
        query: The search query string

    Returns:
        Search results as a string
    """
    return "..."

# Or force tool use
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(
    tools,
    tool_choice="required"  # Must use a tool
)
```

#### Issue 3: State Not Persisting
**Symptoms**: Agent forgets previous context
**Solution**:
```python
# Always use same thread_id
config = {"configurable": {"thread_id": "consistent-id"}}

# Verify checkpointer is attached
agent = graph.compile(checkpointer=memory)

# Check state
state = agent.get_state(config)
print(state.values)
```

### Performance Tips
- Use streaming for better UX
- Limit context window with message trimming
- Cache tool results when possible
- Use smaller models for routing decisions

---

## Resources

### Official
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph Platform](https://www.langchain.com/langgraph)

### Tutorials
- [Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
- [Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)

### Community
- [Discord](https://discord.gg/langchain)
- [LangChain Blog](https://blog.langchain.dev/)

---

## Related Integrations

| Next Step | Why | Link |
|-----------|-----|------|
| MCP Tools | Universal tool standard | [mcp.md](./mcp.md) |
| RAG Pipeline | Document Q&A | [rag.md](./rag.md) |
| CrewAI | Team-based agents | [crewai.md](./crewai.md) |
| Production Agents | Deploy to prod | [production-agents.md](./production-agents.md) |

---

*Part of [Luno-AI Integration Hub](../_index.md) | Agentic AI Track*
