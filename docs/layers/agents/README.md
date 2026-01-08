# Agentic AI: Autonomous Systems

> **LLMs that take action** - from simple tool use to multi-agent systems.

---

## Layer Navigation

| Layer | Content | Status |
|-------|---------|--------|
| L0 | [Overview](#overview) | This file |
| L1 | [Concepts](./concepts.md) | Pending |
| L2 | [Deep Dive](./deep-dive.md) | Pending |
| L3 | [Labs](../../labs/agents/) | Pending |
| L4 | [Advanced](./advanced.md) | Pending |

---

## Overview

Agents are LLMs that can perceive, reason, and act. They don't just generate text - they use tools, maintain memory, and accomplish complex goals autonomously.

```
                    AGENT ARCHITECTURE

    ┌─────────────────────────────────────────────┐
    │                   AGENT                      │
    │  ┌─────────┐  ┌─────────┐  ┌─────────────┐  │
    │  │ Perceive│  │ Reason  │  │    Act      │  │
    │  │         │  │ (LLM)   │  │  (Tools)    │  │
    │  └────┬────┘  └────┬────┘  └──────┬──────┘  │
    │       │            │              │         │
    │       └────────────┼──────────────┘         │
    │                    │                        │
    │              ┌─────▼─────┐                  │
    │              │  Memory   │                  │
    │              └───────────┘                  │
    └─────────────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │    ENVIRONMENT      │
              │  Files, APIs, DBs   │
              └─────────────────────┘
```

---

## Agent Frameworks

### Comparison

| Framework | Approach | Best For |
|-----------|----------|----------|
| **LangGraph** | Graph-based workflows | Complex, stateful agents |
| **CrewAI** | Role-based teams | Multi-agent collaboration |
| **AutoGen** | Conversational | Research, experimentation |
| **Claude Tools** | Native tool use | Simple, reliable agents |
| **Agency Swarm** | Custom agents | Production systems |
| **OpenAI Assistants** | Hosted | Quick prototyping |

### LangGraph (Recommended)

```python
from langgraph.graph import StateGraph

# Define agent state
class AgentState(TypedDict):
    messages: list
    next_action: str

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("think", think_node)
workflow.add_node("act", act_node)
workflow.add_edge("think", "act")
```

**Key Concepts:**
- **Nodes**: Functions that process state
- **Edges**: Control flow between nodes
- **Checkpoints**: Persistent state for long-running agents
- **Human-in-the-loop**: Pause for approval

### CrewAI

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert at web research"
)

writer = Agent(
    role="Writer",
    goal="Write clear reports",
    backstory="Technical writer"
)

crew = Crew(agents=[researcher, writer], tasks=[...])
result = crew.kickoff()
```

---

## MCP (Model Context Protocol)

**Standardized way for LLMs to use tools**

```
┌─────────────┐         ┌─────────────┐
│    LLM      │  MCP    │   Server    │
│             │◀───────▶│  (Tools)    │
│  "Read file"│         │  filesystem │
│             │         │  database   │
└─────────────┘         │  api        │
                        └─────────────┘
```

### Key MCP Servers

| Server | Purpose |
|--------|---------|
| `filesystem` | Read/write files |
| `sqlite` | Database queries |
| `github` | Repo operations |
| `brave-search` | Web search |
| `memory` | Persistent memory |

### Using MCP with Claude

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
    }
  }
}
```

---

## Tool Use Patterns

### Simple Function Calling

```python
tools = [{
    "name": "search",
    "description": "Search the web",
    "parameters": {
        "query": {"type": "string"}
    }
}]

response = client.chat(
    model="gpt-4",
    messages=[...],
    tools=tools
)
```

### ReAct Pattern

**Reason + Act iteratively**

```
Thought: I need to find the current weather
Action: search("weather in NYC")
Observation: 72°F, sunny
Thought: Now I have the answer
Action: respond("It's 72°F and sunny in NYC")
```

### Plan + Execute

**Upfront planning, then execution**

```
1. Plan: Break task into steps
2. Execute: Run each step
3. Replan: Adjust if needed
```

---

## Memory Systems

| Type | Duration | Use Case |
|------|----------|----------|
| **Buffer** | Session | Recent conversation |
| **Summary** | Session | Compressed context |
| **Vector** | Persistent | Semantic search |
| **Knowledge Graph** | Persistent | Relationships |

```python
# Vector memory with LangChain
from langchain.memory import VectorStoreRetrieverMemory

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever()
)
```

---

## Multi-Agent Patterns

### 1. Hierarchical

```
         ┌───────────────┐
         │   Supervisor  │
         │   Agent       │
         └───────┬───────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌───────┐   ┌───────┐   ┌───────┐
│Worker │   │Worker │   │Worker │
│   1   │   │   2   │   │   3   │
└───────┘   └───────┘   └───────┘
```

### 2. Collaborative

```
┌───────┐       ┌───────┐
│Agent A│◀─────▶│Agent B│
└───┬───┘       └───┬───┘
    │               │
    └───────┬───────┘
            ▼
       ┌───────┐
       │Agent C│
       └───────┘
```

### 3. Debate/Critique

```
┌──────────┐    propose    ┌──────────┐
│ Proposer │──────────────▶│  Critic  │
└──────────┘               └────┬─────┘
      ▲                         │
      │         critique        │
      └─────────────────────────┘
```

---

## Building Agents: Best Practices

1. **Start simple** - Single agent, few tools
2. **Clear tool descriptions** - LLMs read these
3. **Error handling** - Tools fail, plan for it
4. **Logging** - Track agent reasoning
5. **Human oversight** - For critical actions
6. **Rate limiting** - Prevent runaway agents
7. **Testing** - Unit test tools, integration test flows

---

## Common Pitfalls

| Problem | Solution |
|---------|----------|
| Agent loops forever | Max iterations, timeout |
| Wrong tool choice | Better descriptions |
| Lost context | Summarization, RAG |
| Expensive runs | Smaller model for routing |
| Hallucinated actions | Constrained action space |

---

## Labs

| Notebook | Focus |
|----------|-------|
| `01-tool-use-basics.ipynb` | Function calling |
| `02-langgraph-intro.ipynb` | Graph agents |
| `03-react-agent.ipynb` | ReAct pattern |
| `04-crewai-team.ipynb` | Multi-agent |
| `05-mcp-integration.ipynb` | MCP servers |
| `06-memory-systems.ipynb` | Persistent memory |

---

## Next Steps

- L1: [Agent Architectures](./concepts.md)
- L2: [LangGraph Deep Dive](./deep-dive.md)
- Related: [LLMs](../llms/README.md), [Tools](../tools-ecosystem/README.md)

---

*"The best agent is one you can trust to act on your behalf."*
