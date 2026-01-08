# Multi-Agent Systems Integration

> **Build systems where multiple AI agents collaborate**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Multiple specialized agents working together |
| **Why** | Complex tasks, division of labor, better results |
| **Patterns** | Hierarchical, peer-to-peer, swarm |
| **Frameworks** | LangGraph, CrewAI, AutoGen, custom |

### Architecture Patterns

```
Hierarchical              Peer-to-Peer              Swarm
┌─────────┐              ┌─────────┐              ┌───┐ ┌───┐
│ Manager │              │ Agent A │◄────────────▶│ A │ │ B │
└────┬────┘              └────┬────┘              └─┬─┘ └─┬─┘
     │                        │                     │     │
┌────┴────┐              ┌────┴────┐              ┌─┴─┐ ┌─┴─┐
│         │              │         │              │ C │ │ D │
▼         ▼              ▼         ▼              └───┘ └───┘
Agent A   Agent B       Agent B   Agent C         All connected
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **LLM** | OpenAI API or local models |
| **Python** | 3.10+ |
| **Understanding** | Basic agent concepts |

---

## Quick Start (30 min)

### LangGraph Multi-Agent

```bash
pip install langgraph langchain-openai
```

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str

# Define agents
researcher = ChatOpenAI(model="gpt-4o-mini").bind(
    system="You are a research agent. Find information."
)
writer = ChatOpenAI(model="gpt-4o-mini").bind(
    system="You are a writer. Create content from research."
)
reviewer = ChatOpenAI(model="gpt-4o-mini").bind(
    system="You are a reviewer. Check quality and accuracy."
)

def research_node(state: AgentState):
    response = researcher.invoke(state["messages"])
    return {"messages": [response], "next_agent": "writer"}

def writer_node(state: AgentState):
    response = writer.invoke(state["messages"])
    return {"messages": [response], "next_agent": "reviewer"}

def reviewer_node(state: AgentState):
    response = reviewer.invoke(state["messages"])
    return {"messages": [response], "next_agent": "end"}

def route(state: AgentState):
    return state["next_agent"]

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("researcher", research_node)
workflow.add_node("writer", writer_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("researcher")
workflow.add_conditional_edges("researcher", route, {"writer": "writer"})
workflow.add_conditional_edges("writer", route, {"reviewer": "reviewer"})
workflow.add_conditional_edges("reviewer", route, {"end": END})

graph = workflow.compile()

# Run
result = graph.invoke({
    "messages": [{"role": "user", "content": "Write about quantum computing"}],
    "next_agent": "researcher"
})
```

---

## Learning Path

### L0: Basic Multi-Agent (2-3 hours)
- [ ] Understand agent roles
- [ ] Build 2-agent system
- [ ] Implement handoffs
- [ ] Test collaboration

### L1: Advanced Patterns (4-6 hours)
- [ ] Hierarchical systems
- [ ] Dynamic routing
- [ ] Shared memory
- [ ] Error handling

### L2: Production Systems (1-2 days)
- [ ] Scalable architecture
- [ ] Monitoring
- [ ] Human-in-the-loop
- [ ] Cost optimization

---

## Code Examples

### CrewAI Team

```python
from crewai import Agent, Task, Crew

# Define specialized agents
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, comprehensive information",
    backstory="Expert at finding and synthesizing information",
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging, well-structured content",
    backstory="Skilled writer with attention to detail",
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Ensure quality and accuracy",
    backstory="Experienced editor with high standards",
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the topic: {topic}",
    agent=researcher,
    expected_output="Comprehensive research findings"
)

writing_task = Task(
    description="Write an article based on the research",
    agent=writer,
    expected_output="Well-written article",
    context=[research_task]
)

editing_task = Task(
    description="Review and improve the article",
    agent=editor,
    expected_output="Polished, publication-ready article",
    context=[writing_task]
)

# Create crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "AI in healthcare"})
```

### Supervisor Pattern

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal

class State(TypedDict):
    messages: list
    next: str

supervisor = ChatOpenAI(model="gpt-4o").bind(
    system="""You are a supervisor managing a team:
    - researcher: for finding information
    - coder: for writing code
    - writer: for creating content

    Based on the task, decide which agent should act next.
    Respond with just the agent name or 'FINISH' if done."""
)

def supervisor_node(state: State):
    response = supervisor.invoke(state["messages"])
    next_agent = response.content.strip().lower()
    return {"next": next_agent}

def researcher_node(state: State):
    # Research implementation
    return {"messages": state["messages"] + [{"role": "assistant", "content": "Research done"}]}

def coder_node(state: State):
    # Coding implementation
    return {"messages": state["messages"] + [{"role": "assistant", "content": "Code written"}]}

def writer_node(state: State):
    # Writing implementation
    return {"messages": state["messages"] + [{"role": "assistant", "content": "Content created"}]}

def route(state: State) -> Literal["researcher", "coder", "writer", "end"]:
    if state["next"] == "finish":
        return "end"
    return state["next"]

# Build graph
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route, {
    "researcher": "researcher",
    "coder": "coder",
    "writer": "writer",
    "end": END
})

# After each worker, go back to supervisor
for worker in ["researcher", "coder", "writer"]:
    workflow.add_edge(worker, "supervisor")

graph = workflow.compile()
```

### Shared Memory System

```python
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class SharedMemory:
    def __init__(self):
        self.short_term = ConversationBufferMemory()
        self.long_term = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory="./agent_memory"
        )
        self.task_results = {}

    def add_message(self, agent: str, message: str):
        self.short_term.save_context(
            {"agent": agent},
            {"output": message}
        )

    def store_knowledge(self, content: str, metadata: dict):
        self.long_term.add_texts([content], metadatas=[metadata])

    def retrieve_relevant(self, query: str, k: int = 5):
        return self.long_term.similarity_search(query, k=k)

    def set_task_result(self, task_id: str, result: dict):
        self.task_results[task_id] = result

    def get_task_result(self, task_id: str):
        return self.task_results.get(task_id)

# Usage in agents
memory = SharedMemory()

def agent_with_memory(agent_name: str, task: str):
    # Retrieve relevant context
    context = memory.retrieve_relevant(task)

    # Agent processes task with context
    result = f"Processed: {task}"

    # Store result
    memory.add_message(agent_name, result)
    memory.store_knowledge(result, {"agent": agent_name, "task": task})

    return result
```

### Dynamic Agent Creation

```python
from langchain_openai import ChatOpenAI

class AgentFactory:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini")

    def create_agent(self, role: str, skills: list):
        system_prompt = f"""You are a {role} with the following skills:
        {', '.join(skills)}

        Use your expertise to help with tasks in your domain."""

        return self.llm.bind(system=system_prompt)

    def create_team(self, task_description: str):
        # Analyze task and create appropriate agents
        planner = self.llm.bind(
            system="Analyze tasks and determine what roles are needed."
        )

        response = planner.invoke(
            f"What roles are needed for: {task_description}"
        )

        # Parse and create agents dynamically
        roles = self._parse_roles(response.content)
        return [self.create_agent(role, skills) for role, skills in roles.items()]

    def _parse_roles(self, response: str):
        # Parse LLM response into roles and skills
        return {
            "researcher": ["web search", "analysis"],
            "developer": ["coding", "debugging"]
        }
```

---

## Design Patterns

| Pattern | Use Case | Pros | Cons |
|---------|----------|------|------|
| Hierarchical | Complex workflows | Clear control | Bottleneck |
| Peer-to-Peer | Collaborative | Flexible | Complex coordination |
| Swarm | Simple tasks | Scalable | Chaotic |
| Pipeline | Sequential | Predictable | Rigid |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Agents loop | Add max iterations, clear exit conditions |
| Poor coordination | Add shared memory, clearer roles |
| High latency | Parallelize independent tasks |
| Inconsistent output | Standardize message formats |

---

## Resources

- [LangGraph Multi-Agent](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [CrewAI Docs](https://docs.crewai.com/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [Multi-Agent Patterns](https://www.anthropic.com/research/building-effective-agents)

---

*Part of [Luno-AI](../../README.md) | Agentic AI Track*
