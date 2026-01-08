# CrewAI Integration

> **Role-based multi-agent framework for rapid prototyping**

---

## Overview

| Aspect | Details |
|--------|---------|
| **What** | Framework for creating teams of AI agents with defined roles |
| **Why** | Fastest way to prototype multi-agent systems |
| **Best For** | Quick demos, role-based workflows, team simulations |
| **Trade-off** | Less control than LangGraph, hits scaling limits |

### CrewAI vs LangGraph

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| Setup time | Minutes | Hours |
| Customization | Limited | Full control |
| Production | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Learning curve | Easy | Medium |
| Best for | Prototypes | Production |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.10+ |
| **API Key** | OpenAI or other LLM provider |
| **Knowledge** | Basic Python, LLM concepts |

---

## Quick Start (15 min)

```bash
pip install crewai crewai-tools
```

```python
from crewai import Agent, Task, Crew, LLM

# Define LLM
llm = LLM(model="gpt-4o-mini")

# Create agents with roles
researcher = Agent(
    role="Senior Researcher",
    goal="Find comprehensive information on given topics",
    backstory="Expert at finding and synthesizing information",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear, engaging content from research",
    backstory="Skilled at making complex topics accessible",
    llm=llm,
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest developments in {topic}",
    expected_output="Detailed research notes with key findings",
    agent=researcher
)

writing_task = Task(
    description="Write a blog post based on the research",
    expected_output="A 500-word blog post",
    agent=writer,
    context=[research_task]  # Depends on research
)

# Create crew and run
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": "AI agents in 2025"})
print(result)
```

---

## Learning Path

### L0: Basic Crew (1 hour)
- [ ] Install CrewAI
- [ ] Create 2 agents with roles
- [ ] Define sequential tasks
- [ ] Run your first crew

### L1: Add Tools (2-3 hours)
- [ ] Add web search tool
- [ ] Add file reading tool
- [ ] Create custom tools
- [ ] Handle tool errors

### L2: Advanced Patterns (4-6 hours)
- [ ] Hierarchical crews (manager + workers)
- [ ] Parallel task execution
- [ ] Memory and context sharing
- [ ] Custom LLM providers

### L3: Production (1 day)
- [ ] Error handling and retries
- [ ] Logging and monitoring
- [ ] Async execution
- [ ] Integration with other systems

---

## Code Examples

### Adding Tools

```python
from crewai_tools import SerperDevTool, FileReadTool

search_tool = SerperDevTool()  # Web search
file_tool = FileReadTool()      # Read files

researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert researcher",
    tools=[search_tool, file_tool],
    llm=llm
)
```

### Hierarchical Crew

```python
from crewai import Process

manager = Agent(
    role="Project Manager",
    goal="Coordinate team to complete projects",
    backstory="Experienced manager",
    llm=llm,
    allow_delegation=True
)

crew = Crew(
    agents=[manager, researcher, writer],
    tasks=[main_task],
    process=Process.hierarchical,
    manager_agent=manager
)
```

### Custom Tool

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Evaluates mathematical expressions"
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"
```

---

## Integration Points

| Integration | Method |
|-------------|--------|
| **OpenAI** | Native support |
| **Ollama** | `LLM(model="ollama/llama3.2")` |
| **Claude** | `LLM(model="anthropic/claude-3-sonnet")` |
| **LangChain Tools** | Wrap with `crewai.tools` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Agents looping | Add clearer goals, use `max_iter` |
| Tasks failing | Check tool permissions, add retries |
| Slow execution | Use faster models, reduce verbosity |
| Memory issues | Limit context, use summarization |

---

## Resources

- [CrewAI Docs](https://docs.crewai.com/)
- [CrewAI GitHub](https://github.com/joaomdmoura/crewAI)
- [Examples](https://github.com/joaomdmoura/crewAI-examples)

---

*Part of [Luno-AI](../../README.md) | Agentic AI Track*
