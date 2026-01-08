# Agentic AI: Projects & Comparisons

> **Hands-on projects and framework comparisons for AI Agents**

---

## Project Ideas

### Beginner Projects (L0-L1)

#### Project 1: Simple Tool-Using Agent
**Goal**: Agent that can search and calculate

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 2-3 hours |
| Technologies | LangGraph or OpenAI SDK |
| Skills | Tool definition, basic agents |

**Tasks**:
- [ ] Define calculator tool
- [ ] Define search tool (mock or real)
- [ ] Create agent with tools
- [ ] Test multi-step reasoning
- [ ] Add conversation memory

**Starter Code (LangGraph)**:
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def calculate(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for {query}: ..."

llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [calculate, search])

result = agent.invoke({"messages": [("user", "What is 25 * 4?")]})
```

---

#### Project 2: Web Scraper Agent
**Goal**: Agent that extracts info from websites

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐ Beginner |
| Time | 3-4 hours |
| Technologies | Agent + web tools |
| Skills | Tool creation, web scraping |

**Tasks**:
- [ ] Create URL fetcher tool
- [ ] Create content extractor tool
- [ ] Build agent with tools
- [ ] Extract specific information
- [ ] Handle errors gracefully

---

### Intermediate Projects (L2)

#### Project 3: Research Assistant
**Goal**: Agent that researches topics comprehensively

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | LangGraph + search APIs |
| Skills | Multi-step workflows |

**Tasks**:
- [ ] Query decomposition
- [ ] Parallel searches
- [ ] Source validation
- [ ] Information synthesis
- [ ] Report generation

**Architecture**:
```
Query → Planner → Searcher → Validator → Synthesizer → Report
           ↓          ↓
      Sub-queries   Multiple
                    Sources
```

---

#### Project 4: Code Review Agent
**Goal**: Automated code review with suggestions

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 6-8 hours |
| Technologies | Agent + file tools |
| Skills | Code analysis, tool chaining |

**Tasks**:
- [ ] Read file tool
- [ ] Analyze code structure
- [ ] Check for issues (security, style)
- [ ] Generate suggestions
- [ ] Create PR comment

---

#### Project 5: Personal Assistant with MCP
**Goal**: Agent connected to your tools via MCP

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐ Intermediate |
| Time | 4-6 hours |
| Technologies | MCP + LangGraph |
| Skills | MCP integration |

**Tasks**:
- [ ] Set up MCP filesystem server
- [ ] Connect agent to MCP
- [ ] Read/write files
- [ ] Integrate calendar (optional)
- [ ] Natural language file management

---

### Advanced Projects (L3-L4)

#### Project 6: Multi-Agent Debate System
**Goal**: Agents that debate and reach conclusions

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 1-2 days |
| Technologies | LangGraph or CrewAI |
| Skills | Multi-agent orchestration |

**Tasks**:
- [ ] Create debater agents (pro/con)
- [ ] Create moderator agent
- [ ] Implement turn-taking
- [ ] Track arguments
- [ ] Generate conclusion

**Architecture**:
```
Topic
  ↓
┌─────────────────────┐
│     Moderator       │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌───────┐    ┌───────┐
│ Pro   │◄──►│ Con   │
│ Agent │    │ Agent │
└───────┘    └───────┘
           │
           ▼
      Conclusion
```

---

#### Project 7: Autonomous Coding Agent
**Goal**: Agent that writes, tests, and fixes code

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | LangGraph + code tools |
| Skills | Code execution, iteration |

**Tasks**:
- [ ] Write code based on spec
- [ ] Execute in sandbox
- [ ] Check for errors
- [ ] Self-correct and retry
- [ ] Run tests
- [ ] Iterate until passing

---

#### Project 8: Production Agent System
**Goal**: Reliable, monitored agent deployment

| Aspect | Details |
|--------|---------|
| Difficulty | ⭐⭐⭐ Advanced |
| Time | 2-3 days |
| Technologies | LangGraph + LangSmith |
| Skills | Production patterns |

**Tasks**:
- [ ] Human-in-the-loop approval
- [ ] Checkpointing (persistence)
- [ ] Error recovery
- [ ] Tracing and monitoring
- [ ] Rate limiting
- [ ] Cost tracking

---

## Framework Comparisons

### Comparison 1: Agent Framework Showdown

**Question**: Which framework for your use case?

| Framework | Approach | Best For | Learning Curve | Production |
|-----------|----------|----------|----------------|------------|
| **LangGraph** | Graph-based | Complex workflows | Medium | ⭐⭐⭐⭐⭐ |
| **CrewAI** | Role-based | Team prototypes | Easy | ⭐⭐⭐ |
| **AutoGen** | Conversational | Research | Medium | ⭐⭐⭐ |
| **OpenAI SDK** | Minimal | Simple agents | Easy | ⭐⭐⭐⭐ |
| **Swarm** | Handoffs | Learning | Easy | ⭐⭐ |

**Lab Exercise**: Build same agent in all 4 frameworks.

```python
# Same task in different frameworks

# --- LangGraph ---
from langgraph.prebuilt import create_react_agent
agent_lg = create_react_agent(llm, tools)

# --- CrewAI ---
from crewai import Agent, Task, Crew
agent_crew = Agent(role="Assistant", tools=tools)

# --- OpenAI SDK ---
from openai_agents import Agent
agent_oai = Agent(model="gpt-4o-mini", tools=tools)
```

---

### Comparison 2: CrewAI vs LangGraph Deep Dive

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| **Philosophy** | Role-based teams | Graph workflows |
| **Setup** | Minutes | Hours |
| **Customization** | Limited | Full control |
| **Debugging** | Hard | Built-in tracing |
| **Scale** | Hits wall at 6-12 months | Production-ready |
| **Persistence** | Limited | PostgreSQL, Redis |
| **Human-in-loop** | Basic | Full support |

**Lab Exercise**: Build research agent in both, compare complexity.

---

### Comparison 3: Tool Standards

**Question**: How to connect agents to tools?

| Standard | Adoption | Features | Best For |
|----------|----------|----------|----------|
| **MCP** | High (Anthropic) | Universal, open | Cross-platform |
| **Function Calling** | High (OpenAI) | JSON schema | OpenAI ecosystem |
| **LangChain Tools** | High | Large ecosystem | LangChain/Graph |
| **Custom** | - | Full control | Specific needs |

**Lab Exercise**: Connect same tool via MCP and function calling.

---

### Comparison 4: Memory Systems

**Question**: How should agents remember?

| Type | Duration | Use Case | Implementation |
|------|----------|----------|----------------|
| **Buffer** | Session | Conversation | List of messages |
| **Summary** | Session | Long convos | Summarize + forget |
| **Vector** | Persistent | Knowledge | Vector DB |
| **Entity** | Persistent | Facts | Knowledge graph |
| **Episodic** | Persistent | Events | MemGPT pattern |

**Lab Exercise**: Implement all 4 memory types, compare behavior.

---

## Hands-On Labs

### Lab 1: Basic ReAct Agent (2 hours)
```
Define Tools → Create Agent → Test Reasoning → Add Memory
```

### Lab 2: Multi-Tool Agent (4 hours)
```
Web Search → Calculator → File Access → Chain Together
```

### Lab 3: LangGraph Workflow (4 hours)
```
Design Graph → Add Nodes → Conditional Edges → Run
```

### Lab 4: CrewAI Team (3 hours)
```
Define Roles → Create Agents → Assign Tasks → Execute Crew
```

### Lab 5: MCP Integration (4 hours)
```
Start MCP Server → Connect Client → Use Tools → Build Agent
```

### Lab 6: Framework Comparison (6 hours)
```
Same Task → LangGraph → CrewAI → OpenAI SDK → Compare
```

---

## Agent Design Patterns

### Pattern 1: ReAct (Reasoning + Acting)
```
Thought → Action → Observation → Thought → Action → Answer
```

### Pattern 2: Plan & Execute
```
Plan all steps → Execute step 1 → Execute step 2 → ... → Done
```

### Pattern 3: Reflexion
```
Attempt → Evaluate → Reflect on errors → Retry with learning
```

### Pattern 4: Supervisor
```
Supervisor receives task → Delegates to specialists → Aggregates results
```

### Pattern 5: Debate
```
Agent A argues → Agent B counters → Moderator synthesizes
```

---

## Assessment Rubric

| Criteria | Points | Description |
|----------|--------|-------------|
| **Functionality** | 35 | Does agent complete tasks? |
| **Reliability** | 25 | Handles errors, retries? |
| **Architecture** | 20 | Clean design, separation |
| **Documentation** | 10 | Clear instructions |
| **Innovation** | 10 | Creative solutions |

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [CrewAI Docs](https://docs.crewai.com/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [AutoGen](https://microsoft.github.io/autogen/)

---

*Part of [Luno-AI](../../../README.md) | Agentic AI Track*
