# Framework Decision Flowchart

Use this flowchart to choose the right agentic AI framework for your use case.

---

## Quick Decision Tree

```
                           START
                             │
                             ▼
          ┌─────────────────────────────────────┐
          │  Is production reliability critical? │
          └─────────────────────────────────────┘
                   │                    │
                  YES                   NO
                   │                    │
                   ▼                    ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  Need complex        │   │  Need quick          │
    │  branching/loops?    │   │  prototyping?        │
    └──────────────────────┘   └──────────────────────┘
         │           │              │           │
        YES          NO            YES          NO
         │           │              │           │
         ▼           ▼              ▼           ▼
    ┌────────┐  ┌────────┐    ┌────────┐  ┌────────┐
    │LANGGRAPH│ │LANGGRAPH│   │ CREWAI │  │ Dynamic│
    └────────┘  └────────┘    └────────┘  │ convo? │
                                          └────────┘
                                           │      │
                                          YES     NO
                                           │      │
                                           ▼      ▼
                                      ┌────────┐ ┌────────┐
                                      │AUTOGEN │ │ CREWAI │
                                      └────────┘ └────────┘
```

---

## Detailed Decision Matrix

### Step 1: Assess Your Requirements

Answer these questions with scores 1-5:

| Question | Score |
|----------|-------|
| How critical is fault tolerance? | __/5 |
| How complex is your workflow? | __/5 |
| How quickly do you need to ship? | __/5 |
| How important is observability? | __/5 |
| How dynamic are agent interactions? | __/5 |

### Step 2: Match Scores to Framework

**If fault tolerance > 4 AND complexity > 3:**
→ **LangGraph** (persistence, checkpointing, HITL)

**If time-to-ship > 4 AND complexity < 3:**
→ **CrewAI** (fastest setup, intuitive)

**If dynamic interactions > 4:**
→ **AutoGen** (conversational, flexible)

---

## Use Case → Framework Mapping

### Content & Marketing
```
Content Pipeline     → CrewAI (role-based, sequential)
SEO Research        → CrewAI (researcher + writer agents)
Social Media        → CrewAI (fast iteration)
```

### Customer Support
```
Simple FAQ Bot      → CrewAI (quick setup)
Complex Support     → LangGraph (state, persistence)
Enterprise Support  → LangGraph (Klarna uses it)
```

### Development & Code
```
Code Review         → AutoGen (code execution)
Test Generation     → LangGraph (Uber uses it)
Developer Tools     → AutoGen (coding copilots)
```

### Research & Analysis
```
Quick Research      → CrewAI (researcher agent)
Deep Research       → LangGraph (multi-step, branching)
Academic Research   → AutoGen (dynamic collaboration)
```

### Enterprise Workflows
```
Approval Flows      → LangGraph (HITL, persistence)
Compliance          → LangGraph (audit trail)
Azure Integration   → AutoGen (Microsoft ecosystem)
```

---

## Framework Strengths Radar

```
                 Ease of Use
                     5
                     │
                   4 │ ★ CrewAI
                   3 │   ★ AutoGen
                   2 │     ★ LangGraph
Production ────1─────┼─────1──── Speed
Readiness        2   │   2       to Ship
★ LangGraph      3   │   3       ★ CrewAI
★ AutoGen        4   │   4       ★ AutoGen
★ CrewAI         5   │   5       ★ LangGraph
                     │
               Flexibility
         ★ LangGraph = 5
         ★ AutoGen = 4
         ★ CrewAI = 3
```

---

## Red Flags: When NOT to Use

### Don't Use CrewAI If:
- You need complex conditional branching
- Fault tolerance is critical
- You need deep observability
- Workflow requires frequent human intervention

### Don't Use LangGraph If:
- You need to ship in days, not weeks
- Team lacks graph/state management experience
- Simple linear workflow
- No persistence requirements

### Don't Use AutoGen If:
- You need structured, repeatable workflows
- Built-in persistence is required
- Team unfamiliar with async Python
- Need maximum community support

---

## Team Expertise Consideration

```
Team Experience        Recommended Framework
─────────────────────────────────────────────
ML/AI Engineers        → LangGraph or AutoGen
Python Developers      → Any (start with CrewAI)
Non-Technical PM       → CrewAI (YAML config)
Microsoft Shop         → AutoGen
LangChain Users        → LangGraph
```

---

## Cost Considerations

| Framework | Open Source | Enterprise Option | Cloud Costs |
|-----------|-------------|-------------------|-------------|
| CrewAI | Yes (MIT) | CrewAI Enterprise | Variable |
| LangGraph | Yes (MIT) | LangSmith | Usage-based |
| AutoGen | Yes (MIT) | Azure integration | Azure pricing |

---

## Final Recommendation Algorithm

```python
def choose_framework(requirements: dict) -> str:
    """
    Requirements keys:
    - complexity: 1-5
    - production_critical: bool
    - time_to_ship: "days" | "weeks" | "months"
    - team_expertise: "beginner" | "intermediate" | "expert"
    - dynamic_conversations: bool
    - persistence_needed: bool
    """

    if requirements["production_critical"] and requirements["persistence_needed"]:
        return "LangGraph"

    if requirements["time_to_ship"] == "days":
        return "CrewAI"

    if requirements["dynamic_conversations"]:
        return "AutoGen"

    if requirements["complexity"] >= 4:
        return "LangGraph"

    if requirements["team_expertise"] == "beginner":
        return "CrewAI"

    # Default for balanced requirements
    return "CrewAI"  # Fastest to validate, then migrate if needed
```

---

## Migration Path

```
┌────────────┐
│  Start     │
│  with      │──────────────────────────────────────┐
│  CrewAI    │                                      │
└─────┬──────┘                                      │
      │                                             │
      │ (Outgrow simple workflows?)                 │
      ▼                                             │
┌─────────────────────────────────────┐            │
│ Need complex state/persistence?     │            │
└─────────────────────────────────────┘            │
      │                    │                        │
     YES                   NO                       │
      │                    │                        │
      ▼                    ▼                        │
┌────────────┐      ┌────────────┐                 │
│ Migrate to │      │ Need more  │                 │
│ LangGraph  │      │ dynamic?   │                 │
└────────────┘      └────────────┘                 │
                          │      │                  │
                         YES     NO                 │
                          │      │                  │
                          ▼      └──────────────────┘
                    ┌────────────┐
                    │ Try        │
                    │ AutoGen    │
                    └────────────┘
```

---

*Decision guide for Luno-AI Agentic AI Track*
