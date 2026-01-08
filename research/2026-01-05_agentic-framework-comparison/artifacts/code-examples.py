"""
Agentic Framework Code Examples
==============================

Working examples for CrewAI, LangGraph, and AutoGen
implementing the same use case: Research + Writing Pipeline

Requirements:
    pip install crewai langgraph autogen-agentchat autogen-ext openai

"""

# =============================================================================
# CREWAI EXAMPLE
# =============================================================================

def crewai_example():
    """CrewAI: Role-based team for research and writing."""

    from crewai import Agent, Task, Crew, Process

    # Define agents with roles
    researcher = Agent(
        role='Senior Research Analyst',
        goal='Uncover comprehensive information about the given topic',
        backstory="""You are an expert researcher with years of experience
        in finding, analyzing, and synthesizing information from various sources.
        You excel at identifying key insights and trends.""",
        verbose=True,
        allow_delegation=False,
        # tools=[search_tool],  # Add your tools here
    )

    writer = Agent(
        role='Content Writer',
        goal='Create engaging and informative content based on research',
        backstory="""You are a skilled writer who transforms complex research
        into clear, engaging content. You have a talent for making technical
        topics accessible to a broad audience.""",
        verbose=True,
        allow_delegation=False,
    )

    # Define tasks
    research_task = Task(
        description="""Research the topic: {topic}

        Your task:
        1. Gather comprehensive information about the topic
        2. Identify key trends and insights
        3. Find relevant statistics and examples
        4. Note any controversies or debates

        Provide a detailed research summary.""",
        expected_output='A comprehensive research summary with key findings',
        agent=researcher,
    )

    writing_task = Task(
        description="""Based on the research provided, write an article about: {topic}

        Requirements:
        1. Clear introduction explaining the topic
        2. Main body with organized sections
        3. Practical examples or applications
        4. Conclusion with key takeaways

        Make it engaging and informative.""",
        expected_output='A well-structured article of 500-800 words',
        agent=writer,
        context=[research_task],  # This task depends on research
    )

    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

    # Execute
    result = crew.kickoff(inputs={'topic': 'Multi-Agent AI Systems'})
    return result


# =============================================================================
# LANGGRAPH EXAMPLE
# =============================================================================

def langgraph_example():
    """LangGraph: Graph-based workflow for research and writing."""

    from typing import TypedDict, Annotated
    from langgraph.graph import StateGraph, END
    from langchain_openai import ChatOpenAI

    # Define state schema
    class ResearchState(TypedDict):
        topic: str
        research: str
        article: str
        status: str

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    # Define nodes
    def research_node(state: ResearchState) -> ResearchState:
        """Conduct research on the topic."""
        prompt = f"""You are a senior research analyst.
        Research the following topic comprehensively: {state['topic']}

        Provide:
        1. Key findings and insights
        2. Relevant statistics
        3. Current trends
        4. Notable examples

        Be thorough and factual."""

        response = llm.invoke(prompt)
        return {
            "research": response.content,
            "status": "research_complete"
        }

    def writing_node(state: ResearchState) -> ResearchState:
        """Write article based on research."""
        prompt = f"""You are a skilled content writer.
        Based on this research: {state['research']}

        Write an engaging article about: {state['topic']}

        Requirements:
        - Clear introduction
        - Organized body sections
        - Practical examples
        - Strong conclusion

        Target: 500-800 words."""

        response = llm.invoke(prompt)
        return {
            "article": response.content,
            "status": "complete"
        }

    def should_continue(state: ResearchState) -> str:
        """Determine next step based on status."""
        if state.get("status") == "research_complete":
            return "writer"
        return END

    # Build graph
    workflow = StateGraph(ResearchState)

    # Add nodes
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", writing_node)

    # Add edges
    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges(
        "researcher",
        should_continue,
        {"writer": "writer", END: END}
    )
    workflow.add_edge("writer", END)

    # Compile (add checkpointer for persistence)
    # from langgraph.checkpoint.sqlite import SqliteSaver
    # memory = SqliteSaver.from_conn_string(":memory:")
    # app = workflow.compile(checkpointer=memory)

    app = workflow.compile()

    # Execute
    result = app.invoke({
        "topic": "Multi-Agent AI Systems",
        "research": "",
        "article": "",
        "status": "start"
    })

    return result


# =============================================================================
# AUTOGEN EXAMPLE
# =============================================================================

async def autogen_example():
    """AutoGen: Conversational agents for research and writing."""

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # Initialize model client
    model = OpenAIChatCompletionClient(model="gpt-4o")

    # Define agents
    researcher = AssistantAgent(
        name="researcher",
        model_client=model,
        system_message="""You are a Senior Research Analyst.
        Your role is to research topics thoroughly and provide comprehensive findings.
        Include key insights, statistics, trends, and examples.
        When you've completed your research, say 'RESEARCH COMPLETE' and summarize.""",
    )

    writer = AssistantAgent(
        name="writer",
        model_client=model,
        system_message="""You are a skilled Content Writer.
        Your role is to take research findings and create engaging articles.
        Wait for the researcher to complete their work before writing.
        When you've finished the article, say 'ARTICLE COMPLETE'.""",
    )

    reviewer = AssistantAgent(
        name="reviewer",
        model_client=model,
        system_message="""You are an Editor who reviews content quality.
        Provide brief feedback on the article.
        When satisfied, say 'APPROVED - TASK COMPLETE'.""",
    )

    # Create team with termination condition
    termination = TextMentionTermination("TASK COMPLETE")

    team = RoundRobinGroupChat(
        participants=[researcher, writer, reviewer],
        termination_condition=termination,
        max_turns=10,
    )

    # Execute
    result = await team.run(
        task="Research and write an article about Multi-Agent AI Systems"
    )

    return result


# =============================================================================
# COMPARISON: SAME TASK, THREE APPROACHES
# =============================================================================

def compare_frameworks():
    """
    Summary of key differences when implementing the same pipeline:

    CREWAI:
    - Pros: Intuitive, role-based, quick setup
    - Cons: Less control over execution flow
    - Best for: Business workflows, content pipelines

    LANGGRAPH:
    - Pros: Fine-grained control, persistence, branching
    - Cons: More complex setup, steeper learning curve
    - Best for: Production systems, complex workflows

    AUTOGEN:
    - Pros: Natural conversations, flexible collaboration
    - Cons: Less structured, harder to debug
    - Best for: Research, dynamic interactions

    Lines of Code Comparison:
    - CrewAI: ~50 lines
    - LangGraph: ~70 lines
    - AutoGen: ~60 lines

    Setup Time:
    - CrewAI: 10 minutes
    - LangGraph: 30 minutes
    - AutoGen: 20 minutes
    """
    pass


# =============================================================================
# ADVANCED PATTERNS
# =============================================================================

# CrewAI with Flows
def crewai_flows_example():
    """CrewAI Flows for event-driven orchestration."""

    from crewai.flow.flow import Flow, start, listen

    class ResearchFlow(Flow):
        @start()
        def initialize(self):
            """Starting point of the flow."""
            return {"topic": "AI Agents", "stage": "init"}

        @listen(initialize)
        def research_phase(self, data):
            """Conduct research."""
            # Execute research crew here
            return {"findings": "Research results...", **data}

        @listen(research_phase)
        def writing_phase(self, data):
            """Write based on research."""
            # Execute writing crew here
            return {"article": "Final article...", **data}

    flow = ResearchFlow()
    return flow.kickoff()


# LangGraph with Human-in-the-Loop
def langgraph_hitl_example():
    """LangGraph with human approval gate."""

    from langgraph.types import interrupt

    def approval_node(state):
        """Pause for human approval."""
        result = interrupt({
            "message": "Please review the research findings",
            "data": state["research"],
            "options": ["approve", "reject", "revise"]
        })

        if result["choice"] == "approve":
            return {"status": "approved"}
        elif result["choice"] == "revise":
            return {"status": "needs_revision", "feedback": result.get("feedback")}
        return {"status": "rejected"}

    # Add to graph workflow
    # workflow.add_node("approval", approval_node)


# AutoGen Swarm Pattern
async def autogen_swarm_example():
    """AutoGen Swarm with agent handoffs."""

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import Swarm
    from autogen_agentchat.messages import HandoffMessage

    # Agents can hand off to each other
    researcher = AssistantAgent(
        name="researcher",
        system_message="""Research the topic. When done, hand off to writer.""",
        handoffs=["writer"],
    )

    writer = AssistantAgent(
        name="writer",
        system_message="""Write the article. When done, hand off to reviewer.""",
        handoffs=["reviewer"],
    )

    reviewer = AssistantAgent(
        name="reviewer",
        system_message="""Review and approve. Say COMPLETE when done.""",
        handoffs=[],
    )

    # swarm = Swarm(participants=[researcher, writer, reviewer])
    # result = await swarm.run(task="Create article about AI agents")


# =============================================================================
# RUNNING THE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("CREWAI EXAMPLE")
    print("=" * 60)
    # Uncomment to run:
    # crewai_result = crewai_example()
    # print(crewai_result)

    print("\n" + "=" * 60)
    print("LANGGRAPH EXAMPLE")
    print("=" * 60)
    # Uncomment to run:
    # langgraph_result = langgraph_example()
    # print(langgraph_result)

    print("\n" + "=" * 60)
    print("AUTOGEN EXAMPLE")
    print("=" * 60)
    # Uncomment to run:
    # autogen_result = asyncio.run(autogen_example())
    # print(autogen_result)

    print("\nExamples ready! Uncomment the ones you want to run.")
    print("Make sure to set OPENAI_API_KEY environment variable.")
