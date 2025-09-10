from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.interrupt_agent import interrupt_agent
from agents.knowledge_base_agent import kb_agent
from agents.langgraph_supervisor_agent import langgraph_supervisor_agent
from agents.rag_assistant import rag_assistant
from agents.research_assistant import research_assistant
from agents.basic_langgraph_agent import basic_langgraph_agent
from agents.self_corrective_rag import self_corrective_rag
from agents.prototype_rag_tool import prototype_rag_tool

from schema import AgentInfo

DEFAULT_AGENT = "research-assistant"

# Type alias to handle LangGraph's different agent patterns
# - @entrypoint functions return Pregel
# - StateGraph().compile() returns CompiledStateGraph
AgentGraph = CompiledStateGraph | Pregel


@dataclass
class Agent:
    description: str
    graph:       AgentGraph


agents: dict[str, Agent] = {
    "chatbot":Agent(
        description = "A simple chatbot.",
        graph       = chatbot
    ),
    "research-assistant":Agent(
        description = "A research assistant with web search and calculator.",
        graph       = research_assistant
    ),
    "rag-assistant":Agent(
        description = "A RAG assistant with access to information in a database.",
        graph       = rag_assistant
    ),
    "command-agent":Agent(
        description = "A command agent.",
        graph       = command_agent),
    "bg-task-agent":Agent(
        description = "A background task agent.",
        graph       = bg_task_agent
    ),
    "langgraph-supervisor-agent":Agent(
        description = "A langgraph supervisor agent",
        graph       = langgraph_supervisor_agent
    ),
    "interrupt-agent":Agent(
        description = "An agent the uses interrupts.",
        graph       = interrupt_agent
    ),
    "knowledge-base-agent":Agent(
        description = "A RAG agent using Amazon Bedrock Knowledge Base",
        graph       = kb_agent
    ),
    "basic_langgraph_agent":Agent(
        description = "A very basic langgraph based chatbot.",
        graph       = basic_langgraph_agent
    ),
    "self_corrective_rag":Agent(
        description = "A self corrective RAG agent.",
        graph       = self_corrective_rag),
    "prototype_rag_tool": Agent(
        description = "prototype rag agent with tool nodes",
        graph       = prototype_rag_tool
    ),
        
}

# returns agent object
def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


# just info not Agent objects
def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
