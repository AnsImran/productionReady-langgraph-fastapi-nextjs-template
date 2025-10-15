from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel


from agents.chatbot import chatbot
from agents.basic_langgraph_agent import basic_langgraph_agent
from agents.self_corrective_rag import self_corrective_rag

from schema import AgentInfo

DEFAULT_AGENT = "basic_langgraph_agent"

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
    "basic_langgraph_agent":Agent(
        description = "A very basic langgraph based chatbot.",
        graph       = basic_langgraph_agent
    ),
    "self_corrective_rag":Agent(
        description = "A self corrective RAG agent.",
        graph       = self_corrective_rag),
}

# returns agent object
def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


# just info not Agent objects
def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
