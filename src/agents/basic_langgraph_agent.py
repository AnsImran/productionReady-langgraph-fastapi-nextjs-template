from langgraph.graph import END, MessagesState, StateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` marks fields added here as optional (PEP 589),
    while inherited fields (like `messages` from MessagesState) keep their original requirements.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """
    # # Safety metadata from LlamaGuard (populated by guard nodes)
    # safety: LlamaGuardOutput

    # # LangGraph-managed remaining step budget for the current run
    # remaining_steps: RemainingSteps


MODEL_SYSTEM_MESSAGE = """ You are a helpful chatbot on the website of an accounting firm.
Your role is to provide users with accurate information about the firm’s services.
The details of services provided by the accounting firm are as follows:

<accounting_services>
{accounting_services}
</accounting_services>
"""


accounting_services = 'null'


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Main model node:
       - Selects the concrete model (from config or default),
       - Runs the tool-enabled chat model,
       - Post-checks the output with LlamaGuard,
       - Enforces step budget if tool calls remain."""
    
    # can later decide to pace it outside the fn, if observed high latency in traces
    model      = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL)) 
    
    system_msg = MODEL_SYSTEM_MESSAGE.format(accounting_services=accounting_services)
    response   = await model.ainvoke(
                                        [SystemMessage(content=system_msg)] + state["messages"],
                                        config=config
                                    )
    
    return {"messages": [response]}



# -------------------------
# BUILD THE GRAPH
# -------------------------


# from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage


builder = StateGraph(AgentState)

builder.add_node("acall_model", acall_model)

builder.set_entry_point("acall_model")
builder.add_edge('acall_model', END)


# Compile the graph with persistent checkpointer and in-memory store
basic_langgraph_agent = builder.compile()#checkpointer=memory, store=across_thread_memory) #checkpointer

    