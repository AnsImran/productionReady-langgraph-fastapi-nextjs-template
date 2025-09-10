# # https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# # {message:2, branch:12}

# # simple poora agent hi compile kar raha hay,
# # compiled graph return kar raha hay!



# -----------------------------------------------------------------------------
# RAG Assistant (LangGraph)
# -----------------------------------------------------------------------------
# Global role in the codebase:
# - Defines a LangGraph agent that:
#     1) screens user input for safety (LlamaGuard),
#     2) calls a tool-enabled chat model (with a Chroma-backed `database_search` tool),
#     3) loops model → tools → model until no further tool calls,
#     4) blocks unsafe outputs and enforces a step budget.
# - The compiled runnable (`rag_assistant`) is registered by the agents package so
#   FastAPI (`service.py`) can invoke/stream it via /invoke and /stream endpoints.
# -----------------------------------------------------------------------------

# simple poora agent hi compile kar raha hay,
# compiled graph return kar raha hay!

from datetime import datetime
from typing import Literal

# LangChain core: chat model interface, message types, runnable composition primitives
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)

# LangGraph: graph building, message-carrying state, terminator token, managed step budget,
# and a prebuilt node to execute tool calls emitted by the model.
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

# Project-local: safety checker (LlamaGuard), retrieval tool (Chroma), model selector and settings
from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import database_search
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` marks fields added here as optional (PEP 589),
    while inherited fields (like `messages` from MessagesState) keep their original requirements.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    # Safety metadata from LlamaGuard (populated by guard nodes)
    safety: LlamaGuardOutput

    # LangGraph-managed remaining step budget for the current run
    remaining_steps: RemainingSteps


# Toolset exposed to the model (Chroma-based database search)
tools = [database_search]


# Stamp the system prompt with today's date (helps model responses reference "today")
current_date = datetime.now().strftime("%B %d, %Y")

# System prompt that defines the assistant persona, scope, and citation policy.
# NOTE: The model is instructed to ONLY use links returned by tools.
instructions = f"""
    You are AcmeBot, a helpful and knowledgeable virtual assistant designed to support employees by retrieving
    and answering questions based on AcmeTech's official Employee Handbook. Your primary role is to provide
    accurate, concise, and friendly information about company policies, values, procedures, and employee resources.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - If you have access to multiple databases, gather information from a diverse range of sources before crafting your response.
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Only use information from the database. Do not use information from outside sources.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    """Compose a runnable pipeline:
       AgentState -> [SystemMessage + state messages] -> tool-enabled model -> AIMessage
       Returned as RunnableSerializable for composability/tracing."""
    bound_model = model.bind_tools(tools)  # enable structured tool calling for `database_search`
    preprocessor = RunnableLambda(
        # Prepend the system prompt before the turn's messages
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    """Create a user-facing AIMessage describing unsafe categories when content is blocked."""
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    """Main model node:
       - Selects the concrete model (from config or default),
       - Runs the tool-enabled chat model,
       - Post-checks the output with LlamaGuard,
       - Enforces step budget if tool calls remain."""
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {
            "messages": [format_safety_message(safety_output)],
            "safety": safety_output,
        }

    # If we're nearly out of steps and the model wants to call tools, exit gracefully
    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    """Pre-generation safety check on user input; stores result in state."""
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    """If input is unsafe, emit a single blocking AIMessage and end."""
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------
# Nodes:
# - guard_input: run LlamaGuard on user input
# - model: call the LLM (tool-enabled)
# - tools: execute any tool calls emitted by the model
# - block_unsafe_content: return a block message if safety failed
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


# If unsafe → block_unsafe_content; else → model
agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

# Compile into a runnable graph that the service invokes/streams
rag_assistant = agent.compile()


















# from datetime import datetime
# from typing import Literal

# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.messages import AIMessage, SystemMessage
# from langchain_core.runnables import (
#     RunnableConfig,
#     RunnableLambda,
#     RunnableSerializable,
# )
# from langgraph.graph import END, MessagesState, StateGraph
# from langgraph.managed import RemainingSteps
# from langgraph.prebuilt import ToolNode

# from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
# from agents.tools import database_search
# from core import get_model, settings


# class AgentState(MessagesState, total=False):
#     """`total=False` is PEP589 specs.

#     documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
#     """

#     safety:          LlamaGuardOutput
#     remaining_steps: RemainingSteps


# tools = [database_search]


# current_date = datetime.now().strftime("%B %d, %Y")
# instructions = f"""
#     You are AcmeBot, a helpful and knowledgeable virtual assistant designed to support employees by retrieving
#     and answering questions based on AcmeTech's official Employee Handbook. Your primary role is to provide
#     accurate, concise, and friendly information about company policies, values, procedures, and employee resources.
#     Today's date is {current_date}.

#     NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

#     A few things to remember:
#     - If you have access to multiple databases, gather information from a diverse range of sources before crafting your response.
#     - Please include markdown-formatted links to any citations used in your response. Only include one
#     or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
#     - Only use information from the database. Do not use information from outside sources.
#     """


# def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
#     bound_model = model.bind_tools(tools)
#     preprocessor = RunnableLambda(
#         lambda state: [SystemMessage(content=instructions)] + state["messages"],
#         name="StateModifier",
#     )
#     return preprocessor | bound_model  # type: ignore[return-value]


# def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
#     content = (
#         f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
#     )
#     return AIMessage(content=content)


# async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
#     m              = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
#     model_runnable = wrap_model(m)
#     response = await model_runnable.ainvoke(state, config)

#     # Run llama guard check here to avoid returning the message if it's unsafe
#     llama_guard   = LlamaGuard()
#     safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
#     if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
#         return {
#             "messages": [format_safety_message(safety_output)],
#             "safety": safety_output,
#         }

#     if state["remaining_steps"] < 2 and response.tool_calls:
#         return {
#             "messages": [
#                 AIMessage(
#                     id=response.id,
#                     content="Sorry, need more steps to process this request.",
#                 )
#             ]
#         }
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}


# async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
#     llama_guard = LlamaGuard()
#     safety_output = await llama_guard.ainvoke("User", state["messages"])
#     return {"safety": safety_output, "messages": []}


# async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
#     safety: LlamaGuardOutput = state["safety"]
#     return {"messages": [format_safety_message(safety)]}


# # Define the graph
# agent = StateGraph(AgentState)
# agent.add_node("model", acall_model)
# agent.add_node("tools", ToolNode(tools))
# agent.add_node("guard_input", llama_guard_input)
# agent.add_node("block_unsafe_content", block_unsafe_content)
# agent.set_entry_point("guard_input")


# # Check for unsafe input and block further processing if found
# def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
#     safety: LlamaGuardOutput = state["safety"]
#     match safety.safety_assessment:
#         case SafetyAssessment.UNSAFE:
#             return "unsafe"
#         case _:
#             return "safe"


# agent.add_conditional_edges(
#     "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
# )

# # Always END after blocking unsafe content
# agent.add_edge("block_unsafe_content", END)

# # Always run "model" after "tools"
# agent.add_edge("tools", "model")


# # After "model", if there are tool calls, run "tools". Otherwise END.
# def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
#     last_message = state["messages"][-1]
#     if not isinstance(last_message, AIMessage):
#         raise TypeError(f"Expected AIMessage, got {type(last_message)}")
#     if last_message.tool_calls:
#         return "tools"
#     return "done"


# agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

# rag_assistant = agent.compile()
