# basic_forced_retriever_agent.py

from __future__ import annotations

from typing import Any, Dict, List, Annotated, Optional
from uuid import uuid4
import json

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# --- Your project imports ---
from vector_databases import get_docs_pinecone, get_docs_timescale
from core import settings  # expects: settings.VEC_CLIENT, settings.PINECONE_VEC_CLIENT, settings.POSTGRES_TIMESCALE_VEC_CLIENT, settings.DEFAULT_EMBEDDING_MODEL
import openai
openai.api_type=openai

# from vector_databases import get_vec_client_timescale
# from memory import get_postgres_connection_string

# # create timescale_db_vec_client (async version)
# vec_client = get_vec_client_timescale(get_postgres_connection_string())


# --------------------------- Minimal shared state ---------------------------
class AgentState(MessagesState):
    question: str
    documents: Optional[List[Dict[str, Any]]]  # normalized docs
    retrieval_source: Optional[str]


# --------------------------- Helpers ---------------------------------------
def _normalize_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert Pinecone/Timescale/LC Document objects into a unified shape:
      {"id": str|None, "text": str|None, "metadata": dict}
    """
    out: List[Dict[str, Any]] = []
    for d in docs or []:
        text = getattr(d, "reranking_field", None)
        if text is None:
            text = getattr(d, "page_content", None)
        meta = getattr(d, "metadata", {}) or {}
        _id = getattr(d, "id", None) or meta.get("id") or meta.get("source_id")
        out.append({"id": _id, "text": text, "metadata": meta})
    return out




# --- helper ---
from typing import Optional
import json

def _format_docs_for_tool_message(
    docs: List[Dict[str, Any]],
    *,
    max_chars_per_doc: Optional[int] = None,   # None => no truncation
    include_metadata: bool = True,             # controls whether metadata is printed
) -> str:
    """
    Render docs for ToolMessage.content.

    For each doc:
      [i] <best source>
      <metadata as 1-line JSON>
      
      <text (optionally truncated)>
    """
    if not docs:
        return "No documents found."

    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        txt = (d.get("text") or "").strip()
        if max_chars_per_doc and len(txt) > max_chars_per_doc:
            txt = txt[:max_chars_per_doc].rstrip() + "…"

        meta = d.get("metadata") or {}
        # stringify metadata (single line, JSON)
        meta_str = json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str)

        src = meta.get("source") or meta.get("url") or meta.get("source_id") or d.get("id") or ""
        header = f"[{i}] {src}".strip() if src else f"[{i}]"

        if include_metadata:
            block = f"{header}\n{meta_str}\n\n{txt}" if txt else f"{header}\n{meta_str}"
        else:
            block = f"{header}\n{txt}" if txt else header

        parts.append(block)

    return "\n\n".join(parts)





def _to_text(content: Any) -> str:
    """Best-effort to turn LC message.content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    texts.append(str(part["text"]))
                elif "text" in part:
                    texts.append(str(part["text"]))
                elif "content" in part:
                    texts.append(str(part["content"]))
            else:
                texts.append(str(part))
        return "\n".join(t for t in texts if t)
    return str(content)


def _get_last_user_text(state: AgentState) -> str:
    """Pull the latest human message text from state.messages for streaming inputs."""
    msgs = state.get("messages") or []
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
            return (_to_text(msg.content) or "").strip()
    return ""


# --------------------------- Tools (write directly to state) ----------------




# --- tools (pinecone) ---
@tool("get_docs_pinecone")
async def tool_get_docs_pinecone(
    query:             Optional[str]  = None,
    # question:          Optional[str]  = None,
    top_k:             int            = 5,
    max_chars_per_doc: Optional[int]  = None,   # <-- new
    include_metadata:  bool           = True,             # <-- new
    config:            RunnableConfig = None,
    tool_call_id:      Annotated[str, InjectedToolCallId] = "",
) -> Command:

    """
    Retrieve top-k docs from Pinecone and store them in state.documents.
    Also appends a ToolMessage tied to the triggering tool_call_id with the docs text.
    """
    
    # q   = query #(query if query is not None else question) or ""
    pc  = (config or {}).get("configurable", {}).get("pc")
    idx = (config or {}).get("configurable", {}).get("idx")

    docs = await get_docs_pinecone(
        query,
        pc                   = pc,
        idx                  = idx,
        embedding_model_name = settings.DEFAULT_EMBEDDING_MODEL,
        max_results          = top_k,
    )
    normalized = _normalize_docs(docs)

    tool_msg = ToolMessage(
        content=_format_docs_for_tool_message(
            normalized,
            max_chars_per_doc = max_chars_per_doc,
            include_metadata  = include_metadata,
        ),
        tool_call_id = tool_call_id,
        name         = "get_docs_pinecone",
    )

    return Command(update={
        "messages":  [tool_msg],
        "documents": normalized,
        "retrieval_source": "pinecone",
        "question":  query,
    })







# --- tools (timescale) ---
@tool("get_docs_timescale")
async def tool_get_docs_timescale(
    query:             Optional[str]  = None,
    # question:          Optional[str]  = None,
    top_k:             int            = 5,
    max_chars_per_doc: Optional[int]  = None,   # <-- new
    include_metadata:  bool           = True,             # <-- new
    config:            RunnableConfig = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    Retrieve top-k docs from Timescale Vector and store them in state.documents.
    Also appends a ToolMessage tied to the triggering tool_call_id with the docs text.
    """


    # q          = query #(query if query is not None else question) or ""
    vec_client = (config or {}).get("configurable", {}).get("vec_client")

    docs = await get_docs_timescale(
        query,
        vec_client,
        embedding_model_name = settings.DEFAULT_EMBEDDING_MODEL,
        max_results          = top_k,
    )
    normalized = _normalize_docs(docs)

    tool_msg = ToolMessage(
        content=_format_docs_for_tool_message(
            normalized,
            max_chars_per_doc = max_chars_per_doc,     # None => full text
            include_metadata  = include_metadata,      # metadata on top
        ),
        tool_call_id = tool_call_id,
        name         = "get_docs_timescale",
    )

    return Command(update={
        "messages":  [tool_msg],
        "documents": normalized,
        "retrieval_source": "timescale",
        "question":  query,
    })



# --------------------------- Router: force a single tool call ---------------
async def route_vec_client(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    No model choice. We programmatically inject one tool_call
    based on settings.VEC_CLIENT. If 'question' is missing (astream case),
    derive it from the latest HumanMessage in state.messages.
    """
    # q = (state.get("question") or "").strip()
    # if not q:
    #     q = _get_last_user_text(state)

    call_id = f"call_{uuid4().hex}"
    top_k   = (config or {}).get("configurable", {}).get("top_k", 5)

    if settings.VEC_CLIENT == settings.PINECONE_VEC_CLIENT:
        tool_name = "get_docs_pinecone"
    elif settings.VEC_CLIENT == settings.POSTGRES_TIMESCALE_VEC_CLIENT:
        tool_name = "get_docs_timescale"
    else:
        # Return empty docs if misconfigured
        return {"documents": [], "retrieval_source": "invalid", "question": state['question']}

    top_k      = (config or {}).get("configurable", {}).get("top_k", 5)
    max_chars  = (config or {}).get("configurable", {}).get("max_chars_per_doc", None)
    inc_meta   = (config or {}).get("configurable", {}).get("include_metadata", True)

    manual_tool_call = AIMessage(
                                content    = "",
                                tool_calls = [{
                                    "name":  tool_name,
                                    "args":  {"query": state['question'], "top_k": top_k, "max_chars_per_doc": max_chars, "include_metadata": inc_meta},
                                    "id":    call_id
                                }],
    )
    return {"messages": [manual_tool_call], "question": state['question']}



# --- router: allow config control (optional) ---



# --------------------------- Build graph ------------------------------------
def build_agent():
    tools = [tool_get_docs_pinecone, tool_get_docs_timescale]
    
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    
    graph.set_entry_point("route_vec_client")
    graph.add_node("route_vec_client",     route_vec_client)
    graph.add_node("docs_retrieval_tool", tool_node)
    
    graph.add_edge("route_vec_client",     "docs_retrieval_tool")
    graph.add_edge("docs_retrieval_tool", END)
    
    return graph.compile()

# --------------------------- Minimal usage ----------------------------------
# Example:
prototype_rag_tool = build_agent()

# from vector_databases import get_vec_client_timescale
# from memory import get_postgres_connection_string

# # create timescale_db_vec_client (async version)
# vec_client = get_vec_client_timescale(get_postgres_connection_string())

# # Non-streaming:
# state = await prototype_rag_tool.ainvoke(
#     {"question": "paye?"},
#     config={"configurable": {"top_k": 5, "vec_client": vec_client}},
# )

# # Streaming callers can pass {"messages": [HumanMessage(...)]} and the router will
# # automatically extract the user text into 'question'/'query'.
