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







from typing import Literal
from langchain_core.messages import AIMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, MessagesState
import openai
import re
from pinecone.db_data.index_asyncio import _IndexAsyncio
from pinecone.pinecone_asyncio import PineconeAsyncio
from timescale_vector import client
from langfuse import Langfuse, get_client
from langchain_core.runnables import RunnableConfig
from timescale_vector import client


from vector_databases import (
                                get_docs_timescale,
                                get_docs_pinecone
)
from core import get_model, settings


from memory import get_postgres_connection_string

MAIN_AGENT_DB_URI = get_postgres_connection_string()
TIMESCALE_DB_URI  = get_postgres_connection_string()
# vec_client = vec_client_timescale(TIMESCALE_DB_URI)



import openai
openai.api_type = "openai"  ##################################################################################################
max_results     = 5

# yahan if lagao k agar vec client name postgres ho to vo vala load otherwose 2sray vaa load.
vec_client_name = settings.VEC_CLIENT
# import os
# from dotenv import load_dotenv

# load_dotenv()  # Load environment variables from a .env file

if settings.LANGFUSE_PUBLIC_KEY:
    langfuse_public_key = settings.LANGFUSE_PUBLIC_KEY #os.environ["langfuse_public_key"]
    langfuse_secret_key = settings.LANGFUSE_SECRET_KEY #os.environ["langfuse_secret_key"]
    langfuse_host       = settings.LANGFUSE_HOST       #os.environ["langfuse_host"]
    
    langfuse = Langfuse(
      secret_key = langfuse_secret_key.get_secret_value(),
      public_key = langfuse_public_key.get_secret_value(),
      host       = langfuse_host
    )
    
    langfuse_cl = get_client(public_key=langfuse_public_key.get_secret_value())




################################################################################################################
## Define Pydantic Schemas for Grading
class GradeRelevance(BaseModel):
    binary_score: str = Field(description="Question is related to accounting, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

class Input_State(BaseModel):
    question: str = Field(description="Question asked by the user.")




################################################################################################################
## Define Prompts
RELEVANCE_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether the user's query/question is related to accounting/accountance or accounting firm. Give a binary score 'yes' or 'no'."),
    ("human", "The query/question of user: \n {query}")
])

# langfuse_RELEVANCE_GRADER_PROMPT = langfuse_cl.get_prompt(
#     "RELEVANCE_GRADER_PROMPT",
#     type="chat"
# )

# RELEVANCE_GRADER_PROMPT = ChatPromptTemplate(
#     langfuse_RELEVANCE_GRADER_PROMPT.get_langchain_prompt(),
#     metadata={"langfuse_prompt": langfuse_RELEVANCE_GRADER_PROMPT}  # exactly like that for linked generation
# )

HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an LLM generation is grounded in a set of retrieved facts. Give a binary score 'yes' or 'no'."),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])

ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer resolves a question. Give a binary score 'yes' or 'no'."),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
])

QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval."),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question.")
])

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot assistant on the website of an accounting firm. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say so. Keep it concise."),
    ("human", "\nQuestion: {question} \nContext: {context} \nAnswer:")
])


# # Load prompts from Langfuse
# langfuse_HALLUCINATION_GRADER_PROMPT = langfuse_cl.get_prompt(
#     "HALLUCINATION_GRADER_PROMPT",
#     type="chat"
# )
# HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate(
#     langfuse_HALLUCINATION_GRADER_PROMPT.get_langchain_prompt(),
#     metadata={"langfuse_prompt": langfuse_HALLUCINATION_GRADER_PROMPT}
# )

# langfuse_ANSWER_GRADER_PROMPT = langfuse_cl.get_prompt(
#     "ANSWER_GRADER_PROMPT",
#     type="chat"
# )
# ANSWER_GRADER_PROMPT = ChatPromptTemplate(
#     langfuse_ANSWER_GRADER_PROMPT.get_langchain_prompt(),
#     metadata={"langfuse_prompt": langfuse_ANSWER_GRADER_PROMPT}
# )

# langfuse_QUERY_REWRITER_PROMPT = langfuse_cl.get_prompt(
#     "QUERY_REWRITER_PROMPT",
#     type="chat"
# )
# QUERY_REWRITER_PROMPT = ChatPromptTemplate(
#     langfuse_QUERY_REWRITER_PROMPT.get_langchain_prompt(),
#     metadata={"langfuse_prompt": langfuse_QUERY_REWRITER_PROMPT}
# )

# langfuse_RAG_PROMPT = langfuse_cl.get_prompt(
#     "RAG_PROMPT",
#     type="chat"
# )
# RAG_PROMPT = ChatPromptTemplate(
#     langfuse_RAG_PROMPT.get_langchain_prompt(),
#     metadata={"langfuse_prompt": langfuse_RAG_PROMPT}
# )





################################################################################################################
## Constants
MAX_RETRIES = 3
VERBOSE = True



#         vec_client_name:      str | None             = None,
#         vec_client:           client.Async | None    = None,
#         pc:                   PineconeAsyncio | None = None,
#         idx:                  _IndexAsyncio | None   = None,
#         embedding_model_name: str                    = "text-embedding-3-small",
#         max_results:          int                    = 5,
# ):
#     """
#     Build and return a Retrieval-Augmented Generation (RAG) agent with self-corrective capabilities.

#     Parameters
#     ----------
#     vec_client_name : str, optional
#         Source of the vector store. Use ``"pinecone"`` or ``"postgres_timescale"``.
#     vec_client : client.Async, optional
#         Asynchronous client instance that connects to the chosen vector store.
#     pc : PineconeAsyncio, optional
#         Initialized asynchronous Pinecone client used when ``vec_client_name`` is
#         ``"pinecone"``.
#     idx : _IndexAsyncio, optional
#         Handle to an existing asynchronous index (e.g., a Pinecone index) that the
#         agent should query.
#     embedding_model_name : str, default "text-embedding-3-small"
#         Name of the embedding model used to encode text before vector search.
#     max_results : int, default 5
#         Maximum number of context chunks to retrieve for each user query.

#     Returns
#     -------
#     RAGAgent
#         A fully configured RAG agent that:
#           - retrieves up to ``max_results`` relevant context passages from the specified vector store,
#           - uses a lightweight evaluator to self-grade and filter retrieved content,
#           - triggers additional retrieval (e.g., web search) or query reformulation when relevance falls below threshold,
#           - and generates a final response grounded in the most accurate and refined context, using the OpenAI Chat Completion API.
#     """


#     ################################################################################################################
## Define Graph State and Configuration
class AgentState(MessagesState):
    question:         str
    # documents:        list[Document]
    documents:        Optional[List[Dict[str, Any]]]  # normalized docs
    candidate_answer: str
    retries:          int
    retrieval_source: Optional[str]

class GraphConfig(BaseModel):
    max_retries: int = MAX_RETRIES

 




################################################################################################################
## Query Relevance
async def query_relevance(state: AgentState, config: RunnableConfig) -> AgentState:
    question = convert_to_messages(state["messages"])[-1].content
    return {"question": question}


## isay history deni ho tu simply last 3,4 messages ko keys str main convert kr k, pass kr dena isay ...
## yani last 4 messages + append the current question and that's it!
## yani question vala string hi fn k andr update ho jaye ga ...
async def query_relevance_router(state: Input_State, config: RunnableConfig) -> Literal["route_vec_client", "cant_help"]:
    question = state.question
    if VERBOSE:
        print("---CHECK RELEVANCE---")

    # Initialize the model
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    grader = (RELEVANCE_GRADER_PROMPT | llm.with_structured_output(GradeRelevance)).with_config(tags=["skip_stream"])
    relevance_grade: GradeRelevance = await grader.ainvoke({"query": question})

    if relevance_grade.binary_score == "yes":
        if VERBOSE: print("---DECISION: QUERY/QUESTION <IS RELATED> TO ACCOUNTING---")
        return "route_vec_client"
    else:
        if VERBOSE: print("---DECISION: QUERY/QUESTION <IS NOT RELATED> TO ACCOUNTING---")
        return "cant_help"


################################################################################################################
################################################################################################################

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

    call_id = f"call_{uuid4().hex}"
    top_k   = (config or {}).get("configurable", {}).get("top_k", 5)

    if settings.VEC_CLIENT == settings.PINECONE_VEC_CLIENT:
        tool_name = "get_docs_pinecone"
    elif settings.VEC_CLIENT == settings.POSTGRES_TIMESCALE_VEC_CLIENT:
        tool_name = "get_docs_timescale"
    else:
        # Return empty docs if misconfigured
        return {"documents": [], "retrieval_source": "invalid", "question": state["question"]}

    top_k      = (config or {}).get("configurable", {}).get("top_k", 5)
    max_chars  = (config or {}).get("configurable", {}).get("max_chars_per_doc", None)
    inc_meta   = (config or {}).get("configurable", {}).get("include_metadata", True)

    manual_tool_call = AIMessage(
                                content    = "",
                                tool_calls = [{
                                    "name":  tool_name,
                                    "args":  {"query": state["question"], "top_k": top_k, "max_chars_per_doc": max_chars, "include_metadata": inc_meta},
                                    "id":    call_id
                                }],
    )
    return {"messages": [manual_tool_call], "question": state["question"]}
            
################################################################################################################
################################################################################################################
## Answer Generation
async def generate(state: AgentState, config: RunnableConfig) -> AgentState:
    if VERBOSE:
        print("---GENERATE---")

    # Initialize the model
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    rag_chain = RAG_PROMPT | llm | StrOutputParser()
    generation = await rag_chain.ainvoke({"context": state["documents"], "question": state["question"]})
    return {"retries": state.get("retries", 0) + 1, "candidate_answer": generation}


################################################################################################################
## Answer Grading
async def grade_generation_v_documents_and_question(state: AgentState, config: RunnableConfig) -> Literal["generate", "transform_query", "cant_help", "finalize_response"]:
    if VERBOSE:
        print("---CHECK HALLUCINATIONS---")

    # Initialize the model
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    halluc_grader = (HALLUCINATION_GRADER_PROMPT | llm.with_structured_output(GradeHallucinations)).with_config(tags=["skip_stream"])
    hall_result = await halluc_grader.ainvoke({"documents": state["documents"], "generation": state["candidate_answer"]})

    if hall_result.binary_score == "no":
        if VERBOSE: print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "generate" if state.get("retries", 0) < config.get("configurable", {}).get("max_retries", MAX_RETRIES) else "cant_help"

    if VERBOSE:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

    answer_grader = (ANSWER_GRADER_PROMPT | llm.with_structured_output(GradeAnswer)).with_config(tags=["skip_stream"])
    answer_result = await answer_grader.ainvoke({"question": state["question"], "generation": state["candidate_answer"]})

    if answer_result.binary_score == "yes":
        if VERBOSE: print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "finalize_response"
    else:
        if VERBOSE: print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "transform_query" if state.get("retries", 0) < config.get("configurable", {}).get("max_retries", MAX_RETRIES) else "cant_help"


################################################################################################################
## Transform Query
async def transform_query(state: AgentState, config: RunnableConfig) -> AgentState:
    if VERBOSE:
        print("---TRANSFORM QUERY---")

    # Initialize the model
    llm = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

    rewriter = QUERY_REWRITER_PROMPT | llm | StrOutputParser()
    better_question = await rewriter.ainvoke({"question": state["question"]})
    return {"question": better_question}


################################################################################################################
## Fallback Handler
async def cant_help(state: AgentState, config: RunnableConfig) -> AgentState:
    return {"candidate_answer": "Sorry, I cannot help you in this matter."}


################################################################################################################
## Finalize Response
async def finalize_response(state: AgentState, config: RunnableConfig) -> AgentState:
    if VERBOSE:
        print("---FINALIZING THE RESPONSE---")        
        return {"messages": [AIMessage(content=state["candidate_answer"])]}



################################################################################################################
## Build Graph
graph = StateGraph(AgentState, config_schema=GraphConfig)

tools     = [tool_get_docs_pinecone, tool_get_docs_timescale]
tool_node = ToolNode(tools)


graph.set_entry_point("query_relevance")

graph.add_node("query_relevance",   query_relevance)
graph.add_node("route_vec_client",    route_vec_client)
graph.add_node("docs_retrieval_tool", tool_node)
# graph.add_node("document_search",   document_search)
graph.add_node("generate",          generate)
graph.add_node("transform_query",   transform_query)
graph.add_node("cant_help",         cant_help)
graph.add_node("finalize_response", finalize_response)

graph.add_conditional_edges("query_relevance", query_relevance_router, ["cant_help", "route_vec_client"])#"document_search"])
graph.add_edge("route_vec_client",     "docs_retrieval_tool")
graph.add_edge("docs_retrieval_tool",  "generate")
# graph.add_edge("document_search", "generate")
graph.add_conditional_edges("generate", grade_generation_v_documents_and_question, ["generate", "transform_query", "cant_help", "finalize_response"])
graph.add_edge("transform_query", "route_vec_client")#"document_search"])
graph.add_edge("cant_help", "finalize_response")
graph.add_edge("finalize_response", END)

self_corrective_rag = graph.compile()





