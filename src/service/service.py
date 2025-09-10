import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import CallbackHandler  # type: ignore[import-untyped]
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from timescale_vector import client







# 1. Your project uses a **`src/` layout**, where `service` and `memory` are **sibling packages** under `src`.
# 2. Both folders have an `__init__.py`, making them **top-level Python packages**.
# 3. Python searches packages using `sys.path`; when you run from the repo root with `src` on the path, it can see `memory`.
# 4. **Absolute import** `from memory import …` works because Python finds `src/memory/__init__.py`.
# 5. **Relative import** like `from .memory import …` would fail because there is no `service/memory` subfolder.
# 6. `initialize_database()` and `initialize_store()` are exposed by `memory/__init__.py`.
# 7. They **decide at runtime** which backend to use: SQLite (default), Postgres, or Mongo.
# 8. These functions return **async context managers** for short-term (checkpointer) and long-term (store) memory.
# 9. `service.py` can stay backend-agnostic because it just calls these two initializers.
# 10. FastAPI’s `lifespan` uses them to set up memory for all agents on app startup.
# 11. Python will resolve the import as long as `src` is on `sys.path` (via `-m`, `PYTHONPATH`, or editable install).
# 12. Sibling packages under the same path can always import each other via **absolute imports**.
# 13. That’s why `memory` doesn’t need to be inside the `service` folder.
# 14. The current structure is correct and Pythonic for multi-package projects.
# 15. Relative imports are only needed if the module is a **subfolder** of the importing package.

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from core import settings
from memory import initialize_database, initialize_store, get_postgres_connection_string ###########################################################
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from vector_databases import get_vec_client_timescale






# This line suppresses **LangChain beta warnings** at runtime:
# * `warnings.filterwarnings("ignore", category=LangChainBetaWarning)` tells Python to **ignore all `LangChainBetaWarning`** messages.
# * These warnings appear when using **experimental or beta features** in LangChain.
# * It keeps logs clean but **hides potential upgrade risks**.
# * Remove or change to `"default"` to see the warnings again.
warnings.filterwarnings("ignore", category=LangChainBetaWarning)




# * `logger = logging.getLogger(__name__)` creates a **module-specific logger** using Python’s logging system.
# * `__name__` gives the current module’s name, so logs are **namespaced** (e.g., `service.api`).
# * It allows logs from different modules to be **distinguished and filtered** easily.
# * Logging behavior (format, level, output) depends on the **global logging configuration**.
# * Example log: `ERROR:service.api:Something went wrong`.
logger = logging.getLogger(__name__)




# `verify_bearer` is a **FastAPI dependency** that enforces optional Bearer token authentication:

# 1. Uses `HTTPBearer(auto_error=False)` to parse `Authorization: Bearer <token>` headers.
# 2. If no header is present, `http_auth` is `None`.
# 3. If `settings.AUTH_SECRET` is **unset**, auth is skipped.
# 4. If set, retrieves the real secret via `.get_secret_value()`.
# 5. Compares provided token (`http_auth.credentials`) to the secret.
# 6. If missing or incorrect → raises `HTTPException(401)`.
# 7. Returning `None` means the request is allowed.
# 8. Applied at the router level → all endpoints in that router are protected.
# 9. Provides simple, single-secret, service-wide authentication.
# 10. Acts as a **gatekeeper** without modifying the request object.
def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)




# 1. `lifespan` is an **async context manager** used by FastAPI to run setup and teardown code during the app’s lifetime.
# 2. It initializes **short-term memory** (checkpointer) and **long-term memory** (store) for LangGraph agents.
# 3. Uses `initialize_database()` to get the **checkpointer** for conversation/thread-scoped memory.
# 4. Uses `initialize_store()` to get the **store** for persistent, cross-conversation memory.
# 5. Both are async context managers, ensuring proper connection setup and cleanup.
# 6. If the checkpointer or store has a `.setup()` method, it is awaited to fully prepare the backend (e.g., create tables).
# 7. Retrieves a list of all agent metadata via `get_all_agent_info()`.
# 8. For each agent in the system, calls `get_agent(agent.key)` to get its `AgentGraph` instance.
# 9. Injects the **checkpointer** into the agent for **thread-based chat history tracking**.
# 10. Injects the **store** into the agent for **long-term knowledge storage/retrieval**.
# 11. This wiring allows agents to automatically log, retrieve, and resume conversations.
# 12. The context manager yields, allowing FastAPI to start serving requests.
# 13. If any exception occurs during setup, it is logged and re-raised to stop the app from starting.
# 14. When the app shuts down, the async context managers clean up resources automatically.
# 15. **No vector DB retrievers (like Chroma or Pinecone) are started here**—only memory components for LangGraph agents.

# * `AsyncGenerator[YieldType, SendType]` defines what a generator **yields** and what it can **receive via `.asend()`**.
# * In `lifespan(...) -> AsyncGenerator[None, None]`, it **yields nothing** and **receives nothing** — just manages setup/teardown.
# * Example: `AsyncGenerator[str, None]` means the function yields strings like `yield "hello"` (e.g., a streaming API).
# * So `None, None` isn’t about how many things are initialized — it’s about what’s yielded/sent.

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer and store
    based on settings.
    """
    try:
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            print("Checkpointer:", type(saver).__name__)
            print("Store:", type(store).__name__)

            # Configure agents with both memory components | ALL AGENTS! |
            agents = get_all_agent_info()
            for a in agents:
                agent = get_agent(a.key)
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store

            vec_client = get_vec_client_timescale(get_postgres_connection_string())
            app.state.vec_client = vec_client
            print(type(vec_client))
            yield
                
    except Exception as e:
        logger.error(f"Error during database/store initialization: {e}")
        raise


app    = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])






@router.get("/info")
async def info() -> ServiceMetadata:
    
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    
    return ServiceMetadata(
        agents        = get_all_agent_info(),
        models        = models,
        default_agent = DEFAULT_AGENT,
        default_model = settings.DEFAULT_MODEL,
    )




# A concise summary of what happens in `_handle_input(...)` and the two routes `/invoke` and `/stream`:

# ---

# ### 🔹 `_handle_input(user_input, agent)`

# * Generates a unique `run_id`, and fills in `thread_id` and `user_id` from input (or random UUIDs).
# * Builds a `configurable` dict with `{thread_id, user_id, model}`, and merges `agent_config` if provided (ensuring no key conflicts).
# * Optionally adds a **Langfuse callback** if tracing is enabled.
# * Constructs a `RunnableConfig` with the above data.
# * Checks agent’s state to see if it's **awaiting input (interrupted)**.

#   * If yes → wraps user message in `Command(resume=...)`
#   * If no → wraps it as a `HumanMessage(...)`
# * Returns a tuple of `{input, config}` and `run_id`.

# ---

# ### 🔹 `POST /invoke`

# * Gets the agent (by ID or default).
# * Calls `_handle_input(...)` to prepare config + input.
# * Runs the agent using `agent.ainvoke(...)` with `stream_mode=["updates", "values"]`.
# * Gets the **last event** in the response:

#   * If `"values"` → extracts last message and returns it.
#   * If `"updates"` with `"__interrupt__"` → returns first interrupt as an AI message.
# * Converts the result into a `ChatMessage`, attaches `run_id`, and returns it.

# ---

# ### 🔹 `POST /stream`

# * Also calls `_handle_input(...)`.
# * Streams agent output using `agent.astream(...)` (async generator).
# * Handles:

#   * `updates`: intermediate messages and interrupts
#   * `messages`: token-by-token chunks (optional)
#   * `custom`: metadata messages
# * Converts stream events to `ChatMessage` or tokens, emits as **SSE (`data: {...}`)**.
# * Skips echoed human input and tool-use chunks.
# * Ends with `data: [DONE]`.

async def _handle_input(user_input: UserInput, agent: AgentGraph, vec_client: client.Async|None = None) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id    = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    user_id   = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "model": user_input.model, "user_id": user_id}
    
    if vec_client:
        configurable["vec_client"] = vec_client
                        
    callbacks = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config) # simple dict upadte ki hay

    # simply comes from langchain/langgraph
    config = RunnableConfig(
        configurable = configurable,
        run_id       = run_id,
        callbacks    = callbacks,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    # input ki 2 types ho skti hain. aik type interruption k case main use hogi, 2sri normal message flow k case main
    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = get_agent(agent_id)

    if agent_id == "self_corrective_rag" or "prototype_rag_tool":
        kwargs, run_id    = await _handle_input(user_input, agent, app.state.vec_client)
    else:
        kwargs, run_id    = await _handle_input(user_input, agent, None)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])

        # both conditions must be true. "__interrupt__" key must be present in response dict. AND response_type == "updates"
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            # ye vaqiya mian, aik question hoga, Ai ki traf say k ab aagay kiya krna hay.
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)

        # complete output return kray ga including run_id
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")





# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:8
async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    
    if agent_id == "self_corrective_rag" or "prototype_rag_tool":
        kwargs, run_id    = await _handle_input(user_input, agent, app.state.vec_client)
    else:
        kwargs, run_id    = await _handle_input(user_input, agent, None)

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "custom", "messages"]
        ):
            if not isinstance(stream_event, tuple):
                continue
                
            stream_mode, event = stream_event
            new_messages = []                                                ########
            
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue


                        
                    updates = updates or {}                                   #
                    update_messages = updates.get("messages", [])             #
                    
                    # special cases for using langgraph-supervisor library
                    if node == "supervisor":
                        # Get only the last AIMessage since supervisor includes all previous messages
                        ai_messages = [msg for msg in update_messages if isinstance(msg, AIMessage)]
                        if ai_messages:
                            update_messages = [ai_messages[-1]]
                            
                    if node in ("research_expert", "math_expert"):
                        # By default the sub-agent output is returned as an AIMessage.
                        # Convert it to a ToolMessage so it displays in the UI as a tool response.
                        msg = ToolMessage(
                            content      = update_messages[0].content,
                            name         = node,
                            tool_call_id = "",
                        )
                        update_messages = [msg]
                    
                    new_messages.extend(update_messages)                      #

                    
                    # if node == 'document_search':
                    #     current_docs = [updates['documents'][i].model_dump_json() + "\n \n" for i in range(len(updates['documents']))]
                    #     current_docs = "".join(current_docs)
                    #     new_messages.append(AIMessage(content=current_docs))
                    #     continue



            
            if stream_mode == "custom":
                new_messages = [event]




                
            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/            
            processed_messages                 = []
            current_message:    dict[str, Any] = {}
            
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)    
            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

                
            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"



            
            if stream_mode == "messages":
                if not user_input.stream_tokens: # default=True. see `schema.schema` -> `StreamInput`
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"





# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:10
def _create_ai_message(parts: dict) -> AIMessage:
    sig        = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered   = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:10
def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:10
@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


######################################################################################################################################################
# might add langfuse capability
# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:11
@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    # u can do the similar thing with LANGFUSE as well!
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id = feedback.run_id,
        key    = feedback.key,
        score  = feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:11
@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages:      list[AnyMessage]  = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:11
@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


# https://chatgpt.com/c/6891f4fd-6f08-8324-bebf-b4ea5243ebe3
# message:2 branch:11
app.include_router(router)
