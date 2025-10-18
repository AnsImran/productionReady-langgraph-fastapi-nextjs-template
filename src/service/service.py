import asyncio
import inspect
import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional
from uuid import UUID, uuid4
import hashlib, time

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
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
from psycopg import OperationalError
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
from core.settings import DatabaseType
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
# from vector_databases import get_vec_client_timescale
# from vector_databases import get_vec_client_pgvector






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


async def _safe_exit_context(ctx: Any) -> None:
    """Best-effort closure for async context managers."""
    if ctx is None:
        return
    exit_cb = getattr(ctx, "__aexit__", None)
    if exit_cb is None:
        return
    try:
        await exit_cb(None, None, None)
    except Exception as exc:  # pragma: no cover - cleanup path
        logger.warning("Error while closing context: %s", exc)


def _assign_memory_components(saver: Any, store: Any) -> None:
    """Attach shared memory components to every registered agent."""
    for agent_info in get_all_agent_info():
        agent = get_agent(agent_info.key)
        if saver is not None:
            agent.checkpointer = saver
        if store is not None:
            agent.store = store


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
    app.state.saver_ctxs = []
    app.state.store_ctxs = []
    app.state.refresh_lock = asyncio.Lock()
    try:
        saver_ctx = initialize_database()
        store_ctx = initialize_store()
        try:
            saver = await saver_ctx.__aenter__()
            store = await store_ctx.__aenter__()
        except Exception:
            await _safe_exit_context(saver_ctx)
            await _safe_exit_context(store_ctx)
            raise

        app.state.saver_ctxs.append(saver_ctx)
        app.state.store_ctxs.append(store_ctx)
        app.state.saver = saver
        app.state.store = store

        # Set up both components
        if hasattr(saver, "setup"):  # ignore: union-attr
            await saver.setup()
        if hasattr(store, "setup"):  # ignore: union-attr
            await store.setup()

        print("Checkpointer:", type(saver).__name__)
        print("Store:", type(store).__name__)

        _assign_memory_components(saver, store)

        # vec_client = get_vec_client_timescale(get_postgres_connection_string())
        # app.state.vec_client = vec_client
        # print(type(vec_client))
        yield
    except Exception as e:
        logger.error(f"Error during database/store initialization: {e}")
        raise
    finally:
        for ctx in reversed(getattr(app.state, "store_ctxs", [])):
            await _safe_exit_context(ctx)
        for ctx in reversed(getattr(app.state, "saver_ctxs", [])):
            await _safe_exit_context(ctx)


app    = FastAPI(lifespan=lifespan)

# --- Configure which frontends can call your API ---
# Dev: Next.js on localhost:3000
# Prod: replace with your real domain(s)
DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# # Optional: allow configuring via env var (comma-separated)
# # e.g. FRONTEND_ORIGINS="http://localhost:3000,https://yourapp.com"
# env_origins = [
#     o.strip() for o in os.getenv("FRONTEND_ORIGINS", "").split(",") if o.strip()
# ]
# this and a lot of extra code will be transferred to core.settings
env_origins = None

ALLOWED_ORIGINS = env_origins or DEFAULT_ORIGINS

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_methods=["POST", "OPTIONS"],   # your endpoint is POST /.../stream
#     allow_headers=["*"],                 # accepts JSON + SSE
#     expose_headers=["x-vercel-ai-ui-message-stream"],  # let the browser read this if needed
#     allow_credentials=False,             # set True later if you use cookies across origins
# )

# temporarily allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                         # allow all origins (TEMP ONLY)
    allow_methods=["*"],                         # or ["POST", "OPTIONS"] if you prefer stricter
    allow_headers=["*"],                         # accept any request headers
    expose_headers=["x-vercel-ai-ui-message-stream"],
    allow_credentials=False,                     # must be False when using "*"
    max_age=3600,                                # cache preflight for an hour
)
print('---temporarily allowed all origins---')

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

async def _refresh_memory_components() -> None:
    """Reinitialize LangGraph memory connections after a database failure."""
    if settings.DATABASE_TYPE != DatabaseType.POSTGRES:
        return

    lock: asyncio.Lock = getattr(app.state, "refresh_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state.refresh_lock = lock

    async with lock:
        if not hasattr(app.state, "saver_ctxs"):
            app.state.saver_ctxs = []
        if not hasattr(app.state, "store_ctxs"):
            app.state.store_ctxs = []

        saver_ctx = initialize_database()
        store_ctx = initialize_store()
        try:
            saver = await saver_ctx.__aenter__()
            store = await store_ctx.__aenter__()
        except Exception as exc:
            await _safe_exit_context(saver_ctx)
            await _safe_exit_context(store_ctx)
            logger.error("Failed to refresh LangGraph memory components: %s", exc)
            raise

        app.state.saver_ctxs.append(saver_ctx)
        app.state.store_ctxs.append(store_ctx)
        app.state.saver = saver
        app.state.store = store

        if hasattr(saver, "setup"):  # ignore: union-attr
            await saver.setup()
        if hasattr(store, "setup"):  # ignore: union-attr
            await store.setup()

        logger.warning("Reinitialized LangGraph Postgres connections after failure.")
        _assign_memory_components(saver, store)


async def _aget_state_with_retry(agent: AgentGraph, config: RunnableConfig):
    """Fetch agent state, retrying once if the Postgres connection was dropped."""
    try:
        return await agent.aget_state(config=config)
    except OperationalError as exc:
        if settings.DATABASE_TYPE != DatabaseType.POSTGRES:
            raise
        logger.warning("Checkpointer connection closed; refreshing before retry. %s", exc)
        await _refresh_memory_components()
        return await agent.aget_state(config=config)


async def _invoke_agent_with_retry(
    agent: AgentGraph,
    *,
    kwargs: dict[str, Any],
    stream_mode: Optional[list[str]] = None,
) -> list[tuple[str, Any]]:
    """Call agent.ainvoke with a single automatic retry on connection loss."""
    attempt = 0
    while True:
        try:
            if stream_mode is None:
                return await agent.ainvoke(**kwargs)
            return await agent.ainvoke(**kwargs, stream_mode=stream_mode)
        except OperationalError as exc:
            attempt += 1
            if settings.DATABASE_TYPE != DatabaseType.POSTGRES or attempt > 1:
                raise
            logger.warning("Agent invoke failed due to closed connection; retrying. %s", exc)
            await _refresh_memory_components()


async def _stream_agent_with_retry(
    agent: AgentGraph,
    *,
    kwargs: dict[str, Any],
    stream_mode: list[str],
) -> AsyncGenerator[tuple[str, Any], None]:
    """Yield agent stream events with a single retry if the connection drops."""
    attempt = 0
    while attempt < 2:
        try:
            async for event in agent.astream(**kwargs, stream_mode=stream_mode):
                yield event
            return
        except OperationalError as exc:
            attempt += 1
            if settings.DATABASE_TYPE != DatabaseType.POSTGRES or attempt >= 2:
                raise
            logger.warning("Agent stream failed due to closed connection; retrying. %s", exc)
            await _refresh_memory_components()


async def _handle_input(user_input: UserInput, agent: AgentGraph, vec_client: client.Async|None = None) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id    = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    user_id   = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "model": user_input.model, "user_id": user_id}
    
    # if vec_client:
    #     configurable["vec_client"] = vec_client
                        
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
    state = await _aget_state_with_retry(agent, config)
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

    # if agent_id == "self_corrective_rag" or "prototype_rag_tool":
    #     kwargs, run_id    = await _handle_input(user_input, agent, app.state.vec_client)
    # else:
    #     kwargs, run_id    = await _handle_input(user_input, agent, None)
    kwargs, run_id    = await _handle_input(user_input, agent, None)

    try:
        response_events: list[tuple[str, Any]] = await _invoke_agent_with_retry(
            agent,
            kwargs=kwargs,
            stream_mode=["updates", "values"],
        )  # type: ignore # fmt: skip
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
    
    # if agent_id == "self_corrective_rag" or "prototype_rag_tool":
    #     kwargs, run_id    = await _handle_input(user_input, agent, app.state.vec_client)
    # else:
    #     kwargs, run_id    = await _handle_input(user_input, agent, None)
    kwargs, run_id    = await _handle_input(user_input, agent, None)


    # Build a stable key for this conversation thread
    session_key = f"{agent_id}:{getattr(user_input, 'thread_id', None) or 'no_thread'}:{getattr(user_input, 'user_id', None) or 'anon'}"

    # Global-ish store in app.state: { session_key: {"inputs": set(), "outputs": set(), "ts": float} }
    if not hasattr(app.state, "tool_sse_seen"):
        app.state.tool_sse_seen = {}

    # TTL + light cleanup to avoid unbounded memory
    TTL_SECONDS = 60 * 60  # 1 hour
    NOW = time.time()
    # opportunistic cleanup
    for k, v in list(app.state.tool_sse_seen.items()):
        if (NOW - v.get("ts", NOW)) > TTL_SECONDS:
            app.state.tool_sse_seen.pop(k, None)

    thread_seen = app.state.tool_sse_seen.setdefault(session_key, {"inputs": set(), "outputs": set(), "ts": NOW})
    thread_seen["ts"] = NOW  # touch


    # >>> NEW: ids + flags for UI message stream
    msg_id   = f"msg_{uuid4().hex}"
    text_id  = f"{msg_id}_t0"

    started  = False
    finished = False
    started_tool  = False  # sent text-start?
    finished_tool = False  # sent text-end/finish?



    try:
        async for stream_event in _stream_agent_with_retry(
            agent,
            kwargs=kwargs,
            stream_mode=["updates", "custom", "messages"],
        ):
            if not isinstance(stream_event, tuple):
                continue

            stream_mode, event = stream_event
            new_messages = []

            if stream_mode == "updates":
                for node, updates in (event or {}).items():
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates or []:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue

                    update_messages = (updates or {}).get("messages", [])

                    if node == "supervisor":
                        ai_messages = [m for m in update_messages if isinstance(m, AIMessage)]
                        if ai_messages:
                            update_messages = [ai_messages[-1]]

                    if node in ("research_expert", "math_expert"):
                        msg = ToolMessage(content=update_messages[0].content, name=node, tool_call_id="")
                        update_messages = [msg]

                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            processed_messages = []
            current_message: dict[str, Any] = {}

            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    current_message[key] = value
                else:
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    if isinstance(message, str):
                        chat_message        = ChatMessage(type="ai", content=message)
                        chat_message.run_id = str(run_id)
                    else:
                        chat_message        = langchain_to_chat_message(message)
                        chat_message.run_id = str(run_id)
                    # chat_message        = langchain_to_chat_message(message)
                    # chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    # >>> CHANGED: UI stream error shape
                    yield f"data: {json.dumps({'type': 'error', 'errorText': 'Unexpected error'})}\n\n"
                    continue
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue




                if chat_message.type == "ai" and chat_message.tool_calls:
                    for tool_call in chat_message.tool_calls:
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args") or {}
                        raw_id    = tool_call.get("id")

                        # Stable fingerprint for assistant->tool input
                        fp_source = raw_id or f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                        fp_in = hashlib.sha1(fp_source.encode("utf-8")).hexdigest()

                        # Skip if this exact input was already streamed for this thread
                        if fp_in in thread_seen["inputs"]:
                            continue
                        thread_seen["inputs"].add(fp_in)

                        # Use original id if present; otherwise deterministic from fp
                        tool_call_id = raw_id or f"call_{fp_in}"

                        started_tool = True
                        yield f"data: {json.dumps({'type': 'start'})}\n\n"
                        yield f"data: {json.dumps({'type': 'start-step'})}\n\n"
                        yield f"data: {json.dumps({'type': 'tool-input-start', 'toolCallId': tool_call_id, 'toolName': tool_name})}\n\n"

                        if tool_args:
                            yield f"data: {json.dumps({'type': 'tool-input-delta', 'toolCallId': tool_call_id, 'inputTextDelta': json.dumps(tool_args)})}\n\n"

                        yield f"data: {json.dumps({'type': 'tool-input-available', 'toolCallId': tool_call_id, 'toolName': tool_name, 'input': tool_args})}\n\n"

                if chat_message.type == "tool":
                    raw_tool_call_id = chat_message.tool_call_id  # may be None
                    # Parse output for stable hashing
                    try:
                        tool_output: Any = json.loads(chat_message.content)
                    except json.JSONDecodeError:
                        tool_output = chat_message.content

                    out_norm = json.dumps(tool_output, sort_keys=True) if isinstance(tool_output, (dict, list)) else str(tool_output)
                    fp_source = f"{raw_tool_call_id or ''}:{out_norm}"
                    fp_out = hashlib.sha1(fp_source.encode("utf-8")).hexdigest()

                    # Skip if this exact output was already streamed for this thread
                    if fp_out in thread_seen["outputs"]:
                        continue
                    thread_seen["outputs"].add(fp_out)

                    tool_call_id = raw_tool_call_id or f"call_{fp_out}"

                    yield f"data: {json.dumps({'type': 'tool-output-available', 'toolCallId': tool_call_id, 'output': tool_output})}\n\n"
                    yield f"data: {json.dumps({'type': 'finish-step'})}\n\n"
                    finished_tool = True

                    

                # yield f"data: {json.dumps({'type': 'data-message', 'data': chat_message.model_dump()})}\n\n"





            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                if not isinstance(msg, AIMessageChunk):
                    continue

                content = remove_tool_calls(msg.content)
                if content:
                    token_text = convert_message_content_to_string(content)

                    # >>> NEW: open the UI stream on first token
                    if not started:
                        if started_tool == False:
                          yield f"data: {json.dumps({'type': 'start', 'messageId': msg_id})}\n\n"

                        yield f"data: {json.dumps({"type": "start-step"})}\n\n"
                        yield f"data: {json.dumps({'type': 'text-start', 'id': text_id})}\n\n"
                        started = True

                    # >>> CHANGED: token -> text-delta
                    yield f"data: {json.dumps({'type': 'text-delta', 'id': text_id, 'delta': token_text})}\n\n"

        # >>> NEW: after the loop, close text if we opened it
        if started and not finished:
            yield f"data: {json.dumps({'type': 'text-end', 'id': text_id})}\n\n"
            yield f"data: {json.dumps({"type": "finish-step"})}\n\n"
            yield f"data: {json.dumps({'type': 'finish'})}\n\n"
            finished = True

    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        # >>> CHANGED: UI stream error shape
        yield f"data: {json.dumps({'type': 'error', 'errorText': 'Internal server error'})}\n\n"
    finally:
        # >>> keep your termination marker last
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
async def stream(user_input: StreamInput, request:Request, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    # >>> NEW: add the UI message stream header (no middleware)
    headers = {
        "x-vercel-ai-ui-message-stream": "v1",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
    }

    # If auth is enabled and dependency set user_id, prefer it
    jwt_sub = getattr(request.state, "jwt_sub", None)
    if jwt_sub:
        user_input.user_id = jwt_sub
    # print('user_id from stream endpoint', user_input.user_id)
    
    obj = await request.json()
    # convert to indented JSON string
    pretty = json.dumps(obj, indent=2, ensure_ascii=False)

    print(pretty)



    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
        headers=headers,  # <<< added
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
async def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = await _aget_state_with_retry(
            agent,
            RunnableConfig(configurable={"thread_id": input.thread_id}),
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
