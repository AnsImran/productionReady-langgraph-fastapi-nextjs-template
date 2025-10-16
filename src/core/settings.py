# =============================================================================
# GLOBAL OBJECTIVE (in the whole codebase)
# -----------------------------------------------------------------------------
# This module is the single source of truth for configuration:
# - Loads environment variables (from a discovered .env and the OS env)
# - Validates/normalizes values with Pydantic (URLs, secrets, enums)
# - Determines which LLM providers are "active" and which model names are
#   exposed to the service (DEFAULT_MODEL, AVAILABLE_MODELS) for /info
# - Validates provider-specific constraints (notably Azure OpenAI deployments)
# - Centralizes service, tracing, and database settings used by service.py:
#     * /info uses DEFAULT_MODEL & AVAILABLE_MODELS
#     * auth uses AUTH_SECRET
#     * /health uses LANGFUSE_* flags
#     * app lifespan/memory init reads DATABASE_* settings
# =============================================================================


# -----------------------------------------------------------------------------
# BLOCK: Standard library imports
# Purpose: basic types/utilities for enums, JSON parsing, typing annotations.
# -----------------------------------------------------------------------------
from enum import StrEnum                      # String-valued enums for stable identifiers
from json import loads                        # Parse JSON text for Azure deployment map (when passed as string)
from typing import Annotated, Any             # Annotated (attach validators), Any for post-init signature


# -----------------------------------------------------------------------------
# BLOCK: 3rd-party libraries for env and validation
# Purpose: locate .env and define/validate settings via Pydantic.
# -----------------------------------------------------------------------------
from dotenv import find_dotenv                # Walk up directories to locate a .env file

from pydantic import (                        # Pydantic V2 building blocks
    BeforeValidator,                          # Run a callable before field validation (e.g., URL coercion)
    Field,                                    # Configure field defaults/metadata (e.g., default_factory for dict)
    HttpUrl,                                  # Strong URL type for validation/normalization
    SecretStr,                                # Wrapper for sensitive strings (masked in logs/dumps)
    TypeAdapter,                              # Ad-hoc validator/coercer (used for the URL checker)
    computed_field,                           # Expose @property value as a field in dumps/schema
)
from pydantic_settings import BaseSettings, SettingsConfigDict
# BaseSettings: env-aware model
# SettingsConfigDict: behavior for env parsing (extra, env_file, etc.)


# -----------------------------------------------------------------------------
# BLOCK: Internal enums/models
# Purpose: provider and model enumerations used to compute defaults/availability.
# -----------------------------------------------------------------------------
from schema.models import (
    AllModelEnum,                             # Type alias: union of all model-enum types
    AllEmbeddingModelEnum,                    ###########################################
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenAIEmbeddingModelName,
    OpenRouterModelName,
    Provider,                                 # Provider enum (OPENAI, ANTHROPIC, etc.)
    VertexAIModelName,
)

from schema.vec_clients import (
    AllVecClientEnum,
    Postgres_Timescale_Vec_Client_Name,
    Pinecone_Vec_Client_Name,
    Postgres_PgVector_Vec_Client_Name,
)


# -----------------------------------------------------------------------------
# BLOCK: DatabaseType enum
# Purpose: canonical identifiers used by memory initialization to choose backend.
# -----------------------------------------------------------------------------
class DatabaseType(StrEnum):
    SQLITE   = "sqlite"    # default simple file-based checkpointer/store
    POSTGRES = "postgres"  # production-grade relational backend
    MONGO    = "mongo"     # document DB option


# -----------------------------------------------------------------------------
# BLOCK: Utility validator
# Purpose: enforce/normalize HTTP(S) URLs on string fields via Annotated + BeforeValidator.
# -----------------------------------------------------------------------------
def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)           # Build a one-off adapter for HttpUrl
    return str(http_url_adapter.validate_python(x))   # Validate+normalize to a string


# -----------------------------------------------------------------------------
# BLOCK: Settings model (core of configuration)
# Purpose: read env, validate, compute active providers/models, expose helpers.
# -----------------------------------------------------------------------------
# `find_dotenv()` doesn’t just look in the same folder — it:
# 1. Starts in the current file’s directory
# 2. Walks up the directory tree
# 3. Returns the **first `.env` it finds**
# So even if `.env` is in your project root and `settings.py` is deep inside `src/` or `configs/`, it will still find it.
class Settings(BaseSettings):
    # --- Pydantic env parsing config (how envs are read/handled) ---
    model_config = SettingsConfigDict(
                    env_file          = find_dotenv(),  # Auto-discover .env for local/dev convenience
                    env_file_encoding = "utf-8",        # Read .env as UTF-8
                    env_ignore_empty  = True,           # Treat empty env vars as missing (avoid false "configured")
                    extra             = "ignore",       # Unknown env vars won't fail startup
                    validate_default  = False,          # Defer default validation to instantiation time
    )
    # --- Mode/host/port/auth (used by service.py startup & auth dependency) ---
    MODE: str | None = None                  # e.g., "dev" toggles dev-only behavior (reload)
    HOST: str = "0.0.0.0"                    # Bind address for FastAPI server
    PORT: int = 8080                         # Port for FastAPI server
    AUTH_SECRET: SecretStr | None = None     # Optional API secret for bearer auth on all routes

    # --- Provider credentials & toggles (presence "activates" providers) ---
    OPENAI_API_KEY:                 SecretStr | None = None
    DEEPSEEK_API_KEY:               SecretStr | None = None
    ANTHROPIC_API_KEY:              SecretStr | None = None
    GOOGLE_API_KEY:                 SecretStr | None = None
    GOOGLE_APPLICATION_CREDENTIALS: SecretStr | None = None  # Used for VertexAI auth
    GROQ_API_KEY:                   SecretStr | None = None
    USE_AWS_BEDROCK:                bool             = False  # Flag-activated provider
    OLLAMA_MODEL:                   str       | None = None   # Setting any model name "activates" Ollama
    OLLAMA_BASE_URL:                str       | None = None   # Optional override for Ollama endpoint
    USE_FAKE_MODEL:                 bool             = False  # Testing/demo-only provider
    OPENROUTER_API_KEY:             str       | None = None

    # --- Model selection surface advertised to clients (/info) ---
    # If DEFAULT_MODEL is None, it will be set in model_post_init
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    # That set() there is just creating an empty set — and right now, it’s being typed as set[AllModelEnum] so that later you can fill it with enum members        # from     any of your model name enums.
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    DEFAULT_EMBEDDING_MODEL: AllEmbeddingModelEnum | None = None#####################################################################################

    # --- OpenAI-compatible shim (self-hosted or alt vendors with OpenAI API shape) ---
    COMPATIBLE_MODEL:    str       | None = None
    COMPATIBLE_API_KEY:  SecretStr | None = None
    COMPATIBLE_BASE_URL: str       | None = None

    # --- Extra service/tooling API keys (used by tools, not core service) ---
    OPENWEATHERMAP_API_KEY: SecretStr | None = None

    # --- LangChain/LangSmith tracing configuration ---
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_PROJECT:    str = "default"
    LANGCHAIN_ENDPOINT:   Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )
    LANGCHAIN_API_KEY: SecretStr | None = None

    # --- Langfuse tracing configuration (used by /health and callbacks) ---
    LANGFUSE_TRACING:    bool             = False
    LANGFUSE_HOST:       Annotated[str, BeforeValidator(check_str_is_http)] = "https://cloud.langfuse.com"  #"http://localhost:3000" 
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None

    # --- Database configuration (used by memory initialization in app lifespan) ---
    DATABASE_TYPE: DatabaseType = (
        DatabaseType.SQLITE
    )  # Options: DatabaseType.SQLITE or DatabaseType.POSTGRES
    SQLITE_DB_PATH: str = "checkpoints.db"  # File path for SQLite checkpointer/store

    # PostgreSQL Configuration (effective when DATABASE_TYPE=POSTGRES)
    POSTGRES_USER:     str       | None = None
    POSTGRES_PASSWORD: SecretStr | None = None
    POSTGRES_HOST:     str       | None = None
    POSTGRES_PORT:     int       | None = None
    POSTGRES_DB:       str       | None = None

    # MongoDB Configuration (effective when DATABASE_TYPE=MONGO)
    MONGO_HOST:        str       | None = None
    MONGO_PORT:        int       | None = None
    MONGO_DB:          str       | None = None
    MONGO_USER:        str       | None = None
    MONGO_PASSWORD:    SecretStr | None = None
    MONGO_AUTH_SOURCE: str       | None = None

    # --- Azure OpenAI specifics (validated when Azure is active) ---
    AZURE_OPENAI_API_KEY:        SecretStr | None = None
    AZURE_OPENAI_ENDPOINT:       str       | None = None
    AZURE_OPENAI_API_VERSION:    str              = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_MAP: dict[str, str]   = Field(
        default_factory=dict, description="Map of model names to Azure deployment IDs"
    )

    VEC_CLIENT:                    AllVecClientEnum | None = None  # type: ignore[assignment]
    POSTGRES_TIMESCALE_VEC_CLIENT: AllVecClientEnum | None = None #Postgres_Timescale_Vec_Client_Name.POSTGRES_TIMESCALE_VEC_CLIENT
    PINECONE_VEC_CLIENT:           AllVecClientEnum | None = None #Pinecone_Vec_Client_Name.PINECONE_VEC_CLIENT
    POSTGRES_PGVECTOR_VEC_CLIENT: AllVecClientEnum | None = None #Postgres_PgVector_Vec_Client_Name.POSTGRES_PGVECTOR_VEC_CLIENT
    
    
    

    # --- Post-construction hook: compute active providers/models & validate combos ---
    def model_post_init(self, __context: Any) -> None:
        self.POSTGRES_TIMESCALE_VEC_CLIENT = Postgres_Timescale_Vec_Client_Name.POSTGRES_TIMESCALE_VEC_CLIENT
        self.PINECONE_VEC_CLIENT           = Pinecone_Vec_Client_Name.PINECONE_VEC_CLIENT
        self.POSTGRES_PGVECTOR_VEC_CLIENT = Postgres_PgVector_Vec_Client_Name.POSTGRES_PGVECTOR_VEC_CLIENT
        

        
        # Map each Provider to the presence/flag that "activates" it
        api_keys = {
            Provider.OPENAI:            self.OPENAI_API_KEY,
            Provider.OPENAI_COMPATIBLE: self.COMPATIBLE_BASE_URL and self.COMPATIBLE_MODEL,
            Provider.DEEPSEEK:          self.DEEPSEEK_API_KEY,
            Provider.ANTHROPIC:         self.ANTHROPIC_API_KEY,
            Provider.GOOGLE:            self.GOOGLE_API_KEY,
            Provider.VERTEXAI:          self.GOOGLE_APPLICATION_CREDENTIALS,
            Provider.GROQ:              self.GROQ_API_KEY,
            Provider.AWS:               self.USE_AWS_BEDROCK,
            Provider.OLLAMA:            self.OLLAMA_MODEL,
            Provider.FAKE:              self.USE_FAKE_MODEL,
            Provider.AZURE_OPENAI:      self.AZURE_OPENAI_API_KEY,
            Provider.OPENROUTER:        self.OPENROUTER_API_KEY,
        }
        # Keep only providers with truthy activation signals
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            # The service cannot function without at least one LLM backend
            raise ValueError("At least one LLM API key must be provided.")

        # For each active provider:
        # - Set DEFAULT_MODEL if still unset (first one wins)
        # - Populate AVAILABLE_MODELS with that provider's entire enum
        # - Run any provider-specific validation (Azure)
        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4_1_NANO   ###############################
                        self.DEFAULT_EMBEDDING_MODEL = OpenAIEmbeddingModelName.TEXT_EMBEDDING_3_SMALL         ####### zroori nahi k aap llm or embedding model aik hi provider ka use kro
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))
                case Provider.OPENAI_COMPATIBLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAICompatibleName.OPENAI_COMPATIBLE
                    self.AVAILABLE_MODELS.update(set(OpenAICompatibleName))
                case Provider.DEEPSEEK:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = DeepseekModelName.DEEPSEEK_CHAT
                    self.AVAILABLE_MODELS.update(set(DeepseekModelName))
                case Provider.ANTHROPIC:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AnthropicModelName.HAIKU_3
                    self.AVAILABLE_MODELS.update(set(AnthropicModelName))
                case Provider.GOOGLE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GoogleModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(GoogleModelName))
                case Provider.VERTEXAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = VertexAIModelName.GEMINI_20_FLASH
                    self.AVAILABLE_MODELS.update(set(VertexAIModelName))
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B
                    self.AVAILABLE_MODELS.update(set(GroqModelName))
                case Provider.AWS:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AWSModelName.BEDROCK_HAIKU
                    self.AVAILABLE_MODELS.update(set(AWSModelName))
                case Provider.OLLAMA:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OllamaModelName.OLLAMA_GENERIC
                    self.AVAILABLE_MODELS.update(set(OllamaModelName))
                case Provider.OPENROUTER:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenRouterModelName.GEMINI_25_FLASH
                    self.AVAILABLE_MODELS.update(set(OpenRouterModelName))
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE
                    self.AVAILABLE_MODELS.update(set(FakeModelName))
                case Provider.AZURE_OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = AzureOpenAIModelName.AZURE_GPT_4O_MINI
                    self.AVAILABLE_MODELS.update(set(AzureOpenAIModelName))
                    # Validate Azure OpenAI settings if Azure provider is available
                    if not self.AZURE_OPENAI_API_KEY:
                        raise ValueError("AZURE_OPENAI_API_KEY must be set")
                    if not self.AZURE_OPENAI_ENDPOINT:
                        raise ValueError("AZURE_OPENAI_ENDPOINT must be set")
                    if not self.AZURE_OPENAI_DEPLOYMENT_MAP:
                        raise ValueError("AZURE_OPENAI_DEPLOYMENT_MAP must be set")

                    # Parse deployment map if it's a string
                    if isinstance(self.AZURE_OPENAI_DEPLOYMENT_MAP, str):
                        try:
                            self.AZURE_OPENAI_DEPLOYMENT_MAP = loads(
                                self.AZURE_OPENAI_DEPLOYMENT_MAP
                            )
                        except Exception as e:
                            # Surface clear error when JSON is malformed
                            raise ValueError(f"Invalid AZURE_OPENAI_DEPLOYMENT_MAP JSON: {e}")

                    # Validate required deployments exist
                    required_models = {"gpt-4o", "gpt-4o-mini"}
                    missing_models = required_models - set(self.AZURE_OPENAI_DEPLOYMENT_MAP.keys())
                    if missing_models:
                        raise ValueError(f"Missing required Azure deployments: {missing_models}")
                case _:
                    # Defensive guard in case Provider enum is extended without handling here
                    raise ValueError(f"Unknown provider: {provider}")

    # -----------------------------------------------------------------------------
    # BLOCK: Computed helpers
    # Purpose: add derived fields/flags used by client/server entrypoints.
    # -----------------------------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"  # Derived endpoint string (client convenience)

    def is_dev(self) -> bool:
        return self.MODE == "dev"                 # Single place to check "developer mode"


# -----------------------------------------------------------------------------
# BLOCK: Module-level instantiation
# Purpose: eagerly load/validate settings so imports elsewhere can rely on `settings`.
# -----------------------------------------------------------------------------
settings = Settings()
