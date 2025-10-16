from typing import Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

from schema.models import (
    AllEmbeddingModelEnum,
    OpenAIEmbeddingModelName,
)

load_dotenv()


def get_vec_client_pgvector(
    PGVECTOR_CONNECTION: str,
    collection_name: str = "services",
    embedding_model_name: AllEmbeddingModelEnum = OpenAIEmbeddingModelName.TEXT_EMBEDDING_3_SMALL,
    *,
    create_if_missing: bool = False,
    texts: Optional[list[str]] = None,
    metadatas: Optional[list[dict]] = None,
    use_jsonb: bool = True,
) -> PGVector:
    """
    Initializes and returns a LangChain PGVector vector store client.

    Args:
        PGVECTOR_CONNECTION:  str  =  ''   # e.g. 'postgresql+psycopg2://user:pass@host:5432/db'
        collection_name:      str  =  'services'
        embedding_model_name: AllEmbeddingModelEnum = 'text-embedding-3-small'
        create_if_missing:    bool =  False
            - If True, will create the collection by inserting `texts` (and optional `metadatas`).
              If False, it will connect to an existing collection.
        texts:                Optional[List[str]]
            - Required when `create_if_missing=True`.
        metadatas:            Optional[List[dict]]
            - Optional metadata parallel to `texts`.
        use_jsonb:            bool = True
            - Store metadatas as JSONB in Postgres (recommended).

    Returns:
        PGVector: An initialized LangChain PGVector store client.

    Raises:
        ValueError: If `create_if_missing=True` but `texts` is not provided.
    """

    # Map your enum to the actual OpenAI embedding model id (extend as needed)
    if embedding_model_name == OpenAIEmbeddingModelName.TEXT_EMBEDDING_3_SMALL:
        model_id = "text-embedding-3-small"
        # embedding_dims = 1536  # FYI, not required by PGVector, but kept for parity with your pattern
    else:
        # Fallback: assume enum value is already a valid model name string
        model_id = str(embedding_model_name)

    embeddings = OpenAIEmbeddings(model=model_id)

    if create_if_missing:
        if not texts:
            raise ValueError("`texts` must be provided when create_if_missing=True.")
        # Create the collection (or upsert into it) by inserting texts
        return PGVector.from_texts(
            connection_string=PGVECTOR_CONNECTION,
            embedding=embeddings,
            texts=texts,
            metadatas=metadatas,
            collection_name=collection_name,
            use_jsonb=use_jsonb,
        )

    # Connect to an existing collection/index
    return PGVector.from_existing_index(
        embedding=embeddings,
        collection_name=collection_name,
        connection_string=PGVECTOR_CONNECTION,
        use_jsonb=use_jsonb,
    )
