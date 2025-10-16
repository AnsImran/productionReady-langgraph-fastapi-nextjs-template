import os
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from schema.models import (
    AllEmbeddingModelEnum,
    OpenAIEmbeddingModelName,
)

from memory import get_postgres_connection_string


async def get_docs_pgvector(
        question: str,
        vec_client: PGVector = None,
        embedding_model_name: str = "text-embedding-3-small",
        max_results: int = 5,
    ) -> list[Document]:
    """
    Performs semantic search on a PGVector collection using LangChain's PGVector client.

    Args:
        question:              The input query or user question.
        vec_client:            An initialized PGVector client.
        embedding_model_name:  The OpenAI embedding model to use for query embedding.
        max_results:           Maximum number of similar documents to retrieve.

    Returns:
        list[Document]: List of LangChain Document objects (each with .page_content and .metadata)
    """

    if vec_client is None:
        # Auto-connect if client not provided
        COLLECTION = os.getenv("PGVECTOR_COLLECTION", "services")
        connection_string = get_postgres_connection_string()

        embeddings = OpenAIEmbeddings(model=embedding_model_name)
        vec_client = PGVector.from_existing_index(
            embedding=embeddings,
            connection_string=connection_string,
            collection_name=COLLECTION,
            use_jsonb=True,
        )

    # Perform similarity search (PGVector returns Documents directly)
    docs = await vec_client.asimilarity_search(question, k=max_results)
    return docs


# # Example Usage
# from modules.sub_agent_fns.get_docs_pgvector import get_docs_pgvector
#
# question = "contact information of your firm?"
# docs = await get_docs_pgvector(question, vec_client=pgvec_client)
# docs
