from typing import Optional, List, Union
from pinecone.db_data.index_asyncio import _IndexAsyncio
from pinecone.pinecone_asyncio import PineconeAsyncio
from pinecone.core.openapi.db_data.model.query_response import QueryResponse
from pinecone.inference.models.rerank_result import RerankResult
import logging

from vector_databases.local_pinecone.query_pinecone import async_query_pinecone
from vector_databases.local_pinecone.result_reranker_pinecone import async_rerank_results_pinecone

async def async_query_and_rerank_pinecone(
    query: str,
    idx: _IndexAsyncio,
    pc: PineconeAsyncio,
    FILTER_SOURCE: Optional[List[List]] = None,
    embedding_model_name: str = "text-embedding-3-small",
    TOP_K: int = 4,
    metadata_inclusion: bool = True,
    namespace: str = "__default__",
    reranker_model: str = "bge-reranker-v2-m3",
    rank_field_name: str = "reranking_field",
    top_n: int = 4,
    return_documents: bool = True
) -> Optional[RerankResult]:
    """
    Performs a two-stage pipeline: 
    1. Vector search using a Pinecone index.
    2. Reranking the results using a reranker model.

    Args:
        query (str): The search query string.
        idx (_IndexAsyncio): The Pinecone index instance for the initial vector search.
        pc (PineconeAsyncio): The Pinecone client instance for reranking.
        FILTER_SOURCE (Optional[List[List]], optional): Filter for narrowing down documents. Format: [['field'], ['val1', 'val2']]. Defaults to None.
        embedding_model_name (str, optional): Embedding model name for query embedding. Defaults to "text-embedding-3-small".
        TOP_K (int, optional): Number of top results to retrieve from Pinecone. Defaults to 4.
        metadata_inclusion (bool, optional): Whether to include metadata in results. Defaults to True.
        namespace (str, optional): Namespace to query within the Pinecone index. Defaults to "__default__".
        reranker_model (str, optional): Reranker model name. Defaults to "bge-reranker-v2-m3".
        rank_field_name (str, optional): Metadata field used for reranking. Defaults to "reranking_field".
        top_n (int, optional): Number of top reranked results to return. Defaults to 4.
        return_documents (bool, optional): Whether to include documents in the reranked result. Defaults to True.

    Returns:
        Optional[RerankResult]: Reranked results, or None if the query step failed.
    """
    try:
        # Step 1: Perform Pinecone vector query
        query_results: QueryResponse = await async_query_pinecone(
            query=query,
            idx=idx,
            FILTER_SOURCE=FILTER_SOURCE,
            embedding_model_name=embedding_model_name,
            TOP_K=TOP_K,
            metadata_inclusion=metadata_inclusion,
            namespace=namespace
        )

        if not query_results or not query_results.get("matches"):
            return None  # No valid matches or an error occurred during query

        # Step 2: Rerank the matched results
        reranked: RerankResult = await async_rerank_results(
            results=query_results,
            query=query,
            pc=pc,
            model=reranker_model,
            rank_field_name=rank_field_name,
            top_n=top_n,
            return_documents=return_documents
        )

        return reranked

    except Exception as e:
        logging.error(f"Error in async_query_and_rerank: {e}", exc_info=True)
        return None
