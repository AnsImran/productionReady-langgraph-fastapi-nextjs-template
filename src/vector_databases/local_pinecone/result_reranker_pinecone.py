from typing import List, Dict, Any
from pinecone.core.openapi.db_data.model.query_response import QueryResponse
from pinecone.inference.models.rerank_result import RerankResult
from pinecone.pinecone_asyncio import PineconeAsyncio

async def async_rerank_results_pinecone(
    results: QueryResponse,
    query: str,
    pc: PineconeAsyncio,
    model: str = "bge-reranker-v2-m3",
    rank_field_name: str = "reranking_field",
    top_n: int = 4,
    return_documents: bool = True
) -> RerankResult:
    """
    Asynchronously rerank Pinecone query results using a specified reranker model.

    Args:
        results (QueryResponse): The query results returned from Pinecone, containing document matches.
        query (str): The query string used for reranking relevance.
        pc (Any): An instance of the Pinecone client that supports inference methods.
        model (str, optional): The name of the reranker model to use. Defaults to "bge-reranker-v2-m3".
        rank_field_name (str, optional): The name of the field used for reranking. Defaults to "reranking_field".
        top_n (int, optional): The number of top documents to return after reranking. Defaults to 4.
        return_documents (bool, optional): Whether to include document data in the rerank result. Defaults to True.

    Returns:
        RerankResult: The result of the reranking operation, including top-ranked documents and scores.
    """

    # Transform Pinecone matches into a format suitable for reranking
    transformed_documents: List[Dict[str, str]] = [
        {
            'id': match['id'],
            rank_field_name: '; '.join(f"{key}: {value}" for key, value in match['metadata'].items())
        }
        for match in results['matches']
    ]

    # Perform reranking using the specified model and parameters
    reranked_results: RerankResult = await pc.inference.rerank(
        model=model,
        query=query,
        documents=transformed_documents,
        rank_fields=[rank_field_name],
        top_n=top_n,
        return_documents=return_documents,
    )

    return reranked_results

# example usage
# reranked_results_field = await async_rerank_results(results, query, pc)
# reranked_results_field
