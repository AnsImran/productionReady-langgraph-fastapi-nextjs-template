from pinecone.core.openapi.inference.model.document import Document
from vector_databases.local_pinecone.query_and_rerank_pinecone import async_query_and_rerank_pinecone


async def get_docs_pinecone(
        question: str,
        pc=None, 
        idx=None, 
        embedding_model_name: str = "text-embedding-3-small", 
        max_results: int = 5,
        ) -> list[Document]:    
    
    response = await async_query_and_rerank_pinecone(question, pc=pc, idx=idx, TOP_K=max_results, top_n=max_results, embedding_model_name=embedding_model_name)

    return [response.rerank_result.data[i]['document'] for i in range(len(response.data))]


# # Example usage
# from modules.sub_agent_fns.get_docs_pinecone import get_docs_pinecone

# question = 'contact information of your firm?'
# docs = await get_docs(question, pc=pc, idx=idx, embedding_model_name=embedding_model_name)
# docs



