import openai
from langchain_core.documents import Document
from core.settings import settings
from timescale_vector import client



async def get_docs_timescale(
        question:             str, 
        vec_client:           client.Async  = None, 
        embedding_model_name: str = "text-embedding-3-small", 
        max_results:          int = 5,
        ) -> list[Document]:
    
    response  = openai.embeddings.create(input=[question], model=embedding_model_name)
    query_vec = response.data[0].embedding
    results   = await vec_client.search(query_vec, limit=max_results)

    return [Document(page_content=row[2], metadata=row[1]) for row in results]


# # Example Usage
# from modules.sub_agent_fns.get_docs_timescale import get_docs_timescale

# question = 'contact information of your firm?'
# docs = await get_docs_timescale(question, vec_client=vec_client)
# docs


