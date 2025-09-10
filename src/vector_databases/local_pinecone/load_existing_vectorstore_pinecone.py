from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone.db_data.index_asyncio import _IndexAsyncio

async def async_load_existing_vectorstore_pinecone(index_object: _IndexAsyncio, embedding_model: str = "text-embedding-3-small") -> PineconeVectorStore:
    """
    Loads an existing Pinecone vector store using OpenAI embeddings.

    This function assumes the index has already been created and populated
    with vector data.

    Args:
        index_object (_IndexAsyncio): A Pinecone asyncio index object from the Pinecone client.
        embedding_model (str): Name of the OpenAI embedding model to use.

    Returns:
        PineconeVectorStore: A LangChain-compatible vector store wrapper around the given index.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)

    return PineconeVectorStore(
        index=index_object,
        embedding=embeddings
    )
