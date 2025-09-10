from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

async def async_add_documents_to_vectorstore_pinecone(documents: list[Document], index_name: str = 'pinecone-reranker') -> PineconeVectorStore:
    """
    Asynchronously adds documents to a Pinecone vector store using OpenAI embeddings.

    This function creates a new Pinecone vector store or adds documents to an existing one
    under the specified index name.

    Args:
        documents (List[Document]): A list of LangChain `Document` objects to embed and store.
        index_name (str): The name of the Pinecone index where the embeddings will be stored.

    Returns:
        PineconeVectorStore: The resulting Pinecone vector store containing the embedded documents.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = await PineconeVectorStore.afrom_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )

    return vectorstore


# example usage

# vs = await async_add_documents_to_vectorstore(documents, 'pinecone-reranker')

# # note kindly wait a bit after adding new chunks, they might not immediately show up in the `vector_count`
# await idx.describe_index_stats()