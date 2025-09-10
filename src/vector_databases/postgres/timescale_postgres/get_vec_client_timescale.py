from timescale_vector import client

from schema.models import (
    AllEmbeddingModelEnum,                    ###########################################
    OpenAIEmbeddingModelName,
)


def get_vec_client_timescale(TIMESCALE_DB_URI:str, table_name:str='services', embedding_model_name:AllEmbeddingModelEnum=OpenAIEmbeddingModelName.TEXT_EMBEDDING_3_SMALL) -> client.Async:
    """
    Initializes and returns an asynchronous Timescale vector database client.

    Args:
        TIMESCALE_DB_URI:     str =                    ''
        table_name:           str                   =  'services'
        embedding_model_name: AllEmbeddingModelEnum =  'text-embedding-3-small'
    Returns:
        client.Async: An initialized asynchronous Timescale vector client.
    
    Raises:
        KeyError: If the 'TIMESCALE_DB_URI' environment variable is not set.
    """
    if embedding_model_name == OpenAIEmbeddingModelName.TEXT_EMBEDDING_3_SMALL:
        embedding_dims: int = 1536
    # elif ...:

    # Initialize and return the async vector database client
    return client.Async(
        TIMESCALE_DB_URI,
        table_name,
        embedding_dims
    )
