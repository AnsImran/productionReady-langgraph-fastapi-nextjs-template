from typing import Union, List, Optional
from langchain_openai import OpenAIEmbeddings
from pinecone.db_data.index_asyncio import _IndexAsyncio
import logging
from pinecone.core.openapi.db_data.model.query_response import QueryResponse

async def async_query_pinecone(
    query:              str,
    idx:                _IndexAsyncio,
    FILTER_SOURCE:      Optional[List[List]] = None,
    embedding_model_name:         str = "text-embedding-3-small",
    TOP_K:              int = 4,
    metadata_inclusion: bool = True,
    namespace:          str = "__default__"
) -> QueryResponse:
    """
    Asynchronously embeds a query using OpenAI embeddings and queries a Pinecone index.

    Returns:
        dict or None: The query response from the Pinecone index, or None if an error occurred.
    """
    try:
        # Validate FILTER_SOURCE
        if FILTER_SOURCE and (not isinstance(FILTER_SOURCE, list) or len(FILTER_SOURCE) != 2):
            raise ValueError("FILTER_SOURCE must be a list of two lists, e.g., [['field'], ['value1', 'value2']]")

        # Initialize the OpenAI embeddings model
        try:
            embeddings = OpenAIEmbeddings(model=embedding_model_name)
        except Exception as e:
            logging.error(f"Failed to initialize OpenAIEmbeddings with model '{embedding_model_name}': {e}")
            return None

        # Generate the query embedding
        try:
            query_embedding = await embeddings.aembed_query(query)
        except Exception as e:
            logging.error(f"Failed to embed query '{query}': {e}")
            return None

        # Build the filter
        filter_dict = None
        if FILTER_SOURCE:
            try:
                field, values = FILTER_SOURCE
                filter_dict = {field[0]: {"$in": values}}
            except Exception as e:
                logging.error(f"Invalid FILTER_SOURCE format: {e}")
                return None

        # Perform the asynchronous query on the Pinecone index
        try:
            response = await idx.query(
                vector=query_embedding,
                top_k=TOP_K,
                filter=filter_dict,
                include_metadata=metadata_inclusion,
                namespace=namespace
            )
        except Exception as e:
            logging.error(f"Pinecone query failed: {e}")
            return None

        if not response.get('matches'):
            print("No results returned. Are you sure your filters are spelled correctly?\n", "Warning: If a metadata field or its corresponding value does not exist in the index, the query will return no results. Please double-check your filters.")
        return response



    except Exception as e:
        logging.error(f"Unexpected error in query_pinecone_with_embedding: {e}")
        return None



# # example usage
# query = 'contract?'
# filter_sources = None


# # filter_sources = [['source'], ['to_split\\Modele-de-contrat-engagement-16072024.pdf', 'to_split\\Quantum Computing Explained _part_13.pdf']]
# filter_sources = [['general_scientific_field'], ['administrative_docs', 'computer_science']]
# # filter_sources = [['bitch'], ['administrative_docs', 'computer_science']] # this query will return no results

# results = await query_pinecone_with_embedding(query, idx, FILTER_SOURCE=filter_sources)
# results



