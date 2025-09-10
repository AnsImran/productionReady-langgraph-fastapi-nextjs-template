import os
import asyncio

from pinecone import PineconeAsyncio, ServerlessSpec

from dotenv import load_dotenv


# load_dotenv()  # Load environment variables from a .env file
# DEFAULT_INDEX     = os.environ["INDEX_NAME"]

DEFAULT_INDEX = "pinecone-reranker"


async def get_async_index_pinecone(
    PINECONE_API_KEY:    str | None = None,
    index_name: str        = DEFAULT_INDEX,
    dimension:  int        = 1536,
    metric:     str        = "cosine"
):
    """
    Initializes an async Pinecone client and index.
    - Creates the index if it doesn't exist.
    - Waits for it to become READY.
    Returns: (PineconeAsyncio client, IndexAsyncio handle)
    """
    # 1. Start async client
    pc = PineconeAsyncio(PINECONE_API_KEY=PINECONE_API_KEY)

    # 2. Build your serverless spec
    cloud  = os.getenv("PINECONE_CLOUD", "aws")
    region = os.getenv("PINECONE_REGION", "us-east-1")
    spec   = ServerlessSpec(cloud=cloud, region=region)

    # 3. Check/create index
    existing = await pc.list_indexes()
    if index_name not in existing.names():
        await pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec
        )

    # 4. Wait until it's ready
    desc = await pc.describe_index(index_name)
    while not desc.status["ready"]:
        await asyncio.sleep(1)
        desc = await pc.describe_index(index_name)

    # 5. Grab your async index handle
    idx: IndexAsyncio = pc.IndexAsyncio(name=index_name, host=desc.host)
    return pc, idx




# # # Example usage

# # creating/connecting to index
# from modules.vector_databases.pinecone.get_async_index import get_async_index

# # if index not already present, it'll be created
# pc, idx = await get_async_index(index_name='bla-bla-bla')

# # await idx.close()
# # await pc.close()

# # try:
# #     await idx.describe_index_stats()
# # except Exception as e:
# #     print(f"RuntimeError: {repr(e)}")

# # try:
# #     await pc.list_indexes()
# # except Exception as e:
# #     print(f"RuntimeError: {repr(e)}")

# # await idx.describe_index_stats()

# # # deleting an index
# # # returns nothing but will delete for sure
# # await pc.delete_index(name='bla-bla-bla')




