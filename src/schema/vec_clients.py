from enum import StrEnum, auto
from typing import TypeAlias

class Postgres_Timescale_Vec_Client_Name(StrEnum):
    """Postgres vector client name."""
    POSTGRES_TIMESCALE_VEC_CLIENT = "postgres_timescale_vec_client"

class Pinecone_Vec_Client_Name(StrEnum):
    """Pinecone vector client name."""
    PINECONE_VEC_CLIENT = "pinecone_vec_client"

class Postgres_PgVector_Vec_Client_Name(StrEnum):
    """Postgres vector client name."""
    POSTGRES_PGVECTOR_VEC_CLIENT = "postgres_pgvector_vec_client"

AllVecClientEnum: TypeAlias = (
    Postgres_Timescale_Vec_Client_Name
    | Pinecone_Vec_Client_Name
    | Postgres_PgVector_Vec_Client_Name
)
