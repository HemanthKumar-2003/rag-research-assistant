import psycopg
from pgvector.psycopg import register_vector

from rag_assistant.database.connection import (
    get_database_url,
)

EMBEDDING_DIMENSION = 384


def initialize_database() -> None:
    """Create the pgvector extension and application tables."""

    with psycopg.connect(
        get_database_url()
    ) as connection:
        connection.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )

        register_vector(connection)

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                created_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW()
            )
            """
        )

        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id TEXT PRIMARY KEY,

                document_id TEXT NOT NULL
                    REFERENCES documents(document_id)
                    ON DELETE CASCADE,

                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,

                embedding vector({EMBEDDING_DIMENSION})
                    NOT NULL,

                created_at TIMESTAMPTZ
                    NOT NULL DEFAULT NOW(),

                UNIQUE(document_id, chunk_index)
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_document_chunks_embedding_hnsw

            ON document_chunks

            USING hnsw (
                embedding vector_cosine_ops
            )
            """
        )