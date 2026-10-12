from pgvector import Vector

from rag_assistant.chunking.models import DocumentChunk
from rag_assistant.database.connection import (
    get_connection,
)
from rag_assistant.embeddings.models import (
    EmbeddedChunk,
    SearchResult,
)


def save_document(
    document_id: str,
    source_name: str,
    page_count: int,
) -> None:
    """Insert or update document metadata."""

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                document_id,
                source_name,
                page_count
            )
            VALUES (%s, %s, %s)

            ON CONFLICT (document_id)
            DO UPDATE SET
                source_name = EXCLUDED.source_name,
                page_count = EXCLUDED.page_count
            """,
            (
                document_id,
                source_name,
                page_count,
            ),
        )

def save_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
) -> None:
    """Persist document chunks and their embeddings."""

    if not embedded_chunks:
        return

    with get_connection() as connection:
        for item in embedded_chunks:
            chunk = item.chunk

            connection.execute(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    page_number,
                    chunk_index,
                    text,
                    embedding
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )

                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    page_number = EXCLUDED.page_number,
                    chunk_index = EXCLUDED.chunk_index,
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.page_number,
                    chunk.chunk_index,
                    chunk.text,
                    Vector(list(item.embedding)),
                ),
            )

def search_chunks(
    query_embedding: tuple[float, ...],
    top_k: int = 5,
) -> list[SearchResult]:
    """Retrieve chunks using cosine distance in PostgreSQL."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    query_vector = Vector(
        list(query_embedding)
    )

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.chunk_id,
                c.document_id,
                d.source_name,
                c.page_number,
                c.chunk_index,
                c.text,
                1 - (c.embedding <=> %s)
                    AS similarity
            FROM document_chunks AS c

            JOIN documents AS d
                ON d.document_id = c.document_id

            ORDER BY c.embedding <=> %s

            LIMIT %s
            """,
            (
                query_vector,
                query_vector,
                top_k,
            ),
        ).fetchall()

    return [
        SearchResult(
            chunk=DocumentChunk(
                chunk_id=row[0],
                document_id=row[1],
                source_name=row[2],
                page_number=row[3],
                chunk_index=row[4],
                text=row[5],
            ),
            score=float(row[6]),
        )
        for row in rows
    ]