import numpy as np

from rag_assistant.embeddings.embedder import TextEmbedder
from rag_assistant.embeddings.models import (
    EmbeddedChunk,
    SearchResult,
)


def cosine_similarity(
    first: tuple[float, ...],
    second: tuple[float, ...],
) -> float:
    """Calculate cosine similarity between two vectors."""

    vector_a = np.asarray(
        first,
        dtype=np.float32,
    )

    vector_b = np.asarray(
        second,
        dtype=np.float32,
    )

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Vectors must have matching dimensions."
        )

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError(
            "Cosine similarity is undefined "
            "for zero vectors."
        )

    return float(
        np.dot(vector_a, vector_b)
        / (norm_a * norm_b)
    )


def semantic_search(
    query: str,
    embedded_chunks: list[EmbeddedChunk],
    embedder: TextEmbedder,
    top_k: int = 5,
) -> list[SearchResult]:
    """Retrieve chunks most semantically similar to a query."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if not embedded_chunks:
        return []

    query_embedding = embedder.embed_texts(
        [query]
    )[0]

    results = [
        SearchResult(
            chunk=item.chunk,
            score=cosine_similarity(
                query_embedding,
                item.embedding,
            ),
        )
        for item in embedded_chunks
    ]

    results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return results[:top_k]