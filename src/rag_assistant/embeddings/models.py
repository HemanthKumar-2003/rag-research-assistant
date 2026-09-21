from dataclasses import dataclass

from rag_assistant.chunking.models import DocumentChunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A document chunk paired with its embedding vector."""

    chunk: DocumentChunk
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved document chunk with its similarity score."""

    chunk: DocumentChunk
    score: float