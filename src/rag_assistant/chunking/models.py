from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrieval-ready piece of an ingested document."""

    chunk_id: str
    document_id: str
    source_name: str
    page_number: int
    chunk_index: int
    text: str