from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """Structured representation of one extracted PDF page."""

    document_id: str
    source_name: str
    page_number: int
    text: str