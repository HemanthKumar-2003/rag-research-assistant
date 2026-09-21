from rag_assistant.chunking import (
    ChunkingConfig,
    chunk_pages,
)
from rag_assistant.ingestion.models import DocumentPage


def make_page(text: str) -> DocumentPage:
    return DocumentPage(
        document_id="test-document",
        source_name="test.pdf",
        page_number=1,
        text=text,
    )


def test_chunk_pages_preserves_metadata() -> None:
    page = make_page(
        "Deep learning is useful for image classification."
    )

    chunks = chunk_pages([page])

    assert len(chunks) == 1

    assert chunks[0].document_id == "test-document"
    assert chunks[0].source_name == "test.pdf"
    assert chunks[0].page_number == 1


def test_chunks_respect_maximum_size() -> None:
    text = " ".join(
        f"Sentence {index} contains useful information."
        for index in range(100)
    )

    page = make_page(text)

    config = ChunkingConfig(
        max_chars=300,
        overlap_chars=50,
    )

    chunks = chunk_pages(
        [page],
        config=config,
    )

    assert len(chunks) > 1

    assert all(
        len(chunk.text) <= config.max_chars
        for chunk in chunks
    )


def test_empty_pages_are_skipped() -> None:
    page = make_page("")

    chunks = chunk_pages([page])

    assert chunks == []


def test_chunk_ids_are_deterministic() -> None:
    page = make_page(
        "Transformers use attention mechanisms."
    )

    first_result = chunk_pages([page])
    second_result = chunk_pages([page])

    assert (
        first_result[0].chunk_id
        == second_result[0].chunk_id
    )


def test_chunk_indices_are_sequential() -> None:
    text = " ".join(
        f"This is sentence number {index}."
        for index in range(100)
    )

    page = make_page(text)

    chunks = chunk_pages(
        [page],
        config=ChunkingConfig(
            max_chars=250,
            overlap_chars=40,
        ),
    )

    indices = [
        chunk.chunk_index
        for chunk in chunks
    ]

    assert indices == list(range(len(chunks)))