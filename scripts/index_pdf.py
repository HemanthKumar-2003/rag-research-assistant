import argparse
from pathlib import Path

from rag_assistant.chunking import (
    ChunkingConfig,
    chunk_pages,
)
from rag_assistant.database import (
    initialize_database,
    save_document,
    save_embedded_chunks,
)
from rag_assistant.embeddings import TextEmbedder
from rag_assistant.ingestion import extract_pdf_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Index a PDF into PostgreSQL "
            "using vector embeddings."
        )
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
    )

    args = parser.parse_args()

    print("Initialising database...")
    initialize_database()

    print("Extracting PDF...")
    pages = extract_pdf_pages(
        args.pdf_path
    )

    print(
        f"Extracted {len(pages)} pages."
    )

    print("Chunking document...")

    chunks = chunk_pages(
        pages,
        config=ChunkingConfig(
            max_chars=1200,
            overlap_chars=200,
        ),
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Loading embedding model...")

    embedder = TextEmbedder()

    print("Generating embeddings...")

    embedded_chunks = embedder.embed_chunks(
        chunks
    )

    document_id = pages[0].document_id

    save_document(
        document_id=document_id,
        source_name=pages[0].source_name,
        page_count=len(pages),
    )

    print("Saving vectors to PostgreSQL...")

    save_embedded_chunks(
        embedded_chunks
    )

    print()
    print("Indexing complete.")
    print(
        f"Document: {pages[0].source_name}"
    )
    print(
        f"Document ID: {document_id}"
    )
    print(
        f"Stored chunks: {len(chunks)}"
    )


if __name__ == "__main__":
    main()