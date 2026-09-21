import argparse
from pathlib import Path

from rag_assistant.chunking import (
    ChunkingConfig,
    chunk_pages,
)
from rag_assistant.embeddings import (
    TextEmbedder,
    semantic_search,
)
from rag_assistant.ingestion import extract_pdf_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run semantic search over a PDF."
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
    )

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    print("Extracting PDF...")

    pages = extract_pdf_pages(
        args.pdf_path
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

    print(
        f"Embedding dimension: "
        f"{embedder.dimension}"
    )

    print("Creating embeddings...")

    embedded_chunks = embedder.embed_chunks(
        chunks
    )

    print("Searching...")

    results = semantic_search(
        query=args.query,
        embedded_chunks=embedded_chunks,
        embedder=embedder,
        top_k=args.top_k,
    )

    print()

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)

        print(
            f"Rank: {rank} | "
            f"Score: {result.score:.4f}"
        )

        print(
            f"Source: {result.chunk.source_name} | "
            f"Page: {result.chunk.page_number} | "
            f"Chunk: {result.chunk.chunk_index}"
        )

        print("-" * 80)

        print(result.chunk.text)

        print()


if __name__ == "__main__":
    main()