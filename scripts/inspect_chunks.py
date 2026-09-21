import argparse
from pathlib import Path
from statistics import mean

from rag_assistant.chunking import (
    ChunkingConfig,
    chunk_pages,
)
from rag_assistant.ingestion import extract_pdf_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect document chunking."
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
    )

    args = parser.parse_args()

    pages = extract_pdf_pages(
        args.pdf_path
    )

    config = ChunkingConfig(
        max_chars=1200,
        overlap_chars=200,
    )

    chunks = chunk_pages(
        pages,
        config=config,
    )

    lengths = [
        len(chunk.text)
        for chunk in chunks
    ]

    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")

    if lengths:
        print(
            f"Average chunk length: "
            f"{mean(lengths):.1f}"
        )

        print(
            f"Smallest chunk: "
            f"{min(lengths)}"
        )

        print(
            f"Largest chunk: "
            f"{max(lengths)}"
        )

    print()

    for chunk in chunks[:5]:
        print("=" * 80)

        print(
            f"Chunk: {chunk.chunk_index} | "
            f"Page: {chunk.page_number} | "
            f"Length: {len(chunk.text)}"
        )

        print(
            f"Chunk ID: {chunk.chunk_id}"
        )

        print("-" * 80)

        print(chunk.text)

        print()


if __name__ == "__main__":
    main()