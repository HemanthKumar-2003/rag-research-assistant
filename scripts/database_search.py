import argparse

from rag_assistant.database import (
    search_chunks,
)
from rag_assistant.embeddings import TextEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Search indexed document chunks."
        )
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

    print("Loading embedding model...")

    embedder = TextEmbedder()

    print("Embedding query...")

    query_embedding = embedder.embed_texts(
        [args.query]
    )[0]

    print("Searching database...")
    print()

    results = search_chunks(
        query_embedding=query_embedding,
        top_k=args.top_k,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 80)

        print(
            f"Rank: {rank} | "
            f"Similarity: {result.score:.4f}"
        )

        print(
            f"Source: {result.chunk.source_name} | "
            f"Page: {result.chunk.page_number}"
        )

        print(
            f"Chunk: {result.chunk.chunk_index}"
        )

        print("-" * 80)

        print(
            result.chunk.text
        )

        print()


if __name__ == "__main__":
    main()