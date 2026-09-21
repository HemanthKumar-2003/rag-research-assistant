import argparse
from pathlib import Path

from rag_assistant.ingestion import extract_pdf_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect extracted PDF text."
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF file.",
    )

    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf_path)

    print(f"Extracted {len(pages)} pages.")
    print()

    for page in pages[:3]:
        print("=" * 80)
        print(
            f"Source: {page.source_name} | "
            f"Page: {page.page_number}"
        )
        print("=" * 80)
        print(page.text[:1000])
        print()


if __name__ == "__main__":
    main()