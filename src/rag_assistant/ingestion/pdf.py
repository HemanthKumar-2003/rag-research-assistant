import hashlib
from pathlib import Path

from pypdf import PdfReader

from rag_assistant.ingestion.cleaning import clean_text
from rag_assistant.ingestion.models import DocumentPage
from rag_assistant.ingestion.validation import validate_pdf


def generate_document_id(file_path: str | Path) -> str:
    """Generate a stable SHA-256 identifier from file contents."""

    path = Path(file_path)

    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()

class PDFIngestionError(RuntimeError):
    """Base exception for PDF ingestion failures."""


class EncryptedPDFError(PDFIngestionError):
    """Raised when an encrypted PDF cannot be processed."""


class NoExtractableTextError(PDFIngestionError):
    """Raised when a PDF contains no extractable text."""

def extract_pdf_pages(file_path: str | Path,) -> list[DocumentPage]:
    """Extract cleaned text from a PDF while preserving page metadata."""

    path = validate_pdf(file_path)

    document_id = generate_document_id(path)

    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:
        raise PDFIngestionError(
            f"Unable to open PDF: {path.name}"
        ) from exc

    if reader.is_encrypted:
        raise EncryptedPDFError(
            f"Encrypted PDFs are not currently supported: {path.name}"
        )

    pages: list[DocumentPage] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        raw_text = page.extract_text() or ""

        cleaned_text = clean_text(raw_text)

        pages.append(
            DocumentPage(
                document_id=document_id,
                source_name=path.name,
                page_number=page_number,
                text=cleaned_text,
            )
        )

    if not any(page.text for page in pages):
        raise NoExtractableTextError(
            f"No extractable text found in {path.name}. "
            "The PDF may be scanned and require OCR."
        )

    return pages