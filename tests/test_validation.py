import pytest

from rag_assistant.ingestion.validation import (
    DocumentValidationError,
    validate_pdf,
)


def test_validate_pdf_rejects_missing_file(
    tmp_path,
) -> None:
    file_path = tmp_path / "missing.pdf"

    with pytest.raises(DocumentValidationError):
        validate_pdf(file_path)


def test_validate_pdf_rejects_non_pdf(
    tmp_path,
) -> None:
    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "Not a PDF",
        encoding="utf-8",
    )

    with pytest.raises(DocumentValidationError):
        validate_pdf(file_path)


def test_validate_pdf_rejects_fake_pdf(
    tmp_path,
) -> None:
    file_path = tmp_path / "fake.pdf"

    file_path.write_text(
        "This is not really a PDF.",
        encoding="utf-8",
    )

    with pytest.raises(DocumentValidationError):
        validate_pdf(file_path)