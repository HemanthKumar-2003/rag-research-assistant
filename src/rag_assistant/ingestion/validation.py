from pathlib import Path


class DocumentValidationError(ValueError):
    """Raised when an input document fails validation."""


def validate_pdf(file_path: str | Path) -> Path:
    """Validate that a path points to a readable PDF file."""

    path = Path(file_path)

    if not path.exists():
        raise DocumentValidationError(
            f"File does not exist: {path}"
        )

    if not path.is_file():
        raise DocumentValidationError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise DocumentValidationError(
            f"Unsupported file type: {path.suffix}"
        )

    if path.stat().st_size == 0:
        raise DocumentValidationError(
            f"PDF file is empty: {path}"
        )

    with path.open("rb") as file:
        header = file.read(5)

    if header != b"%PDF-":
        raise DocumentValidationError(
            f"File does not appear to be a valid PDF: {path}"
        )

    return path