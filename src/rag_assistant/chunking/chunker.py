import hashlib
import re
from dataclasses import dataclass

from rag_assistant.chunking.models import DocumentChunk
from rag_assistant.ingestion.models import DocumentPage


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration controlling document chunk creation."""

    max_chars: int = 1200
    overlap_chars: int = 200

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("max_chars must be at least 100.")

        if self.overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative.")

        if self.overlap_chars >= self.max_chars:
            raise ValueError(
                "overlap_chars must be smaller than max_chars."
            )
        
def _split_by_words(
    text: str,
    max_chars: int,
) -> list[str]:
    """Split text into bounded pieces without normally breaking words."""

    words = text.split()

    if not words:
        return []

    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_length = 0

            pieces.extend(
                word[index : index + max_chars]
                for index in range(0, len(word), max_chars)
            )
            continue

        additional_length = len(word)

        if current:
            additional_length += 1

        if current_length + additional_length <= max_chars:
            current.append(word)
            current_length += additional_length
        else:
            pieces.append(" ".join(current))
            current = [word]
            current_length = len(word)

    if current:
        pieces.append(" ".join(current))

    return pieces

def _split_long_paragraph(
    paragraph: str,
    max_chars: int,
) -> list[str]:
    """Split an oversized paragraph using sentence boundaries."""

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            paragraph,
        )
        if sentence.strip()
    ]

    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""

            pieces.extend(
                _split_by_words(
                    sentence,
                    max_chars,
                )
            )
            continue

        if not current:
            current = sentence
            continue

        candidate = f"{current} {sentence}"

        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return pieces

def _create_units(
    text: str,
    max_chars: int,
) -> list[str]:
    """Create paragraph-preferred text units."""

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n+",
            text,
        )
        if paragraph.strip()
    ]

    units: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(
                _split_long_paragraph(
                    paragraph,
                    max_chars,
                )
            )

    return units

def _get_overlap_tail(
    text: str,
    overlap_chars: int,
) -> str:
    """Return a word-aligned tail for chunk overlap."""

    if overlap_chars <= 0:
        return ""

    if len(text) <= overlap_chars:
        return text

    tail = text[-overlap_chars:]

    first_space = tail.find(" ")

    if first_space != -1:
        tail = tail[first_space + 1 :]

    return tail.strip()

def _merge_units(
    units: list[str],
    config: ChunkingConfig,
) -> list[str]:
    """Merge text units into bounded chunks with overlap."""

    if not units:
        return []

    chunks: list[str] = []
    current = units[0]

    for unit in units[1:]:
        candidate = f"{current}\n\n{unit}"

        if len(candidate) <= config.max_chars:
            current = candidate
            continue

        chunks.append(current)

        available_overlap = (
            config.max_chars
            - len(unit)
            - 2
        )

        overlap_size = min(
            config.overlap_chars,
            max(0, available_overlap),
        )

        overlap = _get_overlap_tail(
            current,
            overlap_size,
        )

        if overlap:
            current = f"{overlap}\n\n{unit}"
        else:
            current = unit

    if current:
        chunks.append(current)

    return chunks

def _generate_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """Generate a deterministic identifier for a chunk."""

    payload = (
        f"{document_id}:"
        f"{page_number}:"
        f"{chunk_index}:"
        f"{text}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:24]


def chunk_pages(
    pages: list[DocumentPage],
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Convert extracted document pages into retrieval-ready chunks."""

    if config is None:
        config = ChunkingConfig()

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        if not page.text.strip():
            continue

        units = _create_units(
            page.text,
            config.max_chars,
        )

        page_chunks = _merge_units(
            units,
            config,
        )

        for text in page_chunks:
            chunk_id = _generate_chunk_id(
                document_id=page.document_id,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    source_name=page.source_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                )
            )

            chunk_index += 1

    return chunks