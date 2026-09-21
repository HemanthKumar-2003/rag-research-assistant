from rag_assistant.ingestion.cleaning import clean_text


def test_clean_text_removes_extra_spaces() -> None:
    raw = "Deep     learning\tis useful."

    cleaned = clean_text(raw)

    assert cleaned == "Deep learning is useful."


def test_clean_text_normalises_blank_lines() -> None:
    raw = "Paragraph one.\n\n\n\nParagraph two."

    cleaned = clean_text(raw)

    assert cleaned == "Paragraph one.\n\nParagraph two."


def test_clean_text_removes_null_characters() -> None:
    raw = "Vision\x00 Transformer"

    cleaned = clean_text(raw)

    assert cleaned == "Vision Transformer"