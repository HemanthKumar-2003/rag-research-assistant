import pytest

from rag_assistant.embeddings import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    vector = (1.0, 0.0, 0.0)

    score = cosine_similarity(
        vector,
        vector,
    )

    assert score == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    first = (1.0, 0.0)
    second = (0.0, 1.0)

    score = cosine_similarity(
        first,
        second,
    )

    assert score == pytest.approx(0.0)


def test_opposite_vectors_have_negative_similarity() -> None:
    first = (1.0, 0.0)
    second = (-1.0, 0.0)

    score = cosine_similarity(
        first,
        second,
    )

    assert score == pytest.approx(-1.0)