from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_assistant.chunking.models import DocumentChunk
from rag_assistant.embeddings.models import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the text embedding model."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

class TextEmbedder:
    """Create normalised embeddings for text and document chunks."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig()

        self.model = SentenceTransformer(
            self.config.model_name
        )

    @property
    def dimension(self) -> int:
        dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension."
            )

        return dimension

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[tuple[float, ...]]:
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        return [
            tuple(float(value) for value in vector)
            for vector in vectors
        ]


    def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embed_texts(texts)

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding,
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        ]