from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.chunking.models import TextChunk


DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-small"


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Store a text chunk together with its embedding vector."""

    chunk: TextChunk
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate embedded chunk data."""

        if not isinstance(self.chunk, TextChunk):
            raise TypeError(
                "chunk must be a TextChunk instance."
            )

        if not self.embedding:
            raise ValueError(
                "Embedding cannot be empty."
            )

        try:
            vector = np.asarray(
                self.embedding,
                dtype=np.float32,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Embedding must contain numeric values."
            ) from error

        if vector.ndim != 1:
            raise ValueError(
                "Embedding must be a one-dimensional vector."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Embedding contains invalid numerical values."
            )


class MultilingualE5Embedder:
    """
    Generate multilingual text embeddings using
    the multilingual-e5-small model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        local_files_only: bool = False,
    ) -> None:

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                "model_name must be a non-empty string."
            )

        if not isinstance(device, str) or not device.strip():
            raise ValueError(
                "device must be a non-empty string."
            )

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
        ):
            raise TypeError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if not isinstance(normalize_embeddings, bool):
            raise TypeError(
                "normalize_embeddings must be a boolean."
            )

        if not isinstance(local_files_only, bool):
            raise TypeError(
                "local_files_only must be a boolean."
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Load the Sentence Transformer model.
        self._model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )

        # Use the updated Sentence Transformers API.
        dimension = self._model.get_embedding_dimension()

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise RuntimeError(
                "Could not determine the embedding dimension."
            )

        self.embedding_dimension = dimension

    # ======================================================
    # INTERNAL EMBEDDING GENERATION
    # ======================================================

    def _encode(
        self,
        texts: list[str],
        prefix: str,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Generate validated embedding vectors."""

        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        prepared_texts = [
            f"{prefix}{text}"
            for text in texts
        ]

        vectors = self._model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        expected_shape = (
            len(texts),
            self.embedding_dimension,
        )

        if vectors.shape != expected_shape:
            raise RuntimeError(
                f"Expected embeddings with shape {expected_shape}, "
                f"but received {vectors.shape}."
            )

        if not np.isfinite(vectors).all():
            raise RuntimeError(
                "Generated embeddings contain invalid numerical values."
            )

        return vectors

    # ======================================================
    # DOCUMENT EMBEDDINGS
    # ======================================================

    def embed_documents(
        self,
        texts: Sequence[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple document texts.

        Uses the E5 passage prefix.
        """

        if (
            not isinstance(texts, Sequence)
            or isinstance(texts, (str, bytes, bytearray))
        ):
            raise TypeError(
                "texts must be a sequence of strings."
            )

        if not isinstance(show_progress_bar, bool):
            raise TypeError(
                "show_progress_bar must be a boolean."
            )

        clean_texts = []

        for index, text in enumerate(texts):

            if not isinstance(text, str):
                raise TypeError(
                    f"Document at index {index} must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"Document at index {index} cannot be empty."
                )

            clean_texts.append(text.strip())

        return self._encode(
            texts=clean_texts,
            prefix="passage: ",
            show_progress_bar=show_progress_bar,
        )

    # ======================================================
    # QUERY EMBEDDING
    # ======================================================

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Generate an embedding for a user question.

        Uses the E5 query prefix.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        vectors = self._encode(
            texts=[query.strip()],
            prefix="query: ",
        )

        return vectors[0]

    # ======================================================
    # TEXT CHUNK EMBEDDINGS
    # ======================================================

    def embed_chunks(
        self,
        chunks: Sequence[TextChunk],
        show_progress_bar: bool = False,
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for TextChunk objects.

        Original chunk metadata is preserved.
        """

        if (
            not isinstance(chunks, Sequence)
            or isinstance(chunks, (str, bytes, bytearray))
        ):
            raise TypeError(
                "chunks must be a sequence of TextChunk objects."
            )

        if not isinstance(show_progress_bar, bool):
            raise TypeError(
                "show_progress_bar must be a boolean."
            )

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, TextChunk):
                raise TypeError(
                    f"Item at index {index} must be a TextChunk."
                )

        if not chunks:
            return []

        vectors = self.embed_documents(
            [chunk.text for chunk in chunks],
            show_progress_bar=show_progress_bar,
        )

        embedded_chunks = []

        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        ):

            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=tuple(
                        float(value)
                        for value in vector
                    ),
                )
            )

        return embedded_chunks