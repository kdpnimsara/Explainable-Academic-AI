import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from src.embeddings.embedder import (
    EmbeddedChunk,
    DEFAULT_MODEL_NAME,
)


@dataclass(frozen=True, slots=True)
class VectorStoreRecord:
    """Represent a retrieved record from ChromaDB."""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    token_count: int
    page_numbers: tuple[int, ...]
    heading: str | None
    distance: float | None = None


class ChromaVectorStore:
    """
    Persistent vector storage for SafeSteps LK.

    Stores text chunks, embeddings and metadata
    using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str | Path = "data/vector_store",
        collection_name: str = "safesteps_chunks",
        embedding_dimension: int = 384,
        distance_metric: str = "cosine",
        embedding_model: str = DEFAULT_MODEL_NAME,
    ) -> None:

        # Validate storage directory.
        if not isinstance(
            persist_directory,
            (str, Path),
        ):
            raise TypeError(
                "persist_directory must be a string or Path."
            )

        if not str(persist_directory).strip():
            raise ValueError(
                "persist_directory cannot be empty."
            )

        # Validate collection name.
        if (
            not isinstance(collection_name, str)
            or not collection_name.strip()
        ):
            raise ValueError(
                "collection_name cannot be empty."
            )

        # Validate embedding dimension.
        if (
            isinstance(embedding_dimension, bool)
            or not isinstance(embedding_dimension, int)
        ):
            raise TypeError(
                "embedding_dimension must be an integer."
            )

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be positive."
            )

        # Validate distance metric.
        if distance_metric not in (
            "cosine",
            "l2",
            "ip",
        ):
            raise ValueError(
                "distance_metric must be cosine, l2 or ip."
            )

        if (
            not isinstance(embedding_model, str)
            or not embedding_model.strip()
        ):
            raise ValueError(
                "embedding_model cannot be empty."
            )

        self.persist_directory = Path(
            persist_directory
        )

        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric
        self.embedding_model = embedding_model

        # Create storage directory.
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Initialize persistent ChromaDB client.
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Create or retrieve the collection.
        self._collection = (
            self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_metric,
                    "embedding_dimension": self.embedding_dimension,
                    "embedding_model": self.embedding_model,
                },
                embedding_function=None,
            )
        )

        # Check existing collection configuration.
        existing_metadata = (
            self._collection.metadata or {}
        )

        existing_dimension = existing_metadata.get(
            "embedding_dimension"
        )

        if (
            existing_dimension is not None
            and existing_dimension != self.embedding_dimension
        ):
            raise ValueError(
                "Existing collection uses a different "
                "embedding dimension."
            )

        existing_model = existing_metadata.get(
            "embedding_model"
        )

        if (
            existing_model is not None
            and existing_model != self.embedding_model
        ):
            raise ValueError(
                "Existing collection uses a different embedding model."
            )

        existing_metric = existing_metadata.get(
            "hnsw:space"
        )

        if (
            existing_metric is not None
            and existing_metric != self.distance_metric
        ):
            raise ValueError(
                "Existing collection uses a different distance metric."
            )

    # ======================================================
    # INTERNAL VALIDATION
    # ======================================================

    def _validate_embedding(
        self,
        embedding: Sequence[float],
    ) -> list[float]:
        """Validate an embedding vector."""

        try:
            vector = np.asarray(
                embedding,
                dtype=np.float32,
            )

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Embedding must contain numeric values."
            ) from error

        if vector.ndim != 1:
            raise ValueError(
                "Embedding must be one-dimensional."
            )

        if len(vector) != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension} dimensions, "
                f"received {len(vector)}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Embedding contains invalid numerical values."
            )

        return [
            float(value)
            for value in vector
        ]

    @staticmethod
    def _validate_ids(
        ids: Sequence[str],
    ) -> list[str]:
        """Validate chunk IDs."""

        if (
            not isinstance(ids, Sequence)
            or isinstance(ids, (str, bytes, bytearray))
        ):
            raise TypeError(
                "ids must be a sequence of strings."
            )

        clean_ids = []

        for index, chunk_id in enumerate(ids):

            if not isinstance(chunk_id, str):
                raise TypeError(
                    f"ID at index {index} must be a string."
                )

            if not chunk_id.strip():
                raise ValueError(
                    "Chunk IDs cannot be empty."
                )

            clean_ids.append(chunk_id)

        if len(clean_ids) != len(set(clean_ids)):
            raise ValueError(
                "Duplicate chunk IDs are not allowed."
            )

        return clean_ids

    # ======================================================
    # ADD / UPDATE EMBEDDINGS
    # ======================================================

    def upsert(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
    ) -> int:
        """
        Insert or update embedded chunks.

        Existing chunk IDs are updated instead
        of creating duplicate records.
        """

        if (
            not isinstance(embedded_chunks, Sequence)
            or isinstance(
                embedded_chunks,
                (str, bytes, bytearray),
            )
        ):
            raise TypeError(
                "embedded_chunks must be a sequence."
            )

        if not embedded_chunks:
            return 0

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for index, item in enumerate(embedded_chunks):

            if not isinstance(item, EmbeddedChunk):
                raise TypeError(
                    f"Item at index {index} must be an EmbeddedChunk."
                )

            chunk = item.chunk

            ids.append(chunk.chunk_id)

            documents.append(chunk.text)

            embeddings.append(
                self._validate_embedding(
                    item.embedding
                )
            )

            # ChromaDB metadata supports scalar values.
            # Page numbers are therefore stored as JSON.
            metadatas.append(
                {
                    "source": chunk.source,
                    "chunk_index": int(chunk.chunk_index),
                    "token_count": int(chunk.token_count),
                    "page_numbers": json.dumps(
                        list(chunk.page_numbers)
                    ),
                    "heading": chunk.heading or "",
                }
            )

        self._validate_ids(ids)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(ids)

    # ======================================================
    # DATABASE COUNT
    # ======================================================

    def count(self) -> int:
        """Return the number of stored chunks."""

        return self._collection.count()

    # ======================================================
    # GET RECORDS BY IDS
    # ======================================================

    def get_by_ids(
        self,
        ids: Sequence[str],
    ) -> list[VectorStoreRecord]:
        """Retrieve stored chunks using their IDs."""

        clean_ids = self._validate_ids(ids)

        if not clean_ids:
            return []

        result = self._collection.get(
            ids=clean_ids,
            include=[
                "documents",
                "metadatas",
            ],
        )

        records = {}

        returned_ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]

        for index, chunk_id in enumerate(returned_ids):

            records[chunk_id] = self._create_record(
                chunk_id=chunk_id,
                text=documents[index],
                metadata=metadatas[index],
            )

        # Preserve the requested order.
        return [
            records[chunk_id]
            for chunk_id in clean_ids
            if chunk_id in records
        ]

    # ======================================================
    # VECTOR SEARCH
    # ======================================================

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[VectorStoreRecord]:
        """
        Search for semantically similar chunks.

        Returns the nearest matching documents
        with their distance scores.
        """

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if where is not None and not isinstance(where, dict):
            raise TypeError(
                "where must be a dictionary or None."
            )

        vector = self._validate_embedding(
            query_embedding
        )

        if not self.count():
            return []

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.count()),
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        ids = result["ids"][0]

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        records = []

        for index, chunk_id in enumerate(ids):

            records.append(
                self._create_record(
                    chunk_id=chunk_id,
                    text=documents[index],
                    metadata=metadatas[index],
                    distance=float(distances[index]),
                )
            )

        return records

    # ======================================================
    # DELETE RECORDS
    # ======================================================

    def delete(
        self,
        ids: Sequence[str],
    ) -> None:
        """Delete selected chunks from the collection."""

        clean_ids = self._validate_ids(ids)

        if not clean_ids:
            return

        self._collection.delete(
            ids=clean_ids
        )

    # ======================================================
    # RECORD CONVERSION
    # ======================================================

    @staticmethod
    def _create_record(
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
        distance: float | None = None,
    ) -> VectorStoreRecord:
        """Convert ChromaDB results into typed records."""

        try:

            page_numbers = tuple(
                json.loads(
                    metadata["page_numbers"]
                )
            )

            return VectorStoreRecord(
                chunk_id=chunk_id,
                text=text,
                source=metadata["source"],
                chunk_index=int(
                    metadata["chunk_index"]
                ),
                token_count=int(
                    metadata["token_count"]
                ),
                page_numbers=page_numbers,
                heading=metadata.get("heading") or None,
                distance=distance,
            )

        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:

            raise RuntimeError(
                "Could not reconstruct stored chunk metadata."
            ) from error