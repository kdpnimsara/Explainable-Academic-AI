
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.storage.vector_store import VectorStoreRecord


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A retrieved document with dense, TF-IDF and hybrid scores."""

    record: VectorStoreRecord
    dense_score: float | None
    tfidf_score: float
    hybrid_score: float


class HybridRetriever:
    """
    Retrieve relevant document chunks using:
    1. Dense semantic retrieval through ChromaDB.
    2. Keyword retrieval through TF-IDF.
    3. Reciprocal Rank Fusion (RRF) for hybrid ranking.

    corpus_records must contain the records available for TF-IDF search.
    """

    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        corpus_records: Sequence[VectorStoreRecord],
        rrf_k: int = 60,
    ) -> None:
        if not callable(getattr(embedder, "embed_query", None)):
            raise TypeError("embedder must provide embed_query().")

        if not callable(getattr(vector_store, "query", None)):
            raise TypeError("vector_store must provide query().")

        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, int)
        ):
            raise TypeError("rrf_k must be an integer.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if (
            not isinstance(corpus_records, Sequence)
            or isinstance(corpus_records, (str, bytes))
        ):
            raise TypeError("corpus_records must be a sequence.")

        self.embedder = embedder
        self.vector_store = vector_store
        self.rrf_k = rrf_k

        self._corpus: tuple[VectorStoreRecord, ...] = ()
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None

        self.refresh_corpus(corpus_records)

    def refresh_corpus(
        self,
        corpus_records: Sequence[VectorStoreRecord],
    ) -> None:
        """Replace the TF-IDF corpus and rebuild its index."""

        if (
            not isinstance(corpus_records, Sequence)
            or isinstance(corpus_records, (str, bytes))
        ):
            raise TypeError("corpus_records must be a sequence.")

        records = tuple(corpus_records)

        for record in records:
            if not isinstance(record, VectorStoreRecord):
                raise TypeError(
                    "Every corpus item must be a VectorStoreRecord."
                )

        ids = [record.chunk_id for record in records]

        if len(ids) != len(set(ids)):
            raise ValueError("Corpus contains duplicate chunk IDs.")

        self._corpus = records
        self._vectorizer = None
        self._tfidf_matrix = None

        if not records:
            return

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
        )

        try:
            matrix = vectorizer.fit_transform(
                [record.text for record in records]
            )
        except ValueError as error:
            if "empty vocabulary" not in str(error).lower():
                raise
            return

        self._vectorizer = vectorizer
        self._tfidf_matrix = matrix

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve and rank the most relevant document chunks.

        Returns results ordered by Reciprocal Rank Fusion score.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Step 1: Dense semantic retrieval.
        query_embedding = self.embedder.embed_query(query)

        dense_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Keep candidate details using chunk IDs.
        candidates: dict[str, dict[str, Any]] = {}

        for rank, record in enumerate(dense_results, start=1):
            distance = record.distance

            dense_score = (
                None if distance is None else 1.0 - float(distance)
            )

            candidates[record.chunk_id] = {
                "record": record,
                "dense_rank": rank,
                "dense_score": dense_score,
                "tfidf_rank": None,
                "tfidf_score": 0.0,
            }

        # Step 2: TF-IDF keyword retrieval.
        if self._vectorizer is not None and self._tfidf_matrix is not None:
            query_vector = self._vectorizer.transform([query])

            scores = cosine_similarity(
                self._tfidf_matrix,
                query_vector,
            ).ravel()

            ranked_indices = np.argsort(-scores, kind="stable")

            tfidf_rank = 0

            for index in ranked_indices:
                score = float(scores[index])

                # Ignore documents with no keyword similarity.
                if score <= 0:
                    break

                tfidf_rank += 1
                record = self._corpus[int(index)]

                if record.chunk_id not in candidates:
                    candidates[record.chunk_id] = {
                        "record": record,
                        "dense_rank": None,
                        "dense_score": None,
                        "tfidf_rank": tfidf_rank,
                        "tfidf_score": score,
                    }
                else:
                    candidates[record.chunk_id]["tfidf_rank"] = (
                        tfidf_rank
                    )
                    candidates[record.chunk_id]["tfidf_score"] = score

                if tfidf_rank >= top_k:
                    break

        # Step 3: Reciprocal Rank Fusion.
        results: list[RetrievedChunk] = []

        for item in candidates.values():
            dense_rank = item["dense_rank"]
            tfidf_rank = item["tfidf_rank"]

            hybrid_score = 0.0

            if dense_rank is not None:
                hybrid_score += 1.0 / (self.rrf_k + dense_rank)

            if tfidf_rank is not None:
                hybrid_score += 1.0 / (self.rrf_k + tfidf_rank)

            results.append(
                RetrievedChunk(
                    record=item["record"],
                    dense_score=item["dense_score"],
                    tfidf_score=item["tfidf_score"],
                    hybrid_score=hybrid_score,
                )
            )

        # Highest hybrid score comes first.
        results.sort(
            key=lambda item: (
                item.hybrid_score,
                item.tfidf_score,
                item.dense_score
                if item.dense_score is not None
                else float("-inf"),
            ),
            reverse=True,
        )

        return results[:top_k]
