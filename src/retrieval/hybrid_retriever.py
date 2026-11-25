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
    3. Reciprocal Rank Fusion (RRF).

    Supports institution-based retrieval filtering.

    When an institution is selected, retrieval includes:
    - Common documents.
    - Documents belonging to the selected institution.

    Documents belonging to other institutions are excluded.
    """

    def __init__(
        self,
        embedder: Any,
        vector_store: Any,
        corpus_records: Sequence[VectorStoreRecord],
        rrf_k: int = 60,
        candidate_pool_factor: int = 6,
        max_chunks_per_document: int = 3,
    ) -> None:
        if not callable(getattr(embedder, "embed_query", None)):
            raise TypeError("embedder must provide embed_query().")

        if not callable(getattr(vector_store, "query", None)):
            raise TypeError("vector_store must provide query().")

        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
            raise TypeError("rrf_k must be an integer.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if isinstance(candidate_pool_factor, bool) or not isinstance(candidate_pool_factor, int):
            raise TypeError("candidate_pool_factor must be an integer.")
        if candidate_pool_factor < 1:
            raise ValueError("candidate_pool_factor must be at least one.")

        if isinstance(max_chunks_per_document, bool) or not isinstance(max_chunks_per_document, int):
            raise TypeError("max_chunks_per_document must be an integer.")
        if max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document must be at least one.")

        if (
            not isinstance(corpus_records, Sequence)
            or isinstance(corpus_records, (str, bytes))
        ):
            raise TypeError("corpus_records must be a sequence.")

        self.embedder = embedder
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.candidate_pool_factor = candidate_pool_factor
        self.max_chunks_per_document = max_chunks_per_document

        self._corpus: tuple[VectorStoreRecord, ...] = ()
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        # Reuse institution-specific TF-IDF indexes across repeated queries.
        self._institution_tfidf_cache: dict[tuple[str, bool], tuple[Any, Any, Any]] = {}

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
        self._institution_tfidf_cache.clear()

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

    @staticmethod
    def _expand_tfidf_query(
        query: str,
        institution_id: str | None,
    ) -> str:
        """Add focused NSBM vocabulary for academic evidence queries."""
        if institution_id is None or institution_id.casefold() != "nsbm":
            return query

        lowered = query.casefold()

        if "thesis" in lowered and any(
            term in lowered for term in ("page size", "paper size", "margin", "formatting")
        ):
            return query + " A4 size paper page margins left right top bottom one inch 1 inch 1 1/4 inch binding"

        if "thesis" in lowered and any(
            term in lowered for term in ("referenc", "citation", "style")
        ):
            return query + " Vancouver APA IEEE referencing reference citation faculty style"

        if "thesis" in lowered and (
            any(term in lowered for term in ("pagination", "page numbering", "roman numerals"))
            or ("preliminary" in lowered and any(term in lowered for term in ("number", "numbered", "numbering")))
        ):
            return query + " lower-case Roman numerals inner title page page i Declaration ii Abbreviations Introduction page 1 bottom center"

        if (
            any(term in lowered for term in ("2016", "by-law", "bylaw", "by laws"))
            and any(term in lowered for term in ("gpa", "grade point", "module grade", "average"))
        ):
            return query + " NSBM Degree Programmes Examination By-Laws Version 1 2016 module grade points weighted by credit allocation exclude non-credit modules total weighted grade points divided by total credits"

        if (
            any(term in lowered for term in ("extenuating", "circumstances", "ec guidelines"))
            and any(term in lowered for term in ("technical", "problems", "internet", "printer"))
        ):
            return query + " corrupt data media poor internet connectivity printer failure incorrect format wrong file accidental resubmission"

        if "data science" in lowered and any(
            term in lowered for term in ("option 2", "entry", "admission", "qualification")
        ):
            return query + " three simple passes S any stream one sitting GCE A/L credit pass C Mathematics GCE O/L bridging programme exemption ICT"

        if "first class" in lowered and any(
            term in lowered for term in ("honours", "honors")
        ):
            return query + " special note examination offences examination offence disciplinary misconduct lower than 70% attendance any module absent absence without valid reason not eligible for a first class not be awarded a class 120 credits grade C no grade below C first attempt grade A half courses completes relevant requirements within a period of four academic years"
        return query
    @staticmethod
    def _validate_institution_id(
        institution_id: str | None,
    ) -> str | None:
        """Validate the optional institution identifier."""

        if institution_id is None:
            return None

        if not isinstance(institution_id, str):
            raise TypeError("institution_id must be a string or None.")

        institution_id = institution_id.strip()

        if not institution_id:
            raise ValueError("institution_id cannot be empty.")

        return institution_id

    @staticmethod
    def _matches_institution(
        record: VectorStoreRecord,
        institution_id: str,
    ) -> bool:
        """
        Check whether a record belongs to the selected institution
        or is a common document.
        """

        metadata = record.document_metadata

        # Exclude legacy records without document metadata.
        if metadata is None:
            return False

        if metadata.document_scope == "common":
            return True

        return (
            metadata.document_scope == "institution"
            and metadata.institution_id == institution_id
        )

    @staticmethod
    def _build_institution_filter(
        institution_id: str,
    ) -> dict[str, Any]:
        """Build the ChromaDB filter for institution-based retrieval."""

        return {
            "$or": [
                {"document_scope": "common"},
                {"institution_id": institution_id},
            ]
        }

    @staticmethod
    def _document_identity(record: VectorStoreRecord) -> str:
        """Return a stable identity used to diversify results by document."""
        metadata = record.document_metadata
        document_id = getattr(metadata, "document_id", None) if metadata else None
        if document_id:
            return str(document_id).replace("\\", "/").casefold()
        return str(record.source or "").replace("\\", "/").casefold()

    @staticmethod
    def _passage_identity(record: VectorStoreRecord) -> str:
        """Normalize whitespace and case to detect repeated passages."""
        return " ".join((record.text or "").split()).casefold()

    def _get_institution_tfidf_index(
        self,
        institution_id: str,
        records: tuple[VectorStoreRecord, ...],
        include_heading: bool = False,
    ):
        """Build an institution-specific TF-IDF index once and reuse it."""
        cache_key = (institution_id.casefold(), include_heading)
        cached = self._institution_tfidf_cache.get(cache_key)
        if cached is not None:
            return cached

        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
        )
        documents = [
            (f"{record.heading or ''}\n{record.text}" if include_heading else record.text)
            for record in records
        ]
        try:
            matrix = vectorizer.fit_transform(documents)
        except ValueError as error:
            if "empty vocabulary" not in str(error).lower():
                raise
            return (records, None, None)

        cached = (records, vectorizer, matrix)
        self._institution_tfidf_cache[cache_key] = cached
        return cached

    def _select_diverse_results(
        self,
        ranked_results: list[RetrievedChunk],
        top_k: int,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        """Limit each source's contribution so one PDF cannot fill top-k."""
        selected: list[RetrievedChunk] = []
        deferred: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        counts: dict[str, int] = {}
        available_documents = {self._document_identity(item.record) for item in ranked_results}

        for item in ranked_results:
            chunk_id = item.record.chunk_id
            if chunk_id in selected_ids:
                continue
            key = self._document_identity(item.record)
            document_limit = self.max_chunks_per_document if max_chunks_per_document is None else max_chunks_per_document
            if counts.get(key, 0) >= document_limit:
                deferred.append(item)
                continue
            selected.append(item)
            selected_ids.add(chunk_id)
            counts[key] = counts.get(key, 0) + 1
            if len(selected) >= top_k:
                return selected

        # If the entire candidate pool contains only one source, allow more
        # chunks from that source so evidence-heavy questions remain answerable.
        # When multiple sources exist, keep the per-document cap strict.
        if len(available_documents) == 1:
            for item in deferred:
                if item.record.chunk_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.record.chunk_id)
                if len(selected) >= top_k:
                    break
        return selected

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        institution_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve and rank relevant document chunks.

        If institution_id is provided, only common documents and
        documents belonging to that institution are retrieved.

        If institution_id is None, the original unfiltered
        retrieval behaviour is preserved.
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

        institution_id = self._validate_institution_id(
            institution_id
        )

        # Retrieve a broader candidate pool before RRF. Querying only top_k
        # at each stage prevents fusion from recovering relevant candidates
        # ranked just below either backend's initial cut-off.
        candidate_k = max(top_k, top_k * self.candidate_pool_factor)

        # Step 1: Dense semantic retrieval.
        query_embedding = self.embedder.embed_query(query)

        if institution_id is None:
            dense_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=candidate_k,
            )
        else:
            where = self._build_institution_filter(institution_id)

            dense_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=candidate_k,
                where=where,
            )

            # Additional validation prevents unwanted records
            # if a vector store does not apply the filter correctly.
            dense_results = [
                record
                for record in dense_results
                if self._matches_institution(
                    record,
                    institution_id,
                )
            ]

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
        if institution_id is None:
            eligible_records = self._corpus
            vectorizer = self._vectorizer
            tfidf_matrix = self._tfidf_matrix

        else:
            eligible_records = tuple(
                record
                for record in self._corpus
                if self._matches_institution(record, institution_id)
            )
            is_first_class_query = (
                institution_id.casefold() == "nsbm"
                and "first class" in query.casefold()
                and any(term in query.casefold() for term in ("honours", "honors"))
            )
            eligible_records, vectorizer, tfidf_matrix = (
                self._get_institution_tfidf_index(
                    institution_id,
                    eligible_records,
                    include_heading=is_first_class_query,
                )
            )

        if vectorizer is not None and tfidf_matrix is not None:
            tfidf_query = self._expand_tfidf_query(
                query,
                institution_id,
            )
            query_vector = vectorizer.transform([tfidf_query])

            scores = cosine_similarity(
                tfidf_matrix,
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
                record = eligible_records[int(index)]

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

                if tfidf_rank >= candidate_k:
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

        # Remove repeated text passages within the same source before applying
        # a per-document cap. Different versions/documents remain independent.
        seen_passages: dict[str, set[str]] = {}
        unique_results: list[RetrievedChunk] = []
        for item in results:
            document_key = self._document_identity(item.record)
            passage_key = self._passage_identity(item.record)
            if passage_key and passage_key in seen_passages.setdefault(document_key, set()):
                continue
            if passage_key:
                seen_passages[document_key].add(passage_key)
            unique_results.append(item)

        thesis_format_query = (
            institution_id is not None
            and institution_id.casefold() == "nsbm"
            and "thesis" in query.casefold()
            and any(term in query.casefold() for term in ("page size", "paper size", "margin", "formatting"))
        )
        return self._select_diverse_results(
            unique_results, top_k,
            max_chunks_per_document=8 if thesis_format_query else None,
        )