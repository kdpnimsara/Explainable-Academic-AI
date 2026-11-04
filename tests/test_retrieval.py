import pytest

from src.metadata.document_metadata import DocumentMetadata
from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.vector_store import VectorStoreRecord


def make_record(
    chunk_id,
    text,
    source="sample.pdf",
    document_metadata=None,
):
    return VectorStoreRecord(
        chunk_id=chunk_id,
        text=text,
        source=source,
        chunk_index=1,
        token_count=len(text.split()),
        page_numbers=(1,),
        heading=None,
        distance=None,
        document_metadata=document_metadata,
    )


def make_metadata(
    document_id,
    scope,
    institution_id=None,
):
    return DocumentMetadata(
        document_id=document_id,
        document_scope=scope,
        document_category="academic_regulation",
        document_title=f"Test document {document_id}",
        institution_id=institution_id,
    )


class FakeEmbedder:
    def embed_query(self, query):
        return [1.0, 0.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def query(
        self,
        query_embedding,
        top_k=5,
        where=None,
    ):
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "top_k": top_k,
                "where": where,
            }
        )

        return self.results[:top_k]


@pytest.fixture
def sample_records():
    return [
        make_record(
            "chunk_1",
            "Flood preparedness and emergency shelter information.",
        ),
        make_record(
            "chunk_2",
            "Landslide warnings and evacuation procedures.",
        ),
        make_record(
            "chunk_3",
            "Flood emergency shelter and evacuation planning.",
        ),
    ]


def test_hybrid_retrieval_returns_results(sample_records):
    store = FakeVectorStore(sample_records[:2])

    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        corpus_records=sample_records,
    )

    results = retriever.retrieve(
        "flood emergency shelter",
        top_k=3,
    )

    assert len(results) == 3
    assert all(result.hybrid_score > 0 for result in results)


def test_hybrid_retrieval_combines_both_methods(sample_records):
    store = FakeVectorStore(sample_records[:2])

    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        corpus_records=sample_records,
    )

    results = retriever.retrieve(
        "flood emergency shelter",
        top_k=3,
    )

    result_ids = {
        item.record.chunk_id
        for item in results
    }

    assert "chunk_1" in result_ids
    assert "chunk_3" in result_ids


def test_retrieval_rejects_empty_query(sample_records):
    retriever = HybridRetriever(
        FakeEmbedder(),
        FakeVectorStore([]),
        sample_records,
    )

    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retrieval_rejects_invalid_top_k(sample_records):
    retriever = HybridRetriever(
        FakeEmbedder(),
        FakeVectorStore([]),
        sample_records,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "flood preparedness",
            top_k=0,
        )


def test_empty_corpus_uses_dense_results(sample_records):
    store = FakeVectorStore(sample_records[:1])

    retriever = HybridRetriever(
        FakeEmbedder(),
        store,
        [],
    )

    results = retriever.retrieve(
        "flood",
        top_k=2,
    )

    assert len(results) == 1
    assert results[0].record.chunk_id == "chunk_1"
    assert results[0].tfidf_score == 0.0


def test_duplicate_corpus_ids_are_rejected(sample_records):
    with pytest.raises(ValueError):
        HybridRetriever(
            FakeEmbedder(),
            FakeVectorStore([]),
            [sample_records[0], sample_records[0]],
        )


# Institution-specific retrieval tests


@pytest.fixture
def institution_records():
    common_metadata = make_metadata(
        "common_001",
        "common",
    )

    nsbm_metadata = make_metadata(
        "nsbm_001",
        "institution",
        "NSBM",
    )

    other_metadata = make_metadata(
        "other_001",
        "institution",
        "OTHER_UNIVERSITY",
    )

    return [
        make_record(
            "common_001",
            "University academic regulations and disaster preparedness emergency policies.",
            "common_disaster_guide.pdf",
            common_metadata,
        ),
        make_record(
            "nsbm_001",
            "NSBM academic regulations examination and student registration.",
            "nsbm_regulations.pdf",
            nsbm_metadata,
        ),
        make_record(
            "other_001",
            "OTHER UNIVERSITY academic regulations examination and registration.",
            "other_regulations.pdf",
            other_metadata,
        ),
        make_record(
            "legacy_001",
            "Academic regulations examination and student registration.",
            "legacy_document.pdf",
        ),
    ]


def test_institution_retrieval_includes_common_and_selected_documents(
    institution_records,
):
    store = FakeVectorStore(institution_records)

    retriever = HybridRetriever(
        FakeEmbedder(),
        store,
        institution_records,
    )

    results = retriever.retrieve(
        "academic regulations and disaster preparedness",
        top_k=5,
        institution_id="NSBM",
    )

    result_ids = {
        item.record.chunk_id
        for item in results
    }

    assert "common_001" in result_ids
    assert "nsbm_001" in result_ids
    assert "other_001" not in result_ids
    assert "legacy_001" not in result_ids


def test_institution_filter_is_passed_to_vector_store(
    institution_records,
):
    store = FakeVectorStore(institution_records)

    retriever = HybridRetriever(
        FakeEmbedder(),
        store,
        institution_records,
    )

    retriever.retrieve(
        "academic regulations",
        top_k=5,
        institution_id="NSBM",
    )

    assert store.calls[0]["where"] == {
        "$or": [
            {"document_scope": "common"},
            {"institution_id": "NSBM"},
        ]
    }


def test_institution_retrieval_excludes_other_universities_from_tfidf(
    institution_records,
):
    store = FakeVectorStore([])

    retriever = HybridRetriever(
        FakeEmbedder(),
        store,
        institution_records,
    )

    results = retriever.retrieve(
        "academic regulations disaster preparedness",
        top_k=5,
        institution_id="NSBM",
    )

    result_ids = {
        item.record.chunk_id
        for item in results
    }

    assert "common_001" in result_ids
    assert "nsbm_001" in result_ids
    assert "other_001" not in result_ids
    assert "legacy_001" not in result_ids


@pytest.mark.parametrize(
    "institution_id",
    [
        "",
        "   ",
    ],
)
def test_institution_retrieval_rejects_invalid_institution_id(
    sample_records,
    institution_id,
):
    retriever = HybridRetriever(
        FakeEmbedder(),
        FakeVectorStore(sample_records),
        sample_records,
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "academic regulations",
            institution_id=institution_id,
        )


def test_institution_retrieval_allows_none_institution_id(
    sample_records,
):
    retriever = HybridRetriever(
        FakeEmbedder(),
        FakeVectorStore(sample_records),
        sample_records,
    )

    results = retriever.retrieve(
        "flood preparedness",
        institution_id=None,
    )

    assert len(results) > 0


def test_institution_retrieval_rejects_non_string_id(sample_records):
    retriever = HybridRetriever(
        FakeEmbedder(),
        FakeVectorStore(sample_records),
        sample_records,
    )

    with pytest.raises(TypeError):
        retriever.retrieve(
            "academic regulations",
            institution_id=123,
        )