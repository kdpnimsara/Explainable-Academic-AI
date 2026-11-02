
import pytest

from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.vector_store import VectorStoreRecord


def make_record(chunk_id, text, source="sample.pdf"):
    return VectorStoreRecord(
        chunk_id=chunk_id,
        text=text,
        source=source,
        chunk_index=1,
        token_count=len(text.split()),
        page_numbers=(1,),
        heading=None,
        distance=None,
    )


class FakeEmbedder:
    def embed_query(self, query):
        return [1.0, 0.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def query(self, query_embedding, top_k=5):
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

    result_ids = {item.record.chunk_id for item in results}

    # Dense and TF-IDF candidates should both be represented.
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
        retriever.retrieve("flood preparedness", top_k=0)


def test_empty_corpus_uses_dense_results(sample_records):
    store = FakeVectorStore(sample_records[:1])

    retriever = HybridRetriever(
        FakeEmbedder(),
        store,
        [],
    )

    results = retriever.retrieve("flood", top_k=2)

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
