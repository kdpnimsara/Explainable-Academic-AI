import numpy as np
import pytest

import src.embeddings.embedder as embedder_module

from src.embeddings.embedder import (
    DEFAULT_MODEL_NAME,
    MultilingualE5Embedder,
)

from src.ingestion.chunking.models import TextChunk


# ======================================================
# FAKE MODEL FOR UNIT TESTS
# ======================================================

@pytest.fixture
def fake_model(monkeypatch):
    """Provide a deterministic fake embedding model."""

    instances = []

    class FakeSentenceTransformer:

        def __init__(
            self,
            model_name,
            device=None,
            local_files_only=False,
        ):

            self.model_name = model_name
            self.device = device
            self.local_files_only = local_files_only

            self.last_texts = None
            self.last_kwargs = None
            self.custom_output = None

            instances.append(self)

        def get_embedding_dimension(self):
            return 4

        def encode(self, sentences, **kwargs):

            self.last_texts = list(sentences)
            self.last_kwargs = kwargs

            if self.custom_output is not None:
                return self.custom_output

            return np.full(
                (len(sentences), 4),
                0.5,
                dtype=np.float32,
            )

    monkeypatch.setattr(
        embedder_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    return instances


@pytest.fixture
def embedder(fake_model):

    return MultilingualE5Embedder(
        batch_size=8,
        device="cpu",
    )


# ======================================================
# TEST DATA
# ======================================================

def create_chunk(
    chunk_id="chunk_001",
    source="disaster_act.pdf",
    chunk_index=1,
    text="Flood preparedness information.",
    page_numbers=(1,),
):

    return TextChunk(
        chunk_id=chunk_id,
        source=source,
        chunk_index=chunk_index,
        text=text,
        token_count=5,
        page_numbers=page_numbers,
        heading="Flood Preparedness",
    )


# ======================================================
# MODEL INITIALIZATION TESTS
# ======================================================

def test_model_initialization(fake_model):

    embedder = MultilingualE5Embedder()

    assert len(fake_model) == 1

    assert fake_model[0].model_name == DEFAULT_MODEL_NAME

    assert embedder.embedding_dimension == 4

    assert embedder.batch_size == 32


def test_model_uses_configured_settings(fake_model):

    MultilingualE5Embedder(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        batch_size=16,
        local_files_only=True,
    )

    model = fake_model[0]

    assert model.device == "cpu"
    assert model.local_files_only is True


# ======================================================
# DOCUMENT EMBEDDING TESTS
# ======================================================

def test_document_embeddings_are_generated(embedder):

    texts = [
        "Sri Lanka disaster preparedness.",
        "Flood safety information.",
    ]

    vectors = embedder.embed_documents(texts)

    assert vectors.shape == (2, 4)

    assert np.isfinite(vectors).all()


def test_document_embedding_uses_passage_prefix(
    embedder,
    fake_model,
):

    embedder.embed_documents(
        [
            "Flood preparedness.",
            "Landslide safety.",
        ]
    )

    model = fake_model[0]

    assert model.last_texts == [
        "passage: Flood preparedness.",
        "passage: Landslide safety.",
    ]


def test_document_embedding_preserves_order(embedder):

    texts = [
        "First document.",
        "Second document.",
        "Third document.",
    ]

    vectors = embedder.embed_documents(texts)

    assert vectors.shape[0] == len(texts)


def test_empty_documents_return_empty_matrix(embedder):

    vectors = embedder.embed_documents([])

    assert vectors.shape == (0, 4)

    assert vectors.dtype == np.float32


def test_reject_string_instead_of_document_sequence(embedder):

    with pytest.raises(TypeError):

        embedder.embed_documents(
            "This should be a list."
        )


def test_reject_non_string_document(embedder):

    with pytest.raises(TypeError):

        embedder.embed_documents(
            ["Valid text.", 123]
        )


def test_reject_empty_document(embedder):

    with pytest.raises(ValueError):

        embedder.embed_documents(
            ["Valid text.", "   "]
        )


# ======================================================
# QUERY EMBEDDING TESTS
# ======================================================

def test_query_embedding_is_generated(embedder):

    vector = embedder.embed_query(
        "What should people do during a flood?"
    )

    assert vector.shape == (4,)

    assert np.isfinite(vector).all()


def test_query_embedding_uses_query_prefix(
    embedder,
    fake_model,
):

    embedder.embed_query(
        "How can people prepare for landslides?"
    )

    assert fake_model[0].last_texts == [
        "query: How can people prepare for landslides?"
    ]


def test_reject_empty_query(embedder):

    with pytest.raises(ValueError):

        embedder.embed_query("   ")


def test_reject_non_string_query(embedder):

    with pytest.raises(TypeError):

        embedder.embed_query(123)


# ======================================================
# CHUNK EMBEDDING TESTS
# ======================================================

def test_chunk_embeddings_are_generated(embedder):

    chunks = [
        create_chunk(
            chunk_id="chunk_001",
            text="Flood safety information.",
        ),
        create_chunk(
            chunk_id="chunk_002",
            source="emergency_plan.pdf",
            chunk_index=2,
            text="Emergency response information.",
            page_numbers=(2, 3),
        ),
    ]

    results = embedder.embed_chunks(chunks)

    assert len(results) == 2

    assert len(results[0].embedding) == 4

    assert len(results[1].embedding) == 4


def test_chunk_metadata_is_preserved(embedder):

    original_chunk = create_chunk()

    results = embedder.embed_chunks(
        [original_chunk]
    )

    embedded = results[0]

    assert embedded.chunk is original_chunk

    assert embedded.chunk.chunk_id == "chunk_001"

    assert embedded.chunk.source == "disaster_act.pdf"

    assert embedded.chunk.page_numbers == (1,)

    assert embedded.chunk.heading == "Flood Preparedness"


def test_chunk_order_is_preserved(embedder):

    chunks = [
        create_chunk(
            chunk_id="chunk_001",
            chunk_index=1,
        ),
        create_chunk(
            chunk_id="chunk_002",
            chunk_index=2,
        ),
    ]

    results = embedder.embed_chunks(chunks)

    assert [
        result.chunk.chunk_id
        for result in results
    ] == [
        "chunk_001",
        "chunk_002",
    ]


def test_empty_chunks_return_empty_list(embedder):

    results = embedder.embed_chunks([])

    assert results == []


def test_reject_invalid_chunk(embedder):

    with pytest.raises(TypeError):

        embedder.embed_chunks(
            ["Not a TextChunk"]
        )


# ======================================================
# CONFIGURATION VALIDATION TESTS
# ======================================================

def test_reject_invalid_batch_size(fake_model):

    with pytest.raises(ValueError):

        MultilingualE5Embedder(
            batch_size=0
        )

    assert len(fake_model) == 0


def test_reject_negative_batch_size(fake_model):

    with pytest.raises(ValueError):

        MultilingualE5Embedder(
            batch_size=-1
        )


def test_reject_invalid_batch_size_type(fake_model):

    with pytest.raises(TypeError):

        MultilingualE5Embedder(
            batch_size=1.5
        )


# ======================================================
# EMBEDDING OUTPUT VALIDATION TESTS
# ======================================================

def test_reject_incorrect_embedding_dimension(
    embedder,
    fake_model,
):

    fake_model[0].custom_output = np.ones(
        (1, 3),
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError):

        embedder.embed_documents(
            ["Test document."]
        )


def test_reject_non_finite_embeddings(
    embedder,
    fake_model,
):

    fake_model[0].custom_output = np.array(
        [[np.nan, 0.5, 0.5, 0.5]],
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError):

        embedder.embed_documents(
            ["Test document."]
        )


# ======================================================
# REAL MODEL INTEGRATION TEST
# ======================================================

def test_real_multilingual_e5_model():

    try:

        real_embedder = MultilingualE5Embedder(
            device="cpu",
            local_files_only=True,
        )

    except OSError:

        pytest.skip(
            "Multilingual E5 model is not available locally."
        )

    document_vectors = real_embedder.embed_documents(
        [
            "Sri Lanka disaster preparedness information.",
        ]
    )

    query_vector = real_embedder.embed_query(
        "What is disaster preparedness?"
    )

    assert document_vectors.shape == (1, 384)

    assert query_vector.shape == (384,)

    assert np.isfinite(document_vectors).all()

    assert np.isfinite(query_vector).all()

    assert np.linalg.norm(query_vector) == pytest.approx(
        1.0,
        abs=1e-5,
    )