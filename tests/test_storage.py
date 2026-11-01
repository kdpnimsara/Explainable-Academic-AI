import numpy as np
import pytest

from src.embeddings.embedder import EmbeddedChunk
from src.ingestion.chunking.models import TextChunk
from src.storage.vector_store import ChromaVectorStore


@pytest.fixture
def store(tmp_path):

    return ChromaVectorStore(
        persist_directory=tmp_path / "chroma_db",
        collection_name="test_safesteps",
        embedding_dimension=4,
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
    heading="Flood Preparedness",
):

    return TextChunk(
        chunk_id=chunk_id,
        source=source,
        chunk_index=chunk_index,
        text=text,
        token_count=5,
        page_numbers=page_numbers,
        heading=heading,
    )


def create_embedded_chunk(
    chunk=None,
    embedding=(1.0, 0.0, 0.0, 0.0),
):

    if chunk is None:
        chunk = create_chunk()

    return EmbeddedChunk(
        chunk=chunk,
        embedding=embedding,
    )


# ======================================================
# INITIALIZATION TESTS
# ======================================================

def test_storage_initialization(store):

    assert store.count() == 0


def test_storage_creates_database_directory(tmp_path):

    directory = tmp_path / "new_database"

    ChromaVectorStore(
        persist_directory=directory,
        collection_name="test_chunks",
        embedding_dimension=4,
    )

    assert directory.exists()
    assert directory.is_dir()


def test_reject_invalid_embedding_dimension(tmp_path):

    with pytest.raises(ValueError):

        ChromaVectorStore(
            persist_directory=tmp_path,
            embedding_dimension=0,
        )


def test_reject_invalid_distance_metric(tmp_path):

    with pytest.raises(ValueError):

        ChromaVectorStore(
            persist_directory=tmp_path,
            distance_metric="invalid",
        )


# ======================================================
# UPSERT TESTS
# ======================================================

def test_upsert_stores_embedded_chunks(store):

    chunks = [
        create_embedded_chunk(
            create_chunk(
                chunk_id="chunk_001",
            ),
            (1.0, 0.0, 0.0, 0.0),
        ),
        create_embedded_chunk(
            create_chunk(
                chunk_id="chunk_002",
                source="emergency_plan.pdf",
                chunk_index=2,
                text="Emergency response information.",
                page_numbers=(2, 3),
            ),
            (0.0, 1.0, 0.0, 0.0),
        ),
    ]

    stored = store.upsert(chunks)

    assert stored == 2
    assert store.count() == 2


def test_upsert_empty_list(store):

    result = store.upsert([])

    assert result == 0
    assert store.count() == 0


def test_reject_invalid_embedded_chunk(store):

    with pytest.raises(TypeError):

        store.upsert(
            ["invalid"]
        )


def test_reject_duplicate_chunk_ids(store):

    chunks = [
        create_embedded_chunk(
            create_chunk(
                chunk_id="duplicate",
            )
        ),
        create_embedded_chunk(
            create_chunk(
                chunk_id="duplicate",
                text="Another document.",
            )
        ),
    ]

    with pytest.raises(ValueError):

        store.upsert(chunks)


def test_reject_wrong_embedding_dimension(store):

    invalid_item = create_embedded_chunk(
        embedding=(1.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError):

        store.upsert(
            [invalid_item]
        )


def test_upsert_updates_existing_chunk(store):

    first = create_embedded_chunk(
        create_chunk(
            chunk_id="chunk_001",
            text="Original information.",
        )
    )

    store.upsert([first])

    updated = create_embedded_chunk(
        create_chunk(
            chunk_id="chunk_001",
            text="Updated information.",
        )
    )

    store.upsert([updated])

    assert store.count() == 1

    results = store.get_by_ids(
        ["chunk_001"]
    )

    assert results[0].text == "Updated information."


# ======================================================
# RETRIEVAL TESTS
# ======================================================

def test_get_by_ids_returns_correct_records(store):

    chunks = [
        create_embedded_chunk(
            create_chunk(
                chunk_id="chunk_001",
            )
        ),
        create_embedded_chunk(
            create_chunk(
                chunk_id="chunk_002",
                source="neop.pdf",
                chunk_index=2,
                text="Emergency planning information.",
                page_numbers=(2, 3),
            )
        ),
    ]

    store.upsert(chunks)

    results = store.get_by_ids(
        ["chunk_002", "chunk_001"]
    )

    assert len(results) == 2

    assert results[0].chunk_id == "chunk_002"
    assert results[1].chunk_id == "chunk_001"


def test_metadata_is_preserved(store):

    chunk = create_chunk(
        chunk_id="metadata_test",
        source="disaster_act.pdf",
        page_numbers=(2, 4, 5),
        heading="Disaster Management",
    )

    store.upsert(
        [
            create_embedded_chunk(
                chunk=chunk
            )
        ]
    )

    result = store.get_by_ids(
        ["metadata_test"]
    )[0]

    assert result.source == "disaster_act.pdf"

    assert result.page_numbers == (2, 4, 5)

    assert result.heading == "Disaster Management"

    assert result.chunk_index == 1

    assert result.token_count == 5


def test_missing_ids_are_ignored(store):

    store.upsert(
        [create_embedded_chunk()]
    )

    results = store.get_by_ids(
        ["missing_chunk"]
    )

    assert results == []


def test_empty_ids_return_empty_list(store):

    assert store.get_by_ids([]) == []


# ======================================================
# VECTOR QUERY TESTS
# ======================================================

def test_query_returns_most_similar_chunk(store):

    chunks = [
        create_embedded_chunk(
            create_chunk(
                chunk_id="flood_chunk",
                text="Flood safety information.",
            ),
            (1.0, 0.0, 0.0, 0.0),
        ),
        create_embedded_chunk(
            create_chunk(
                chunk_id="landslide_chunk",
                source="landslide.pdf",
                text="Landslide preparedness information.",
            ),
            (0.0, 1.0, 0.0, 0.0),
        ),
    ]

    store.upsert(chunks)

    results = store.query(
        query_embedding=(1.0, 0.0, 0.0, 0.0),
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].chunk_id == "flood_chunk"

    assert results[0].distance is not None

    assert results[0].distance <= results[1].distance


def test_query_respects_top_k(store):

    chunks = [
        create_embedded_chunk(
            create_chunk(
                chunk_id=f"chunk_{index}",
                chunk_index=index,
            ),
            (1.0, 0.0, 0.0, 0.0),
        )
        for index in range(1, 4)
    ]

    store.upsert(chunks)

    results = store.query(
        query_embedding=(1.0, 0.0, 0.0, 0.0),
        top_k=2,
    )

    assert len(results) == 2


def test_query_supports_metadata_filter(store):

    chunks = [
        create_embedded_chunk(
            create_chunk(
                chunk_id="flood",
                source="flood.pdf",
            ),
            (1.0, 0.0, 0.0, 0.0),
        ),
        create_embedded_chunk(
            create_chunk(
                chunk_id="landslide",
                source="landslide.pdf",
            ),
            (1.0, 0.0, 0.0, 0.0),
        ),
    ]

    store.upsert(chunks)

    results = store.query(
        query_embedding=(1.0, 0.0, 0.0, 0.0),
        top_k=5,
        where={
            "source": "flood.pdf"
        },
    )

    assert len(results) == 1

    assert results[0].chunk_id == "flood"


def test_query_empty_collection(store):

    results = store.query(
        query_embedding=(1.0, 0.0, 0.0, 0.0)
    )

    assert results == []


def test_reject_invalid_query_dimension(store):

    with pytest.raises(ValueError):

        store.query(
            query_embedding=(1.0, 0.0, 0.0)
        )


def test_reject_invalid_top_k(store):

    with pytest.raises(ValueError):

        store.query(
            query_embedding=(1.0, 0.0, 0.0, 0.0),
            top_k=0,
        )


def test_reject_non_finite_query(store):

    with pytest.raises(ValueError):

        store.query(
            query_embedding=(
                np.nan,
                0.0,
                0.0,
                0.0,
            )
        )


# ======================================================
# DELETE TESTS
# ======================================================

def test_delete_chunks(store):

    chunks = [
        create_embedded_chunk(
            create_chunk(
                chunk_id="chunk_001",
            )
        ),
        create_embedded_chunk(
            create_chunk(
                chunk_id="chunk_002",
                chunk_index=2,
            )
        ),
    ]

    store.upsert(chunks)

    assert store.count() == 2

    store.delete(
        ["chunk_001"]
    )

    assert store.count() == 1

    remaining = store.get_by_ids(
        ["chunk_002"]
    )

    assert len(remaining) == 1


def test_delete_empty_list(store):

    store.delete([])

    assert store.count() == 0