from src.metadata.document_metadata import DocumentMetadata
from src.ingestion.chunking.models import TextChunk
from src.embeddings.embedder import EmbeddedChunk
from src.storage.vector_store import ChromaVectorStore


def create_institution_chunk(
    chunk_id,
    scope,
    institution_id=None,
):
    metadata = DocumentMetadata(
        document_id=chunk_id,
        document_scope=scope,
        document_category="academic_regulation",
        document_title=f"Test document {chunk_id}",
        institution_id=institution_id,
    )

    chunk = TextChunk(
        chunk_id=chunk_id,
        source=f"{chunk_id}.pdf",
        chunk_index=1,
        text="University academic administration information.",
        token_count=6,
        page_numbers=(1,),
        heading="Academic Administration",
        document_metadata=metadata,
    )

    return EmbeddedChunk(
        chunk=chunk,
        embedding=(1.0, 0.0, 0.0, 0.0),
    )


def test_institution_filter_returns_common_and_selected_documents(
    tmp_path,
):
    store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma_db",
        collection_name="institution_filter_test",
        embedding_dimension=4,
    )

    chunks = [
        create_institution_chunk("common_001", "common"),
        create_institution_chunk("nsbm_001", "institution", "NSBM"),
        create_institution_chunk(
            "other_001",
            "institution",
            "OTHER_UNIVERSITY",
        ),
    ]

    store.upsert(chunks)

    results = store.query(
        query_embedding=(1.0, 0.0, 0.0, 0.0),
        top_k=5,
        where={
            "$or": [
                {"document_scope": "common"},
                {"institution_id": "NSBM"},
            ]
        },
    )

    result_ids = {record.chunk_id for record in results}

    assert result_ids == {"common_001", "nsbm_001"}
    assert "other_001" not in result_ids