from collections import Counter

from src.embeddings.embedder import MultilingualE5Embedder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.vector_store import ChromaVectorStore


COLLECTION_NAME = "multi_university_admin_chunks"

INSTITUTION_QUERIES = {
    "NSBM": (
        "NSBM examination regulations, instructions to candidates "
        "and repeat coursework procedures"
    ),
    "SLIIT": (
        "SLIIT semester module registration timelines "
        "and examination rules"
    ),
}


def load_all_records(store):
    """Load all existing records from the ChromaDB collection."""

    raw_data = store._collection.get(
        include=["documents", "metadatas"]
    )

    records = []

    for chunk_id, document, metadata in zip(
        raw_data["ids"],
        raw_data["documents"],
        raw_data["metadatas"],
    ):
        record = store._create_record(
            chunk_id=chunk_id,
            text=document,
            metadata=metadata,
        )

        records.append(record)

    return records


def is_valid_for_institution(record, institution_id):
    """Check whether a record belongs to the selected institution."""

    metadata = record.document_metadata

    if metadata is None:
        return False

    if metadata.document_scope == "common":
        return True

    return (
        metadata.document_scope == "institution"
        and metadata.institution_id == institution_id
    )


def main():

    print("=" * 65)
    print("NSBM + SLIIT RETRIEVAL VERIFICATION")
    print("=" * 65)

    # 1. Connect to the existing persistent database.
    store = ChromaVectorStore(
        persist_directory="data/vector_store",
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model="intfloat/multilingual-e5-small",
    )

    print("\nCollection:", COLLECTION_NAME)
    print("Database record count:", store.count())

    # 2. Load existing records.
    records = load_all_records(store)

    print("Loaded records:", len(records))

    if len(records) != store.count():
        raise RuntimeError("Loaded record count does not match database count.")

    # 3. Display institution distribution.
    scope_counts = Counter()
    institution_counts = Counter()

    for record in records:
        metadata = record.document_metadata

        if metadata is None:
            scope_counts["legacy"] += 1
            continue

        scope_counts[metadata.document_scope] += 1

        if metadata.institution_id:
            institution_counts[metadata.institution_id] += 1

    print("\nDocument scope distribution:")
    for scope, count in scope_counts.items():
        print(f"{scope}: {count}")

    print("\nInstitution distribution:")
    for institution, count in sorted(institution_counts.items()):
        print(f"{institution}: {count}")

    # 4. Initialize the embedding model.
    print("\nLoading embedding model...")

    embedder = MultilingualE5Embedder(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
        local_files_only=True,
    )

    # 5. Initialize the hybrid retriever.
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        corpus_records=records,
    )

    print("\n" + "=" * 65)
    print("TEST 1: INSTITUTION-SPECIFIC RETRIEVAL")
    print("=" * 65)

    institution_test_passed = True

    for institution_id, query in INSTITUTION_QUERIES.items():

        results = retriever.retrieve(
            query=query,
            top_k=8,
            institution_id=institution_id,
        )

        invalid_records = [
            result.record
            for result in results
            if not is_valid_for_institution(
                result.record,
                institution_id,
            )
        ]

        own_records = [
            result.record
            for result in results
            if (
                result.record.document_metadata is not None
                and result.record.document_metadata.document_scope
                == "institution"
                and result.record.document_metadata.institution_id
                == institution_id
            )
        ]

        print(f"\nInstitution: {institution_id}")
        print("Retrieved records:", len(results))
        print("Own institution records:", len(own_records))
        print("Invalid records:", len(invalid_records))

        for result in results[:3]:
            metadata = result.record.document_metadata

            if metadata:
                print(
                    "  -",
                    metadata.document_scope,
                    metadata.institution_id,
                    metadata.document_category,
                )

        if not results or invalid_records or not own_records:
            institution_test_passed = False
            print("RESULT: FAILED")
        else:
            print("RESULT: PASSED")

    # 6. Final report.
    print()
    print("=" * 65)
    print("FINAL VERIFICATION REPORT — NSBM + SLIIT")
    print("=" * 65)
    print(
        "Institution-specific retrieval:",
        "PASSED" if institution_test_passed else "FAILED",
    )

    if institution_test_passed:
        print("\nALL NSBM + SLIIT RETRIEVAL CHECKS PASSED")
    else:
        print("\nRETRIEVAL CHECKS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()