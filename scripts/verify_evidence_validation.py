
"""Verify evidence validation using the real NSBM ChromaDB records."""

from pathlib import Path

from src.embeddings.embedder import MultilingualE5Embedder
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
)
from src.storage.vector_store import ChromaVectorStore
from src.generation.evidence_validator import EvidenceValidator


MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "multi_university_admin_chunks"


def load_all_records(store):
    """Load existing vector database records."""

    raw_data = store._collection.get(
        include=["documents", "metadatas"]
    )

    return [
        store._create_record(
            chunk_id=chunk_id,
            text=document,
            metadata=metadata,
        )
        for chunk_id, document, metadata in zip(
            raw_data["ids"],
            raw_data["documents"],
            raw_data["metadatas"],
        )
    ]


def main():
    print("=" * 65)
    print("REAL NSBM EVIDENCE VALIDATION")
    print("=" * 65)

    store = ChromaVectorStore(
        persist_directory=Path("data/vector_store"),
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL_NAME,
    )

    print("\nDatabase records:", store.count())

    records = load_all_records(store)

    if not records:
        raise RuntimeError("No records found in the vector database.")

    embedder = MultilingualE5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
        local_files_only=True,
    )

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        corpus_records=records,
    )

    validator = EvidenceValidator()

    test_cases = [
        {
            "name": "SUPPORTED",
            "question": (
                "How long must a candidate wait before "
                "leaving the examination hall?"
            ),
            "expected": True,
        },
        {
            "name": "UNSUPPORTED",
            "question": (
                "What is the exact amount of the NSBM "
                "library late return fine for 2026?"
            ),
            "expected": False,
        },
    ]

    all_passed = True

    for case in test_cases:

        question = case["question"]

        print("\n" + "=" * 65)
        print(case["name"])
        print("Question:", question)

        chunks = retriever.retrieve(
            query=question,
            top_k=5,
            institution_id="NSBM",
        )

        result = validator.validate(
            question,
            retrieved_chunks=chunks,
        )

        print("\nRetrieved documents:", len(chunks))

        for index, chunk in enumerate(chunks, start=1):
            print(
                f"{index}. {chunk.record.source} "
                f"| Pages: {chunk.record.page_numbers}"
            )

        print("\nValidation result:")
        print("Sufficient:", result.sufficient)
        print("Coverage:", round(result.coverage, 3))
        print("Matched terms:", result.matched_terms)
        print("Missing terms:", result.missing_terms)
        print("Reason:", result.reason)

        passed = result.sufficient == case["expected"]

        print(
            "\nTEST:",
            "PASSED" if passed else "FAILED",
        )

        if not passed:
            all_passed = False

    print("\n" + "=" * 65)
    print("FINAL VALIDATION REPORT")
    print("=" * 65)

    if all_passed:
        print("ALL REAL-DOCUMENT CHECKS PASSED")
    else:
        print("SOME REAL-DOCUMENT CHECKS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
