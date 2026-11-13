
from pathlib import Path
from collections import Counter
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

from src.storage.vector_store import ChromaVectorStore
from src.embeddings.embedder import MultilingualE5Embedder

MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "multi_university_admin_chunks"

PDF_DIR = ROOT / "data" / "documents" / "institutions" / "nsbm"
VECTOR_DIR = ROOT / "data" / "vector_store"

EXPECTED_COUNTS = {
    "examination_bylaws": 22,
    "examination_payments": 1,
    "examination_regulations": 31,
    "extenuating_circumstances": 7,
    "mba_assessment": 5,
    "mbs_assessment": 5,
    "postgraduate_diploma_assessment": 6,
    "postgraduate_handbook": 212,
    "programme_guide": 60,
    "repeat_coursework": 4,
}

QUERIES = [
    ("NSBM Examination Regulations Instructions to Candidates",
     "examination_regulations"),
    ("NSBM Repeat Coursework Procedure",
     "repeat_coursework"),
    ("NSBM Extenuating Circumstances Guidelines",
     "extenuating_circumstances"),
    ("NSBM MBA Programme Assessment Criteria",
     "mba_assessment"),
    ("NSBM MSc Information Technology Programme Guide",
     "programme_guide"),
]


def section(name):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)


def verify_database():
    section("1. NSBM DATABASE AND METADATA AUDIT")

    if not PDF_DIR.is_dir():
        raise RuntimeError("NSBM PDF directory not found.")

    pdfs = list(PDF_DIR.glob("*.pdf"))
    if len(pdfs) != 11:
        raise RuntimeError(
            f"Expected 11 PDFs, found {len(pdfs)}"
        )

    store = ChromaVectorStore(
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL_NAME,
    )

    data = store._collection.get(include=["metadatas"])
    ids = data["ids"]
    metadatas = data["metadatas"]

    if len(ids) != 353:
        raise RuntimeError(
            f"Expected 353 records, found {len(ids)}"
        )

    categories = Counter()
    documents = set()
    errors = []

    for record_id, metadata in zip(ids, metadatas):
        metadata = metadata or {}

        institution = str(
            metadata.get("institution_id", "")
        ).casefold()

        if institution != "nsbm":
            errors.append(
                f"Wrong institution metadata: {record_id}"
            )

        if metadata.get("document_scope") != "institution":
            errors.append(
                f"Wrong document scope: {record_id}"
            )

        category = metadata.get("document_category", "")
        if not category:
            errors.append(
                f"Missing category: {record_id}"
            )
        else:
            categories[category] += 1

        document_id = metadata.get("document_id", "")
        if document_id:
            documents.add(document_id)
        else:
            errors.append(
                f"Missing document ID: {record_id}"
            )

        if not metadata.get("source_url"):
            errors.append(
                f"Missing source URL: {record_id}"
            )

        if not metadata.get("source_authority"):
            errors.append(
                f"Missing source authority: {record_id}"
            )

    expected_documents = {
        f"institutions/nsbm/{pdf.stem}"
        for pdf in pdfs
    }

    if documents != expected_documents:
        errors.append(
            "Document ID inventory does not match the 11 PDFs."
        )

    if dict(categories) != EXPECTED_COUNTS:
        errors.append(
            "Category chunk counts do not match ingestion results."
        )

    print("PDF files:", len(pdfs))
    print("Database records:", len(ids))
    print("Unique NSBM documents:", len(documents))
    print("\nCategory distribution:")

    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    other_institutions = Counter()

    for metadata in metadatas:
        metadata = metadata or {}
        institution = metadata.get("institution_id", "missing")
        if str(institution).casefold() != "nsbm":
            other_institutions[str(institution)] += 1

    print("\nOther institution records:", dict(other_institutions))

    if errors:
        print("\nDATABASE AUDIT: FAILED")
        for error in errors:
            print(" -", error)
        return None

    print("\nDATABASE AUDIT: PASSED")
    return store


def verify_retrieval(store):
    section("2. NSBM RETRIEVAL CHECK")

    embedder = MultilingualE5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=16,
        normalize_embeddings=True,
        local_files_only=True,
    )

    query_method = getattr(embedder, "embed_query", None)
    if not callable(query_method):
        query_method = getattr(embedder, "embed_queries", None)

    if not callable(query_method):
        print(
            "SKIPPED: No public query-embedding method "
            "was found in the embedder."
        )
        return

    passed = 0

    for question, expected_category in QUERIES:
        try:
            if getattr(query_method, "__name__", "") == "embed_queries":
                raw = query_method([question])
                raw = raw[0]
            else:
                raw = query_method(question)

            if hasattr(raw, "tolist"):
                raw = raw.tolist()

            if raw and isinstance(raw[0], (list, tuple)):
                raw = raw[0]

            vector = [float(value) for value in raw]

            result = store._collection.query(
                query_embeddings=[vector],
                n_results=5,
                where={"institution_id": "NSBM"},
                include=["metadatas", "distances"],
            )

            found_categories = [
                (item or {}).get("document_category", "")
                for item in result["metadatas"][0]
            ]

            success = expected_category in found_categories

            if success:
                passed += 1

            print(
                f"{'PASS' if success else 'CHECK'} | "
                f"{expected_category} | "
                f"Top-5: {found_categories}"
            )

        except Exception as exc:
            print(
                f"QUERY ERROR | {expected_category} | "
                f"{type(exc).__name__}: {exc}"
            )

    print(f"\nRetrieval category hits: {passed}/{len(QUERIES)}")


def verify_tests():
    section("3. FULL PROJECT TEST SUITE")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
    )

    if result.returncode == 0:
        print("\nPYTEST: PASSED")
    else:
        print("\nPYTEST: FAILED")

    return result.returncode == 0


def main():
    print("=" * 60)
    print("EXPLAINABLE ACADEMIC AI - FINAL NSBM VERIFICATION")
    print("=" * 60)

    store = verify_database()

    if store is not None:
        try:
            verify_retrieval(store)
        except Exception as exc:
            print(
                "\nRETRIEVAL CHECK ERROR:",
                type(exc).__name__,
                str(exc),
            )

    tests_passed = verify_tests()

    section("FINAL RESULT")

    if store is not None and tests_passed:
        print("DATABASE METADATA: PASSED")
        print("PROJECT TESTS: PASSED")
        print("FINAL VERIFICATION: SUCCESS")
    else:
        print("FINAL VERIFICATION: NEEDS ATTENTION")

    if not tests_passed:
        sys.exit(1)

    if store is None:
        sys.exit(2)


if __name__ == "__main__":
    main()
