"""Remove stale SLIIT vectors after validating the current manifest inventory.

This script rebuilds deterministic chunk IDs only (no embedding generation).
It deletes only obsolete SLIIT chunk IDs after verifying that all current
manifest-approved SLIIT chunk IDs already exist in Chroma and after backing up
the complete vector-store directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ingest_sliit_complete import (  # noqa: E402
    BACKUPS as DEFAULT_BACKUPS,
    COLLECTION,
    MODEL,
    VECTOR,
    prepare_chunks,
)
from src.storage.vector_store import ChromaVectorStore  # noqa: E402


def is_sliit_metadata(metadata: dict | None) -> bool:
    metadata = metadata or {}
    institution = str(metadata.get("institution_id", "")).casefold()
    document_id = str(metadata.get("document_id", "")).casefold()
    return institution == "sliit" or document_id.startswith("institutions/sliit/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and remove stale SLIIT chunk IDs.")
    parser.add_argument("--backup-root", default=str(DEFAULT_BACKUPS))
    args = parser.parse_args()

    if not VECTOR.is_dir():
        raise RuntimeError(f"Vector database not found: {VECTOR}")

    prepared = prepare_chunks(dry_run=True)
    _, sources, excluded, chunks, expected_documents, *_ = prepared
    expected_ids = {chunk.chunk_id for chunk in chunks}
    if len(expected_ids) != len(chunks):
        raise RuntimeError("Current SLIIT manifest produced duplicate chunk IDs; aborting.")

    store = ChromaVectorStore(
        persist_directory=VECTOR,
        collection_name=COLLECTION,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL,
    )
    before = store._collection.get(include=["metadatas"])
    current_sliit: set[str] = set()
    other_ids: set[str] = set()
    current_documents: set[str] = set()
    for chunk_id, metadata in zip(before["ids"], before["metadatas"]):
        metadata = metadata or {}
        if is_sliit_metadata(metadata):
            current_sliit.add(chunk_id)
            if metadata.get("document_id"):
                current_documents.add(metadata["document_id"])
        else:
            other_ids.add(chunk_id)

    missing = expected_ids - current_sliit
    stale = current_sliit - expected_ids
    if missing:
        raise RuntimeError(
            "Cannot prune: current SLIIT chunks are missing from the vector store. "
            f"Missing={len(missing)}; examples={sorted(missing)[:5]}"
        )

    print("\nSLIIT CURRENT-INVENTORY CHECK")
    print("Manifest-approved documents:", len(sources))
    print("Archive-only sources:", len(excluded))
    print("Expected current chunks:", len(expected_ids))
    print("Existing SLIIT chunks:", len(current_sliit))
    print("Stale SLIIT chunks to remove:", len(stale))
    print("Other-institution chunks to preserve:", len(other_ids))
    print("Expected manifest document IDs:", len(expected_documents))

    if not args.apply:
        print("DRY RUN ONLY: no vector database records changed. Re-run with --apply.")
        return 0

    backup_root = Path(args.backup_root)
    if not backup_root.exists():
        backup_root = ROOT / "data" / "backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / ("vector_store_before_sliit_stale_cleanup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if backup.exists():
        raise RuntimeError(f"Backup destination already exists: {backup}")

    print("\nBacking up vector database before stale cleanup...")
    shutil.copytree(VECTOR, backup)
    print("Backup:", backup)

    if stale:
        store.delete(sorted(stale))

    after = store._collection.get(include=["metadatas"])
    final_sliit: set[str] = set()
    final_other: set[str] = set()
    final_documents: set[str] = set()
    for chunk_id, metadata in zip(after["ids"], after["metadatas"]):
        metadata = metadata or {}
        if is_sliit_metadata(metadata):
            final_sliit.add(chunk_id)
            if metadata.get("document_id"):
                final_documents.add(metadata["document_id"])
        else:
            final_other.add(chunk_id)

    if final_sliit != expected_ids:
        raise RuntimeError("Post-cleanup SLIIT chunk inventory verification failed.")
    if final_other != other_ids:
        raise RuntimeError("Non-SLIIT records changed during cleanup.")
    if final_documents != expected_documents:
        raise RuntimeError("Post-cleanup SLIIT document inventory does not match manifest.")

    report = {
        "institution_id": "SLIIT",
        "manifest_documents": len(sources),
        "archive_only_sources": len(excluded),
        "expected_chunks": len(expected_ids),
        "previous_sliit_chunks": len(current_sliit),
        "stale_chunks_removed": len(stale),
        "final_sliit_chunks": len(final_sliit),
        "other_institution_chunks_preserved": len(final_other),
        "final_database_chunks": store.count(),
        "backup_path": str(backup),
        "status": "PASSED",
        "created_at": datetime.now().astimezone().isoformat(),
    }
    report_path = ROOT / "data" / "evaluation" / "sliit_stale_cleanup_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSLIIT STALE CLEANUP COMPLETED")
    print("Stale SLIIT chunks removed:", len(stale))
    print("SLIIT chunks after cleanup:", len(final_sliit))
    print("SLIIT documents after cleanup:", len(final_documents))
    print("Other-institution records preserved:", len(final_other))
    print("Database chunks after cleanup:", store.count())
    print("Report:", report_path)
    print("CLEANUP: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
