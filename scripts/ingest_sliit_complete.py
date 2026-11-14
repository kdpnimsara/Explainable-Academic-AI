from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from src.metadata.document_metadata import DocumentMetadata
from src.ingestion.pdf_loader import PDFLoader, PDFPage
from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker
from src.ingestion.chunking.tokenizer_adapter import HuggingFaceTokenizerAdapter
from src.embeddings.embedder import MultilingualE5Embedder
from src.storage.vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]
SLIIT_ROOT = ROOT / "data" / "documents" / "institutions" / "sliit"
MANIFEST = SLIIT_ROOT / "sliit_sources_manifest.json"
VECTOR = ROOT / "data" / "vector_store"
BACKUPS = ROOT / "data" / "backups"
REPORTS = ROOT / "data" / "evaluation"
MODEL = "intfloat/multilingual-e5-small"
COLLECTION = "multi_university_admin_chunks"
CHUNK_SIZE = 240
CHUNK_OVERLAP = 32
TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm"}
PDF_SUFFIX = ".pdf"
MIN_TEXT_CHARS = 500


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_source_path(record: dict) -> Path:
    local_path = record.get("local_path")
    if not isinstance(local_path, str) or not local_path.strip():
        raise RuntimeError(f"Manifest record has no local_path: {record.get('title')!r}")

    # Manifest paths are project-relative. Resolve and prevent path traversal.
    candidate = (ROOT / Path(local_path.replace("\\", "/"))).resolve()
    try:
        candidate.relative_to(SLIIT_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Source escapes the SLIIT directory: {local_path}") from exc
    return candidate


def document_version(record: dict, path: Path) -> str | None:
    label = f"{record.get('title', '')} {path.name}"
    match = re.search(r"\b(20\d{2})(?:[._-](\d+))?\b", label)
    if not match:
        return None
    return f"{match.group(1)}.{match.group(2)}" if match.group(2) else match.group(1)


def make_metadata(record: dict, path: Path) -> DocumentMetadata:
    relative = path.relative_to(SLIIT_ROOT).as_posix()
    source_url = record.get("source_url")
    if not isinstance(source_url, str) or not source_url.startswith(("https://", "http://")):
        raise RuntimeError(f"Invalid source_url for {relative}: {source_url!r}")

    host = (urlparse(source_url).hostname or "").lower()
    if not (host == "sliit.lk" or host.endswith(".sliit.lk")):
        raise RuntimeError(f"Non-SLIIT source host in manifest: {source_url}")

    category = record.get("category") or path.parent.name
    title = record.get("title") or path.stem.replace("_", " ").replace("-", " ")
    document_id = f"institutions/sliit/{Path(relative).with_suffix('').as_posix()}"

    return DocumentMetadata(
        document_id=document_id,
        document_scope="institution",
        institution_id="SLIIT",
        document_category=str(category),
        document_title=str(title),
        document_version=document_version(record, path),
        source_authority="Sri Lanka Institute of Information Technology (SLIIT)",
        source_url=source_url,
    )


def load_manifest_sources() -> tuple[dict, list[tuple[dict, Path]], list[tuple[dict, Path]]]:
    if not MANIFEST.is_file():
        raise RuntimeError(f"SLIIT manifest not found: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("institution_id") != "SLIIT":
        raise RuntimeError("Manifest institution_id is not SLIIT; aborting.")

    records = manifest.get("sources")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Manifest has no source records.")

    selected: list[tuple[dict, Path]] = []
    excluded: list[tuple[dict, Path]] = []
    seen_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("Manifest contains a non-object source record.")
        path = safe_source_path(record)
        rel = path.relative_to(SLIIT_ROOT).as_posix()
        if rel in seen_paths:
            raise RuntimeError(f"Duplicate local_path in manifest: {rel}")
        seen_paths.add(rel)

        if not path.is_file():
            raise RuntimeError(f"Manifest source is missing locally: {rel}")
        expected_hash = record.get("sha256")
        if expected_hash and sha256(path).lower() != str(expected_hash).lower():
            raise RuntimeError(f"SHA-256 mismatch: {rel}")

        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES | {PDF_SUFFIX}:
            raise RuntimeError(f"Unsupported SLIIT source file type: {rel}")

        if record.get("status") not in (None, "downloaded"):
            raise RuntimeError(f"Source is not marked downloaded: {rel} ({record.get('status')})")

        if record.get("index", True) is False:
            excluded.append((record, path))
        else:
            selected.append((record, path))

    return manifest, selected, excluded


def build_pages(record: dict, path: Path, metadata: DocumentMetadata, loader: PDFLoader) -> list[PDFPage]:
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        pages = loader.load(path.resolve(), document_metadata=metadata)
        if not pages:
            raise RuntimeError(f"PDF has no extractable text pages: {path.name}")
        return pages

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) < MIN_TEXT_CHARS:
        raise RuntimeError(f"Text snapshot is too short ({len(text)} chars): {path.name}")
    lowered = text.casefold()
    bad_markers = ("access blocked", "403 forbidden", "cloudflare ray id", "captcha")
    if any(marker in lowered for marker in bad_markers):
        raise RuntimeError(f"Blocked/error page found in source: {path.name}")

    return [
        PDFPage(
            source=str(path.resolve()),
            page_number=1,
            text=text,
            document_metadata=metadata,
        )
    ]


def prepare_chunks(dry_run: bool = True):
    manifest, sources, excluded = load_manifest_sources()
    print("=" * 72)
    print("SLIIT-ONLY MANIFEST-DRIVEN INGESTION — DRY RUN" if dry_run else "SLIIT-ONLY MANIFEST-DRIVEN INGESTION — APPLY")
    print("=" * 72)
    print("Manifest source records:", len(sources) + len(excluded))
    print("Sources selected for indexing:", len(sources))
    print("Archive-only / context-only sources:", len(excluded))
    if excluded:
        for record, path in excluded:
            print(f"  [NOT INDEXED] {record.get('title', path.name)} | {path.name}")
    print("PDFs:", sum(p.suffix.lower() == PDF_SUFFIX for _, p in sources))
    print("Text snapshots:", sum(p.suffix.lower() in TEXT_SUFFIXES for _, p in sources))
    print("Vector collection:", COLLECTION)
    print("Existing vector records will not be deleted.")

    print("\nLoading local tokenizer...")
    tokenizer = HuggingFaceTokenizerAdapter.from_pretrained(
        MODEL,
        local_files_only=True,
    )
    chunker = TextChunker(
        config=ChunkingConfig(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
        tokenizer=tokenizer,
    )
    loader = PDFLoader()

    all_chunks = []
    total_pages = 0
    category_documents = Counter()
    category_chunks = Counter()
    type_documents = Counter()

    print("\nLoading and chunking SLIIT sources...")
    for index, (record, path) in enumerate(sources, start=1):
        metadata = make_metadata(record, path)
        pages = build_pages(record, path, metadata, loader)
        chunks, stats = chunker.process(pages)
        if not chunks:
            raise RuntimeError(f"No chunks generated: {path.relative_to(SLIIT_ROOT)}")
        if any(chunk.document_metadata != metadata for chunk in chunks):
            raise RuntimeError(f"Metadata mismatch: {path.name}")
        if any(chunk.document_metadata.institution_id != "SLIIT" for chunk in chunks):
            raise RuntimeError(f"Non-SLIIT chunk metadata: {path.name}")

        total_pages += len(pages)
        all_chunks.extend(chunks)
        category = metadata.document_category
        category_documents[category] += 1
        category_chunks[category] += len(chunks)
        type_documents[path.suffix.lower()] += 1
        print(
            f"[{index}/{len(sources)}] {category} | {path.name} | "
            f"pages={len(pages)} | chunks={len(chunks)}"
        )

    ids = [chunk.chunk_id for chunk in all_chunks]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate chunk IDs detected within SLIIT ingestion batch.")

    expected_documents = {
        make_metadata(record, path).document_id for record, path in sources
    }
    actual_documents = {
        chunk.document_metadata.document_id
        for chunk in all_chunks
        if chunk.document_metadata is not None
    }
    if actual_documents != expected_documents:
        raise RuntimeError("Chunk document inventory does not match the manifest.")

    print("\n" + "-" * 72)
    print("SLIIT DRY-RUN SUMMARY")
    print("-" * 72)
    print("Documents validated:", len(sources))
    print("PDF documents:", type_documents[PDF_SUFFIX])
    print("Text snapshots:", sum(type_documents[s] for s in TEXT_SUFFIXES))
    print("Extracted pages / text units:", total_pages)
    print("Unique chunks planned:", len(all_chunks))
    print("Unique documents represented:", len(actual_documents))
    print("Category document counts:")
    for category, count in sorted(category_documents.items()):
        print(f"  {category}: {count} documents, {category_chunks[category]} chunks")

    return manifest, sources, excluded, all_chunks, expected_documents, total_pages, category_documents, category_chunks


def apply_ingestion(prepared) -> None:
    manifest, sources, excluded, chunks, expected_documents, total_pages, category_documents, category_chunks = prepared
    if not VECTOR.is_dir():
        raise RuntimeError(f"Existing Chroma database not found: {VECTOR}")

    print("\nGenerating SLIIT embeddings...")
    embedder = MultilingualE5Embedder(
        model_name=MODEL,
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
        local_files_only=True,
    )
    embedded = embedder.embed_chunks(chunks, show_progress_bar=True)
    if len(embedded) != len(chunks):
        raise RuntimeError("Embedding count mismatch; no vector records were changed.")

    BACKUPS.mkdir(parents=True, exist_ok=True)
    backup = BACKUPS / ("vector_store_before_sliit_ingestion_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if backup.exists():
        raise RuntimeError(f"Backup destination already exists: {backup}")
    print("\nBacking up current vector database...")
    shutil.copytree(VECTOR, backup)
    print("Backup:", backup)

    store = ChromaVectorStore(
        persist_directory=VECTOR,
        collection_name=COLLECTION,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL,
    )

    before = store.count()
    previous = store._collection.get(include=["metadatas"])
    other_ids: set[str] = set()
    sliit_before_ids: set[str] = set()
    existing_metadata_by_id = {}

    for record_id, raw_metadata in zip(previous["ids"], previous["metadatas"]):
        metadata = raw_metadata or {}
        institution = str(metadata.get("institution_id", "")).casefold()
        document_id = str(metadata.get("document_id", "")).casefold()
        is_sliit = institution == "sliit" or document_id.startswith("institutions/sliit/")
        if is_sliit:
            sliit_before_ids.add(record_id)
        else:
            other_ids.add(record_id)
        existing_metadata_by_id[record_id] = metadata

    new_ids = {item.chunk.chunk_id for item in embedded}
    collisions = new_ids.intersection(other_ids)
    if collisions:
        raise RuntimeError(
            "Chunk ID collision with another institution; no upsert performed. "
            f"Examples: {sorted(collisions)[:5]}"
        )

    print("Vector records before:", before)
    print("Existing SLIIT chunk IDs:", len(sliit_before_ids))
    print("Other-institution records protected:", len(other_ids))
    print("SLIIT chunks to upsert:", len(new_ids))

    print("\nUpserting SLIIT chunks in batches...")
    stored = 0
    batch_size = 5000
    for start in range(0, len(embedded), batch_size):
        batch = embedded[start:start + batch_size]
        batch_stored = store.upsert(batch)
        stored += batch_stored
        print("  Upserted", stored, "/", len(embedded), "chunks")
    if stored != len(embedded):
        raise RuntimeError("Upsert count mismatch.")

    check = store._collection.get(ids=list(new_ids), include=["metadatas"])
    if set(check["ids"]) != new_ids:
        raise RuntimeError("Some SLIIT chunk IDs were not found after upsert.")

    found_documents: set[str] = set()
    stored_categories = Counter()
    for record_id, raw_metadata in zip(check["ids"], check["metadatas"]):
        metadata = raw_metadata or {}
        if str(metadata.get("institution_id", "")).casefold() != "sliit":
            raise RuntimeError(f"Wrong institution metadata after upsert: {record_id}")
        if metadata.get("document_scope") != "institution":
            raise RuntimeError(f"Wrong document scope after upsert: {record_id}")
        for field in ("document_id", "document_category", "document_title", "source_authority", "source_url"):
            if not metadata.get(field):
                raise RuntimeError(f"Missing metadata field {field!r}: {record_id}")
        found_documents.add(metadata["document_id"])
        stored_categories[metadata["document_category"]] += 1

    if found_documents != expected_documents:
        raise RuntimeError("Stored SLIIT document inventory does not match the manifest.")

    final = store._collection.get(include=["metadatas"])
    final_other_ids: set[str] = set()
    final_sliit_ids: set[str] = set()
    for record_id, raw_metadata in zip(final["ids"], final["metadatas"]):
        metadata = raw_metadata or {}
        institution = str(metadata.get("institution_id", "")).casefold()
        document_id = str(metadata.get("document_id", "")).casefold()
        if institution == "sliit" or document_id.startswith("institutions/sliit/"):
            final_sliit_ids.add(record_id)
        else:
            final_other_ids.add(record_id)

    if final_other_ids != other_ids:
        raise RuntimeError("Other-institution records changed; inspect the vector backup.")
    if not new_ids.issubset(final_sliit_ids):
        raise RuntimeError("Final SLIIT chunk verification failed.")
    if store.count() < before:
        raise RuntimeError("Vector record count unexpectedly decreased.")

    report = {
        "institution_id": "SLIIT",
        "institution_name": "Sri Lanka Institute of Information Technology",
        "manifest_path": str(MANIFEST.relative_to(ROOT)),
        "manifest_source_count": len(sources) + len(excluded),
        "indexed_source_count": len(sources),
        "archive_only_source_count": len(excluded),
        "archive_only_sources": [record.get("local_path") for record, _ in excluded],
        "pdf_count": sum(path.suffix.lower() == PDF_SUFFIX for _, path in sources),
        "text_snapshot_count": sum(path.suffix.lower() in TEXT_SUFFIXES for _, path in sources),
        "extracted_page_or_text_units": total_pages,
        "unique_documents": len(expected_documents),
        "chunks_upserted": stored,
        "vector_records_before": before,
        "vector_records_after": store.count(),
        "sliit_chunk_ids_before": len(sliit_before_ids),
        "sliit_chunk_ids_after": len(final_sliit_ids),
        "other_institution_records_preserved": len(final_other_ids),
        "category_document_counts": dict(category_documents),
        "category_chunk_counts": dict(category_chunks),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "backup_path": str(backup.relative_to(ROOT)),
        "note": "SLIIT chunks upserted only. No existing vector records were deleted. Other-institution IDs were verified unchanged.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS / "sliit_ingestion_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("SLIIT INGESTION COMPLETED")
    print("=" * 72)
    print("Documents indexed:", len(expected_documents))
    print("Archive-only sources:", len(excluded))
    print("PDFs:", report["pdf_count"])
    print("Text snapshots:", report["text_snapshot_count"])
    print("Chunks upserted:", stored)
    print("Vector records before:", before)
    print("Vector records after:", store.count())
    print("SLIIT chunk IDs after:", len(final_sliit_ids))
    print("Other-institution records preserved:", len(final_other_ids))
    print("Report:", report_path)
    print("Backup:", backup)
    print("INGESTION: PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manifest-driven SLIIT-only ingestion for PDFs and text snapshots."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create embeddings and upsert SLIIT chunks. Default is a read-only dry run.",
    )
    args = parser.parse_args()

    try:
        prepared = prepare_chunks(dry_run=not args.apply)
        if not args.apply:
            print("\nDRY RUN ONLY: no embeddings created and no vector database records changed.")
            print("After reviewing this report, rerun with --apply to ingest SLIIT only.")
            return 0
        apply_ingestion(prepared)
        return 0
    except Exception as exc:  # noqa: BLE001 - print actionable CLI error
        print(f"\nSLIIT INGESTION FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
