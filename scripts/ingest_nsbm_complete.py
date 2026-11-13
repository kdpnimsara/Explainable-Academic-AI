
from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib
import json
import re
import shutil

from pypdf import PdfReader

from src.metadata.document_metadata import DocumentMetadata
from src.ingestion.pdf_loader import PDFLoader, PDFPage
from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker
from src.ingestion.chunking.tokenizer_adapter import HuggingFaceTokenizerAdapter
from src.embeddings.embedder import MultilingualE5Embedder
from src.storage.vector_store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
NSBM = ROOT / "data" / "documents" / "institutions" / "nsbm"
MANIFEST = NSBM / "nsbm_sources_manifest.json"
VECTOR = ROOT / "data" / "vector_store"
BACKUPS = ROOT / "data" / "backups"

MODEL = "intfloat/multilingual-e5-small"
COLLECTION = "multi_university_admin_chunks"

EXCLUDED = {
    "examinations/NSBM_Examination_Portal_Public_Information.md":
        "Link directory only.",
    "student_services/NSBM_Student_Portal_Public_Directory.md":
        "Login-protected portal landing page.",
    "student_services/NSBM_Library_Technical_Support.md":
        "Insufficient substantive information.",
    "student_services/NSBM_Library_Using_the_Library.md":
        "Conflicting policy values; await Library confirmation.",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def legacy_metadata(path):
    name = re.sub(r"[^a-z0-9]+", " ", path.stem.casefold()).strip()

    rules = [
        (
            "examination regulations instructions",
            "examination_regulations",
            "NSBM Examination Regulations and Instructions to Candidates",
            None,
            "NSBM Examination Department",
            "https://drive.google.com/file/d/1GcAjKhhqRokWMHgHt6528yMfBy2dlqy4/view",
        ),
        (
            "revision to nsbm ec guidelines",
            "extenuating_circumstances",
            "NSBM Extenuating Circumstances Guidelines",
            "August 2023",
            "NSBM Examination Department",
            "https://drive.google.com/file/d/1WhZ6qOo0WCbZz4h0hXUOe7mHUgaxeN6n/view",
        ),
        (
            "repeat coursework procedure",
            "repeat_coursework",
            "NSBM Repeat Coursework Procedure",
            None,
            "NSBM Examination Department",
            "https://drive.google.com/file/d/1hax7fB7vsN7B00sXgmTly6VVUhKur4yL/view",
        ),
        (
            "revision of examination payments",
            "examination_payments",
            "NSBM Examination Payments",
            "January 2023",
            "NSBM Examination Department",
            "https://drive.google.com/file/d/1oBs1XRbv3-hXhEEcP1fyoGhXnoQLSfJb/view",
        ),
        (
            "degree programmes exam by laws",
            "examination_bylaws",
            "NSBM Degree Programmes Examination By-Laws",
            "26 July 2016 (Version 1; 17.2 intake onwards)",
            "NSBM Examination Department",
            "https://drive.google.com/file/d/18X4qyr1138KX6lbo-SAokh1rQTLZ34di/view",
        ),
        (
            "criteria for assessment of mba",
            "mba_assessment",
            "NSBM MBA Programme Assessment Criteria",
            None,
            "NSBM Examination Department",
            "https://drive.google.com/file/d/1KPmeT5MpvqZyX3hz8yv-56WnbP2yw6Ay/view",
        ),
        (
            "criteria for assessment of mbs",
            "mbs_assessment",
            "NSBM MBS Programme Assessment Criteria",
            None,
            "NSBM Examination Department",
            "https://drive.google.com/file/d/17w53NwCPptm5KDtqw5CXini8K4pDP9f1/view",
        ),
        (
            "criteria for assessment of postgraduate diploma",
            "postgraduate_diploma_assessment",
            "NSBM Postgraduate Diploma Assessment Criteria",
            None,
            "NSBM Examination Department",
            "https://drive.google.com/file/d/1mQjdElYuKnIlobuoThYRA3m1_nx0clv2/view",
        ),
        (
            "mbs mba student handbook",
            "postgraduate_handbook",
            "NSBM MBS and MBA Student Handbook",
            "2022",
            "NSBM Green University",
            "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2023/06/MBS-MBA-guide.pdf",
        ),
        (
            "msc it programme guide",
            "programme_guide",
            "NSBM MSc in Information Technology Programme Guide",
            "2024",
            "NSBM Green University",
            "https://www.nsbm.ac.lk/wp-content/uploads/2024/09/NSBM-MSc-in-IT-Guide.pdf",
        ),
        (
            "postgraduate diploma handbook",
            "postgraduate_handbook",
            "NSBM Postgraduate Diploma Handbook",
            "2025.1",
            "NSBM Green University",
            "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2025/05/POSTGRADUATE-DIPLOMA-2025.1-handbook.pdf",
        ),
    ]

    for prefix, category, title, version, authority, url in rules:
        if prefix in name:
            return {
                "document_id": f"institutions/nsbm/{path.stem}",
                "document_scope": "institution",
                "document_category": category,
                "document_title": title,
                "institution_id": "NSBM",
                "document_version": version,
                "source_authority": authority,
                "source_url": url,
                "file_type": "pdf",
                "index": True,
            }

    raise RuntimeError(f"Unknown original NSBM PDF: {path.name}")


def prepare_manifest():
    if not MANIFEST.exists():
        raise RuntimeError("NSBM manifest not found.")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = {
        item["relative_path"].replace("\\", "/"): dict(item)
        for item in manifest.get("documents", [])
    }

    # Add original 11 PDFs to the new 20-source manifest.
    root_pdfs = list(NSBM.glob("*.pdf"))
    if len(root_pdfs) != 11:
        raise RuntimeError(f"Expected 11 original PDFs; found {len(root_pdfs)}.")

    for path in root_pdfs:
        rel = path.relative_to(NSBM).as_posix()
        record = legacy_metadata(path)
        record["relative_path"] = rel
        record["page_count"] = len(PdfReader(str(path)).pages)
        record["sha256"] = sha256(path)
        record["collection_date"] = "2026-11-13"
        record["curation_note"] = (
            "Original NSBM source. Check document version before answering current-policy questions."
        )
        records[rel] = record

    result = []

    for rel, record in records.items():
        path = NSBM / Path(rel)
        if not path.is_file():
            raise RuntimeError(f"Missing file: {rel}")

        record["institution_id"] = "NSBM"
        record["document_scope"] = "institution"
        record["sha256"] = sha256(path)
        record["index"] = rel not in EXCLUDED
        if rel in EXCLUDED:
            record["index_note"] = EXCLUDED[rel]

        for field in (
            "document_id", "document_category", "document_title",
            "source_authority", "source_url",
        ):
            if not record.get(field):
                raise RuntimeError(f"Missing {field}: {rel}")

        result.append(record)

    actual = {
        p.relative_to(NSBM).as_posix()
        for p in NSBM.rglob("*")
        if p.is_file() and p.suffix.casefold() in {".pdf", ".md"}
    }
    registered = {r["relative_path"] for r in result}

    if actual != registered:
        raise RuntimeError(
            f"Inventory mismatch. Unregistered={actual-registered}; missing={registered-actual}"
        )

    pdfs = sum(r["file_type"] == "pdf" for r in result)
    markdown = sum(r["file_type"] == "markdown" for r in result)
    indexed = [r for r in result if r["index"]]

    if (len(result), pdfs, markdown, len(indexed)) != (31, 16, 15, 27):
        raise RuntimeError(
            f"Expected 31 files (16 PDF + 15 Markdown), 27 to index. "
            f"Found {len(result)}, {pdfs}, {markdown}, {len(indexed)}."
        )

    ids = [r["document_id"] for r in indexed]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate document IDs in manifest.")

    return manifest, result, indexed


def make_metadata(record):
    return DocumentMetadata(
        document_id=record["document_id"],
        document_scope="institution",
        document_category=record["document_category"],
        document_title=record["document_title"],
        institution_id="NSBM",
        document_version=record.get("document_version"),
        source_authority=record["source_authority"],
        source_url=record["source_url"],
    )


def load_source(record, loader):
    path = NSBM / Path(record["relative_path"])
    metadata = make_metadata(record)

    if path.suffix.casefold() == ".pdf":
        pages = loader.load(path.resolve(), document_metadata=metadata)
        if not pages:
            raise RuntimeError(f"No readable pages: {path.name}")
        return pages

    if path.suffix.casefold() == ".md":
        text = path.read_text(encoding="utf-8").strip()
        if len(text) < 500:
            raise RuntimeError(f"Markdown too short: {path.name}")
        return [
            PDFPage(
                source=str(path.resolve()),
                page_number=1,
                text=text,
                document_metadata=metadata,
            )
        ]

    raise RuntimeError(f"Unsupported file: {path}")


def main():
    print("=" * 65)
    print("NSBM COMPLETE INGESTION")
    print("=" * 65)

    if not VECTOR.is_dir():
        raise RuntimeError("Existing Chroma database not found.")

    manifest, records, indexed = prepare_manifest()
    print("Registered files:", len(records))
    print("Files to index:", len(indexed))
    print("Archive-only pages:", len(records) - len(indexed))

    print("\nLoading tokenizer...")
    tokenizer = HuggingFaceTokenizerAdapter.from_pretrained(
        MODEL,
        local_files_only=True,
    )
    chunker = TextChunker(
        config=ChunkingConfig(chunk_size=240, chunk_overlap=32),
        tokenizer=tokenizer,
    )
    loader = PDFLoader()

    print("\nLoading and chunking all NSBM sources...")
    chunks = []
    for record in indexed:
        pages = load_source(record, loader)
        document_chunks, stats = chunker.process(pages)
        if not document_chunks:
            raise RuntimeError(f"No chunks: {record['relative_path']}")
        chunks.extend(document_chunks)
        print(
            f"PASS | {record['document_category']} | "
            f"{record['relative_path']} | "
            f"pages={len(pages)} | chunks={len(document_chunks)}"
        )

    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise RuntimeError("Duplicate chunk IDs detected.")

    print("\nTotal chunks:", len(chunks))
    print("\nGenerating embeddings...")
    embedder = MultilingualE5Embedder(
        model_name=MODEL,
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
        local_files_only=True,
    )
    embedded = embedder.embed_chunks(chunks, show_progress_bar=True)

    if len(embedded) != len(chunks):
        raise RuntimeError("Embedding count mismatch.")

    new_ids = {item.chunk.chunk_id for item in embedded}
    expected_docs = {r["document_id"] for r in indexed}

    print("\nBacking up current Chroma database...")
    BACKUPS.mkdir(parents=True, exist_ok=True)
    backup = BACKUPS / (
        "vector_store_before_nsbm_complete_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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
    old_nsbm = set()
    other_ids = set()

    for rid, md in zip(previous["ids"], previous["metadatas"]):
        md = md or {}
        institution = str(md.get("institution_id", "")).casefold()
        docid = str(md.get("document_id", "")).casefold()
        if institution == "nsbm" or docid.startswith("institutions/nsbm/"):
            old_nsbm.add(rid)
        else:
            other_ids.add(rid)

    print("Records before:", before)
    print("Old NSBM chunks:", len(old_nsbm))
    print("Other institution chunks protected:", len(other_ids))

    print("\nUpserting complete NSBM source set...")
    stored = store.upsert(embedded)
    if stored != len(embedded):
        raise RuntimeError("Upsert count mismatch. Old records retained.")

    check = store._collection.get(ids=list(new_ids), include=["metadatas"])
    if set(check["ids"]) != new_ids:
        raise RuntimeError("New chunk verification failed. Old records retained.")

    found_docs = set()
    categories = Counter()

    for rid, md in zip(check["ids"], check["metadatas"]):
        md = md or {}
        if str(md.get("institution_id", "")).casefold() != "nsbm":
            raise RuntimeError(f"Wrong institution metadata: {rid}")
        if md.get("document_scope") != "institution":
            raise RuntimeError(f"Wrong document scope: {rid}")
        for field in (
            "document_id", "document_category", "document_title",
            "source_authority", "source_url",
        ):
            if not md.get(field):
                raise RuntimeError(f"Missing {field}: {rid}")
        found_docs.add(md["document_id"])
        categories[md["document_category"]] += 1

    if found_docs != expected_docs:
        raise RuntimeError("Document metadata verification failed. Old records retained.")

    stale = sorted(old_nsbm - new_ids)
    if stale:
        print("Removing stale NSBM chunks:", len(stale))
        store.delete(stale)

    final = store._collection.get(include=["metadatas"])
    final_nsbm = set()
    final_other = set()
    final_docs = set()
    final_categories = Counter()

    for rid, md in zip(final["ids"], final["metadatas"]):
        md = md or {}
        institution = str(md.get("institution_id", "")).casefold()
        docid = str(md.get("document_id", "")).casefold()
        if institution == "nsbm" or docid.startswith("institutions/nsbm/"):
            final_nsbm.add(rid)
            final_docs.add(md.get("document_id"))
            final_categories[md.get("document_category", "missing")] += 1
        else:
            final_other.add(rid)

    if final_nsbm != new_ids:
        raise RuntimeError("Final NSBM chunk verification failed.")
    if final_other != other_ids:
        raise RuntimeError("Other-institution records changed.")
    if final_docs != expected_docs:
        raise RuntimeError("Final document inventory mismatch.")

    manifest["documents"] = records
    manifest["document_count"] = len(records)
    manifest["indexed_document_count"] = len(indexed)
    manifest["excluded_from_index"] = [
        {"relative_path": k, "reason": v} for k, v in EXCLUDED.items()
    ]
    manifest["updated_at"] = datetime.now().astimezone().isoformat()

    temp = MANIFEST.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    temp.replace(MANIFEST)

    print("\n" + "=" * 65)
    print("NSBM COMPLETE INGESTION SUCCESSFUL")
    print("=" * 65)
    print("Registered files:", len(records))
    print("Indexed files:", len(indexed))
    print("Archived-only files:", len(records) - len(indexed))
    print("Unique NSBM documents:", len(final_docs))
    print("NSBM chunks:", len(final_nsbm))
    print("Database before:", before)
    print("Database after:", store.count())
    print("Other institution records preserved:", len(final_other))
    print("\nCategory counts:")
    for category, count in sorted(final_categories.items()):
        print(f"  {category}: {count}")
    print("\nManifest updated:", MANIFEST)
    print("Backup:", backup)
    print("INGESTION: PASSED")


if __name__ == "__main__":
    main()
