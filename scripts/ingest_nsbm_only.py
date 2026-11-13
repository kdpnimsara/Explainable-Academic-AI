
from pathlib import Path
from collections import Counter
from datetime import datetime
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metadata.document_metadata import DocumentMetadata
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker
from src.ingestion.chunking.tokenizer_adapter import HuggingFaceTokenizerAdapter
from src.embeddings.embedder import MultilingualE5Embedder
from src.storage.vector_store import ChromaVectorStore


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "multi_university_admin_chunks"

PDF_DIR = PROJECT_ROOT / "data" / "documents" / "institutions" / "nsbm"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
BACKUP_ROOT = PROJECT_ROOT / "data" / "backups"

# Exact expected files. Unexpected/missing files stop ingestion.
DOCUMENTS = {
    "Examination Regulations_Instructions To Candidates.pdf": {
        "category": "examination_regulations",
        "title": "NSBM Examination Regulations and Instructions to Candidates",
        "version": None,
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/1GcAjKhhqRokWMHgHt6528yMfBy2dlqy4/view",
    },
    "Revision to NSBM EC Guidelines to students from Aug2023.pdf": {
        "category": "extenuating_circumstances",
        "title": "NSBM Extenuating Circumstances Guidelines",
        "version": "August 2023",
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/1WhZ6qOo0WCbZz4h0hXUOe7mHUgaxeN6n/view",
    },
    "Repeat coursework Procedure.pdf": {
        "category": "repeat_coursework",
        "title": "NSBM Repeat Coursework Procedure",
        "version": None,
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/1hax7fB7vsN7B00sXgmTly6VVUhKur4yL/view",
    },
    "Revision of Examination Payments Jan2023_Students.pdf": {
        "category": "examination_payments",
        "title": "NSBM Examination Payments",
        "version": "January 2023",
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/1oBs1XRbv3-hXhEEcP1fyoGhXnoQLSfJb/view",
    },
    "NSBM Degree Programmes_Exam By Laws _ Version 1 (17.2 onwards) _Students.pdf": {
        "category": "examination_bylaws",
        "title": "NSBM Degree Programmes Examination By-Laws",
        "version": "Version 1 (17.2 onwards)",
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/18X4qyr1138KX6lbo-SAokh1rQTLZ34di/view",
    },
    "Criteria for Assessment of MBA Programmes (NSBM).pdf": {
        "category": "mba_assessment",
        "title": "NSBM MBA Programme Assessment Criteria",
        "version": None,
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/1KPmeT5MpvqZyX3hz8yv-56WnbP2yw6Ay/view",
    },
    "Criteria for Assessment of MBS Programmes (NSBM).pdf": {
        "category": "mbs_assessment",
        "title": "NSBM MBS Programme Assessment Criteria",
        "version": None,
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/17w53NwCPptm5KDtqw5CXini8K4pDP9f1/view",
    },
    "Criteria for Assessment of Postgraduate Diploma Programmes (NSBM).pdf": {
        "category": "postgraduate_diploma_assessment",
        "title": "NSBM Postgraduate Diploma Programme Assessment Criteria",
        "version": None,
        "authority": "NSBM Examination Department",
        "url": "https://drive.google.com/file/d/1mQjdElYuKnIlobuoThYRA3m1_nx0clv2/view",
    },
    "MBS-MBA-Student-Handbook-2022.pdf": {
        "category": "postgraduate_handbook",
        "title": "NSBM MBS and MBA Student Handbook",
        "version": "2022",
        "authority": "NSBM Green University",
        "url": "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2023/06/MBS-MBA-guide.pdf",
    },
    "NSBM_MSc_IT_Programme_Guide_2024.pdf": {
        "category": "programme_guide",
        "title": "NSBM MSc in Information Technology Programme Guide",
        "version": "2024",
        "authority": "NSBM Green University",
        "url": "https://www.nsbm.ac.lk/wp-content/uploads/2024/09/NSBM-MSc-in-IT-Guide.pdf",
    },
    "NSBM_Postgraduate_Diploma_Handbook_2025.1.pdf": {
        "category": "postgraduate_handbook",
        "title": "NSBM Postgraduate Diploma Handbook",
        "version": "2025.1",
        "authority": "NSBM Green University",
        "url": "https://d3c539pel8wzjz.cloudfront.net/wp-content/uploads/2025/05/POSTGRADUATE-DIPLOMA-2025.1-handbook.pdf",
    },
}


def main():
    print("=" * 60)
    print("NSBM-ONLY DOCUMENT INGESTION")
    print("=" * 60)

    if not PDF_DIR.is_dir():
        raise RuntimeError(f"NSBM PDF directory not found: {PDF_DIR}")

    actual_files = {
        p.name for p in PDF_DIR.glob("*.pdf")
    }
    expected_files = set(DOCUMENTS)

    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)

    if missing or unexpected:
        print("PDF INVENTORY CHECK FAILED.")
        if missing:
            print("Missing files:", missing)
        if unexpected:
            print("Unexpected PDF files:", unexpected)
        raise RuntimeError("PDF inventory must match the expected 11 files.")

    print("PDF inventory: 11/11 verified.")

    # Prepare tokenizer and chunker.
    print("\nLoading tokenizer...")
    tokenizer = HuggingFaceTokenizerAdapter.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    chunker = TextChunker(
        config=ChunkingConfig(
            chunk_size=240,
            chunk_overlap=32,
        ),
        tokenizer=tokenizer,
    )

    loader = PDFLoader()
    all_chunks = []
    total_pages = 0

    # Read and chunk all PDFs before changing the database.
    for filename, info in DOCUMENTS.items():
        pdf_path = PDF_DIR / filename

        metadata = DocumentMetadata(
            document_id=f"institutions/nsbm/{pdf_path.stem}",
            document_scope="institution",
            document_category=info["category"],
            document_title=info["title"],
            institution_id="NSBM",
            document_version=info["version"],
            source_authority=info["authority"],
            source_url=info["url"],
        )

        pages = loader.load(
            pdf_path,
            document_metadata=metadata,
        )

        if not pages:
            raise RuntimeError(f"No readable pages: {filename}")

        chunks, stats = chunker.process(pages)

        if not chunks:
            raise RuntimeError(f"No chunks generated: {filename}")

        total_pages += len(pages)
        all_chunks.extend(chunks)

        print(
            f"Prepared: {filename} | "
            f"pages={len(pages)} | chunks={len(chunks)}"
        )

    if not all_chunks:
        raise RuntimeError("No chunks generated. Database not changed.")

    print(f"\nTotal pages: {total_pages}")
    print(f"Total chunks prepared: {len(all_chunks)}")

    # Generate all embeddings before opening/changing Chroma.
    print("\nLoading embedding model...")
    embedder = MultilingualE5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
        local_files_only=True,
    )

    embedded = embedder.embed_chunks(
        all_chunks,
        show_progress_bar=True,
    )

    if len(embedded) != len(all_chunks):
        raise RuntimeError("Embedding count mismatch. Database not changed.")

    new_ids = {
        item.chunk.chunk_id for item in embedded
    }

    if len(new_ids) != len(embedded):
        raise RuntimeError("Duplicate new chunk IDs detected.")

    print(f"Embeddings prepared: {len(embedded)}")

    # Backup the entire existing vector-store directory.
    # Close other project processes using Chroma before running this script.
    if VECTOR_DIR.exists():
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        backup_dir = BACKUP_ROOT / (
            "vector_store_before_nsbm_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        shutil.copytree(VECTOR_DIR, backup_dir)
        print(f"\nDatabase backup created: {backup_dir}")
    else:
        raise RuntimeError(
            "Existing vector database directory not found. "
            "Stopping rather than creating a new database."
        )

    # Open the existing collection only after backup.
    store = ChromaVectorStore(
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL_NAME,
    )

    before_count = store.count()
    existing = store._collection.get(include=["metadatas"])

    old_nsbm_ids = set()
    for record_id, metadata in zip(
        existing["ids"],
        existing["metadatas"],
    ):
        metadata = metadata or {}
        institution = str(metadata.get("institution_id", "")).casefold()
        document_id = str(metadata.get("document_id", "")).casefold()

        if institution == "nsbm" or document_id.startswith("institutions/nsbm/"):
            old_nsbm_ids.add(record_id)

    print(f"Database records before: {before_count}")
    print(f"Existing NSBM chunks: {len(old_nsbm_ids)}")

    # Upsert new NSBM chunks.
    print("\nUpserting new NSBM chunks...")
    stored = store.upsert(embedded)

    if stored != len(embedded):
        raise RuntimeError(
            f"Upsert count mismatch: expected {len(embedded)}, got {stored}. "
            "Old NSBM chunks have NOT been deleted."
        )

    # Verify every new ID and its metadata before deleting stale NSBM IDs.
    verification = store._collection.get(
        ids=list(new_ids),
        include=["metadatas"],
    )

    verified_ids = set(verification["ids"])
    if verified_ids != new_ids:
        raise RuntimeError(
            "New chunk ID verification failed. "
            "Old NSBM chunks have NOT been deleted."
        )

    for metadata in verification["metadatas"]:
        metadata = metadata or {}
        if (
            str(metadata.get("institution_id", "")).casefold() != "nsbm"
            or metadata.get("document_scope") != "institution"
            or not metadata.get("document_category")
            or not metadata.get("document_id")
            or not metadata.get("source_url")
        ):
            raise RuntimeError(
                "Metadata verification failed. "
                "Old NSBM chunks have NOT been deleted."
            )

    # Delete only stale NSBM records. Never delete other institutions.
    stale_ids = sorted(old_nsbm_ids - new_ids)
    if stale_ids:
        print(f"Removing {len(stale_ids)} stale NSBM chunks...")
        store.delete(stale_ids)

    # Final verification.
    final_data = store._collection.get(include=["metadatas"])
    final_nsbm = {}

    for record_id, metadata in zip(
        final_data["ids"],
        final_data["metadatas"],
    ):
        metadata = metadata or {}
        if str(metadata.get("institution_id", "")).casefold() == "nsbm":
            category = metadata.get("document_category", "missing")
            final_nsbm[record_id] = category

    final_nsbm_ids = set(final_nsbm)

    if final_nsbm_ids != new_ids:
        raise RuntimeError(
            "Final NSBM record set does not match the newly prepared set. "
            f"Expected={len(new_ids)}, found={len(final_nsbm_ids)}"
        )

    after_count = store.count()

    print("\n" + "=" * 60)
    print("NSBM INGESTION SUCCESSFUL")
    print("=" * 60)
    print(f"NSBM documents: {len(DOCUMENTS)}")
    print(f"NSBM chunks: {len(final_nsbm_ids)}")
    print(f"Database records before: {before_count}")
    print(f"Database records after: {after_count}")
    print("NSBM category distribution:")
    for category, count in sorted(Counter(final_nsbm.values()).items()):
        print(f"  {category}: {count}")

    print("\nOther institution records were not targeted for deletion.")
    print(f"Backup: {backup_dir}")


if __name__ == "__main__":
    main()
