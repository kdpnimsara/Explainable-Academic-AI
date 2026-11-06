
from pathlib import Path
from collections import Counter

from src.metadata.document_metadata import DocumentMetadata
from src.ingestion.pdf_loader import PDFLoader

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker
from src.ingestion.chunking.tokenizer_adapter import (
    HuggingFaceTokenizerAdapter,
)

from src.embeddings.embedder import MultilingualE5Embedder
from src.storage.vector_store import ChromaVectorStore


# ======================================================
# PROJECT CONFIGURATION
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_ROOT = PROJECT_ROOT / "data" / "documents"

COLLECTION_NAME = "multi_university_admin_chunks"

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 240
CHUNK_OVERLAP = 32


# ======================================================
# DOCUMENTS TO SKIP
# ======================================================

SKIPPED_DOCUMENTS = {
    (
        "institutions/university_of_moratuwa/"
        "examinations_assessment/By-Law 15-2024_0.pdf"
    ): "Scanned PDF. OCR is required.",

    (
        "institutions/university_of_colombo/"
        "examinations_assessment/examination_manual.pdf"
    ): "Text extraction is garbled. Source needs review.",
}


# ======================================================
# INSTITUTION IDENTIFIERS
# ======================================================

INSTITUTION_IDS = {
    "nsbm": "NSBM",
    "university_of_moratuwa": "UOM",
    "university_of_colombo": "UOC",
    "university_of_sri_jayewardenepura": "USJ",
    "sliit": "SLIIT",
}


# ======================================================
# DOCUMENT METADATA
# ======================================================

def create_document_metadata(
    pdf_path: Path,
) -> DocumentMetadata:

    relative_path = pdf_path.relative_to(
        DOCUMENTS_ROOT
    )

    parts = relative_path.parts

    document_id = relative_path.with_suffix("").as_posix()

    document_title = (
        pdf_path.stem
        .replace("_", " ")
        .replace("-", " ")
    )

    # Common disaster management documents

    if parts[0] == "safesteps":

        return DocumentMetadata(
            document_id=document_id,
            document_scope="common",
            document_category="disaster_management",
            document_title=document_title,
        )

    # Institution-specific documents

    if parts[0] == "institutions":

        if len(parts) < 3:
            raise ValueError(
                f"Invalid institution document path: {relative_path}"
            )

        institution_key = parts[1]

        institution_id = INSTITUTION_IDS.get(
            institution_key
        )

        if institution_id is None:
            raise ValueError(
                f"Unknown institution: {institution_key}"
            )

        category = pdf_path.parent.name

        return DocumentMetadata(
            document_id=document_id,
            document_scope="institution",
            institution_id=institution_id,
            document_category=category,
            document_title=document_title,
        )

    raise ValueError(
        f"Unsupported document location: {relative_path}"
    )


# ======================================================
# FIND PDF DOCUMENTS
# ======================================================

def find_documents():

    common_paths = list(
        (DOCUMENTS_ROOT / "safesteps").rglob("*.pdf")
    )

    institution_paths = list(
        (DOCUMENTS_ROOT / "institutions").rglob("*.pdf")
    )

    paths = sorted(
        common_paths + institution_paths
    )

    selected = []
    skipped = []

    for path in paths:

        relative_path = path.relative_to(
            DOCUMENTS_ROOT
        ).as_posix()

        skip_reason = SKIPPED_DOCUMENTS.get(
            relative_path
        )

        if skip_reason:

            skipped.append(
                (path, skip_reason)
            )

        else:

            selected.append(path)

    return selected, skipped


# ======================================================
# MAIN INGESTION PIPELINE
# ======================================================

def main():

    print("\nStarting Multi-University PDF Ingestion")

    print("----------------------------------------")

    pdf_paths, skipped = find_documents()

    if not pdf_paths:
        raise RuntimeError(
            "No PDF documents found."
        )

    print(
        "\nPDF files found:",
        len(pdf_paths) + len(skipped)
    )

    print(
        "PDFs selected for ingestion:",
        len(pdf_paths)
    )

    print(
        "PDFs skipped:",
        len(skipped)
    )

    # Display skipped documents

    if skipped:

        print("\nSkipped Documents:")

        for path, reason in skipped:

            print(
                f"- {path.name} | {reason}"
            )

    # Initialize PDF loader

    loader = PDFLoader()

    # Initialize tokenizer

    print("\nLoading Hugging Face tokenizer...")

    tokenizer = HuggingFaceTokenizerAdapter.from_pretrained(
        MODEL_NAME,
        local_files_only=False,
    )

    # Initialize chunker

    chunker = TextChunker(
        config=ChunkingConfig(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ),
        tokenizer=tokenizer,
    )

    all_chunks = []

    total_pages = 0

    # ==================================================
    # STEP 1: PDF LOADING AND CHUNKING
    # ==================================================

    print("\nLoading PDFs and creating chunks...")

    for index, pdf_path in enumerate(
        pdf_paths,
        start=1,
    ):

        metadata = create_document_metadata(
            pdf_path
        )

        pages = loader.load(
            pdf_path,
            document_metadata=metadata,
        )

        if not pages:
            raise RuntimeError(
                f"No readable pages found: {pdf_path.name}"
            )

        chunks, statistics = chunker.process(
            pages
        )

        if not chunks:
            raise RuntimeError(
                f"No chunks created: {pdf_path.name}"
            )

        # Verify metadata preservation

        if any(
            chunk.document_metadata != metadata
            for chunk in chunks
        ):
            raise RuntimeError(
                f"Document metadata mismatch: {pdf_path.name}"
            )

        total_pages += len(pages)

        all_chunks.extend(chunks)

        print(
            f"[{index}/{len(pdf_paths)}] "
            f"{pdf_path.name} | "
            f"Pages: {len(pages)} | "
            f"Chunks: {len(chunks)} | "
            f"Scope: {metadata.document_scope} | "
            f"Institution: {metadata.institution_id}"
        )

    print("\nTotal extracted pages:", total_pages)

    print(
        "Total created chunks:",
        len(all_chunks),
    )

    # ==================================================
    # STEP 2: CREATE EMBEDDINGS
    # ==================================================

    print("\nCreating multilingual embeddings...")

    embedder = MultilingualE5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=32,
    )

    embedded_chunks = embedder.embed_chunks(
        all_chunks,
        show_progress_bar=True,
    )

    if len(embedded_chunks) != len(all_chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    print(
        "Embedded chunks:",
        len(embedded_chunks),
    )

    # ==================================================
    # STEP 3: CONNECT TO CHROMADB
    # ==================================================

    print("\nConnecting to ChromaDB...")

    store = ChromaVectorStore(
        persist_directory=(
            PROJECT_ROOT / "data" / "vector_store"
        ),
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL_NAME,
    )

    before_count = store.count()

    print(
        "Existing collection records:",
        before_count,
    )

    # Upsert documents

    stored = store.upsert(
        embedded_chunks
    )

    after_count = store.count()

    # ==================================================
    # STEP 4: INGESTION REPORT
    # ==================================================

    print("\n==========================================")
    print("MULTI-UNIVERSITY INGESTION REPORT")
    print("==========================================")

    print("PDFs processed:", len(pdf_paths))
    print("PDFs skipped:", len(skipped))
    print("Pages extracted:", total_pages)
    print("Chunks created:", len(all_chunks))
    print("Embeddings created:", len(embedded_chunks))
    print("Records upserted:", stored)
    print("Records before:", before_count)
    print("Records after:", after_count)

    # Document scope distribution

    print("\nDocument Scope Distribution:")

    scope_counts = Counter(
        chunk.document_metadata.document_scope
        for chunk in all_chunks
    )

    for scope, count in scope_counts.items():

        print(
            f"{scope}: {count}"
        )

    # Institution distribution

    print("\nInstitution Distribution:")

    institution_counts = Counter(
        chunk.document_metadata.institution_id
        for chunk in all_chunks
        if chunk.document_metadata.institution_id
    )

    for institution, count in institution_counts.items():

        print(
            f"{institution}: {count}"
        )

    # Final validation

    if after_count < before_count:
        raise RuntimeError(
            "Collection record count unexpectedly decreased."
        )

    print("\n==========================================")
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    main()
