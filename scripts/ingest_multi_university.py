
import hashlib
import json
from pathlib import Path
from collections import Counter
from dataclasses import asdict

from src.metadata.document_metadata import DocumentMetadata
from src.ingestion.pdf_loader import PDFLoader

from src.ingestion.chunking.models import (
    ChunkingConfig,
    TextChunk,
)
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

CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "ingestion_chunks"

COLLECTION_NAME = "multi_university_admin_chunks"

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 240
CHUNK_OVERLAP = 32

LOOKUP_BATCH_SIZE = 5000

CACHE_VERSION = 1


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

    if parts[0] == "safesteps":

        return DocumentMetadata(
            document_id=document_id,
            document_scope="common",
            document_category="disaster_management",
            document_title=document_title,
        )

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
# PDF SHA256
# ======================================================

def calculate_file_hash(
    file_path: Path,
) -> str:

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


# ======================================================
# CACHE FILE PATH
# ======================================================

def get_cache_path(
    pdf_path: Path,
) -> Path:

    relative_path = pdf_path.relative_to(
        DOCUMENTS_ROOT
    ).as_posix()

    path_hash = hashlib.sha256(
        relative_path.encode("utf-8")
    ).hexdigest()

    return CACHE_ROOT / f"{path_hash}.json"


# ======================================================
# CACHE FINGERPRINT
# ======================================================

def create_cache_fingerprint(
    pdf_hash: str,
    metadata: DocumentMetadata,
) -> str:

    configuration = {
        "cache_version": CACHE_VERSION,
        "pdf_sha256": pdf_hash,
        "metadata": asdict(metadata),
        "model_name": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }

    serialized = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ======================================================
# SERIALIZE CHUNK
# ======================================================

def serialize_chunk(
    chunk: TextChunk,
) -> dict:

    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "page_numbers": list(chunk.page_numbers),
        "heading": chunk.heading,
        "document_metadata": (
            asdict(chunk.document_metadata)
            if chunk.document_metadata
            else None
        ),
    }


# ======================================================
# DESERIALIZE CHUNK
# ======================================================

def deserialize_chunk(
    data: dict,
) -> TextChunk:

    metadata_data = data.get(
        "document_metadata"
    )

    metadata = (
        DocumentMetadata(**metadata_data)
        if metadata_data
        else None
    )

    return TextChunk(
        chunk_id=data["chunk_id"],
        source=data["source"],
        chunk_index=data["chunk_index"],
        text=data["text"],
        token_count=data["token_count"],
        page_numbers=tuple(data["page_numbers"]),
        heading=data.get("heading"),
        document_metadata=metadata,
    )


# ======================================================
# LOAD CHUNK CACHE
# ======================================================

def load_chunk_cache(
    cache_path: Path,
    fingerprint: str,
):

    if not cache_path.exists():
        return None

    try:

        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            cache_data = json.load(file)

        if cache_data.get("cache_version") != CACHE_VERSION:
            return None

        if cache_data.get("fingerprint") != fingerprint:
            return None

        chunks_data = cache_data["chunks"]

        if not isinstance(chunks_data, list):
            return None

        chunks = [
            deserialize_chunk(item)
            for item in chunks_data
        ]

        page_count = int(
            cache_data["page_count"]
        )

        if page_count <= 0 or not chunks:
            return None

        return chunks, page_count

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):

        print(
            "Cache is invalid. Rebuilding..."
        )

        return None


# ======================================================
# SAVE CHUNK CACHE
# ======================================================

def save_chunk_cache(
    cache_path: Path,
    fingerprint: str,
    chunks: list[TextChunk],
    page_count: int,
):

    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_data = {
        "cache_version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "page_count": page_count,
        "chunks": [
            serialize_chunk(chunk)
            for chunk in chunks
        ],
    }

    temporary_path = cache_path.with_suffix(
        ".tmp"
    )

    try:

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                cache_data,
                file,
                ensure_ascii=False,
                separators=(",", ":"),
            )

        temporary_path.replace(
            cache_path
        )

    finally:

        if temporary_path.exists():
            temporary_path.unlink()


# ======================================================
# LOAD PDF OR CACHE
# ======================================================

def load_or_create_chunks(
    pdf_path: Path,
    metadata: DocumentMetadata,
    loader: PDFLoader,
    chunker: TextChunker,
):

    cache_path = get_cache_path(
        pdf_path
    )

    pdf_hash = calculate_file_hash(
        pdf_path
    )

    fingerprint = create_cache_fingerprint(
        pdf_hash,
        metadata,
    )

    cached_result = load_chunk_cache(
        cache_path,
        fingerprint,
    )

    if cached_result is not None:

        chunks, page_count = cached_result

        if any(
            chunk.document_metadata != metadata
            for chunk in chunks
        ):

            print(
                "Cache metadata mismatch. Rebuilding..."
            )

        else:

            return chunks, page_count, "CACHE HIT"

    # Cache miss: extract PDF and create chunks.

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

    if any(
        chunk.document_metadata != metadata
        for chunk in chunks
    ):

        raise RuntimeError(
            f"Document metadata mismatch: {pdf_path.name}"
        )

    save_chunk_cache(
        cache_path=cache_path,
        fingerprint=fingerprint,
        chunks=chunks,
        page_count=len(pages),
    )

    return chunks, len(pages), "CACHE MISS"


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

    if skipped:

        print("\nSkipped Documents:")

        for path, reason in skipped:

            print(
                f"- {path.name} | {reason}"
            )

    # ==================================================
    # INITIALIZE PDF LOADER
    # ==================================================

    loader = PDFLoader()

    # ==================================================
    # INITIALIZE TOKENIZER
    # ==================================================

    print("\nLoading Hugging Face tokenizer...")

    tokenizer = HuggingFaceTokenizerAdapter.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    # ==================================================
    # INITIALIZE CHUNKER
    # ==================================================

    chunker = TextChunker(
        config=ChunkingConfig(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ),
        tokenizer=tokenizer,
    )

    all_chunks = []

    total_pages = 0

    cache_hits = 0
    cache_misses = 0

    # ==================================================
    # STEP 1: PDF LOADING AND CHUNK CACHING
    # ==================================================

    print("\nLoading PDFs and checking chunk cache...")

    for index, pdf_path in enumerate(
        pdf_paths,
        start=1,
    ):

        metadata = create_document_metadata(
            pdf_path
        )

        chunks, page_count, cache_status = (
            load_or_create_chunks(
                pdf_path=pdf_path,
                metadata=metadata,
                loader=loader,
                chunker=chunker,
            )
        )

        if cache_status == "CACHE HIT":
            cache_hits += 1
        else:
            cache_misses += 1

        total_pages += page_count

        all_chunks.extend(chunks)

        print(
            f"[{index}/{len(pdf_paths)}] "
            f"{pdf_path.name} | "
            f"Pages: {page_count} | "
            f"Chunks: {len(chunks)} | "
            f"{cache_status} | "
            f"Scope: {metadata.document_scope} | "
            f"Institution: {metadata.institution_id}"
        )

    print("\nTotal extracted pages:", total_pages)

    print(
        "Total created chunks:",
        len(all_chunks),
    )

    print(
        "Cache hits:",
        cache_hits,
    )

    print(
        "Cache misses:",
        cache_misses,
    )

    # ==================================================
    # STEP 2: CONNECT TO CHROMADB
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

    # ==================================================
    # STEP 3: IDENTIFY NEW CHUNKS
    # ==================================================

    print("\nChecking existing chunks...")

    existing_ids = set()

    for start in range(
        0,
        len(all_chunks),
        LOOKUP_BATCH_SIZE,
    ):

        batch = all_chunks[
            start:start + LOOKUP_BATCH_SIZE
        ]

        batch_ids = [
            chunk.chunk_id
            for chunk in batch
        ]

        records = store.get_by_ids(
            batch_ids
        )

        existing_ids.update(
            record.chunk_id
            for record in records
        )

        print(
            f"Checked {min(start + len(batch), len(all_chunks))}"
            f"/{len(all_chunks)} chunks"
        )

    new_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.chunk_id not in existing_ids
    ]

    print(
        "\nAlready stored chunks:",
        len(existing_ids)
    )

    print(
        "New chunks to embed:",
        len(new_chunks)
    )

    # ==================================================
    # STEP 4: EMBED ONLY NEW CHUNKS
    # ==================================================

    embedded_chunks = []

    if new_chunks:

        print(
            "\nCreating multilingual embeddings "
            "for new chunks only..."
        )

        embedder = MultilingualE5Embedder(
            model_name=MODEL_NAME,
            device="cpu",
            batch_size=32,
        )

        embedded_chunks = embedder.embed_chunks(
            new_chunks,
            show_progress_bar=True,
        )

        if len(embedded_chunks) != len(new_chunks):

            raise RuntimeError(
                "Embedding count does not match new chunk count."
            )

        print(
            "New embeddings created:",
            len(embedded_chunks),
        )

        stored = store.upsert(
            embedded_chunks
        )

    else:

        print(
            "\nAll chunks already exist. "
            "No embeddings required."
        )

        stored = 0

    after_count = store.count()

    # ==================================================
    # STEP 5: INGESTION REPORT
    # ==================================================

    print("\n==========================================")
    print("MULTI-UNIVERSITY INGESTION REPORT")
    print("==========================================")

    print("PDFs processed:", len(pdf_paths))
    print("PDFs skipped:", len(skipped))
    print("Pages extracted:", total_pages)
    print("Chunks created:", len(all_chunks))
    print("Cache hits:", cache_hits)
    print("Cache misses:", cache_misses)
    print("Existing chunks:", len(existing_ids))
    print("New chunks:", len(new_chunks))
    print("Embeddings created:", len(embedded_chunks))
    print("Records upserted:", stored)
    print("Records before:", before_count)
    print("Records after:", after_count)

    # ==================================================
    # DOCUMENT SCOPE DISTRIBUTION
    # ==================================================

    print("\nDocument Scope Distribution:")

    scope_counts = Counter(
        chunk.document_metadata.document_scope
        for chunk in all_chunks
    )

    for scope, count in scope_counts.items():

        print(
            f"{scope}: {count}"
        )

    # ==================================================
    # INSTITUTION DISTRIBUTION
    # ==================================================

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

    # ==================================================
    # FINAL VALIDATION
    # ==================================================

    if after_count < before_count:

        raise RuntimeError(
            "Collection record count unexpectedly decreased."
        )

    if after_count != before_count + stored:

        raise RuntimeError(
            "Unexpected ChromaDB record count after ingestion."
        )

    print("\n==========================================")
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    main()
