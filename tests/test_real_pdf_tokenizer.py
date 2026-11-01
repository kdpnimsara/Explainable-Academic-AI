from pathlib import Path

import pytest

from src.ingestion.pdf_loader import PDFLoader

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker
from src.ingestion.chunking.tokenizer_adapter import (
    HuggingFaceTokenizerAdapter,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_PDF_NAME = "01_disaster_management_act_2005.pdf"


@pytest.fixture(scope="module")
def sample_pdf():
    """Locate and load the actual SafeSteps LK PDF document."""

    project_root = Path(__file__).resolve().parents[1]

    pdf_path = (
        project_root
        / "data"
        / "documents"
        / "safesteps"
        / SAMPLE_PDF_NAME
    )

    if not pdf_path.is_file():
        pytest.skip("SafeSteps LK sample PDF is unavailable.")

    return PDFLoader().load(pdf_path)


@pytest.fixture(scope="module")
def real_tokenizer():
    """Load the actual cached Hugging Face tokenizer."""

    try:
        return HuggingFaceTokenizerAdapter.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

    except OSError:
        pytest.skip(
            "Hugging Face tokenizer is unavailable locally."
        )


@pytest.fixture(scope="module")
def real_pipeline(sample_pdf, real_tokenizer):
    """Run the complete PDF-to-chunks pipeline."""

    config = ChunkingConfig(
        chunk_size=240,
        chunk_overlap=32,
    )

    chunker = TextChunker(
        config=config,
        tokenizer=real_tokenizer,
    )

    chunks, statistics = chunker.process(sample_pdf)

    return {
        "pages": sample_pdf,
        "chunks": chunks,
        "statistics": statistics,
        "tokenizer": real_tokenizer,
        "config": config,
        "chunker": chunker,
    }


def test_real_pdf_pages_are_extracted(real_pipeline):
    """Verify that actual PDF pages are extracted."""

    pages = real_pipeline["pages"]

    assert len(pages) > 0

    for page in pages:
        assert page.source == SAMPLE_PDF_NAME
        assert page.text.strip()
        assert page.page_number >= 1


def test_real_pdf_generates_chunks(real_pipeline):
    """Verify that the real PDF generates chunks."""

    chunks = real_pipeline["chunks"]

    assert len(chunks) > 0


def test_real_tokenizer_counts_are_accurate(real_pipeline):
    """Verify token counts using the actual tokenizer."""

    tokenizer = real_pipeline["tokenizer"]

    for chunk in real_pipeline["chunks"]:

        actual_count = len(
            tokenizer.encode(chunk.text)
        )

        assert actual_count == chunk.token_count


def test_real_pdf_chunks_respect_model_limit(real_pipeline):
    """Verify that chunks respect the configured token limit."""

    tokenizer = real_pipeline["tokenizer"]
    config = real_pipeline["config"]

    for chunk in real_pipeline["chunks"]:

        token_count = len(
            tokenizer.encode(chunk.text)
        )

        assert token_count <= config.chunk_size


def test_real_pdf_metadata_is_preserved(real_pipeline):
    """Verify that PDF source and page metadata are preserved."""

    for chunk in real_pipeline["chunks"]:

        assert chunk.source == SAMPLE_PDF_NAME

        assert len(chunk.page_numbers) > 0

        assert all(
            number >= 1
            for number in chunk.page_numbers
        )


def test_real_pdf_chunk_ids_are_unique(real_pipeline):
    """Verify that all generated chunk IDs are unique."""

    chunks = real_pipeline["chunks"]

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_real_pdf_statistics_are_correct(real_pipeline):
    """Verify that chunking statistics are accurate."""

    statistics = real_pipeline["statistics"]
    chunks = real_pipeline["chunks"]

    assert statistics.total_chunks == len(chunks)

    assert statistics.total_tokens == sum(
        chunk.token_count
        for chunk in chunks
    )

    assert statistics.total_documents == 1

    assert statistics.total_pages == len(
        real_pipeline["pages"]
    )


def test_real_pdf_chunk_ids_are_deterministic(real_pipeline):
    """Verify that repeated processing generates identical IDs."""

    chunks = real_pipeline["chunks"]

    chunker = real_pipeline["chunker"]
    pages = real_pipeline["pages"]

    repeated_chunks = chunker.chunk_pages(pages)

    first_ids = [
        chunk.chunk_id for chunk in chunks
    ]

    second_ids = [
        chunk.chunk_id for chunk in repeated_chunks
    ]

    assert first_ids == second_ids


def test_real_pdf_content_is_not_empty(real_pipeline):
    """Verify that generated chunks contain valid text."""

    for chunk in real_pipeline["chunks"]:

        assert isinstance(chunk.text, str)
        assert chunk.text.strip()
        assert chunk.token_count > 0