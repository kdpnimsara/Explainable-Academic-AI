from pathlib import Path

import pytest

from src.ingestion.pdf_loader import PDFLoader

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker


class IntegrationWordTokenizer:
    """Deterministic tokenizer for PDF pipeline integration tests."""

    def __init__(self):
        self.vocabulary = {}
        self.reverse_vocabulary = {}

    def encode(self, text: str) -> list[int]:
        token_ids = []

        for word in text.split():

            if word not in self.vocabulary:
                token_id = len(self.vocabulary) + 1
                self.vocabulary[word] = token_id
                self.reverse_vocabulary[token_id] = word

            token_ids.append(self.vocabulary[word])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(
            self.reverse_vocabulary[token_id]
            for token_id in token_ids
        )


@pytest.fixture(scope="module")
def sample_pdf_path():
    """Locate the actual SafeSteps LK sample PDF document."""

    project_root = Path(__file__).resolve().parents[1]

    pdf_path = (
        project_root
        / "data"
        / "documents"
        / "safesteps"
        / "01_disaster_management_act_2005.pdf"
    )

    if not pdf_path.is_file():
        pytest.skip("SafeSteps LK sample PDF is not available.")

    return pdf_path


@pytest.fixture(scope="module")
def pipeline_result(sample_pdf_path):
    """Run the complete PDF-to-chunks pipeline."""

    # Step 1: Load the actual PDF.
    loader = PDFLoader()

    pages = loader.load(sample_pdf_path)

    # Step 2: Configure the chunking engine.
    tokenizer = IntegrationWordTokenizer()

    config = ChunkingConfig(
        chunk_size=384,
        chunk_overlap=64,
    )

    # Step 3: Initialize the main chunker.
    chunker = TextChunker(
        config=config,
        tokenizer=tokenizer,
    )

    # Step 4: Process actual extracted PDF pages.
    chunks, statistics = chunker.process(pages)

    return {
        "pages": pages,
        "chunks": chunks,
        "statistics": statistics,
        "tokenizer": tokenizer,
        "config": config,
    }


def test_real_pdf_text_extraction(pipeline_result):
    """Verify that actual PDF text was extracted."""

    pages = pipeline_result["pages"]

    assert len(pages) > 0

    for page in pages:
        assert page.source == "01_disaster_management_act_2005.pdf"
        assert page.page_number >= 1
        assert page.text.strip()


def test_real_pdf_generates_chunks(pipeline_result):
    """Verify that actual PDF pages generate text chunks."""

    chunks = pipeline_result["chunks"]

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.chunk_id
        assert chunk.token_count > 0


def test_real_pdf_preserves_page_metadata(pipeline_result):
    """Verify source and page information."""

    chunks = pipeline_result["chunks"]

    for chunk in chunks:

        assert (
            chunk.source
            == "01_disaster_management_act_2005.pdf"
        )

        assert len(chunk.page_numbers) > 0

        assert all(
            page_number >= 1
            for page_number in chunk.page_numbers
        )


def test_real_pdf_respects_token_limit(pipeline_result):
    """Verify that generated chunks respect the configured limit."""

    chunks = pipeline_result["chunks"]
    tokenizer = pipeline_result["tokenizer"]
    config = pipeline_result["config"]

    for chunk in chunks:

        actual_token_count = len(
            tokenizer.encode(chunk.text)
        )

        assert actual_token_count == chunk.token_count

        assert actual_token_count <= config.chunk_size


def test_real_pdf_chunk_ids_are_unique(pipeline_result):
    """Verify that every generated chunk has a unique ID."""

    chunks = pipeline_result["chunks"]

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_real_pdf_statistics_are_correct(pipeline_result):
    """Verify chunking statistics."""

    pages = pipeline_result["pages"]
    chunks = pipeline_result["chunks"]
    statistics = pipeline_result["statistics"]

    assert statistics.total_chunks == len(chunks)

    assert statistics.total_tokens == sum(
        chunk.token_count
        for chunk in chunks
    )

    assert statistics.total_documents == 1

    assert statistics.total_pages == len(pages)

    assert statistics.maximum_tokens <= 384