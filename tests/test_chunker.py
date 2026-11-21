import pytest

from src.ingestion.pdf_loader import PDFPage

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker


class FakeWordTokenizer:
    """Provide deterministic tokenization for testing."""

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


@pytest.fixture
def tokenizer():
    return FakeWordTokenizer()


@pytest.fixture
def chunker(tokenizer):

    config = ChunkingConfig(
        chunk_size=8,
        chunk_overlap=2,
    )

    return TextChunker(
        config=config,
        tokenizer=tokenizer,
    )


# ============================================================
# BASIC CHUNKING TESTS
# ============================================================

def test_process_pdf_pages(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "Alpha beta gamma. "
                "Delta epsilon zeta. "
                "Eta theta iota."
            ),
        )
    ]

    chunks, statistics = chunker.process(pages)

    assert len(chunks) > 0

    assert statistics.total_chunks == len(chunks)
    assert statistics.total_documents == 1
    assert statistics.total_pages == 1


def test_preserve_pdf_metadata(chunker):

    pages = [
        PDFPage(
            source="disaster_plan.pdf",
            page_number=4,
            text="Disaster preparedness is important.",
        )
    ]

    chunks = chunker.chunk_pages(pages)

    assert chunks[0].source == "disaster_plan.pdf"
    assert chunks[0].page_numbers == (4,)


def test_generate_unique_chunk_ids(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "Alpha beta gamma. "
                "Delta epsilon zeta. "
                "Eta theta iota."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_ids_are_deterministic(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text="Python supports object oriented programming.",
        )
    ]

    first_result = chunker.chunk_pages(pages)
    second_result = chunker.chunk_pages(pages)

    first_ids = [
        chunk.chunk_id
        for chunk in first_result
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_result
    ]

    assert first_ids == second_ids


def test_token_counts_are_correct(chunker, tokenizer):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "Alpha beta gamma. "
                "Delta epsilon zeta."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    for chunk in chunks:
        assert chunk.token_count == len(
            tokenizer.encode(chunk.text)
        )


def test_chunk_size_is_respected(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "Alpha beta gamma. "
                "Delta epsilon zeta. "
                "Eta theta iota."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    for chunk in chunks:
        assert chunk.token_count <= 8


def test_normalize_text_before_chunking(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text="   Python     is     useful.   ",
        )
    ]

    chunks = chunker.chunk_pages(pages)

    assert chunks[0].text == "Python is useful."


def test_process_multiple_pages(chunker):

    pages = [
        PDFPage("sample.pdf", 1, "First page content."),
        PDFPage("sample.pdf", 2, "Second page content."),
    ]

    chunks, statistics = chunker.process(pages)

    assert statistics.total_pages == 2
    assert statistics.total_documents == 1

    page_numbers = {
        page_number
        for chunk in chunks
        for page_number in chunk.page_numbers
    }

    assert page_numbers == {1, 2}


def test_process_multiple_documents(chunker):

    pages = [
        PDFPage("first.pdf", 1, "First document content."),
        PDFPage("second.pdf", 1, "Second document content."),
    ]

    chunks, statistics = chunker.process(pages)

    assert statistics.total_documents == 2
    assert len(chunks) == 2


def test_calculate_chunking_statistics(chunker):

    pages = [
        PDFPage("sample.pdf", 1, "Alpha beta gamma."),
        PDFPage("sample.pdf", 2, "Delta epsilon zeta."),
    ]

    chunks, statistics = chunker.process(pages)

    assert statistics.total_chunks == len(chunks)

    assert statistics.total_tokens == sum(
        chunk.token_count for chunk in chunks
    )

    assert statistics.minimum_tokens >= 1

    assert statistics.maximum_tokens >= statistics.minimum_tokens

    assert statistics.average_tokens > 0


def test_empty_pages_return_empty_chunks(chunker):

    chunks, statistics = chunker.process([])

    assert chunks == []

    assert statistics.total_chunks == 0
    assert statistics.total_tokens == 0
    assert statistics.total_documents == 0
    assert statistics.total_pages == 0


def test_empty_page_text_is_ignored(chunker):

    pages = [
        PDFPage("sample.pdf", 1, "   "),
    ]

    chunks = chunker.chunk_pages(pages)

    assert chunks == []


# ============================================================
# INPUT VALIDATION TESTS
# ============================================================

def test_reject_invalid_pages_collection(chunker):

    with pytest.raises(TypeError):
        chunker.process("invalid")


def test_reject_invalid_page_object(chunker):

    with pytest.raises(TypeError):
        chunker.process(["invalid"])


def test_reject_empty_pdf_source(chunker):

    pages = [
        PDFPage("", 1, "Some content."),
    ]

    with pytest.raises(ValueError):
        chunker.process(pages)


def test_reject_invalid_page_number(chunker):

    pages = [
        PDFPage("sample.pdf", 0, "Some content."),
    ]

    with pytest.raises(ValueError):
        chunker.process(pages)


def test_reject_invalid_page_text(chunker):

    pages = [
        PDFPage("sample.pdf", 1, None),
    ]

    with pytest.raises(TypeError):
        chunker.process(pages)


def test_reject_invalid_normalizer(tokenizer):

    config = ChunkingConfig()

    with pytest.raises(TypeError):
        TextChunker(
            config=config,
            tokenizer=tokenizer,
            normalizer=object(),
        )


# ============================================================
# HEADING PRESERVATION TESTS
# ============================================================

def test_chunks_preserve_heading_hierarchy(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "2. Risk Profile\n"
                "2.1 Floods\n"
                "Alpha beta gamma delta epsilon zeta eta theta."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    expected_heading = (
        "2. Risk Profile > 2.1 Floods"
    )

    assert len(chunks) > 0

    assert all(
        chunk.heading == expected_heading
        for chunk in chunks
    )


def test_heading_context_updates_for_new_section(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "2. Risk Profile\n"
                "2.1 Floods\n"
                "Alpha beta gamma.\n"
                "2.2 Landslides\n"
                "Delta epsilon zeta."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    first_chunks = [
        chunk for chunk in chunks
        if "Alpha" in chunk.text
    ]

    second_chunks = [
        chunk for chunk in chunks
        if "Delta" in chunk.text
    ]

    assert first_chunks
    assert second_chunks

    assert all(
        chunk.heading == "2. Risk Profile > 2.1 Floods"
        for chunk in first_chunks
    )

    assert all(
        chunk.heading == "2. Risk Profile > 2.2 Landslides"
        for chunk in second_chunks
    )


def test_heading_context_continues_across_pages(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "2. Risk Profile\n"
                "2.1 Floods\n"
                "Alpha beta gamma."
            ),
        ),
        PDFPage(
            source="sample.pdf",
            page_number=2,
            text="Delta epsilon zeta.",
        ),
    ]

    chunks = chunker.chunk_pages(pages)

    expected_heading = "2. Risk Profile > 2.1 Floods"

    assert len(chunks) == 1

    assert chunks[0].heading == expected_heading

    assert chunks[0].page_numbers == (1, 2)

    assert "Alpha beta gamma." in chunks[0].text

    assert "Delta epsilon zeta." in chunks[0].text

    assert chunks[0].token_count <= chunker.config.chunk_size


def test_heading_lines_are_not_chunk_content(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "2. Risk Profile\n"
                "2.1 Floods\n"
                "Alpha beta gamma."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    assert chunks

    for chunk in chunks:
        assert "2. Risk Profile" not in chunk.text
        assert "2.1 Floods" not in chunk.text


def test_chunks_without_headings_keep_none_metadata(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text="Python supports object oriented programming.",
        )
    ]

    chunks = chunker.chunk_pages(pages)

    assert chunks

    assert all(
        chunk.heading is None
        for chunk in chunks
    )


# ============================================================
# CROSS-PAGE CHUNKING TESTS
# ============================================================

def test_merge_content_across_pages_with_same_heading(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "2. Risk Profile\n"
                "2.1 Floods\n"
                "Alpha beta gamma."
            ),
        ),
        PDFPage(
            source="sample.pdf",
            page_number=2,
            text="Delta epsilon zeta.",
        ),
    ]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert "Alpha beta gamma." in chunk.text

    assert "Delta epsilon zeta." in chunk.text

    assert chunk.page_numbers == (1, 2)

    assert chunk.heading == (
        "2. Risk Profile > 2.1 Floods"
    )

    assert chunk.token_count <= chunker.config.chunk_size


def test_do_not_merge_content_when_heading_changes(chunker):

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=1,
            text=(
                "2. Risk Profile\n"
                "2.1 Floods\n"
                "Alpha beta gamma."
            ),
        ),
        PDFPage(
            source="sample.pdf",
            page_number=2,
            text=(
                "2.2 Landslides\n"
                "Delta epsilon zeta."
            ),
        ),
    ]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) == 2

    assert chunks[0].page_numbers == (1,)
    assert chunks[1].page_numbers == (2,)

    assert chunks[0].heading == (
        "2. Risk Profile > 2.1 Floods"
    )

    assert chunks[1].heading == (
        "2. Risk Profile > 2.2 Landslides"
    )