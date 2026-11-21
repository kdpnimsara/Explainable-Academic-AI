import pytest

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.tokenizer_adapter import (
    HuggingFaceTokenizerAdapter,
)
from src.ingestion.chunking.recursive_splitter import (
    RecursiveTextSplitter,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


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
            "Pretrained tokenizer is not available locally."
        )


@pytest.fixture
def splitter(real_tokenizer):
    """Create a splitter using the real tokenizer."""

    config = ChunkingConfig(
        chunk_size=240,
        chunk_overlap=32,
    )

    return RecursiveTextSplitter(
        config=config,
        tokenizer=real_tokenizer,
    )


def test_real_tokenizer_loads(real_tokenizer):
    """Verify the actual pretrained tokenizer."""

    assert real_tokenizer.vocabulary_size == 30522
    assert real_tokenizer.model_max_length == 512


def test_real_tokenizer_produces_tokens(real_tokenizer):
    text = "B+ trees maintain sorted keys."

    token_ids = real_tokenizer.encode(text)

    assert len(token_ids) > 0
    assert all(isinstance(token, int) for token in token_ids)


def test_real_tokenizer_returns_valid_offsets(real_tokenizer):
    text = "The B+ tree keeps keys in sorted order."

    token_ids, offsets = real_tokenizer.encode_with_offsets(text)

    assert len(token_ids) == len(offsets)

    for start, end in offsets:
        assert 0 <= start <= end <= len(text)


def test_offsets_preserve_original_text(real_tokenizer):
    """Verify that offsets refer to the original characters."""

    text = "The B+ Tree keeps Academic Records."

    _, offsets = real_tokenizer.encode_with_offsets(text)

    extracted_text = [
        text[start:end]
        for start, end in offsets
        if start != end
    ]

    assert "The" in extracted_text
    assert "Tree" in extracted_text
    assert "Academic" in extracted_text


def test_real_tokenizer_splitter_creates_multiple_chunks(splitter):
    """Verify splitting with a real pretrained tokenizer."""

    sentence = (
        "the b tree stores sorted keys and supports "
        "efficient range queries. "
    )

    text = sentence * 40

    chunks = splitter.split_text(text)

    assert len(chunks) > 1


def test_real_tokenizer_respects_chunk_size(
    splitter,
    real_tokenizer,
):
    """Verify actual token counts of generated chunks."""

    sentence = (
        "the b tree stores sorted keys and supports "
        "efficient range queries. "
    )

    text = sentence * 40

    chunks = splitter.split_text(text)

    for chunk in chunks:

        actual_token_count = len(
            real_tokenizer.encode(chunk)
        )

        assert actual_token_count <= 240


def test_real_tokenizer_overlap_does_not_exceed_limit(
    splitter,
    real_tokenizer,
):
    """Verify that overlap does not exceed the token limit."""

    sentence = (
        "the b tree stores sorted keys and supports "
        "efficient range queries. "
    )

    chunks = splitter.split_text(sentence * 40)

    assert len(chunks) > 1

    for chunk in chunks:
        assert (
            len(real_tokenizer.encode(chunk))
            <= splitter.config.chunk_size
        )


def test_real_tokenizer_handles_empty_text(splitter):
    assert splitter.split_text("") == []
    assert splitter.split_text("   ") == []


def test_real_tokenizer_handles_short_text(splitter):
    text = "the b tree stores sorted keys."

    chunks = splitter.split_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text

def test_offset_splitting_preserves_original_text(real_tokenizer):
    """Verify exact original text preservation during token splitting."""

    config = ChunkingConfig(
        chunk_size=48,
        chunk_overlap=0,
    )

    splitter = RecursiveTextSplitter(
        config=config,
        tokenizer=real_tokenizer,
    )

    text = (
        "B+Tree,AcademicRecords,"
        "MathematicalExplainability,ExactCapitalization,"
    ) * 30

    chunks = splitter.split_text(text)

    assert len(chunks) > 1

    # No characters should be lost or modified.
    assert "".join(chunks) == text


def test_offset_overlap_preserves_original_capitalization(
    real_tokenizer,
):
    """Verify capitalization is preserved during overlap."""

    config = ChunkingConfig(
        chunk_size=48,
        chunk_overlap=8,
    )

    splitter = RecursiveTextSplitter(
        config=config,
        tokenizer=real_tokenizer,
    )

    sentence = (
        "B+Tree Maintains Academic Records "
        "With ExactCapitalization. "
    )

    chunks = splitter.split_text(sentence * 30)

    assert len(chunks) > 1

    combined_text = "\n".join(chunks)

    assert "B+Tree" in combined_text
    assert "Academic" in combined_text
    assert "ExactCapitalization" in combined_text

    for chunk in chunks:
        assert len(real_tokenizer.encode(chunk)) <= 48