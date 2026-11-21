import re

import pytest

from src.ingestion.chunking.tokenizer_adapter import (
    HuggingFaceTokenizerAdapter,
)


MODEL_NAME = "intfloat/multilingual-e5-small"


class FakeFastTokenizer:
    """Simulate a Hugging Face fast tokenizer."""

    is_fast = True
    model_max_length = 512
    vocab_size = 30522

    def encode(
        self,
        text,
        add_special_tokens=False,
        verbose=True,
    ):

        return [
            index
            for index, _ in enumerate(
                re.finditer(r"\S+", text),
                start=1,
            )
        ]

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return "decoded text"

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=True,
    ):

        matches = list(
            re.finditer(r"\S+", text)
        )

        return {
            "input_ids": list(
                range(1, len(matches) + 1)
            ),
            "offset_mapping": [
                match.span()
                for match in matches
            ],
        }


@pytest.fixture
def adapter():
    return HuggingFaceTokenizerAdapter(
        FakeFastTokenizer()
    )


@pytest.fixture(scope="module")
def real_huggingface_adapter():
    """
    Load a locally cached Hugging Face tokenizer.

    Skip the optional real-model test when the model
    is unavailable locally. Do not download during tests.
    """

    try:
        return HuggingFaceTokenizerAdapter.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )

    except OSError:
        pytest.skip(
            "Hugging Face tokenizer is not available locally."
        )


# ======================================================
# INITIALIZATION TESTS
# ======================================================

def test_initialize_fast_tokenizer(adapter):

    assert adapter.model_max_length == 512

    assert adapter.vocabulary_size == 30522


# ======================================================
# ENCODING TESTS
# ======================================================

def test_encode_text(adapter):

    result = adapter.encode(
        "Python is useful"
    )

    assert len(result) == 3

    assert all(
        isinstance(token, int)
        for token in result
    )


def test_decode_tokens(adapter):

    result = adapter.decode(
        [1, 2, 3]
    )

    assert isinstance(result, str)

    assert result == "decoded text"


def test_encode_with_offsets(adapter):

    text = "B+ Tree keeps keys sorted."

    token_ids, offsets = adapter.encode_with_offsets(
        text
    )

    assert len(token_ids) == len(offsets)

    extracted_parts = [
        text[start:end]
        for start, end in offsets
    ]

    assert extracted_parts == [
        "B+",
        "Tree",
        "keeps",
        "keys",
        "sorted.",
    ]


def test_offsets_preserve_original_capitalization(adapter):

    text = "Academic PDF Document"

    _, offsets = adapter.encode_with_offsets(
        text
    )

    original_parts = [
        text[start:end]
        for start, end in offsets
    ]

    assert original_parts[0] == "Academic"

    assert original_parts[1] == "PDF"

    assert original_parts[2] == "Document"


# ======================================================
# INPUT VALIDATION TESTS
# ======================================================

def test_reject_invalid_text(adapter):

    with pytest.raises(TypeError):
        adapter.encode(123)


def test_reject_invalid_offset_input(adapter):

    with pytest.raises(TypeError):
        adapter.encode_with_offsets(None)


def test_reject_slow_tokenizer():

    class SlowTokenizer:

        is_fast = False

        def encode(self, text):
            return []

        def decode(self, token_ids):
            return ""

    with pytest.raises(ValueError):
        HuggingFaceTokenizerAdapter(
            SlowTokenizer()
        )


def test_reject_invalid_tokenizer():

    with pytest.raises(TypeError):
        HuggingFaceTokenizerAdapter(
            object()
        )


# ======================================================
# REAL HUGGING FACE TOKENIZER TEST
# ======================================================

def test_real_huggingface_tokenizer(
    real_huggingface_adapter,
):
    """
    Test the actual locally cached multilingual E5 tokenizer.
    """

    adapter = real_huggingface_adapter

    text = "A B+ tree keeps keys in sorted order."

    token_ids, offsets = adapter.encode_with_offsets(
        text
    )

    assert len(token_ids) > 0

    assert len(token_ids) == len(offsets)

    # Original characters must remain accessible.
    assert (
        text[offsets[0][0]:offsets[0][1]]
        == "A"
    )

    assert adapter.vocabulary_size == 250002

    for start, end in offsets:

        assert 0 <= start <= end <= len(text)


# ======================================================
# LONG TEXT WARNING HANDLING TEST
# ======================================================

def test_long_text_does_not_truncate_with_offsets(adapter):

    text = "Disaster preparedness information. " * 200

    token_ids, offsets = adapter.encode_with_offsets(
        text
    )

    assert len(token_ids) > 512

    assert len(token_ids) == len(offsets)

    assert offsets[-1][1] <= len(text)


def test_long_text_encoding_does_not_truncate(adapter):

    text = "Flood preparedness information. " * 200

    token_ids = adapter.encode(text)

    assert len(token_ids) > 512