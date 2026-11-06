import pytest

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.recursive_splitter import (
    RecursiveTextSplitter,
)


class FakeWordTokenizer:
    """Simple tokenizer for deterministic unit tests."""

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


class FakeCharacterTokenizer:
    """Tokenizer used to test long unbroken text."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.fixture
def tokenizer():
    return FakeWordTokenizer()


def create_splitter(
    tokenizer,
    chunk_size=8,
    chunk_overlap=2,
):
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return RecursiveTextSplitter(
        config=config,
        tokenizer=tokenizer,
    )


def test_empty_text_returns_empty_list(tokenizer):
    splitter = create_splitter(tokenizer)

    assert splitter.split_text("") == []
    assert splitter.split_text("   ") == []


def test_short_text_remains_single_chunk(tokenizer):
    splitter = create_splitter(tokenizer)

    text = "Python is useful"

    result = splitter.split_text(text)

    assert len(result) == 1
    assert result[0] == text


def test_recursive_sentence_splitting(tokenizer):
    splitter = create_splitter(
        tokenizer,
        chunk_size=7,
        chunk_overlap=2,
    )

    text = (
        "Alpha beta gamma. "
        "Delta epsilon zeta. "
        "Eta theta iota."
    )

    result = splitter.split_text(text)

    assert len(result) == 3
    assert "Alpha beta gamma." in result[0]
    assert "Delta epsilon zeta." in result[1]
    assert "Eta theta iota." in result[2]


def test_paragraph_boundaries_are_preserved(tokenizer):
    splitter = create_splitter(
        tokenizer,
        chunk_size=6,
        chunk_overlap=1,
    )

    text = (
        "First paragraph here.\n\n"
        "Second paragraph here.\n\n"
        "Third paragraph here."
    )

    result = splitter.split_text(text)

    assert len(result) == 3

    combined = " ".join(result)

    assert combined.index("First") < combined.index("Second")
    assert combined.index("Second") < combined.index("Third")


def test_chunk_token_limit_is_respected(tokenizer):
    splitter = create_splitter(
        tokenizer,
        chunk_size=8,
        chunk_overlap=2,
    )

    text = (
        "alpha bravo charlie delta echo foxtrot "
        "golf hotel india juliet kilo lima"
    )

    result = splitter.split_text(text)

    assert len(result) >= 2

    for chunk in result:
        assert len(tokenizer.encode(chunk)) <= 8


def test_token_overlap_is_applied(tokenizer):
    splitter = create_splitter(
        tokenizer,
        chunk_size=8,
        chunk_overlap=2,
    )

    text = (
        "alpha bravo charlie delta echo foxtrot "
        "golf hotel india juliet kilo lima"
    )

    result = splitter.split_text(text)

    assert len(result) >= 2

    previous_tokens = tokenizer.encode(result[0])
    next_tokens = tokenizer.encode(result[1])

    assert previous_tokens[-2:] == next_tokens[:2]


def test_zero_overlap(tokenizer):
    splitter = create_splitter(
        tokenizer,
        chunk_size=4,
        chunk_overlap=0,
    )

    result = splitter.split_text(
        "one two three four five six seven eight"
    )

    assert len(result) == 2

    first_tokens = tokenizer.encode(result[0])
    second_tokens = tokenizer.encode(result[1])

    assert first_tokens[-1] != second_tokens[0]


def test_long_unbroken_text_uses_token_fallback():
    tokenizer = FakeCharacterTokenizer()

    splitter = create_splitter(
        tokenizer,
        chunk_size=10,
        chunk_overlap=2,
    )

    text = "x" * 35

    result = splitter.split_text(text)

    assert len(result) > 1

    for chunk in result:
        assert len(tokenizer.encode(chunk)) <= 10


def test_non_string_input_is_rejected(tokenizer):
    splitter = create_splitter(tokenizer)

    with pytest.raises(TypeError):
        splitter.split_text(123)


def test_invalid_configuration_type_is_rejected(tokenizer):
    with pytest.raises(TypeError):
        RecursiveTextSplitter(
            config="invalid",
            tokenizer=tokenizer,
        )


def test_invalid_tokenizer_is_rejected():
    config = ChunkingConfig()

    with pytest.raises(TypeError):
        RecursiveTextSplitter(
            config=config,
            tokenizer=object(),
        )


def test_invalid_tokenizer_output_is_rejected():
    class InvalidTokenizer:

        def encode(self, text):
            return ["invalid"]

        def decode(self, token_ids):
            return "text"

    config = ChunkingConfig()

    splitter = RecursiveTextSplitter(
        config=config,
        tokenizer=InvalidTokenizer(),
    )

    with pytest.raises(TypeError):
        splitter.split_text("Some sample text")


def test_minimum_chunk_size_validation():
    config = ChunkingConfig(
        chunk_size=6,
        chunk_overlap=4,
        min_chunk_size=3,
    )

    with pytest.raises(ValueError):
        RecursiveTextSplitter(
            config=config,
            tokenizer=FakeWordTokenizer(),
        )