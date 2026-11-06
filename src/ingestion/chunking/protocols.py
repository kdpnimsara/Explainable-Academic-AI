from collections.abc import Sequence
from typing import Protocol


class TokenizerProtocol(Protocol):
    """Interface for interchangeable tokenizers."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text."""
        ...


class EmbeddingModelProtocol(Protocol):
    """Interface for interchangeable embedding models."""

    def embed(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        """Convert text inputs into dense numerical vectors."""
        ...