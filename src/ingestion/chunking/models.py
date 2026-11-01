from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration settings for the chunking engine."""

    chunk_size: int = 384
    chunk_overlap: int = 64
    min_chunk_size: int = 1

    def __post_init__(self) -> None:
        """Validate chunking configuration."""

        integer_fields = {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
        }

        for field_name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

        if self.chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if not 0 < self.min_chunk_size <= self.chunk_size:
            raise ValueError(
                "min_chunk_size must be between 1 and chunk_size."
            )


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Represent a processed text chunk with its metadata."""

    chunk_id: str
    source: str
    chunk_index: int
    text: str
    token_count: int
    page_numbers: tuple[int, ...]
    heading: str | None = None

    def __post_init__(self) -> None:
        """Validate chunk data."""

        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty.")

        if not self.source.strip():
            raise ValueError("source cannot be empty.")

        if not self.text.strip():
            raise ValueError("chunk text cannot be empty.")

        if self.chunk_index < 1:
            raise ValueError("chunk_index must be at least 1.")

        if self.token_count < 1:
            raise ValueError("token_count must be positive.")

        if not self.page_numbers:
            raise ValueError(
                "At least one PDF page number is required."
            )

        if any(
            isinstance(number, bool)
            or not isinstance(number, int)
            or number < 1
            for number in self.page_numbers
        ):
            raise ValueError("Invalid PDF page number.")

        if tuple(sorted(set(self.page_numbers))) != self.page_numbers:
            raise ValueError(
                "page_numbers must be unique and sorted."
            )


@dataclass(frozen=True, slots=True)
class ChunkingStatistics:
    """Store statistics about the chunking process."""

    total_chunks: int
    total_tokens: int
    average_tokens: float
    minimum_tokens: int
    maximum_tokens: int
    total_documents: int
    total_pages: int