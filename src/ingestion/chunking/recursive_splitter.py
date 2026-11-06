import re

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.protocols import TokenizerProtocol


class RecursiveTextSplitter:
    """Split text recursively while preserving original text."""

    SEPARATORS = (
        r"\n[ \t]*\n+",
        r"\n+",
        r"(?<=[.!?])\s+",
        r"\s+",
    )

    def __init__(
        self,
        config: ChunkingConfig,
        tokenizer: TokenizerProtocol,
    ) -> None:
        """Initialize using dependency injection."""

        if not isinstance(config, ChunkingConfig):
            raise TypeError(
                "config must be a ChunkingConfig object."
            )

        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError(
                "Tokenizer must implement encode()."
            )

        if not callable(getattr(tokenizer, "decode", None)):
            raise TypeError(
                "Tokenizer must implement decode()."
            )

        self.config = config
        self.tokenizer = tokenizer

        self.content_limit = (
            config.chunk_size - config.chunk_overlap
        )

        if config.min_chunk_size > self.content_limit:
            raise ValueError(
                "min_chunk_size cannot exceed the available "
                "chunk content capacity."
            )

    @staticmethod
    def _validate_token_ids(token_ids: list[int]) -> None:
        """Validate token ID values."""

        if not isinstance(token_ids, list):
            raise TypeError(
                "Tokenizer must return a list of token IDs."
            )

        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            raise TypeError(
                "Token IDs must contain integers only."
            )

    def _encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""

        token_ids = self.tokenizer.encode(text)

        self._validate_token_ids(token_ids)

        return token_ids

    def _decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""

        decoded_text = self.tokenizer.decode(token_ids)

        if not isinstance(decoded_text, str):
            raise TypeError(
                "Tokenizer decode() must return a string."
            )

        return decoded_text

    def _encode_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]] | None:
        """Get token IDs and original character positions."""

        offset_encoder = getattr(
            self.tokenizer,
            "encode_with_offsets",
            None,
        )

        # Support tokenizers without offset mapping.
        if not callable(offset_encoder):
            return None

        result = offset_encoder(text)

        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "encode_with_offsets() must return "
                "(token_ids, offsets)."
            )

        token_ids, offsets = result

        self._validate_token_ids(token_ids)

        if len(token_ids) != len(offsets):
            raise ValueError(
                "Token IDs and offsets must have equal lengths."
            )

        validated_offsets: list[tuple[int, int]] = []

        previous_start = 0
        previous_end = 0

        for offset in offsets:

            if not isinstance(offset, (tuple, list)):
                raise TypeError(
                    "Each offset must be a tuple or list."
                )

            if len(offset) != 2:
                raise ValueError(
                    "Each offset must contain start and end positions."
                )

            start, end = offset

            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
            ):
                raise TypeError(
                    "Offset positions must be integers."
                )

            if not 0 <= start <= end <= len(text):
                raise ValueError(
                    "Tokenizer returned invalid character offsets."
                )

            if start < previous_start or end < previous_end:
                raise ValueError(
                    "Tokenizer offsets must be ordered."
                )

            validated_offsets.append((start, end))

            previous_start = start
            previous_end = end

        return token_ids, validated_offsets

    @staticmethod
    def _slice_by_offsets(
        text: str,
        offsets: list[tuple[int, int]],
        start_index: int,
        end_index: int,
    ) -> str:
        """Extract original text using token character positions."""

        selected_offsets = [
            offset
            for offset in offsets[start_index:end_index]
            if offset[0] < offset[1]
        ]

        if not selected_offsets:
            return ""

        char_start = selected_offsets[0][0]
        char_end = selected_offsets[-1][1]

        return text[char_start:char_end]

    def _count_tokens(self, text: str) -> int:
        """Calculate actual token count."""

        return len(self._encode(text))

    def _split_preserving_separator(
        self,
        text: str,
        separator: str,
    ) -> list[str]:
        """Split text while retaining separator characters."""

        matches = list(re.finditer(separator, text))

        if not matches:
            return [text]

        parts: list[str] = []
        start = 0

        for match in matches:

            end = match.end()
            part = text[start:end]

            if part:
                parts.append(part)

            start = end

        remaining = text[start:]

        if remaining:
            parts.append(remaining)

        return parts

    def _split_recursively(
        self,
        text: str,
        separator_index: int = 0,
    ) -> list[str]:
        """Recursively split using progressively smaller boundaries."""

        if self._count_tokens(text) <= self.content_limit:
            return [text]

        for index in range(
            separator_index,
            len(self.SEPARATORS),
        ):

            separator = self.SEPARATORS[index]

            parts = self._split_preserving_separator(
                text,
                separator,
            )

            if len(parts) <= 1:
                continue

            result: list[str] = []

            for part in parts:

                if self._count_tokens(part) <= self.content_limit:
                    result.append(part)

                else:
                    result.extend(
                        self._split_recursively(
                            part,
                            index + 1,
                        )
                    )

            return result

        return self._split_by_token_limit(text)

    def _split_by_token_limit(
        self,
        text: str,
    ) -> list[str]:
        """Split long text using original character offsets."""

        offset_result = self._encode_with_offsets(text)

        # Offset-based implementation for fast tokenizers.
        if offset_result is not None:

            token_ids, offsets = offset_result

            chunks: list[str] = []

            start = 0

            while start < len(token_ids):

                end = min(
                    start + self.content_limit,
                    len(token_ids),
                )

                candidate = self._slice_by_offsets(
                    text,
                    offsets,
                    start,
                    end,
                )

                # Ensure the extracted original text satisfies
                # the actual tokenizer's token limit.
                while (
                    end > start
                    and self._count_tokens(candidate)
                    > self.content_limit
                ):

                    end -= 1

                    candidate = self._slice_by_offsets(
                        text,
                        offsets,
                        start,
                        end,
                    )

                if end == start:
                    raise ValueError(
                        "Unable to create a valid original-text chunk."
                    )

                if candidate.strip():
                    chunks.append(candidate.strip())

                start = end

            return chunks

        # Fallback for tokenizers without offset support.
        token_ids = self._encode(text)

        if not token_ids:
            return []

        chunks = []

        for start in range(
            0,
            len(token_ids),
            self.content_limit,
        ):

            token_window = token_ids[
                start:start + self.content_limit
            ]

            decoded_text = self._decode(
                token_window
            ).strip()

            if not decoded_text:
                continue

            if self._count_tokens(decoded_text) > self.content_limit:
                raise ValueError(
                    "Decoded token window exceeds the content limit."
                )

            chunks.append(decoded_text)

        return chunks

    def _merge_small_chunks(
        self,
        chunks: list[str],
    ) -> list[str]:
        """Merge small adjacent chunks when sufficient capacity exists."""

        index = 0

        while index < len(chunks):

            current_count = self._count_tokens(chunks[index])

            if (
                current_count >= self.config.min_chunk_size
                or len(chunks) == 1
            ):
                index += 1
                continue

            if index + 1 < len(chunks):

                merged = chunks[index] + chunks[index + 1]

                if self._count_tokens(merged) <= self.content_limit:
                    chunks[index] = merged.strip()
                    del chunks[index + 1]
                    continue

            elif index > 0:

                merged = chunks[index - 1] + chunks[index]

                if self._count_tokens(merged) <= self.content_limit:
                    chunks[index - 1] = merged.strip()
                    chunks.pop()
                    continue

            index += 1

        return chunks

    def _pack_segments(
        self,
        segments: list[str],
    ) -> list[str]:
        """Combine smaller segments without exceeding content capacity."""

        chunks: list[str] = []
        current_chunk = ""

        for segment in segments:

            if not segment:
                continue

            if self._count_tokens(segment) > self.content_limit:
                raise ValueError(
                    "A segment exceeds the available content capacity."
                )

            if not current_chunk:
                current_chunk = segment
                continue

            candidate = current_chunk + segment

            if self._count_tokens(candidate) <= self.content_limit:
                current_chunk = candidate

            else:

                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                current_chunk = segment

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return self._merge_small_chunks(chunks)

    @staticmethod
    def _join_overlap(
        overlap_text: str,
        current_text: str,
    ) -> str:
        """Join overlap text while maintaining readable spacing."""

        if not overlap_text:
            return current_text

        if not current_text:
            return overlap_text

        if overlap_text[-1].isspace() or current_text[0].isspace():
            return overlap_text + current_text

        return overlap_text + " " + current_text

    def _extract_original_suffix(
        self,
        text: str,
        overlap_size: int,
    ) -> str:
        """Extract overlap from original text without decoding tokens."""

        offset_result = self._encode_with_offsets(text)

        if offset_result is None:

            token_ids = self._encode(text)

            overlap_ids = token_ids[-overlap_size:]

            return self._decode(overlap_ids).strip()

        token_ids, offsets = offset_result

        if not token_ids:
            return ""

        start_index = max(
            0,
            len(token_ids) - overlap_size,
        )

        # Expand backward when tokens belong to the same
        # contiguous original word.
        while start_index > 0:

            previous_end = offsets[start_index - 1][1]
            current_start = offsets[start_index][0]

            if previous_end == current_start:
                start_index -= 1
            else:
                break

        return self._slice_by_offsets(
            text,
            offsets,
            start_index,
            len(token_ids),
        ).strip()

    def _apply_overlap(
        self,
        chunks: list[str],
    ) -> list[str]:
        """Apply token-aware overlap while preserving original text."""

        if not chunks:
            return []

        if self.config.chunk_overlap == 0:
            return chunks

        result = [chunks[0]]

        for index in range(1, len(chunks)):

            previous_chunk = chunks[index - 1]

            previous_token_count = self._count_tokens(
                previous_chunk
            )

            maximum_overlap = min(
                self.config.chunk_overlap,
                previous_token_count,
            )

            current_chunk = chunks[index]

            selected_chunk = current_chunk

            for overlap_size in range(
                maximum_overlap,
                0,
                -1,
            ):

                overlap_text = self._extract_original_suffix(
                    previous_chunk,
                    overlap_size,
                )

                candidate = self._join_overlap(
                    overlap_text,
                    current_chunk,
                )

                if (
                    self._count_tokens(candidate)
                    <= self.config.chunk_size
                ):
                    selected_chunk = candidate.strip()
                    break

            result.append(selected_chunk)

        return result

    def split_text(self, text: str) -> list[str]:
        """Generate token-limited chunks with original-text preservation."""

        if not isinstance(text, str):
            raise TypeError(
                "Input text must be a string."
            )

        if not text.strip():
            return []

        text = text.strip()

        # 1. Recursive structure-aware splitting.
        segments = self._split_recursively(text)

        # 2. Combine segments within the available capacity.
        base_chunks = self._pack_segments(segments)

        # 3. Apply original-text-preserving overlap.
        final_chunks = self._apply_overlap(base_chunks)

        # 4. Validate all final chunk sizes.
        for chunk in final_chunks:

            if self._count_tokens(chunk) > self.config.chunk_size:
                raise ValueError(
                    "Final chunk exceeds the configured token limit."
                )

        return final_chunks