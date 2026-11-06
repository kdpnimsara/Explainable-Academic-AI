import hashlib
import json
from collections.abc import Sequence

from src.ingestion.pdf_loader import PDFPage

from src.ingestion.chunking.models import (
    ChunkingConfig,
    ChunkingStatistics,
    TextChunk,
)

from src.ingestion.chunking.protocols import TokenizerProtocol
from src.ingestion.chunking.normalizer import TextNormalizer

from src.ingestion.chunking.heading_detector import (
    HeadingDetector,
    HeadingMatch,
)

from src.ingestion.chunking.recursive_splitter import (
    RecursiveTextSplitter,
)

from src.metadata.document_metadata import DocumentMetadata


class TextChunker:
    """Coordinate PDF normalization, heading detection and chunking."""

    def __init__(
        self,
        config: ChunkingConfig,
        tokenizer: TokenizerProtocol,
        normalizer: TextNormalizer | None = None,
        heading_detector: HeadingDetector | None = None,
    ) -> None:
        """Initialize the chunking engine using dependency injection."""

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

        self.normalizer = (
            normalizer
            if normalizer is not None
            else TextNormalizer()
        )

        if not callable(
            getattr(self.normalizer, "normalize_pages", None)
        ):
            raise TypeError(
                "Normalizer must implement normalize_pages()."
            )

        self.heading_detector = (
            heading_detector
            if heading_detector is not None
            else HeadingDetector()
        )

        if not callable(
            getattr(self.heading_detector, "detect", None)
        ):
            raise TypeError(
                "Heading detector must implement detect()."
            )

        self.splitter = RecursiveTextSplitter(
            config=self.config,
            tokenizer=self.tokenizer,
        )

    # ========================================================
    # DOCUMENT IDENTITY
    # ========================================================

    @staticmethod
    def _document_key(
        source: str,
        document_metadata: DocumentMetadata | None,
    ) -> str:
        """Create a unique key for document-level processing."""

        if document_metadata is None:
            return f"source:{source}"

        identity = {
            "document_id": document_metadata.document_id,
            "document_scope": document_metadata.document_scope,
            "institution_id": document_metadata.institution_id,
        }

        return json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        )

    # ========================================================
    # PAGE VALIDATION
    # ========================================================

    def _validate_pages(
        self,
        pages: Sequence[PDFPage],
    ) -> tuple[PDFPage, ...]:
        """Validate PDF page inputs."""

        if not isinstance(pages, Sequence) or isinstance(
            pages,
            (str, bytes, bytearray),
        ):
            raise TypeError(
                "pages must be a sequence of PDFPage objects."
            )

        validated_pages: list[PDFPage] = []

        for page in pages:

            if not isinstance(page, PDFPage):
                raise TypeError(
                    "Every item must be a PDFPage object."
                )

            if (
                not isinstance(page.source, str)
                or not page.source.strip()
            ):
                raise ValueError(
                    "PDF source cannot be empty."
                )

            if (
                isinstance(page.page_number, bool)
                or not isinstance(page.page_number, int)
                or page.page_number < 1
            ):
                raise ValueError(
                    "PDF page number must be a positive integer."
                )

            if not isinstance(page.text, str):
                raise TypeError(
                    "PDF page text must be a string."
                )

            validated_pages.append(page)

        return tuple(validated_pages)

    # ========================================================
    # TOKENIZATION
    # ========================================================

    def _encode(self, text: str) -> list[int]:
        """Encode text and validate tokenizer output."""

        token_ids = self.tokenizer.encode(text)

        if not isinstance(token_ids, list):
            raise TypeError(
                "Tokenizer encode() must return a list."
            )

        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            raise TypeError(
                "Tokenizer must return integer token IDs."
            )

        return token_ids

    # ========================================================
    # CHUNK ID GENERATION
    # ========================================================

    @staticmethod
    def _generate_chunk_id(
        source: str,
        page_numbers: tuple[int, ...],
        chunk_index: int,
        text: str,
        heading: str | None = None,
        document_metadata: DocumentMetadata | None = None,
    ) -> str:
        """Generate a deterministic SHA-256 chunk identifier."""

        metadata_identity = None

        if document_metadata is not None:
            metadata_identity = (
                document_metadata.to_chroma_metadata()
            )

        identity_data = {
            "source": source,
            "page_numbers": list(page_numbers),
            "chunk_index": chunk_index,
            "text": text,
            "heading": heading,
            "document_metadata": metadata_identity,
        }

        serialized_data = json.dumps(
            identity_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            serialized_data.encode("utf-8")
        ).hexdigest()

    # ========================================================
    # HEADING MANAGEMENT
    # ========================================================

    @staticmethod
    def _format_heading_path(
        heading_stack: Sequence[HeadingMatch],
    ) -> str | None:
        """Build a hierarchical heading path."""

        if not heading_stack:
            return None

        return " > ".join(
            heading.heading
            for heading in heading_stack
        )

    def _extract_page_segments(
        self,
        page: PDFPage,
        heading_stack: list[HeadingMatch],
    ) -> list[tuple[str, str | None]]:
        """Separate page content according to heading context."""

        segments: list[tuple[str, str | None]] = []

        content_lines: list[str] = []

        def flush_segment() -> None:

            segment_text = "\n".join(
                content_lines
            ).strip()

            if segment_text:

                heading_path = self._format_heading_path(
                    heading_stack
                )

                segments.append(
                    (segment_text, heading_path)
                )

            content_lines.clear()

        cleaned_text = page.text

        for line_number, line in enumerate(
            cleaned_text.splitlines(),
            start=1,
        ):

            detected_heading = self.heading_detector.detect(
                line,
                line_number=line_number,
            )

            if detected_heading is None:
                content_lines.append(line)
                continue

            flush_segment()

            while (
                heading_stack
                and heading_stack[-1].level
                >= detected_heading.level
            ):
                heading_stack.pop()

            heading_stack.append(
                detected_heading
            )

        flush_segment()

        return segments

    # ========================================================
    # INITIAL CHUNK CREATION
    # ========================================================

    def _create_chunks(
        self,
        pages: Sequence[PDFPage],
    ) -> list[TextChunk]:
        """Create metadata-preserving chunks from PDF pages."""

        normalized_pages = self.normalizer.normalize_pages(
            pages
        )

        chunks: list[TextChunk] = []

        heading_contexts: dict[
            str,
            list[HeadingMatch],
        ] = {}

        document_indexes: dict[str, int] = {}

        for page in normalized_pages:

            document_key = self._document_key(
                page.source,
                page.document_metadata,
            )

            heading_stack = heading_contexts.setdefault(
                document_key,
                [],
            )

            segments = self._extract_page_segments(
                page,
                heading_stack,
            )

            for segment_text, heading_path in segments:

                page_chunks = self.splitter.split_text(
                    segment_text
                )

                for chunk_text in page_chunks:

                    chunk_text = chunk_text.strip()

                    if not chunk_text:
                        continue

                    token_ids = self._encode(
                        chunk_text
                    )

                    token_count = len(token_ids)

                    if token_count == 0:
                        raise ValueError(
                            "Tokenizer returned no tokens for a "
                            "non-empty chunk."
                        )

                    if token_count > self.config.chunk_size:
                        raise ValueError(
                            "Generated chunk exceeds the configured "
                            "token limit."
                        )

                    document_indexes[document_key] = (
                        document_indexes.get(document_key, 0) + 1
                    )

                    chunk_index = document_indexes[
                        document_key
                    ]

                    page_numbers = (
                        page.page_number,
                    )

                    chunk_id = self._generate_chunk_id(
                        source=page.source,
                        page_numbers=page_numbers,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        heading=heading_path,
                        document_metadata=page.document_metadata,
                    )

                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            source=page.source,
                            chunk_index=chunk_index,
                            text=chunk_text,
                            token_count=token_count,
                            page_numbers=page_numbers,
                            heading=heading_path,
                            document_metadata=page.document_metadata,
                        )
                    )

        return self._merge_cross_page_chunks(
            chunks
        )

    # ========================================================
    # CROSS-PAGE CHUNKING
    # ========================================================

    def _merge_cross_page_chunks(
        self,
        chunks: Sequence[TextChunk],
    ) -> list[TextChunk]:
        """Merge eligible chunks while preserving document metadata."""

        records: list[
            tuple[
                str,
                str,
                tuple[int, ...],
                str | None,
                int,
                DocumentMetadata | None,
                str,
            ]
        ] = []

        for chunk in chunks:

            document_key = self._document_key(
                chunk.source,
                chunk.document_metadata,
            )

            if records:

                (
                    previous_source,
                    previous_text,
                    previous_pages,
                    previous_heading,
                    previous_token_count,
                    previous_metadata,
                    previous_document_key,
                ) = records[-1]

                same_document = (
                    previous_document_key == document_key
                    and previous_source == chunk.source
                )

                same_heading = (
                    previous_heading == chunk.heading
                )

                same_metadata = (
                    previous_metadata == chunk.document_metadata
                )

                consecutive_pages = (
                    chunk.page_numbers[0]
                    == previous_pages[-1] + 1
                )

                if (
                    same_document
                    and same_heading
                    and same_metadata
                    and consecutive_pages
                ):

                    combined_text = (
                        previous_text.rstrip()
                        + "\n\n"
                        + chunk.text.lstrip()
                    )

                    combined_token_count = len(
                        self._encode(combined_text)
                    )

                    if (
                        combined_token_count
                        <= self.config.chunk_size
                    ):

                        combined_pages = tuple(
                            sorted(
                                set(
                                    previous_pages
                                    + chunk.page_numbers
                                )
                            )
                        )

                        records[-1] = (
                            previous_source,
                            combined_text,
                            combined_pages,
                            previous_heading,
                            combined_token_count,
                            previous_metadata,
                            previous_document_key,
                        )

                        continue

            records.append(
                (
                    chunk.source,
                    chunk.text,
                    chunk.page_numbers,
                    chunk.heading,
                    chunk.token_count,
                    chunk.document_metadata,
                    document_key,
                )
            )

        # Rebuild chunks with updated metadata and IDs.
        merged_chunks: list[TextChunk] = []

        document_indexes: dict[str, int] = {}

        for (
            source,
            text,
            page_numbers,
            heading,
            token_count,
            document_metadata,
            document_key,
        ) in records:

            document_indexes[document_key] = (
                document_indexes.get(document_key, 0) + 1
            )

            chunk_index = document_indexes[document_key]

            chunk_id = self._generate_chunk_id(
                source=source,
                page_numbers=page_numbers,
                chunk_index=chunk_index,
                text=text,
                heading=heading,
                document_metadata=document_metadata,
            )

            merged_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source=source,
                    chunk_index=chunk_index,
                    text=text,
                    token_count=token_count,
                    page_numbers=page_numbers,
                    heading=heading,
                    document_metadata=document_metadata,
                )
            )

        return merged_chunks

    # ========================================================
    # STATISTICS
    # ========================================================

    @staticmethod
    def _calculate_statistics(
        chunks: Sequence[TextChunk],
        pages: Sequence[PDFPage],
    ) -> ChunkingStatistics:
        """Calculate statistics for the chunking process."""

        token_counts = [
            chunk.token_count
            for chunk in chunks
        ]

        total_chunks = len(chunks)

        total_tokens = sum(token_counts)

        average_tokens = (
            total_tokens / total_chunks
            if total_chunks
            else 0.0
        )

        minimum_tokens = (
            min(token_counts)
            if token_counts
            else 0
        )

        maximum_tokens = (
            max(token_counts)
            if token_counts
            else 0
        )

        document_keys = {
            TextChunker._document_key(
                page.source,
                page.document_metadata,
            )
            for page in pages
        }

        total_documents = len(document_keys)

        total_pages = len(
            {
                (
                    TextChunker._document_key(
                        page.source,
                        page.document_metadata,
                    ),
                    page.page_number,
                )
                for page in pages
            }
        )

        return ChunkingStatistics(
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            average_tokens=average_tokens,
            minimum_tokens=minimum_tokens,
            maximum_tokens=maximum_tokens,
            total_documents=total_documents,
            total_pages=total_pages,
        )

    # ========================================================
    # PUBLIC METHODS
    # ========================================================

    def process(
        self,
        pages: Sequence[PDFPage],
    ) -> tuple[list[TextChunk], ChunkingStatistics]:
        """Process PDF pages and return chunks with statistics."""

        validated_pages = self._validate_pages(
            pages
        )

        chunks = self._create_chunks(
            validated_pages
        )

        statistics = self._calculate_statistics(
            chunks=chunks,
            pages=validated_pages,
        )

        return chunks, statistics

    def chunk_pages(
        self,
        pages: Sequence[PDFPage],
    ) -> list[TextChunk]:
        """Return only the generated text chunks."""

        chunks, _ = self.process(pages)

        return chunks