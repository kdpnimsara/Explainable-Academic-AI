from src.ingestion.pdf_loader import PDFPage

from src.ingestion.chunking.models import ChunkingConfig
from src.ingestion.chunking.chunker import TextChunker


class FakeWordTokenizer:
    """Provide deterministic word-level tokenization."""

    def encode(self, text: str) -> list[int]:
        return list(range(1, len(text.split()) + 1))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(
            str(token_id)
            for token_id in token_ids
        )


def test_preserve_original_pdf_page_content():

    chunker = TextChunker(
        config=ChunkingConfig(
            chunk_size=64,
            chunk_overlap=0,
        ),
        tokenizer=FakeWordTokenizer(),
    )

    pages = [
        PDFPage(
            source="sample.pdf",
            page_number=7,
            text=(
                "The instructions refer to Page 1 of 4 "
                "in the guide.\n"
                "This is regular body content."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    assert chunks

    combined_text = "\n".join(
        chunk.text
        for chunk in chunks
    )

    assert "Page 1 of 4" in combined_text
    assert "regular body content" in combined_text

    assert all(
        chunk.source == "sample.pdf"
        for chunk in chunks
    )

    assert all(
        7 in chunk.page_numbers
        for chunk in chunks
    )