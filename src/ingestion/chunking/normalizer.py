import re
import unicodedata
from collections.abc import Sequence

from src.ingestion.pdf_loader import PDFPage


class TextNormalizer:
    """Normalize PDF text while preserving document structure."""

    def normalize_text(
        self,
        text: str,
        preserve_layout: bool = False,
    ) -> str:

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        if not isinstance(preserve_layout, bool):
            raise TypeError(
                "preserve_layout must be a boolean."
            )

        if not text or not text.strip():
            return ""

        # Preserve mathematical and scientific Unicode.
        text = unicodedata.normalize("NFC", text)

        # Normalize line endings.
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove invisible formatting characters.
        text = text.replace("\u00ad", "")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")

        # Normalize non-breaking spaces.
        text = text.replace("\u00a0", " ")
        text = text.replace("\u202f", " ")

        # Preserve Unicode joining characters.
        text = "".join(
            character
            for character in text
            if (
                character in "\n\t\u200c\u200d"
                or unicodedata.category(character)[0] != "C"
            )
        )

        lines = text.split("\n")

        if preserve_layout:

            # Preserve leading spaces and column alignment.
            # Only remove unnecessary trailing whitespace.
            normalized_lines = [
                line.rstrip(" \t")
                for line in lines
            ]

            normalized_text = "\n".join(
                normalized_lines
            )

            # Limit excessive blank lines.
            normalized_text = re.sub(
                r"\n{3,}",
                "\n\n",
                normalized_text,
            )

            # Remove outer blank lines without removing indentation.
            return normalized_text.strip("\n")

        # Plain-text normalization.
        normalized_lines: list[str] = []

        for line in lines:

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            )

            normalized_lines.append(
                line.strip()
            )

        normalized_text = "\n".join(
            normalized_lines
        )

        normalized_text = re.sub(
            r"\n{3,}",
            "\n\n",
            normalized_text,
        )

        return normalized_text.strip()

    def normalize_page(
        self,
        page: PDFPage,
    ) -> PDFPage:

        if not isinstance(page, PDFPage):
            raise TypeError(
                "Input must be a PDFPage object."
            )

        normalized_text = self.normalize_text(
            page.text,
            preserve_layout=(
                page.extraction_mode == "layout"
            ),
        )

        return PDFPage(
            source=page.source,
            page_number=page.page_number,
            text=normalized_text,
            extraction_mode=page.extraction_mode,
        )

    def normalize_pages(
        self,
        pages: Sequence[PDFPage],
    ) -> list[PDFPage]:

        if not isinstance(pages, Sequence) or isinstance(
            pages,
            (str, bytes, bytearray),
        ):
            raise TypeError(
                "pages must be a sequence of PDFPage objects."
            )

        normalized_pages: list[PDFPage] = []

        for page in pages:

            normalized_page = self.normalize_page(page)

            if normalized_page.text.strip():
                normalized_pages.append(
                    normalized_page
                )

        return normalized_pages