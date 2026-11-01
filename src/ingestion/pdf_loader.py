from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PdfReadError


ExtractionMode = Literal["plain", "layout"]


@dataclass(frozen=True)
class PDFPage:
    """Store extracted PDF text and page metadata."""

    source: str
    page_number: int
    text: str
    extraction_mode: ExtractionMode = "plain"


class PDFLoader:
    """Extract text from single and multiple PDF documents."""

    SUPPORTED_MODES = ("plain", "layout")

    def __init__(
        self,
        extraction_mode: ExtractionMode = "plain",
    ) -> None:

        if extraction_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported extraction mode: {extraction_mode}"
            )

        self.extraction_mode = extraction_mode

    # ========================================================
    # PAGE TEXT EXTRACTION
    # ========================================================

    def _extract_page_text(self, page) -> str:
        """Extract text using the selected extraction strategy."""

        if self.extraction_mode == "layout":

            text = page.extract_text(
                extraction_mode="layout"
            )

        else:
            text = page.extract_text()

        return (text or "").strip()

    # ========================================================
    # SINGLE PDF LOADING
    # ========================================================

    def load(
        self,
        file_path: str | Path,
    ) -> list[PDFPage]:
        """Load and extract text from a single PDF document."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Not a file: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Please provide a PDF file."
            )

        try:

            reader = PdfReader(str(path))

            if reader.is_encrypted:

                if reader.decrypt("") == 0:
                    raise ValueError(
                        f"Password-protected PDF: {path.name}"
                    )

            pages: list[PDFPage] = []

            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):

                text = self._extract_page_text(page)

                if not text:
                    continue

                pages.append(
                    PDFPage(
                        source=path.name,
                        page_number=page_number,
                        text=text,
                        extraction_mode=self.extraction_mode,
                    )
                )

            if not pages:
                raise ValueError(
                    f"No extractable text found in {path.name}"
                )

            return pages

        except PdfReadError as error:

            raise ValueError(
                f"Could not read PDF: {error}"
            ) from error

    # ========================================================
    # MULTIPLE PDF LOADING
    # ========================================================

    def load_many(
        self,
        file_paths: Sequence[str | Path],
    ) -> list[PDFPage]:
        """
        Load multiple PDF documents.

        Documents are processed in the supplied order.
        Duplicate filenames are rejected to prevent
        ambiguous source metadata.
        """

        if not isinstance(
            file_paths,
            Sequence,
        ) or isinstance(
            file_paths,
            (str, bytes, bytearray),
        ):

            raise TypeError(
                "file_paths must be a sequence of PDF paths."
            )

        if not file_paths:
            return []

        paths = [
            Path(file_path)
            for file_path in file_paths
        ]

        # Check duplicate filenames before loading.
        filenames = [
            path.name.casefold()
            for path in paths
        ]

        if len(filenames) != len(set(filenames)):

            raise ValueError(
                "Duplicate PDF filenames are not allowed."
            )

        all_pages: list[PDFPage] = []

        # Load each document independently.
        for path in paths:

            document_pages = self.load(path)

            all_pages.extend(
                document_pages
            )

        return all_pages

    # ========================================================
    # DIRECTORY PDF LOADING
    # ========================================================

    def load_directory(
        self,
        directory_path: str | Path,
    ) -> list[PDFPage]:
        """
        Load every PDF document from a directory.

        PDF files are processed alphabetically.
        Non-PDF files are ignored.
        """

        directory = Path(directory_path)

        if not directory.exists():

            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():

            raise ValueError(
                f"Not a directory: {directory}"
            )

        # Find all PDF files in the directory.
        pdf_paths = sorted(
            (
                path
                for path in directory.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower() == ".pdf"
                )
            ),
            key=lambda path: (
                path.name.casefold()
            ),
        )

        # Reuse the multiple PDF loading implementation.
        return self.load_many(pdf_paths)