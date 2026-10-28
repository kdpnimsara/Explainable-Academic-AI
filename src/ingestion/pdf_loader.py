from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class PDFPage:
    """Store the text and metadata of one PDF page."""

    source: str
    page_number: int
    text: str


class PDFLoader:
    """Read text from PDF documents."""

    def load(self, file_path: str | Path) -> list[PDFPage]:
        """Extract text and metadata from a PDF file."""

        path = Path(file_path)

        # Check whether the file exists.
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # Check whether the path refers to a file.
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Check the file extension.
        if path.suffix.lower() != ".pdf":
            raise ValueError("Please provide a PDF file.")

        try:
            # Open the PDF document.
            reader = PdfReader(str(path))

            # Check whether the PDF is encrypted.
            if reader.is_encrypted:
                if reader.decrypt("") == 0:
                    raise ValueError(
                        f"Password-protected PDF: {path.name}"
                    )

            pages: list[PDFPage] = []

            # Read each page separately.
            for page_number, page in enumerate(
                reader.pages, start=1
            ):
                text = (page.extract_text() or "").strip()

                # Save pages that contain readable text.
                if text:
                    pdf_page = PDFPage(
                        source=path.name,
                        page_number=page_number,
                        text=text,
                    )

                    pages.append(pdf_page)

            # Check whether any text was extracted.
            if not pages:
                raise ValueError(
                    f"No extractable text found in {path.name}"
                )

            return pages

        except PdfReadError as error:
            raise ValueError(
                f"Could not read PDF: {error}"
            ) from error