import pytest

from pypdf.errors import PdfReadError

from src.ingestion.pdf_loader import PDFLoader


class FakePage:
    """Create a simple page for testing."""

    def __init__(self, text: str | None):
        self.text = text

    def extract_text(self) -> str | None:
        return self.text


class FakePDFReader:
    """Simulate a PDF reader for testing."""

    def __init__(self, file_path: str):
        self.is_encrypted = False
        self.pages = [
            FakePage("First page text"),
            FakePage("   "),
            FakePage("Third page text"),
        ]


def test_load_pdf_text_and_metadata(tmp_path, monkeypatch):
    """Check whether PDF text and page details are extracted."""

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"Fake PDF content")

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        FakePDFReader,
    )

    loader = PDFLoader()
    pages = loader.load(pdf_file)

    assert len(pages) == 2

    assert pages[0].source == "sample.pdf"
    assert pages[0].page_number == 1
    assert pages[0].text == "First page text"

    assert pages[1].page_number == 3
    assert pages[1].text == "Third page text"


def test_missing_pdf_file():
    """Check whether a missing file raises an error."""

    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("missing_file.pdf")


def test_invalid_file_type(tmp_path):
    """Check whether a non-PDF file is rejected."""

    text_file = tmp_path / "notes.txt"

    text_file.write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    loader = PDFLoader()

    with pytest.raises(ValueError):
        loader.load(text_file)


def test_pdf_without_readable_text(tmp_path, monkeypatch):
    """Check whether a PDF without readable text is rejected."""

    pdf_file = tmp_path / "empty.pdf"
    pdf_file.write_bytes(b"Fake PDF content")

    class EmptyPDFReader:

        def __init__(self, file_path: str):
            self.is_encrypted = False
            self.pages = [
                FakePage("   "),
                FakePage(None),
            ]

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        EmptyPDFReader,
    )

    loader = PDFLoader()

    with pytest.raises(
        ValueError,
        match="No extractable text",
    ):
        loader.load(pdf_file)


def test_directory_path_is_rejected(tmp_path):
    """Check whether a directory is rejected as a PDF file."""

    directory = tmp_path / "folder.pdf"
    directory.mkdir()

    loader = PDFLoader()

    with pytest.raises(
        ValueError,
        match="Not a file",
    ):
        loader.load(directory)


def test_password_protected_pdf_is_rejected(
    tmp_path,
    monkeypatch,
):
    """Check whether password-protected PDFs are rejected."""

    pdf_file = tmp_path / "protected.pdf"
    pdf_file.write_bytes(b"Fake PDF content")

    class EncryptedPDFReader:

        def __init__(self, file_path: str):
            self.is_encrypted = True

        def decrypt(self, password: str):
            return 0

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        EncryptedPDFReader,
    )

    loader = PDFLoader()

    with pytest.raises(
        ValueError,
        match="Password-protected PDF",
    ):
        loader.load(pdf_file)


def test_corrupted_pdf_is_handled(tmp_path, monkeypatch):
    """Check whether corrupted PDFs are handled."""

    pdf_file = tmp_path / "corrupted.pdf"
    pdf_file.write_bytes(b"Invalid PDF content")

    class BrokenPDFReader:

        def __init__(self, file_path: str):
            raise PdfReadError("Invalid PDF structure")

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        BrokenPDFReader,
    )

    loader = PDFLoader()

    with pytest.raises(
        ValueError,
        match="Could not read PDF",
    ):
        loader.load(pdf_file)