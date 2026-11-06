from pathlib import Path

import pytest

from src.ingestion.pdf_loader import PDFLoader


class FakePage:
    """Simulate an extracted PDF page."""

    def __init__(self, text):
        self.text = text

    def extract_text(self, extraction_mode=None):
        return self.text


class FakePDFReader:
    """Simulate different PDF documents."""

    def __init__(self, file_path):
        document_name = Path(file_path).stem

        self.is_encrypted = False

        self.pages = [
            FakePage(f"Content from {document_name}, page one."),
            FakePage("   "),
            FakePage(f"Content from {document_name}, page three."),
        ]


@pytest.fixture
def fake_pdf_reader(monkeypatch):
    """Replace the real PDF reader with a fake reader."""

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        FakePDFReader,
    )


def create_fake_pdf(path: Path):
    """Create a placeholder PDF file for testing."""

    path.write_bytes(b"Fake PDF content")

    return path


# ============================================================
# MULTIPLE PDF LOADING TESTS
# ============================================================

def test_load_many_five_pdf_documents(
    tmp_path,
    fake_pdf_reader,
):
    """Verify that five different PDF documents are loaded."""

    pdf_paths = []

    for index in range(1, 6):

        pdf_path = tmp_path / f"document_{index}.pdf"

        create_fake_pdf(pdf_path)

        pdf_paths.append(pdf_path)

    loader = PDFLoader()

    pages = loader.load_many(pdf_paths)

    assert len(pages) == 10

    sources = {
        page.source
        for page in pages
    }

    assert sources == {
        f"document_{index}.pdf"
        for index in range(1, 6)
    }

    for index in range(1, 6):

        document_pages = [
            page
            for page in pages
            if page.source == f"document_{index}.pdf"
        ]

        assert len(document_pages) == 2

        assert [
            page.page_number
            for page in document_pages
        ] == [1, 3]


def test_load_many_preserves_input_order(
    tmp_path,
    fake_pdf_reader,
):
    """Verify that documents follow the supplied input order."""

    names = [
        "third.pdf",
        "first.pdf",
        "second.pdf",
    ]

    paths = [
        create_fake_pdf(tmp_path / name)
        for name in names
    ]

    loader = PDFLoader()

    pages = loader.load_many(paths)

    first_occurrences = []

    for page in pages:

        if page.source not in first_occurrences:
            first_occurrences.append(page.source)

    assert first_occurrences == names


def test_load_many_rejects_duplicate_filenames(tmp_path):
    """Prevent ambiguous document source metadata."""

    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"

    first_directory.mkdir()
    second_directory.mkdir()

    first_pdf = create_fake_pdf(
        first_directory / "notes.pdf"
    )

    second_pdf = create_fake_pdf(
        second_directory / "notes.pdf"
    )

    loader = PDFLoader()

    with pytest.raises(
        ValueError,
        match="Duplicate PDF filenames",
    ):
        loader.load_many([
            first_pdf,
            second_pdf,
        ])


def test_load_many_rejects_invalid_collection():
    """Reject a single string instead of a collection."""

    loader = PDFLoader()

    with pytest.raises(TypeError):
        loader.load_many("document.pdf")


def test_load_many_returns_empty_list_for_empty_input():
    """An empty PDF collection should return an empty result."""

    loader = PDFLoader()

    assert loader.load_many([]) == []


# ============================================================
# DIRECTORY LOADING TESTS
# ============================================================

def test_load_directory_loads_all_pdf_files(
    tmp_path,
    fake_pdf_reader,
):
    """Verify that every PDF in a directory is loaded."""

    documents_directory = tmp_path / "documents"

    documents_directory.mkdir()

    create_fake_pdf(
        documents_directory / "document_c.pdf"
    )

    create_fake_pdf(
        documents_directory / "document_a.pdf"
    )

    create_fake_pdf(
        documents_directory / "document_b.PDF"
    )

    # Non-PDF files must be ignored.
    (documents_directory / "notes.txt").write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    loader = PDFLoader()

    pages = loader.load_directory(
        documents_directory
    )

    assert len(pages) == 6

    sources = []

    for page in pages:

        if page.source not in sources:
            sources.append(page.source)

    assert sources == [
        "document_a.pdf",
        "document_b.PDF",
        "document_c.pdf",
    ]


def test_load_directory_returns_empty_list_without_pdfs(
    tmp_path,
):
    """Verify that a directory without PDFs returns an empty list."""

    documents_directory = tmp_path / "empty_documents"

    documents_directory.mkdir()

    loader = PDFLoader()

    pages = loader.load_directory(
        documents_directory
    )

    assert pages == []


def test_load_directory_rejects_missing_directory(tmp_path):

    loader = PDFLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_directory(
            tmp_path / "missing_directory"
        )


def test_load_directory_rejects_file_path(tmp_path):

    file_path = tmp_path / "document.pdf"

    create_fake_pdf(file_path)

    loader = PDFLoader()

    with pytest.raises(
        ValueError,
        match="Not a directory",
    ):
        loader.load_directory(file_path)