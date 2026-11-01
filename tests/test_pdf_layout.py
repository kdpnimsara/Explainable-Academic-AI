import pytest

from src.ingestion.pdf_loader import PDFLoader


class FakeLayoutPage:
    """Simulate a PDF page with layout extraction."""

    def __init__(self, text):
        self.text = text
        self.calls = []

    def extract_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.text


class FakeLayoutPDFReader:
    """Simulate a PDF document containing multiple pages."""

    def __init__(self, file_path):
        self.is_encrypted = False

        self.pages = [
            FakeLayoutPage("First page content"),
            FakeLayoutPage("   "),
            FakeLayoutPage("Third page content"),
        ]


@pytest.fixture
def pdf_file(tmp_path):
    file = tmp_path / "layout_sample.pdf"
    file.write_bytes(b"Fake PDF content")
    return file


@pytest.mark.parametrize(
    ("mode", "expected_kwargs"),
    [
        ("plain", {}),
        ("layout", {"extraction_mode": "layout"}),
    ],
)
def test_extraction_mode(
    pdf_file,
    monkeypatch,
    mode,
    expected_kwargs,
):
    fake_reader = FakeLayoutPDFReader

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        fake_reader,
    )

    loader = PDFLoader(extraction_mode=mode)

    pages = loader.load(pdf_file)

    assert len(pages) == 2

    assert pages[0].page_number == 1
    assert pages[1].page_number == 3

    assert pages[0].text == "First page content"

    assert pages[0].source == "layout_sample.pdf"


def test_layout_mode_reaches_page_extractor(
    pdf_file,
    monkeypatch,
):
    fake_reader = FakeLayoutPDFReader(
        str(pdf_file)
    )

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        lambda path: fake_reader,
    )

    loader = PDFLoader(extraction_mode="layout")

    loader.load(pdf_file)

    assert fake_reader.pages[0].calls == [
        {"extraction_mode": "layout"}
    ]


def test_plain_mode_uses_default_extraction(
    pdf_file,
    monkeypatch,
):
    fake_reader = FakeLayoutPDFReader(
        str(pdf_file)
    )

    monkeypatch.setattr(
        "src.ingestion.pdf_loader.PdfReader",
        lambda path: fake_reader,
    )

    loader = PDFLoader(extraction_mode="plain")

    loader.load(pdf_file)

    assert fake_reader.pages[0].calls == [{}]


def test_invalid_extraction_mode_is_rejected():

    with pytest.raises(
        ValueError,
        match="Unsupported extraction mode",
    ):
        PDFLoader(extraction_mode="invalid")