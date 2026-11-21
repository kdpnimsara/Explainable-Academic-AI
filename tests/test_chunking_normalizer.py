import pytest

from src.ingestion.pdf_loader import PDFPage
from src.ingestion.chunking.normalizer import TextNormalizer


@pytest.fixture
def normalizer():
    return TextNormalizer()


def test_normalize_multiple_spaces(normalizer):
    text = "Hello     world"

    result = normalizer.normalize_text(text)

    assert result == "Hello world"


def test_preserve_paragraph_boundaries(normalizer):
    text = "First paragraph.\n\nSecond paragraph."

    result = normalizer.normalize_text(text)

    assert result == text


def test_normalize_line_endings(normalizer):
    text = "First line.\r\nSecond line.\rThird line."

    result = normalizer.normalize_text(text)

    assert result == (
        "First line.\nSecond line.\nThird line."
    )


def test_preserve_mathematical_unicode(normalizer):
    text = "x² + H₂O + Ａ"

    result = normalizer.normalize_text(text)

    assert result == text


def test_preserve_unicode_joining_characters(normalizer):
    text = "A\u200dB"

    result = normalizer.normalize_text(text)

    assert result == text


def test_remove_excessive_blank_lines(normalizer):
    text = "First paragraph.\n\n\n\nSecond paragraph."

    result = normalizer.normalize_text(text)

    assert result == "First paragraph.\n\nSecond paragraph."


def test_empty_text(normalizer):
    assert normalizer.normalize_text("") == ""
    assert normalizer.normalize_text("   ") == ""


def test_reject_invalid_text_type(normalizer):
    with pytest.raises(TypeError):
        normalizer.normalize_text(123)


def test_normalize_page_preserves_metadata(normalizer):
    page = PDFPage(
        source="sample.pdf",
        page_number=3,
        text="  Sample    PDF text  ",
    )

    result = normalizer.normalize_page(page)

    assert result.source == "sample.pdf"
    assert result.page_number == 3
    assert result.text == "Sample PDF text"


def test_normalize_pages_ignores_empty_pages(normalizer):
    pages = [
        PDFPage("sample.pdf", 1, "   "),
        PDFPage("sample.pdf", 2, "Useful content"),
    ]

    result = normalizer.normalize_pages(pages)

    assert len(result) == 1
    assert result[0].page_number == 2
    assert result[0].text == "Useful content"


def test_reject_invalid_page_object(normalizer):
    with pytest.raises(TypeError):
        normalizer.normalize_page("invalid")


def test_reject_invalid_pages_collection(normalizer):
    with pytest.raises(TypeError):
        normalizer.normalize_pages("invalid")

def test_layout_normalization_preserves_column_spacing(normalizer):
    text = (
        "Property                                           Rule\n"
        " Maximum keys per node                              3"
    )

    result = normalizer.normalize_text(
        text,
        preserve_layout=True,
    )

    assert "Property                                           Rule" in result

    assert " Maximum keys per node                              3" in result


def test_layout_page_preserves_extraction_mode(normalizer):
    page = PDFPage(
        source="sample.pdf",
        page_number=1,
        text="Heading                    Content",
        extraction_mode="layout",
    )

    result = normalizer.normalize_page(page)

    assert result.extraction_mode == "layout"
    assert "Heading                    Content" in result.text


def test_plain_page_still_collapses_spaces(normalizer):
    page = PDFPage(
        source="sample.pdf",
        page_number=1,
        text="Hello       World",
        extraction_mode="plain",
    )

    result = normalizer.normalize_page(page)

    assert result.text == "Hello World"