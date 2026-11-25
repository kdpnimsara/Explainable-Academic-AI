from types import SimpleNamespace

from src.retrieval.context_matching import (
    chunk_matches_expected_context,
    count_relevant_chunks,
    retrieved_context_matches,
)


def item(source, pages, text):
    return SimpleNamespace(record=SimpleNamespace(
        source=source, page_numbers=pages, text=text, heading=None
    ))


def test_exact_source_and_page_match():
    expected = {"document": "C:\\data\\rule.pdf", "pages": [2], "context": "candidate admission"}
    record = SimpleNamespace(source="rule.pdf", page_numbers=(2,), text="Different text", heading=None)
    assert chunk_matches_expected_context(record, expected)


def test_equivalent_pdf_and_markdown_context_match_by_text():
    expected = {
        "document": "SLIIT_General_Timeline.md",
        "pages": [1],
        "context": "Semester registration four weeks before semester commencement. January June semester December January July December semester May June.",
    }
    retrieved = [item("General-timelines-for-Semester-Module-registration-and-examination.pdf", (1,),
        "Semester registration must be done four weeks before semester commencement. January-June: December or January. July-December: May or June.")]
    assert retrieved_context_matches(retrieved, [expected]) == (1, 1)
    assert count_relevant_chunks(retrieved, [expected]) == 1


def test_unrelated_text_does_not_match_context():
    expected = {"document": "expected.pdf", "pages": [3], "context": "Candidates must submit the completed application form with required supporting documents before the deadline."}
    unrelated = item("other.pdf", (8,), "The library provides books and computer facilities for university students.")
    assert not chunk_matches_expected_context(unrelated.record, expected)
    assert retrieved_context_matches([unrelated], [expected]) == (0, 1)
