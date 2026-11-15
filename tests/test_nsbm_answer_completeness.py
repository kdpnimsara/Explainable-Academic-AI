
from src.generation.answer_generator import AnswerGenerator, SourceReference
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.storage.vector_store import VectorStoreRecord


def make_chunk(text, chunk_id="chunk_nsbm"):
    record = VectorStoreRecord(
        chunk_id=chunk_id,
        text=text,
        source="NSBM_test_source.pdf",
        chunk_index=0,
        token_count=len(text.split()),
        page_numbers=(3, 4),
        heading="NSBM academic requirements",
        distance=0.1,
    )
    return RetrievedChunk(
        record=record,
        dense_score=0.9,
        tfidf_score=0.8,
        hybrid_score=0.03,
    )


def make_source(chunk_id="chunk_nsbm"):
    return SourceReference(
        citation_id="[S1]",
        chunk_id=chunk_id,
        document="NSBM_test_source.pdf",
        page_numbers=(3, 4),
    )


def test_first_class_guard_adds_gpa_when_retrieved_evidence_supports_it():
    question = "What are the requirements for First Class Honours?"
    answer = (
        "A student must earn C or better in 120 credits "
        "and finish within four academic years."
    )
    chunk = make_chunk(
        "First Class: grade C or better in 120 credits; minimum GPA 3.7; "
        "complete within four academic years."
    )

    result = AnswerGenerator._ensure_first_class_completeness(
        question, answer, [chunk], [make_source()]
    )

    assert "minimum GPA requirement is 3.7 [S1]" in result


def test_first_class_guard_does_not_invent_gpa_without_evidence():
    question = "What are the requirements for First Class Honours?"
    answer = (
        "A student must earn C or better in 120 credits "
        "and finish within four academic years."
    )
    chunk = make_chunk(
        "First Class: grade C or better in 120 credits; "
        "complete within four academic years."
    )

    result = AnswerGenerator._ensure_first_class_completeness(
        question, answer, [chunk], [make_source()]
    )

    assert "GPA" not in result
    assert "3.7" not in result


def test_option_2_guard_adds_three_al_passes_with_citation():
    question = (
        "What additional requirements apply under Option 2 for entry "
        "to the NSBM BSc (Honours) in Data Science?"
    )
    answer = (
        "A Credit pass in GCE O/L Mathematics and completion of the "
        "approved bridging programme are required."
    )
    chunk = make_chunk(
        "Option 2: At least three (3) simple passes (S) in any stream "
        "in one sitting at the GCE A/L Examination, and a Credit pass "
        "in Mathematics at the GCE O/L Examination. Completion of the "
        "bridging programme approved by the Specified Authority."
    )

    result = AnswerGenerator._ensure_option_2_completeness(
        question, answer, [chunk], [make_source()]
    )

    assert "at least three (3) simple passes" in result.casefold()
    assert "any stream" in result
    assert "one sitting" in result
    assert "GCE A/L Examination [S1]" in result


def test_option_2_guard_does_not_invent_al_requirement_without_evidence():
    question = (
        "What additional requirements apply under Option 2 for entry "
        "to the NSBM BSc (Honours) in Data Science?"
    )
    answer = "A Credit pass in GCE O/L Mathematics is required."
    chunk = make_chunk(
        "Option 2: A Credit pass in Mathematics at the GCE O/L Examination."
    )

    result = AnswerGenerator._ensure_option_2_completeness(
        question, answer, [chunk], [make_source()]
    )

    assert result == answer
