
from src.generation.answer_generator import GeneratedAnswer
from src.generation.evidence_validator import (
    EvidenceValidator,
    EvidenceValidatingGenerator,
)
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.storage.vector_store import VectorStoreRecord


def make_chunk(text, source="Examination Regulations.pdf"):
    record = VectorStoreRecord(
        chunk_id="chunk_001",
        text=text,
        source=source,
        chunk_index=1,
        token_count=len(text.split()),
        page_numbers=(2,),
        heading="Examination Regulations",
        distance=0.1,
    )

    return RetrievedChunk(
        record=record,
        dense_score=0.9,
        tfidf_score=0.5,
        hybrid_score=0.03,
    )


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate_answer(self, question, retrieved_chunks):
        self.calls.append(question)

        return GeneratedAnswer(
            answer="The examination waiting period is one hour [S1].",
            sources=(),
        )


def test_relevant_evidence_passes_validation():
    validator = EvidenceValidator()

    chunk = make_chunk(
        "Candidates must wait at least one hour before "
        "leaving the examination hall."
    )

    result = validator.validate(
        "How long must a candidate wait before leaving "
        "the examination hall?",
        retrieved_chunks=[chunk],
    )

    assert result.sufficient is True
    assert result.coverage >= 0.35
    assert len(result.matched_terms) >= 2


def test_unrelated_library_fine_evidence_is_rejected():
    validator = EvidenceValidator()

    chunk = make_chunk(
        "Examination regulations describe penalties "
        "for examination misconduct."
    )

    result = validator.validate(
        "What is the exact amount of the NSBM library "
        "late return fine for 2026?",
        retrieved_chunks=[chunk],
    )

    assert result.sufficient is False
    assert result.coverage < 0.35
    assert "2026" in result.reason


def test_missing_requested_year_is_rejected():
    validator = EvidenceValidator()

    chunk = make_chunk(
        "The examination resit deadline is fourteen days."
    )

    result = validator.validate(
        "What is the examination resit deadline for 2026?",
        retrieved_chunks=[chunk],
    )

    assert result.sufficient is False
    assert "2026" in result.reason


def test_empty_evidence_is_rejected():
    validator = EvidenceValidator()

    result = validator.validate(
        "What are the examination regulations?",
        retrieved_chunks=[],
    )

    assert result.sufficient is False
    assert result.coverage == 0.0


def test_wrapper_calls_generator_for_relevant_evidence():
    generator = FakeGenerator()
    wrapped = EvidenceValidatingGenerator(generator)

    chunk = make_chunk(
        "Candidates must wait at least one hour before "
        "leaving the examination hall."
    )

    result = wrapped.generate_answer(
        "How long must a candidate wait before leaving "
        "the examination hall?",
        [chunk],
    )

    assert len(generator.calls) == 1
    assert "one hour" in result.answer


def test_wrapper_does_not_call_generator_for_unsupported_question():
    generator = FakeGenerator()
    wrapped = EvidenceValidatingGenerator(generator)

    chunk = make_chunk(
        "Examination regulations describe penalties "
        "for examination misconduct."
    )

    result = wrapped.generate_answer(
        "What is the exact amount of the NSBM library "
        "late return fine for 2026?",
        [chunk],
    )

    assert generator.calls == []
    assert result.answer == wrapped._abstain().answer
    assert result.sources == ()


def test_wrapper_abstains_when_no_evidence_exists():
    generator = FakeGenerator()
    wrapped = EvidenceValidatingGenerator(generator)

    result = wrapped.generate_answer(
        "What are the examination regulations?",
        [],
    )

    assert generator.calls == []
    assert result.sources == ()


def test_validator_rejects_invalid_threshold():
    import pytest

    with pytest.raises(ValueError):
        EvidenceValidator(min_coverage=0)


def test_validator_rejects_empty_question():
    import pytest

    validator = EvidenceValidator()

    with pytest.raises(ValueError):
        validator.validate("   ")
