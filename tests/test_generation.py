
import pytest

from src.generation.answer_generator import AnswerGenerator
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.storage.vector_store import VectorStoreRecord


class FakeModel:
    def __init__(self, response="Test generated answer [S1]."):
        self.response = response
        self.last_prompt = None
        self.called = False

    def generate(self, prompt):
        self.called = True
        self.last_prompt = prompt
        return self.response


def make_retrieved_chunk():
    record = VectorStoreRecord(
        chunk_id="chunk_001",
        text="Disaster management involves preparedness and response.",
        source="disaster_management_act_2005.pdf",
        chunk_index=1,
        token_count=8,
        page_numbers=(4, 5),
        heading="Disaster Management",
        distance=0.15,
    )

    return RetrievedChunk(
        record=record,
        dense_score=0.85,
        tfidf_score=0.70,
        hybrid_score=0.03,
    )


def test_generate_answer_with_evidence():
    model = FakeModel()
    generator = AnswerGenerator(model)

    result = generator.generate_answer(
        "What is disaster management?",
        [make_retrieved_chunk()],
    )

    assert result.answer == "Test generated answer [S1]."
    assert len(result.sources) == 1
    assert result.sources[0].document == (
        "disaster_management_act_2005.pdf"
    )
    assert result.sources[0].page_numbers == (4, 5)
    assert model.called


def test_prompt_contains_question_and_evidence():
    model = FakeModel()
    generator = AnswerGenerator(model)

    generator.generate_answer(
        "What is disaster management?",
        [make_retrieved_chunk()],
    )

    assert "What is disaster management?" in model.last_prompt
    assert "Disaster management involves preparedness" in (
        model.last_prompt
    )
    assert "[S1]" in model.last_prompt


def test_no_evidence_returns_abstention():
    model = FakeModel()
    generator = AnswerGenerator(model)

    result = generator.generate_answer("Unknown question", [])

    assert result.answer == generator.ABSTENTION_MESSAGE
    assert result.sources == ()
    assert not model.called


def test_empty_question_is_rejected():
    generator = AnswerGenerator(FakeModel())

    with pytest.raises(ValueError):
        generator.generate_answer("   ", [])


def test_invalid_retrieved_chunk_is_rejected():
    generator = AnswerGenerator(FakeModel())

    with pytest.raises(TypeError):
        generator.generate_answer("Test question", ["invalid"])


def test_invalid_model_output_is_rejected():
    generator = AnswerGenerator(FakeModel(response=123))

    with pytest.raises(TypeError):
        generator.generate_answer(
            "Test question",
            [make_retrieved_chunk()],
        )


def test_empty_model_response_returns_abstention():
    generator = AnswerGenerator(FakeModel(response="  "))

    result = generator.generate_answer(
        "Test question",
        [make_retrieved_chunk()],
    )

    assert result.answer == generator.ABSTENTION_MESSAGE
