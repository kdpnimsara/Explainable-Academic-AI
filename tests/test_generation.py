
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


def _retrieved_chunk(chunk_id, text, source, pages, chunk_index=1):
    record = VectorStoreRecord(
        chunk_id=chunk_id,
        text=text,
        source=source,
        chunk_index=chunk_index,
        token_count=max(1, len(text.split())),
        page_numbers=pages,
        heading=None,
        distance=0.1,
    )
    return RetrievedChunk(
        record=record,
        dense_score=0.9,
        tfidf_score=0.8,
        hybrid_score=0.7,
    )


def test_same_pdf_chunks_share_one_reference_and_merge_pages():
    model = FakeModel("The rule applies [S1].")
    generator = AnswerGenerator(model)
    chunks = [
        _retrieved_chunk("c1", "First unique evidence.", "docs/guide.pdf", (2,)),
        _retrieved_chunk("c2", "Second unique evidence.", "docs/guide.pdf", (4,)),
    ]

    result = generator.generate_answer("What is the rule?", chunks)

    assert len(result.sources) == 1
    assert result.sources[0].citation_id == "[S1]"
    assert result.sources[0].page_numbers == (2, 4)
    assert result.sources[0].chunk_ids == ("c1", "c2")
    assert model.last_prompt.count("Evidence Type: Academic Document") == 1
    assert "First unique evidence." in model.last_prompt
    assert "Second unique evidence." in model.last_prompt


def test_repeated_passage_in_same_pdf_is_included_once():
    model = FakeModel("The rule applies [S1].")
    generator = AnswerGenerator(model)
    chunks = [
        _retrieved_chunk("c1", "Same evidence passage.", "docs/guide.pdf", (2,)),
        _retrieved_chunk("c2", "  Same   evidence passage.  ", "docs/guide.pdf", (3,)),
    ]

    result = generator.generate_answer("What is the rule?", chunks)

    assert len(result.sources) == 1
    assert result.sources[0].page_numbers == (2, 3)
    assert result.sources[0].chunk_ids == ("c1", "c2")
    assert model.last_prompt.count("Same evidence passage.") == 1


def test_same_filename_in_different_folders_keeps_separate_references():
    model = FakeModel("See [S1] and [S2].")
    generator = AnswerGenerator(model)
    chunks = [
        _retrieved_chunk("c1", "Evidence A.", "faculty-a/guide.pdf", (1,)),
        _retrieved_chunk("c2", "Evidence B.", "faculty-b/guide.pdf", (1,)),
    ]

    result = generator.generate_answer("Compare these rules?", chunks)

    assert len(result.sources) == 2
    assert [item.citation_id for item in result.sources] == ["[S1]", "[S2]"]


def test_unsupported_academic_citation_ids_are_removed():
    model = FakeModel("This is supported [S1]. This is not [S9].")
    generator = AnswerGenerator(model)

    result = generator.generate_answer(
        "What is disaster management?", [make_retrieved_chunk()]
    )

    assert "[S1]" in result.answer
    assert "[S9]" not in result.answer
