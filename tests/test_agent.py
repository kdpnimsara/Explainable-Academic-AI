
import json

import pytest

from src.agents.rag_agent import RAGAgent
from src.generation.answer_generator import (
    GeneratedAnswer,
    SourceReference,
)
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.storage.vector_store import VectorStoreRecord


def make_chunk():
    record = VectorStoreRecord(
        chunk_id="chunk_001",
        text="Disaster preparedness reduces disaster risks.",
        source="disaster_management_act_2005.pdf",
        chunk_index=1,
        token_count=7,
        page_numbers=(4, 5),
        heading="Disaster Preparedness",
        distance=0.15,
    )

    return RetrievedChunk(
        record=record,
        dense_score=0.85,
        tfidf_score=0.70,
        hybrid_score=0.03,
    )


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.chunks[:top_k]


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate_answer(self, question, retrieved_chunks):
        self.calls.append((question, list(retrieved_chunks)))

        if not retrieved_chunks:
            return GeneratedAnswer(
                answer=(
                    "I could not find sufficient supporting "
                    "evidence in the available documents "
                    "to answer this question."
                ),
                sources=(),
            )

        chunk = retrieved_chunks[0]

        source = SourceReference(
            citation_id="[S1]",
            chunk_id=chunk.record.chunk_id,
            document=chunk.record.source,
            page_numbers=chunk.record.page_numbers,
        )

        return GeneratedAnswer(
            answer="Disaster preparedness reduces risks [S1].",
            sources=(source,),
        )


def test_agent_coordinates_retrieval_and_generation():
    retriever = FakeRetriever([make_chunk()])
    generator = FakeGenerator()

    agent = RAGAgent(
        retriever=retriever,
        generator=generator,
    )

    result = agent.ask(
        "What is disaster preparedness?",
        top_k=3,
    )

    assert len(retriever.calls) == 1
    assert len(generator.calls) == 1
    assert result.question == "What is disaster preparedness?"
    assert "reduces risks" in result.answer
    assert len(result.sources) == 1
    assert len(result.retrieved_chunks) == 1


def test_agent_persists_answer_and_sources(tmp_path):
    history_path = tmp_path / "history" / "agent.jsonl"

    agent = RAGAgent(
        retriever=FakeRetriever([make_chunk()]),
        generator=FakeGenerator(),
        history_path=history_path,
    )

    result = agent.ask("What is disaster preparedness?")

    assert history_path.exists()

    saved = json.loads(
        history_path.read_text(encoding="utf-8").splitlines()[0]
    )

    assert saved["question"] == result.question
    assert saved["answer"] == result.answer
    assert saved["sources"][0]["citation_id"] == "[S1]"
    assert saved["sources"][0]["page_numbers"] == [4, 5]
    assert saved["retrieved_chunks"][0]["chunk_id"] == "chunk_001"


def test_agent_appends_multiple_history_records(tmp_path):
    history_path = tmp_path / "agent.jsonl"

    agent = RAGAgent(
        FakeRetriever([make_chunk()]),
        FakeGenerator(),
        history_path=history_path,
    )

    agent.ask("First question?")
    agent.ask("Second question?")

    lines = history_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 2


def test_agent_handles_missing_evidence():
    agent = RAGAgent(
        FakeRetriever([]),
        FakeGenerator(),
    )

    result = agent.ask("Question without evidence?")

    assert result.sources == ()
    assert result.retrieved_chunks == ()
    assert "could not find sufficient" in result.answer.lower()


def test_agent_rejects_empty_question():
    retriever = FakeRetriever()
    agent = RAGAgent(retriever, FakeGenerator())

    with pytest.raises(ValueError):
        agent.ask("   ")

    assert retriever.calls == []


def test_agent_rejects_invalid_top_k():
    agent = RAGAgent(
        FakeRetriever(),
        FakeGenerator(),
    )

    with pytest.raises(ValueError):
        agent.ask("Test question", top_k=0)


def test_agent_rejects_empty_history_path():
    with pytest.raises(ValueError):
        RAGAgent(
            FakeRetriever(),
            FakeGenerator(),
            history_path="   ",
        )
