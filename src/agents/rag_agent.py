
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from src.generation.answer_generator import (
    GeneratedAnswer,
    SourceReference,
)
from src.retrieval.hybrid_retriever import RetrievedChunk


class RetrieverProtocol(Protocol):
    """Interface expected from a hybrid retriever."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        ...


class GeneratorProtocol(Protocol):
    """Interface expected from an answer generator."""

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> GeneratedAnswer:
        ...


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Complete response produced by the RAG agent."""

    question: str
    answer: str
    sources: tuple[SourceReference, ...]
    retrieved_chunks: tuple[RetrievedChunk, ...]
    created_at: str


class RAGAgent:
    """
    Coordinates the complete Retrieval-Augmented Generation workflow.

    Workflow:
        Observation -> Reasoning -> Action -> Persistence
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        generator: GeneratorProtocol,
        history_path: str | Path | None = None,
    ) -> None:

        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError(
                "retriever must provide retrieve()."
            )

        if not callable(
            getattr(generator, "generate_answer", None)
        ):
            raise TypeError(
                "generator must provide generate_answer()."
            )

        if history_path is not None:
            if not str(history_path).strip():
                raise ValueError(
                    "history_path cannot be empty."
                )

            self.history_path = Path(history_path)
        else:
            self.history_path = None

        self.retriever = retriever
        self.generator = generator

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> AgentResponse:
        """
        Process a user question and return an evidence-based answer.
        """

        # Stage 1: Observation
        if not isinstance(question, str):
            raise TypeError(
                "question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # Stage 2: Reasoning
        # Determine that document retrieval is required
        # before generating an answer.

        # Stage 3: Action
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
        )

        if (
            not isinstance(retrieved_chunks, Sequence)
            or isinstance(retrieved_chunks, (str, bytes))
        ):
            raise TypeError(
                "Retriever must return a sequence of chunks."
            )

        for chunk in retrieved_chunks:
            if not isinstance(chunk, RetrievedChunk):
                raise TypeError(
                    "Retriever returned an invalid chunk."
                )

        generated = self.generator.generate_answer(
            question,
            retrieved_chunks,
        )

        if not isinstance(generated, GeneratedAnswer):
            raise TypeError(
                "Generator must return a GeneratedAnswer."
            )

        response = AgentResponse(
            question=question,
            answer=generated.answer,
            sources=generated.sources,
            retrieved_chunks=tuple(retrieved_chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Stage 4: Persistence
        if self.history_path is not None:
            self._persist(response)

        return response

    def _persist(self, response: AgentResponse) -> None:
        """Append an agent response to a JSONL history file."""

        if self.history_path is None:
            return

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "question": response.question,
            "answer": response.answer,
            "created_at": response.created_at,
            "sources": [
                {
                    "citation_id": source.citation_id,
                    "chunk_id": source.chunk_id,
                    "document": source.document,
                    "page_numbers": list(source.page_numbers),
                }
                for source in response.sources
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": chunk.record.chunk_id,
                    "hybrid_score": chunk.hybrid_score,
                }
                for chunk in response.retrieved_chunks
            ],
        }

        with self.history_path.open(
            "a",
            encoding="utf-8",
        ) as history_file:
            history_file.write(
                json.dumps(data, ensure_ascii=False) + "\n"
            )
