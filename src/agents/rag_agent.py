import json
import re
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
class ReasoningPlan:
    """Explainable decision made by the agent before retrieval."""

    intent: str
    retrieval_top_k: int
    rationale: str


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Complete response produced by the RAG agent."""

    question: str
    answer: str
    sources: tuple[SourceReference, ...]
    retrieved_chunks: tuple[RetrievedChunk, ...]
    reasoning_plan: ReasoningPlan
    retrieval_attempts: int
    created_at: str


class RAGAgent:
    """
    Coordinates the complete Retrieval-Augmented Generation workflow.

    Workflow:
        Observation -> Reasoning -> Action -> Persistence
    """

    MAX_AUTOMATIC_TOP_K = 20

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

    def _reason(
        self,
        question: str,
        requested_top_k: int,
    ) -> ReasoningPlan:
        """
        Analyze the question and create a retrieval plan.

        Comparative and multi-aspect questions receive
        a broader retrieval scope than simple factual questions.
        """

        normalized = " ".join(question.lower().split())

        comparison_pattern = (
            r"\b(compare|comparison|contrast|difference|"
            r"differences|versus|vs)\b"
        )

        multi_aspect_pattern = (
            r"\b(list|overview|summarize|summarise|"
            r"steps|multiple|all)\b"
            r"|causes and effects"
            r"|risks and impacts"
            r"|benefits and challenges"
        )

        if re.search(comparison_pattern, normalized):
            intent = "comparative"

            planned_top_k = min(
                requested_top_k * 2,
                max(requested_top_k, self.MAX_AUTOMATIC_TOP_K),
            )

            rationale = (
                "The question requests a comparison. "
                "A broader retrieval scope is selected "
                "to collect evidence for the compared topics."
            )

        elif re.search(multi_aspect_pattern, normalized):
            intent = "multi_aspect"

            planned_top_k = min(
                requested_top_k * 2,
                max(requested_top_k, self.MAX_AUTOMATIC_TOP_K),
            )

            rationale = (
                "The question requests multiple aspects "
                "or an overview. A broader retrieval scope "
                "is selected to collect more supporting evidence."
            )

        else:
            intent = "factual"
            planned_top_k = requested_top_k

            rationale = (
                "The question appears to request a focused fact. "
                "The requested retrieval depth is retained."
            )

        return ReasoningPlan(
            intent=intent,
            retrieval_top_k=planned_top_k,
            rationale=rationale,
        )

    def _retrieve_and_validate(
        self,
        question: str,
        top_k: int,
    ) -> tuple[RetrievedChunk, ...]:
        """Retrieve document chunks and validate the results."""

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

        return tuple(retrieved_chunks)

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> AgentResponse:
        """
        Process a user question using an explainable agentic workflow.
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
        reasoning_plan = self._reason(
            question,
            requested_top_k=top_k,
        )

        # Stage 3: Action
        retrieved_chunks = self._retrieve_and_validate(
            question,
            reasoning_plan.retrieval_top_k,
        )

        retrieval_attempts = 1

        # Retry with broader retrieval if no evidence is found.
        retry_top_k = min(
            reasoning_plan.retrieval_top_k * 2,
            max(
                reasoning_plan.retrieval_top_k,
                self.MAX_AUTOMATIC_TOP_K,
            ),
        )

        if not retrieved_chunks and retry_top_k > reasoning_plan.retrieval_top_k:

            retrieved_chunks = self._retrieve_and_validate(
                question,
                retry_top_k,
            )

            retrieval_attempts += 1

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
            retrieved_chunks=retrieved_chunks,
            reasoning_plan=reasoning_plan,
            retrieval_attempts=retrieval_attempts,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Stage 4: Persistence
        if self.history_path is not None:
            self._persist(response)

        return response

    def _persist(self, response: AgentResponse) -> None:
        """Append an agent response and reasoning trace to JSONL."""

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
            "reasoning": {
                "intent": response.reasoning_plan.intent,
                "retrieval_top_k": (
                    response.reasoning_plan.retrieval_top_k
                ),
                "rationale": response.reasoning_plan.rationale,
                "retrieval_attempts": response.retrieval_attempts,
            },
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