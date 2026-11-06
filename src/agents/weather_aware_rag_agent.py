
"""Weather-aware extension of the existing academic RAG agent.

The original RAGAgent remains unchanged. This subclass adds
weather-aware tool selection, external evidence integration,
and evidence relevance validation.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from src.agents.rag_agent import (
    AgentResponse,
    RAGAgent,
    ReasoningPlan,
)
from src.agents.tool_selector import ExplainableToolSelector
from src.generation.answer_generator import (
    ExternalSourceReference,
    GeneratedAnswer,
)
from src.generation.evidence_validator import (
    EvidenceValidator,
    EvidenceValidatingGenerator,
)
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.weather.evidence import WeatherEvidence
from src.weather.location_resolver import LocationResolver
from src.weather.service import (
    WeatherInformationResult,
    WeatherInformationService,
)


@dataclass(frozen=True, slots=True)
class WeatherAwareAgentResponse(AgentResponse):
    """Extended response containing weather execution details."""

    external_sources: tuple[ExternalSourceReference, ...] = ()
    selected_tools: tuple[str, ...] = (
        "academic_retrieval",
    )
    tool_errors: tuple[str, ...] = ()
    location_name: str | None = None


class WeatherAwareRAGAgent(RAGAgent):
    """Extend the original RAG agent with weather and evidence validation."""

    def __init__(
        self,
        retriever,
        generator,
        history_path=None,
        weather_service=None,
        tool_selector=None,
        location_resolver=None,
        evidence_validator=None,
    ):
        # Avoid wrapping an already validated generator twice.
        if isinstance(generator, EvidenceValidatingGenerator):
            validated_generator = generator
        else:
            validator = (
                evidence_validator
                if evidence_validator is not None
                else EvidenceValidator()
            )

            if not isinstance(validator, EvidenceValidator):
                raise TypeError(
                    "evidence_validator must be an EvidenceValidator."
                )

            validated_generator = EvidenceValidatingGenerator(
                generator=generator,
                validator=validator,
            )

        # Keep the original RAGAgent implementation unchanged.
        # Both academic and weather workflows use the validating adapter.
        super().__init__(
            retriever=retriever,
            generator=validated_generator,
            history_path=history_path,
        )

        self.evidence_validator = validated_generator.validator

        self.weather_service = (
            weather_service
            if weather_service is not None
            else WeatherInformationService()
        )

        self.tool_selector = (
            tool_selector
            if tool_selector is not None
            else ExplainableToolSelector()
        )

        self.location_resolver = (
            location_resolver
            if location_resolver is not None
            else LocationResolver()
        )

        if not callable(
            getattr(self.weather_service, "get_information", None)
        ):
            raise TypeError(
                "weather_service must provide get_information()."
            )

        if not callable(
            getattr(self.tool_selector, "select", None)
        ):
            raise TypeError(
                "tool_selector must provide select()."
            )

        if not callable(
            getattr(self.location_resolver, "resolve", None)
        ):
            raise TypeError(
                "location_resolver must provide resolve()."
            )

    def ask(
        self,
        question: str,
        top_k: int = 5,
        institution_id: str | None = None,
        location_name: str | None = None,
    ) -> AgentResponse | WeatherAwareAgentResponse:
        """Answer academic, weather and combined questions."""

        # Select tools based on the user's question.
        selection = self.tool_selector.select(
            question,
            location_name=location_name,
        )

        requires_weather = (
            "weather_information" in selection.selected_tools
        )

        # Academic-only questions use the original workflow.
        # The generator is already protected by the validation adapter.
        if not requires_weather:
            return super().ask(
                question=question,
                top_k=top_k,
                institution_id=institution_id,
            )

        # Validate inputs for the weather-aware workflow.
        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        institution_id = self._validate_institution_id(
            institution_id
        )

        requires_academic = (
            "academic_retrieval" in selection.selected_tools
        )

        # Resolve the requested weather location.
        requested_location = selection.location_name

        location = self.location_resolver.resolve(
            requested_location
        )

        if location is None and requested_location is None:
            location = self.location_resolver.resolve_from_question(
                question
            )

        # Prepare explainable reasoning.
        if requires_academic:
            base_plan = self._reason(
                question,
                requested_top_k=top_k,
            )

            retrieval_top_k = base_plan.retrieval_top_k

            rationale = (
                selection.rationale + " " + base_plan.rationale
            )
        else:
            retrieval_top_k = 0
            rationale = selection.rationale

        reasoning_plan = ReasoningPlan(
            intent=selection.intent,
            retrieval_top_k=retrieval_top_k,
            rationale=rationale,
        )

        # Retrieve academic evidence only when required.
        retrieved_chunks: tuple[RetrievedChunk, ...] = ()
        retrieval_attempts = 0

        if requires_academic:
            retrieved_chunks = self._retrieve_and_validate(
                question,
                retrieval_top_k,
                institution_id=institution_id,
            )

            retrieval_attempts = 1

            retry_top_k = min(
                retrieval_top_k * 2,
                max(
                    retrieval_top_k,
                    self.MAX_AUTOMATIC_TOP_K,
                ),
            )

            if (
                not retrieved_chunks
                and retry_top_k > retrieval_top_k
            ):
                retrieved_chunks = self._retrieve_and_validate(
                    question,
                    retry_top_k,
                    institution_id=institution_id,
                )

                retrieval_attempts += 1

        # Retrieve weather evidence.
        weather_evidence: tuple[WeatherEvidence, ...] = ()
        tool_errors: tuple[str, ...] = ()

        if location is None:
            tool_errors = (
                "A supported Sri Lankan location could not "
                "be identified. Please specify a location "
                "such as Colombo, Kandy or Galle.",
            )
        else:
            try:
                weather_result = self.weather_service.get_information(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    location_name=location.name,
                    location_district=location.district,
                )

                if not isinstance(
                    weather_result,
                    WeatherInformationResult,
                ):
                    raise TypeError(
                        "Weather service returned an invalid result."
                    )

                weather_evidence = weather_result.evidence

                tool_errors = tuple(
                    f"{error.provider}: {error.message}"
                    for error in weather_result.errors
                )

            except Exception as exc:
                # External weather failures must not crash
                # the academic decision-support workflow.
                tool_errors = (
                    "Weather retrieval failed: "
                    f"{type(exc).__name__}: {exc}",
                )

        # Generate an answer through the evidence-validation adapter.
        # The adapter checks academic and current weather evidence
        # before calling the underlying AnswerGenerator.
        generate_with_weather = getattr(
            self.generator,
            "generate_answer_with_weather",
            None,
        )

        if not callable(generate_with_weather):
            raise TypeError(
                "Weather-aware generation requires "
                "generate_answer_with_weather()."
            )

        generated = generate_with_weather(
            question,
            retrieved_chunks,
            weather_evidence,
        )

        if not isinstance(generated, GeneratedAnswer):
            raise TypeError(
                "Generator must return a GeneratedAnswer."
            )

        answer_text = generated.answer

        # Clearly communicate missing weather information.
        if not weather_evidence:
            if location is None:
                notice = tool_errors[0]
            else:
                notice = (
                    "Current weather information could not be "
                    "verified. This answer therefore does not "
                    "confirm current weather conditions or their "
                    "impact on university activities."
                )

            if requires_academic and retrieved_chunks:
                answer_text += "\n\n" + notice
            else:
                answer_text = notice

        elif tool_errors:
            answer_text += (
                "\n\nSome weather information providers were "
                "unavailable. Only the available cited evidence "
                "has been used."
            )

        # Build the explainable response.
        response = WeatherAwareAgentResponse(
            question=question,
            answer=answer_text,
            sources=generated.sources,
            retrieved_chunks=retrieved_chunks,
            reasoning_plan=reasoning_plan,
            retrieval_attempts=retrieval_attempts,
            created_at=datetime.now(timezone.utc).isoformat(),
            institution_id=institution_id,
            external_sources=generated.external_sources,
            selected_tools=selection.selected_tools,
            tool_errors=tool_errors,
            location_name=location.name if location else None,
        )

        # Persist the explainable execution trace.
        if self.history_path is not None:
            self._persist(response)

        return response

    def _persist(self, response: AgentResponse) -> None:
        """Persist the original history fields and weather trace."""

        if self.history_path is None:
            return

        # Preserve the original parent persistence for responses
        # produced through the academic-only workflow.
        if not isinstance(response, WeatherAwareAgentResponse):
            super()._persist(response)
            return

        self.history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "question": response.question,
            "answer": response.answer,
            "created_at": response.created_at,
            "institution_id": response.institution_id,
            "location_name": response.location_name,
            "selected_tools": list(response.selected_tools),
            "tool_errors": list(response.tool_errors),
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
            "external_sources": [
                {
                    "citation_id": source.citation_id,
                    "evidence_type": source.evidence_type,
                    "source_name": source.source_name,
                    "source_url": source.source_url,
                    "data_timestamp": source.data_timestamp,
                    "valid_until": source.valid_until,
                }
                for source in response.external_sources
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
