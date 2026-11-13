
"""Integration tests for the weather-aware RAG agent."""

import json
from datetime import datetime, timezone

import pytest

from src.agents.rag_agent import AgentResponse
from src.agents.weather_aware_rag_agent import (
    WeatherAwareRAGAgent,
    WeatherAwareAgentResponse,
)
from src.generation.answer_generator import AnswerGenerator
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.storage.vector_store import VectorStoreRecord
from src.weather.evidence import forecast_to_evidence
from src.weather.models import WeatherForecast
from src.weather.service import WeatherInformationResult

from tests.test_agent import FakeRetriever
from tests.test_generation import FakeModel


NOW = datetime.now(timezone.utc)


def make_chunk():
    """Create examination-related evidence for weather agent tests."""

    text = (
        "NSBM university examination regulations describe "
        "university examination procedures. Candidates must follow "
        "examination hall attendance rules and student conduct "
        "requirements. The document also discusses heavy rain, "
        "rain, and how rain may affect university examinations "
        "and academic activities in Colombo."
    )

    record = VectorStoreRecord(
        chunk_id="nsbm_examination_001",
        text=text,
        source="NSBM_Examination_Regulations.pdf",
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


def make_weather_evidence():
    """Create fresh weather evidence for integration tests."""

    forecast = WeatherForecast(
        latitude=6.9271,
        longitude=79.8612,
        timezone_name="Asia/Colombo",
        source_url="https://api.open-meteo.com/v1/forecast",
        retrieved_at=NOW,
        current_time=NOW,
        temperature_c=24.6,
        relative_humidity_percent=97.0,
        apparent_temperature_c=29.9,
        precipitation_mm=0.0,
        weather_code=3,
        wind_speed_kmh=6.1,
    )

    evidence = forecast_to_evidence(
        forecast,
        citation_id="[W1]",
        location_name="Colombo",
        now=NOW,
    )

    assert evidence is not None
    return evidence


def make_weather_result():
    """Create a successful weather service result."""

    return WeatherInformationResult(
        location="Colombo",
        evidence=(make_weather_evidence(),),
        errors=(),
    )


class FakeWeatherService:
    """Fake weather service without external API calls."""

    def __init__(self, result=None, exception=None):
        self.result = result or make_weather_result()
        self.exception = exception
        self.calls = []

    def get_information(
        self,
        latitude,
        longitude,
        location_name=None,
        now=None,
        location_district=None,
    ):
        self.calls.append({
            "latitude": latitude,
            "longitude": longitude,
            "location_name": location_name,
            "location_district": location_district,
        })

        if self.exception:
            raise self.exception

        return self.result


def make_agent(
    chunks=None,
    weather_service=None,
    history_path=None,
):
    """Create a weather-aware agent using test dependencies."""

    retriever = FakeRetriever(
        chunks if chunks is not None else [make_chunk()]
    )

    model = FakeModel(
        "The weather information is available [W1]. "
        "Academic evidence supports the response [S1]."
    )

    generator = AnswerGenerator(model)

    weather = weather_service or FakeWeatherService()

    agent = WeatherAwareRAGAgent(
        retriever=retriever,
        generator=generator,
        weather_service=weather,
        history_path=history_path,
    )

    return agent, retriever, generator, weather


def test_weather_only_question_uses_weather_tool():
    agent, retriever, generator, weather = make_agent()

    result = agent.ask(
        "What is the weather in Colombo?"
    )

    assert isinstance(result, WeatherAwareAgentResponse)

    assert result.selected_tools == (
        "weather_information",
    )

    assert result.location_name == "Colombo"
    assert result.retrieved_chunks == ()

    assert retriever.calls == []

    assert len(weather.calls) == 1
    assert weather.calls[0]["latitude"] == 6.9271
    assert weather.calls[0]["longitude"] == 79.8612

    assert len(result.external_sources) == 1
    assert result.external_sources[0].citation_id == "[W1]"

    assert "[W1]" in generator.model.last_prompt


def test_combined_question_uses_both_tools():
    agent, retriever, generator, weather = make_agent(
        chunks=[make_chunk()]
    )

    result = agent.ask(
        "Will heavy rain affect university examinations in Colombo?",
        institution_id="NSBM",
    )

    assert isinstance(result, WeatherAwareAgentResponse)

    assert result.selected_tools == (
        "weather_information",
        "academic_retrieval",
    )

    assert len(retriever.calls) == 1
    assert retriever.institution_ids == ["NSBM"]

    assert result.institution_id == "NSBM"
    assert len(result.retrieved_chunks) == 1

    assert len(weather.calls) == 1
    assert weather.calls[0]["location_name"] == "Colombo"
    assert weather.calls[0]["location_district"] == "Colombo"

    assert len(result.sources) == 1
    assert len(result.external_sources) == 1

    assert "[S1]" in generator.model.last_prompt
    assert "[W1]" in generator.model.last_prompt


def test_academic_question_preserves_original_agent():
    agent, retriever, generator, weather = make_agent()

    result = agent.ask(
        "What are the examination regulations?"
    )

    assert isinstance(result, AgentResponse)

    # The original RAGAgent workflow is preserved.
    assert not isinstance(result, WeatherAwareAgentResponse)

    assert len(retriever.calls) == 1
    assert weather.calls == []

    assert len(result.sources) == 1
    assert generator.model.called


def test_unknown_weather_location_is_not_guessed():
    agent, retriever, generator, weather = make_agent()

    result = agent.ask(
        "What is the current weather?"
    )

    assert isinstance(result, WeatherAwareAgentResponse)

    assert weather.calls == []
    assert retriever.calls == []

    assert result.external_sources == ()

    assert "location could not be identified" in (
        result.answer.lower()
    )

    assert len(result.tool_errors) == 1


def test_weather_service_failure_does_not_crash_agent():
    service = FakeWeatherService(
        exception=RuntimeError("Simulated provider failure")
    )

    agent, retriever, generator, weather = make_agent(
        weather_service=service
    )

    result = agent.ask(
        "What is the weather in Colombo?"
    )

    assert isinstance(result, WeatherAwareAgentResponse)

    assert result.external_sources == ()
    assert len(result.tool_errors) == 1

    assert "could not be verified" in result.answer.lower()

    # Weather failure does not trigger academic retrieval.
    assert retriever.calls == []


def test_agent_persists_weather_execution_trace(tmp_path):
    history_path = tmp_path / "history" / "weather_agent.jsonl"

    agent, retriever, generator, weather = make_agent(
        history_path=history_path
    )

    result = agent.ask(
        "What is the weather in Colombo?"
    )

    assert history_path.exists()

    saved = json.loads(
        history_path.read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )

    assert saved["question"] == result.question

    assert saved["selected_tools"] == [
        "weather_information"
    ]

    assert saved["location_name"] == "Colombo"

    assert len(saved["external_sources"]) == 1

    assert (
        saved["external_sources"][0]["citation_id"]
        == "[W1]"
    )


def test_agent_records_academic_and_weather_sources(tmp_path):
    history_path = tmp_path / "combined_history.jsonl"

    agent, retriever, generator, weather = make_agent(
        history_path=history_path
    )

    result = agent.ask(
        "Will heavy rain affect university examinations in Colombo?"
    )

    saved = json.loads(
        history_path.read_text(
            encoding="utf-8"
        ).splitlines()[0]
    )

    assert len(saved["sources"]) == 1
    assert saved["sources"][0]["citation_id"] == "[S1]"

    assert len(saved["external_sources"]) == 1
    assert saved["external_sources"][0]["citation_id"] == "[W1]"

    assert len(result.retrieved_chunks) == 1


def test_agent_rejects_invalid_top_k():
    agent, retriever, generator, weather = make_agent()

    with pytest.raises(ValueError):
        agent.ask(
            "What is the weather in Colombo?",
            top_k=0,
        )

    assert retriever.calls == []
    assert weather.calls == []


def test_agent_rejects_empty_question():
    agent, retriever, generator, weather = make_agent()

    with pytest.raises(ValueError):
        agent.ask("   ")

    assert retriever.calls == []
    assert weather.calls == []


def test_agent_accepts_explicit_location():
    agent, retriever, generator, weather = make_agent()

    result = agent.ask(
        "What is the weather?",
        location_name="Kandy",
    )

    assert isinstance(result, WeatherAwareAgentResponse)

    assert result.location_name == "Kandy"

    assert weather.calls[0]["latitude"] == 7.2906
    assert weather.calls[0]["longitude"] == 80.6337
    assert weather.calls[0]["location_district"] == "Kandy"


def test_agent_handles_weather_and_academic_evidence():
    agent, retriever, generator, weather = make_agent()

    result = agent.ask(
        "Will rain affect university examinations in Colombo?"
    )

    assert len(result.sources) == 1
    assert len(result.external_sources) == 1

    assert result.external_sources[0].source_name == (
        "Open-Meteo (model-based weather data)"
    )

    assert result.reasoning_plan.intent == "weather_academic"

    assert result.selected_tools == (
        "weather_information",
        "academic_retrieval",
    )

def test_combined_academic_question_without_location_preserves_answer():
    agent, retriever, generator, weather = make_agent()

    result = agent.ask(
        "Has NSBM officially cancelled examinations today because of heavy rain?",
        institution_id="NSBM",
    )

    assert isinstance(result, WeatherAwareAgentResponse)
    assert len(retriever.calls) == 1
    assert weather.calls == []
    assert len(result.tool_errors) == 1

    assert "location could not be identified" not in result.answer.lower()
    assert "please specify a location" not in result.answer.lower()

    assert result.answer != result.tool_errors[0]