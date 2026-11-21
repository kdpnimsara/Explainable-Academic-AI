
from datetime import datetime, timedelta, timezone

import pytest

from src.generation.answer_generator import AnswerGenerator
from src.weather.models import WeatherForecast
from src.weather.evidence import forecast_to_evidence

from tests.test_generation import FakeModel, make_retrieved_chunk


NOW = datetime.now(timezone.utc)


def make_weather_evidence():
    """Create fresh weather evidence for testing."""

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


def test_weather_only_answer_generation():
    model = FakeModel("Colombo weather is available [W1].")
    generator = AnswerGenerator(model)

    result = generator.generate_answer_with_weather(
        "What is the weather in Colombo?",
        [],
        [make_weather_evidence()],
    )

    assert model.called
    assert "[W1]" in model.last_prompt
    assert result.sources == ()
    assert len(result.external_sources) == 1
    assert result.external_sources[0].citation_id == "[W1]"


def test_combined_academic_and_weather_evidence():
    model = FakeModel(
        "Weather conditions should be considered "
        "alongside university regulations [S1] [W1]."
    )

    generator = AnswerGenerator(model)

    result = generator.generate_answer_with_weather(
        "Will rain affect university examinations?",
        [make_retrieved_chunk()],
        [make_weather_evidence()],
    )

    assert "[S1]" in model.last_prompt
    assert "[W1]" in model.last_prompt
    assert len(result.sources) == 1
    assert len(result.external_sources) == 1

    assert result.sources[0].document == (
        "disaster_management_act_2005.pdf"
    )


def test_weather_source_timestamp_is_preserved():
    generator = AnswerGenerator(
        FakeModel("Weather information available [W1].")
    )

    evidence = make_weather_evidence()

    result = generator.generate_answer_with_weather(
        "Weather in Colombo?",
        [],
        [evidence],
    )

    assert (
        result.external_sources[0].data_timestamp
        == evidence.data_timestamp.isoformat()
    )


def test_stale_weather_evidence_is_not_used():
    """Stale weather information must not reach the LLM."""

    model = FakeModel()
    generator = AnswerGenerator(model)

    old_time = datetime.now(timezone.utc) - timedelta(hours=4)

    stale_forecast = WeatherForecast(
        latitude=6.9271,
        longitude=79.8612,
        timezone_name="Asia/Colombo",
        source_url="https://api.open-meteo.com/v1/forecast",
        retrieved_at=datetime.now(timezone.utc),
        current_time=old_time,
        temperature_c=24.6,
        relative_humidity_percent=97.0,
        apparent_temperature_c=29.9,
        precipitation_mm=0.0,
        weather_code=3,
        wind_speed_kmh=6.1,
    )

    evidence = forecast_to_evidence(
        stale_forecast,
        citation_id="[W1]",
        location_name="Colombo",
        now=old_time,
    )

    # Evidence can be created relative to its own timestamp.
    assert evidence is not None

    result = generator.generate_answer_with_weather(
        "What is the weather in Colombo?",
        [],
        [evidence],
    )

    # The generator must reject it because it is now more than
    # three hours old.
    assert result.answer == generator.ABSTENTION_MESSAGE
    assert result.external_sources == ()
    assert model.called is False


def test_no_evidence_returns_abstention():
    model = FakeModel()
    generator = AnswerGenerator(model)

    result = generator.generate_answer_with_weather(
        "Unknown question?",
        [],
        [],
    )

    assert result.answer == generator.ABSTENTION_MESSAGE
    assert result.sources == ()
    assert result.external_sources == ()
    assert not model.called


def test_invalid_weather_evidence_is_rejected():
    generator = AnswerGenerator(FakeModel())

    with pytest.raises(TypeError):
        generator.generate_answer_with_weather(
            "Weather question?",
            [],
            ["invalid"],
        )


def test_existing_academic_method_remains_compatible():
    model = FakeModel()
    generator = AnswerGenerator(model)

    result = generator.generate_answer(
        "What is disaster preparedness?",
        [make_retrieved_chunk()],
    )

    assert model.called
    assert len(result.sources) == 1
    assert result.external_sources == ()


def test_prompt_distinguishes_official_and_model_based_evidence():
    generator = AnswerGenerator(FakeModel())

    generator.generate_answer_with_weather(
        "What is the weather in Colombo?",
        [],
        [make_weather_evidence()],
    )

    assert "model-based" in generator.model.last_prompt.lower()
    assert "direct" in generator.model.last_prompt.lower()
