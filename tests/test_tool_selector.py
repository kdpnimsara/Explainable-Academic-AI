
import pytest

from src.agents.tool_selector import ExplainableToolSelector


def test_selects_academic_tool():
    selector = ExplainableToolSelector()

    result = selector.select(
        "What are the examination regulations?"
    )

    assert result.intent == "academic"
    assert result.selected_tools == ("academic_retrieval",)
    assert result.requires_academic_documents is True
    assert result.requires_weather is False


def test_selects_weather_tool():
    selector = ExplainableToolSelector()

    result = selector.select(
        "What is the weather in Colombo?"
    )

    assert result.intent == "weather"
    assert result.selected_tools == ("weather_information",)
    assert result.requires_weather is True
    assert result.location_name == "Colombo"


def test_selects_both_tools_for_weather_academic_question():
    selector = ExplainableToolSelector()

    result = selector.select(
        "Will heavy rain affect university examinations?"
    )

    assert result.intent == "weather_academic"
    assert result.requires_weather is True
    assert result.requires_academic_documents is True

    assert result.selected_tools == (
        "weather_information",
        "academic_retrieval",
    )


def test_extracts_known_location():
    selector = ExplainableToolSelector()

    result = selector.select(
        "Give me the weather forecast for Kandy."
    )

    assert result.location_name == "Kandy"


def test_accepts_explicit_location():
    selector = ExplainableToolSelector()

    result = selector.select(
        "What is the weather?",
        location_name="Galle",
    )

    assert result.location_name == "Galle"


def test_does_not_guess_multiple_locations():
    selector = ExplainableToolSelector()

    result = selector.select(
        "Compare weather in Colombo and Kandy."
    )

    assert result.location_name is None


def test_provides_reasoning_rationale():
    selector = ExplainableToolSelector()

    result = selector.select(
        "What is the weather in Colombo?"
    )

    assert result.rationale
    assert "weather" in result.rationale.lower()


def test_rejects_empty_question():
    selector = ExplainableToolSelector()

    with pytest.raises(ValueError):
        selector.select("   ")


def test_rejects_invalid_question_type():
    selector = ExplainableToolSelector()

    with pytest.raises(TypeError):
        selector.select(123)


def test_rejects_empty_location():
    selector = ExplainableToolSelector()

    with pytest.raises(ValueError):
        selector.select(
            "What is the weather?",
            location_name="   ",
        )
