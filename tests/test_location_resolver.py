
import pytest

from src.weather.location_resolver import LocationResolver


def test_resolves_colombo():
    location = LocationResolver().resolve("Colombo")

    assert location.name == "Colombo"
    assert location.district == "Colombo"
    assert location.latitude == 6.9271


def test_resolves_case_insensitively():
    location = LocationResolver().resolve("KANDY")

    assert location.name == "Kandy"


def test_resolves_location_with_extra_spaces():
    location = LocationResolver().resolve("  Nuwara   Eliya  ")

    assert location.name == "Nuwara Eliya"


def test_resolves_location_from_question():
    location = LocationResolver().resolve_from_question(
        "What is the weather in Galle?"
    )

    assert location.name == "Galle"


def test_does_not_guess_multiple_locations():
    location = LocationResolver().resolve_from_question(
        "Compare the weather in Colombo and Kandy."
    )

    assert location is None


def test_returns_none_for_unknown_location():
    location = LocationResolver().resolve("Unknown City")

    assert location is None


def test_returns_none_without_location():
    location = LocationResolver().resolve_from_question(
        "What is the current weather?"
    )

    assert location is None


def test_does_not_match_partial_city_name():
    location = LocationResolver().resolve_from_question(
        "What is the weather in ColomboXYZ?"
    )

    assert location is None


def test_rejects_invalid_location_type():
    with pytest.raises(TypeError):
        LocationResolver().resolve(123)


def test_rejects_invalid_question_type():
    with pytest.raises(TypeError):
        LocationResolver().resolve_from_question(123)
