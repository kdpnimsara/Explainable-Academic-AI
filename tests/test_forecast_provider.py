
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from src.weather.models import WeatherForecast
from src.weather.forecast_provider import (
    ForecastProviderError,
    OpenMeteoForecastProvider,
)


def make_api_response():
    """Create a sample Open-Meteo response."""

    base = datetime(
        2026, 11, 6, 0, 0, tzinfo=timezone.utc
    )

    times = [
        base + timedelta(hours=i)
        for i in range(24)
    ]

    return {
        "timezone": "Asia/Colombo",
        "current": {
            "time": base.timestamp(),
            "temperature_2m": 24.6,
            "relative_humidity_2m": 97,
            "apparent_temperature": 29.9,
            "precipitation": 0.0,
            "weather_code": 3,
            "wind_speed_10m": 6.1,
        },
        "hourly": {
            "time": [t.timestamp() for t in times],
            "temperature_2m": [24.0 + i * 0.1 for i in range(24)],
            "precipitation_probability": [10] * 24,
        },
    }


def test_forecast_provider_parses_current_weather(monkeypatch):
    provider = OpenMeteoForecastProvider()
    payload = make_api_response()

    monkeypatch.setattr(
        provider,
        "_download_json",
        lambda url: payload,
    )

    result = provider.fetch_forecast(6.9271, 79.8612)

    assert result.temperature_c == 24.6
    assert result.relative_humidity_percent == 97
    assert result.apparent_temperature_c == 29.9
    assert result.precipitation_mm == 0.0
    assert result.weather_code == 3
    assert result.wind_speed_kmh == 6.1


def test_forecast_provider_parses_hourly_data(monkeypatch):
    provider = OpenMeteoForecastProvider()

    monkeypatch.setattr(
        provider,
        "_download_json",
        lambda url: make_api_response(),
    )

    result = provider.fetch_forecast(6.9271, 79.8612)

    assert len(result.hourly_times) == 24
    assert len(result.hourly_temperatures_c) == 24
    assert len(
        result.hourly_precipitation_probability_percent
    ) == 24


def test_forecast_provider_preserves_source_and_timezone(monkeypatch):
    provider = OpenMeteoForecastProvider()

    monkeypatch.setattr(
        provider,
        "_download_json",
        lambda url: make_api_response(),
    )

    result = provider.fetch_forecast(6.9271, 79.8612)

    assert result.timezone_name == "Asia/Colombo"
    assert "api.open-meteo.com" in result.source_url
    assert result.retrieved_at.tzinfo is not None


def test_forecast_provider_builds_correct_request_url():
    provider = OpenMeteoForecastProvider()

    url = provider._build_url(6.9271, 79.8612)
    parameters = parse_qs(urlsplit(url).query)

    assert parameters["latitude"] == ["6.9271"]
    assert parameters["longitude"] == ["79.8612"]
    assert parameters["timezone"] == ["auto"]
    assert parameters["forecast_hours"] == ["24"]


def test_forecast_provider_rejects_invalid_latitude():
    provider = OpenMeteoForecastProvider()

    with pytest.raises(ValueError):
        provider.fetch_forecast(100, 79.8612)


def test_forecast_provider_rejects_boolean_coordinates():
    provider = OpenMeteoForecastProvider()

    with pytest.raises(TypeError):
        provider.fetch_forecast(True, 79.8612)


def test_forecast_provider_rejects_mismatched_hourly_data(
    monkeypatch,
):
    provider = OpenMeteoForecastProvider()
    payload = make_api_response()

    payload["hourly"]["temperature_2m"].pop()

    monkeypatch.setattr(
        provider,
        "_download_json",
        lambda url: payload,
    )

    with pytest.raises(ForecastProviderError):
        provider.fetch_forecast(6.9271, 79.8612)


def test_forecast_provider_rejects_invalid_temperature(
    monkeypatch,
):
    provider = OpenMeteoForecastProvider()
    payload = make_api_response()

    payload["current"]["temperature_2m"] = "invalid"

    monkeypatch.setattr(
        provider,
        "_download_json",
        lambda url: payload,
    )

    with pytest.raises(ForecastProviderError):
        provider.fetch_forecast(6.9271, 79.8612)


def test_forecast_provider_accepts_missing_optional_values(
    monkeypatch,
):
    provider = OpenMeteoForecastProvider()
    payload = make_api_response()

    payload["current"]["temperature_2m"] = None
    payload["current"]["precipitation"] = None

    monkeypatch.setattr(
        provider,
        "_download_json",
        lambda url: payload,
    )

    result = provider.fetch_forecast(6.9271, 79.8612)

    assert result.temperature_c is None
    assert result.precipitation_mm is None


def test_weather_forecast_freshness():
    now = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)

    forecast = WeatherForecast(
        latitude=6.9271,
        longitude=79.8612,
        timezone_name="Asia/Colombo",
        source_url="https://api.open-meteo.com/v1/forecast",
        retrieved_at=now,
        current_time=now,
        temperature_c=24.6,
        relative_humidity_percent=97,
        apparent_temperature_c=29.9,
        precipitation_mm=0.0,
        weather_code=3,
        wind_speed_kmh=6.1,
    )

    assert forecast.is_fresh(now) is True

    old_time = now + timedelta(hours=4)

    assert forecast.is_fresh(old_time) is False
