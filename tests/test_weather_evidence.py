
from datetime import datetime, timedelta, timezone

import pytest

from src.weather.models import WeatherAlert, WeatherForecast
from src.weather.evidence import (
    WeatherEvidence,
    alert_to_evidence,
    forecast_to_evidence,
)


NOW = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)


def make_alert(**overrides):
    data = {
        "identifier": "alert-001",
        "sender": "info@meteo.gov.lk",
        "sent_at": NOW,
        "status": "Actual",
        "message_type": "Alert",
        "scope": "Public",
        "event": "Lightning",
        "urgency": "Immediate",
        "severity": "Severe",
        "certainty": "Possible",
        "headline": "Lightning Warning",
        "description": "Severe lightning expected.",
        "source_url": "https://was.meteo.gov.lk/cap/en/test.xml",
        "retrieved_at": NOW,
        "effective_at": NOW,
        "expires_at": NOW + timedelta(hours=2),
        "instruction": "Stay indoors.",
        "area_descriptions": ("Colombo District",),
    }

    data.update(overrides)
    return WeatherAlert(**data)


def make_forecast(**overrides):
    data = {
        "latitude": 6.9271,
        "longitude": 79.8612,
        "timezone_name": "Asia/Colombo",
        "source_url": "https://api.open-meteo.com/v1/forecast",
        "retrieved_at": NOW,
        "current_time": NOW,
        "temperature_c": 24.6,
        "relative_humidity_percent": 97.0,
        "apparent_temperature_c": 29.9,
        "precipitation_mm": 0.0,
        "weather_code": 3,
        "wind_speed_kmh": 6.1,
        "hourly_times": (NOW, NOW + timedelta(hours=1)),
        "hourly_temperatures_c": (24.6, 25.0),
        "hourly_precipitation_probability_percent": (10.0, 20.0),
    }

    data.update(overrides)
    return WeatherForecast(**data)


def test_alert_to_evidence():
    evidence = alert_to_evidence(
        make_alert(),
        "[W1]",
        now=NOW + timedelta(minutes=30),
    )

    assert evidence is not None
    assert evidence.citation_id == "[W1]"
    assert evidence.evidence_type == "official_warning"
    assert evidence.source_name == (
        "Sri Lanka Department of Meteorology"
    )
    assert "Lightning" in evidence.content
    assert "Stay indoors." in evidence.content
    assert "Colombo District" in evidence.content


def test_expired_alert_is_not_converted():
    alert = make_alert(
        expires_at=NOW - timedelta(minutes=1)
    )

    evidence = alert_to_evidence(
        alert,
        "[W1]",
        now=NOW,
    )

    assert evidence is None


def test_alert_evidence_preserves_expiry():
    evidence = alert_to_evidence(
        make_alert(),
        "[W1]",
        now=NOW,
    )

    assert evidence is not None
    assert evidence.valid_until == NOW + timedelta(hours=2)
    assert evidence.is_current(NOW) is True


def test_forecast_to_evidence():
    evidence = forecast_to_evidence(
        make_forecast(),
        "[W2]",
        location_name="Colombo",
        now=NOW,
    )

    assert evidence is not None
    assert evidence.citation_id == "[W2]"
    assert evidence.evidence_type == "model_forecast"
    assert evidence.location == "Colombo"
    assert "24.6 °C" in evidence.content
    assert "97.0 %" in evidence.content
    assert "not necessarily a direct" in evidence.content


def test_stale_forecast_is_not_converted():
    old_forecast = make_forecast(
        current_time=NOW - timedelta(hours=4),
        retrieved_at=NOW,
    )

    evidence = forecast_to_evidence(
        old_forecast,
        "[W1]",
        now=NOW,
    )

    assert evidence is None


def test_forecast_evidence_has_source_and_timestamp():
    evidence = forecast_to_evidence(
        make_forecast(),
        "[W1]",
        now=NOW,
    )

    assert evidence is not None
    assert "open-meteo.com" in evidence.source_url
    assert evidence.data_timestamp == NOW
    assert evidence.retrieved_at == NOW


def test_forecast_evidence_checks_freshness():
    evidence = forecast_to_evidence(
        make_forecast(),
        "[W1]",
        now=NOW,
    )

    assert evidence is not None
    assert evidence.is_current(NOW) is True
    assert evidence.is_current(
        NOW + timedelta(hours=4)
    ) is False


def test_evidence_rejects_invalid_url():
    with pytest.raises(ValueError):
        WeatherEvidence(
            citation_id="[W1]",
            evidence_type="official_warning",
            title="Test warning",
            content="Test content",
            source_name="Test source",
            source_url="http://example.com",
            data_timestamp=NOW,
            retrieved_at=NOW,
        )


def test_evidence_rejects_naive_timestamp():
    with pytest.raises(ValueError):
        WeatherEvidence(
            citation_id="[W1]",
            evidence_type="official_warning",
            title="Test warning",
            content="Test content",
            source_name="Test source",
            source_url="https://example.com",
            data_timestamp=datetime(2026, 11, 6, 10, 0),
            retrieved_at=NOW,
        )


def test_evidence_rejects_invalid_validity_window():
    with pytest.raises(ValueError):
        WeatherEvidence(
            citation_id="[W1]",
            evidence_type="official_warning",
            title="Test warning",
            content="Test content",
            source_name="Test source",
            source_url="https://example.com",
            data_timestamp=NOW,
            retrieved_at=NOW,
            valid_from=NOW + timedelta(hours=2),
            valid_until=NOW + timedelta(hours=1),
        )


def test_evidence_rejects_negative_max_age():
    with pytest.raises(ValueError):
        WeatherEvidence(
            citation_id="[W1]",
            evidence_type="model_forecast",
            title="Test forecast",
            content="Test content",
            source_name="Test source",
            source_url="https://example.com",
            data_timestamp=NOW,
            retrieved_at=NOW,
            max_age=timedelta(hours=-1),
        )


def test_invalid_forecast_data_type_is_rejected():
    with pytest.raises(TypeError):
        forecast_to_evidence(
            "invalid",
            "[W1]",
            now=NOW,
        )


def test_invalid_alert_data_type_is_rejected():
    with pytest.raises(TypeError):
        alert_to_evidence(
            "invalid",
            "[W1]",
            now=NOW,
        )
