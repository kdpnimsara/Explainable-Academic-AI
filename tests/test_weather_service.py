
from datetime import datetime, timedelta, timezone

import pytest

from src.weather.models import WeatherAlert, WeatherForecast
from src.weather.cap_provider import WeatherProviderError
from src.weather.forecast_provider import ForecastProviderError
from src.weather.service import WeatherInformationService


NOW = datetime(2026, 11, 6, 10, 0, tzinfo=timezone.utc)


def make_alert():
    return WeatherAlert(
        identifier="alert-001",
        sender="info@meteo.gov.lk",
        sent_at=NOW - timedelta(minutes=10),
        status="Actual",
        message_type="Alert",
        scope="Public",
        event="Lightning",
        urgency="Immediate",
        severity="Severe",
        certainty="Possible",
        headline="Lightning Warning",
        description="Severe lightning expected.",
        source_url="https://was.meteo.gov.lk/cap/en/test.xml",
        retrieved_at=NOW,
        effective_at=NOW - timedelta(minutes=10),
        expires_at=NOW + timedelta(hours=2),
        instruction="Stay indoors.",
        area_descriptions=("Colombo District",),
    )


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
    }

    data.update(overrides)
    return WeatherForecast(**data)


class FakeCAPProvider:
    def __init__(self, alerts=None, error=None):
        self.alerts = alerts if alerts is not None else []
        self.error = error
        self.calls = []

    def fetch_current_alerts(self, now=None):
        self.calls.append(now)

        if self.error:
            raise self.error

        return self.alerts


class FakeForecastProvider:
    def __init__(self, forecast=None, error=None):
        self.forecast = forecast or make_forecast()
        self.error = error
        self.calls = []

    def fetch_forecast(self, latitude, longitude):
        self.calls.append((latitude, longitude))

        if self.error:
            raise self.error

        return self.forecast


def test_service_combines_alerts_and_forecasts():
    cap = FakeCAPProvider([make_alert()])
    forecast = FakeForecastProvider()

    service = WeatherInformationService(
        cap_provider=cap,
        forecast_provider=forecast,
    )

    result = service.get_information(
        6.9271,
        79.8612,
        location_name="Colombo",
        now=NOW,
    )

    assert len(result.evidence) == 2
    assert result.errors == ()
    assert result.has_evidence is True
    assert result.has_errors is False

    assert result.evidence[0].citation_id == "[W1]"
    assert result.evidence[0].evidence_type == "official_warning"

    assert result.evidence[1].citation_id == "[W2]"
    assert result.evidence[1].evidence_type == "model_forecast"

    assert result.location == "Colombo"


def test_service_handles_cap_provider_failure():
    cap = FakeCAPProvider(
        error=WeatherProviderError("CAP feed unavailable")
    )
    forecast = FakeForecastProvider()

    service = WeatherInformationService(cap, forecast)

    result = service.get_information(
        6.9271,
        79.8612,
        now=NOW,
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_type == "model_forecast"
    assert result.has_errors is True
    assert result.is_partial is True
    assert result.errors[0].provider == (
        "Sri Lanka Department of Meteorology"
    )


def test_service_handles_forecast_provider_failure():
    cap = FakeCAPProvider([make_alert()])
    forecast = FakeForecastProvider(
        error=ForecastProviderError("Forecast unavailable")
    )

    service = WeatherInformationService(cap, forecast)

    result = service.get_information(
        6.9271,
        79.8612,
        now=NOW,
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_type == "official_warning"
    assert result.is_partial is True
    assert result.errors[0].provider == "Open-Meteo"


def test_service_handles_both_provider_failures():
    cap = FakeCAPProvider(
        error=WeatherProviderError("CAP unavailable")
    )
    forecast = FakeForecastProvider(
        error=ForecastProviderError("Forecast unavailable")
    )

    service = WeatherInformationService(cap, forecast)

    result = service.get_information(
        6.9271,
        79.8612,
        now=NOW,
    )

    assert result.evidence == ()
    assert len(result.errors) == 2
    assert result.has_evidence is False
    assert result.has_errors is True
    assert result.is_partial is False


def test_service_rejects_stale_forecast():
    cap = FakeCAPProvider()

    forecast = FakeForecastProvider(
        make_forecast(
            current_time=NOW - timedelta(hours=5),
            retrieved_at=NOW,
        )
    )

    service = WeatherInformationService(cap, forecast)

    result = service.get_information(
        6.9271,
        79.8612,
        now=NOW,
    )

    assert result.evidence == ()
    assert len(result.errors) == 1
    assert "outdated" in result.errors[0].message


def test_service_handles_no_active_alerts():
    cap = FakeCAPProvider([])
    forecast = FakeForecastProvider()

    service = WeatherInformationService(cap, forecast)

    result = service.get_information(
        6.9271,
        79.8612,
        now=NOW,
    )

    assert len(result.evidence) == 1
    assert result.evidence[0].citation_id == "[W1]"
    assert result.evidence[0].evidence_type == "model_forecast"
    assert result.errors == ()


def test_service_passes_correct_coordinates():
    cap = FakeCAPProvider()
    forecast = FakeForecastProvider()

    service = WeatherInformationService(cap, forecast)

    service.get_information(
        7.2906,
        80.6337,
        location_name="Kandy",
        now=NOW,
    )

    assert forecast.calls == [(7.2906, 80.6337)]


def test_service_rejects_naive_timestamp():
    service = WeatherInformationService(
        FakeCAPProvider(),
        FakeForecastProvider(),
    )

    with pytest.raises(ValueError):
        service.get_information(
            6.9271,
            79.8612,
            now=datetime(2026, 11, 6, 10, 0),
        )


def test_service_rejects_empty_location_name():
    service = WeatherInformationService(
        FakeCAPProvider(),
        FakeForecastProvider(),
    )

    with pytest.raises(ValueError):
        service.get_information(
            6.9271,
            79.8612,
            location_name="   ",
            now=NOW,
        )



def test_service_filters_warnings_by_district():
    cap = FakeCAPProvider([make_alert()])
    forecast = FakeForecastProvider()

    service = WeatherInformationService(cap, forecast)

    result = service.get_information(
        6.9271,
        79.8612,
        location_name="Colombo",
        location_district="Colombo",
        now=NOW,
    )

    assert len(result.evidence) == 2
    assert result.evidence[0].evidence_type == "official_warning"
    assert result.evidence[0].citation_id == "[W1]"
    assert result.evidence[1].evidence_type == "model_forecast"


def test_service_excludes_warnings_from_other_districts():
    cap = FakeCAPProvider([make_alert()])
    forecast = FakeForecastProvider()

    service = WeatherInformationService(cap, forecast)

    result = service.get_information(
        7.2906,
        80.6337,
        location_name="Kandy",
        location_district="Kandy",
        now=NOW,
    )

    # The Colombo warning must not be presented as a Kandy warning.
    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_type == "model_forecast"
    assert result.evidence[0].citation_id == "[W1]"
    assert result.errors == ()
