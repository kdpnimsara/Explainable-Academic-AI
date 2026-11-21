"""Unified evidence representation for weather information."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from src.weather.models import WeatherAlert, WeatherForecast


# Sri Lanka uses UTC+05:30.
# A fixed offset avoids external timezone database dependencies.
SRI_LANKA_TIMEZONE = timezone(
    timedelta(hours=5, minutes=30),
    name="SLST",
)


def format_sri_lanka_time(value: datetime) -> str:
    """Format a timezone-aware datetime using Sri Lankan local time."""

    if not isinstance(value, datetime):
        raise TypeError("value must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("value must be timezone-aware.")

    return value.astimezone(SRI_LANKA_TIMEZONE).isoformat()


@dataclass(frozen=True, slots=True)
class WeatherEvidence:
    """Represent traceable external weather evidence."""

    citation_id: str
    evidence_type: Literal["official_warning", "model_forecast"]

    title: str
    content: str

    source_name: str
    source_url: str

    data_timestamp: datetime
    retrieved_at: datetime

    valid_from: datetime | None = None
    valid_until: datetime | None = None

    location: str | None = None
    max_age: timedelta | None = None

    def __post_init__(self) -> None:
        """Validate evidence fields and timestamps."""

        if not self.citation_id.strip():
            raise ValueError("citation_id cannot be empty.")

        if self.evidence_type not in {
            "official_warning",
            "model_forecast",
        }:
            raise ValueError("Unsupported evidence type.")

        for name, value in (
            ("title", self.title),
            ("content", self.content),
            ("source_name", self.source_name),
            ("source_url", self.source_url),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty.")

        from urllib.parse import urlparse

        parsed_url = urlparse(self.source_url)

        if parsed_url.scheme != "https" or not parsed_url.hostname:
            raise ValueError("source_url must be a valid HTTPS URL.")

        timestamps = {
            "data_timestamp": self.data_timestamp,
            "retrieved_at": self.retrieved_at,
        }

        if self.valid_from is not None:
            timestamps["valid_from"] = self.valid_from

        if self.valid_until is not None:
            timestamps["valid_until"] = self.valid_until

        for name, value in timestamps.items():
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(
                    f"{name} must be timezone-aware."
                )

        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_from >= self.valid_until
        ):
            raise ValueError(
                "valid_until must be later than valid_from."
            )

        if self.max_age is not None and self.max_age < timedelta(0):
            raise ValueError("max_age cannot be negative.")

    def is_current(
        self,
        now: datetime | None = None,
    ) -> bool:
        """Determine whether the evidence is currently usable."""

        current = now or datetime.now(timezone.utc)

        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")

        current = current.astimezone(timezone.utc)

        if self.valid_from is not None:
            if current < self.valid_from.astimezone(timezone.utc):
                return False

        if self.valid_until is not None:
            if current >= self.valid_until.astimezone(timezone.utc):
                return False

        if self.max_age is not None:
            age = current - self.data_timestamp.astimezone(
                timezone.utc
            )

            if age < timedelta(0) or age > self.max_age:
                return False

        return True


def alert_to_evidence(
    alert: WeatherAlert,
    citation_id: str,
    now: datetime | None = None,
) -> WeatherEvidence | None:
    """Convert a currently valid official alert into evidence."""

    if not isinstance(alert, WeatherAlert):
        raise TypeError("alert must be a WeatherAlert.")

    if not alert.is_current(now):
        return None

    areas = (
        "\n".join(alert.area_descriptions)
        if alert.area_descriptions
        else "Not specified"
    )

    instruction = alert.instruction or "No instruction provided."

    content = (
        "OFFICIAL WEATHER WARNING\n"
        f"Event: {alert.event}\n"
        f"Headline: {alert.headline}\n"
        f"Severity: {alert.severity}\n"
        f"Urgency: {alert.urgency}\n"
        f"Certainty: {alert.certainty}\n"
        f"Issued at: {format_sri_lanka_time(alert.sent_at)}\n"
        f"Expires at: {format_sri_lanka_time(alert.expires_at)}\n"
        f"Description: {alert.description}\n"
        f"Instructions: {instruction}\n"
        f"Affected areas:\n{areas}"
    )

    return WeatherEvidence(
        citation_id=citation_id,
        evidence_type="official_warning",
        title=alert.headline or alert.event,
        content=content,
        source_name="Sri Lanka Department of Meteorology",
        source_url=alert.source_url,
        data_timestamp=alert.sent_at,
        retrieved_at=alert.retrieved_at,
        valid_from=(
            alert.effective_at
            or alert.onset_at
            or alert.sent_at
        ),
        valid_until=alert.expires_at,
        location=areas,
    )


def forecast_to_evidence(
    forecast: WeatherForecast,
    citation_id: str,
    location_name: str | None = None,
    now: datetime | None = None,
    max_age: timedelta = timedelta(hours=3),
) -> WeatherEvidence | None:
    """Convert fresh model-based forecast information into evidence."""

    if not isinstance(forecast, WeatherForecast):
        raise TypeError("forecast must be a WeatherForecast.")

    if max_age < timedelta(0):
        raise ValueError("max_age cannot be negative.")

    if not forecast.is_fresh(now, max_age):
        return None

    location = (
        location_name.strip()
        if location_name and location_name.strip()
        else f"{forecast.latitude}, {forecast.longitude}"
    )

    def display(value, unit: str = "") -> str:
        if value is None:
            return "Not available"

        return f"{value}{unit}"

    lines = [
        "MODEL-BASED WEATHER INFORMATION",
        "This is model-based data, not necessarily a direct "
        "weather-station observation.",
        f"Location: {location}",
        (
            "Data timestamp (Sri Lanka time): "
            f"{format_sri_lanka_time(forecast.current_time)}"
        ),
        f"Temperature: {display(forecast.temperature_c, ' °C')}",
        f"Feels like: {display(forecast.apparent_temperature_c, ' °C')}",
        (
            "Relative humidity: "
            f"{display(forecast.relative_humidity_percent, ' %')}"
        ),
        f"Precipitation: {display(forecast.precipitation_mm, ' mm')}",
        f"Weather code: {display(forecast.weather_code)}",
        f"Wind speed: {display(forecast.wind_speed_kmh, ' km/h')}",
        "",
        "HOURLY FORECAST (SRI LANKA LOCAL TIME):",
    ]

    for timestamp, temperature, probability in zip(
        forecast.hourly_times,
        forecast.hourly_temperatures_c,
        forecast.hourly_precipitation_probability_percent,
    ):
        lines.append(
            f"{format_sri_lanka_time(timestamp)}: "
            f"Temperature {display(temperature, ' °C')}, "
            f"Precipitation probability {display(probability, ' %')}"
        )

    return WeatherEvidence(
        citation_id=citation_id,
        evidence_type="model_forecast",
        title=f"Weather forecast for {location}",
        content="\n".join(lines),
        source_name="Open-Meteo (model-based weather data)",
        source_url=forecast.source_url,
        data_timestamp=forecast.current_time,
        retrieved_at=forecast.retrieved_at,
        valid_from=forecast.current_time,
        location=location,
        max_age=max_age,
    )