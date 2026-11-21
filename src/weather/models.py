
"""Data models for official weather alerts."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True, slots=True)
class WeatherAlert:
    """Represent an official CAP weather alert."""

    identifier: str
    sender: str
    sent_at: datetime

    status: str
    message_type: str
    scope: str

    event: str
    urgency: str
    severity: str
    certainty: str

    headline: str
    description: str

    source_url: str
    retrieved_at: datetime

    effective_at: datetime | None = None
    onset_at: datetime | None = None
    expires_at: datetime | None = None

    instruction: str | None = None
    area_descriptions: tuple[str, ...] = ()

    def is_current(self, now: datetime | None = None) -> bool:
        """Check whether this alert is currently valid."""

        current_time = now or datetime.now(timezone.utc)

        if (
            current_time.tzinfo is None
            or current_time.utcoffset() is None
        ):
            raise ValueError("now must be timezone-aware.")

        if self.status.casefold() != "actual":
            return False

        if self.message_type.casefold() not in {"alert", "update"}:
            return False

        starts_at = (
            self.effective_at
            or self.onset_at
            or self.sent_at
        )

        if starts_at > current_time:
            return False

        if self.expires_at is None:
            return False

        return self.expires_at > current_time




@dataclass(frozen=True, slots=True)
class WeatherForecast:
    """Represent model-based current conditions and weather forecasts.

    Current conditions are model-based values, not necessarily
    direct observations from a local weather station.
    """

    latitude: float
    longitude: float
    timezone_name: str

    source_url: str
    retrieved_at: datetime
    current_time: datetime

    temperature_c: float | None
    relative_humidity_percent: float | None
    apparent_temperature_c: float | None
    precipitation_mm: float | None
    weather_code: int | None
    wind_speed_kmh: float | None

    hourly_times: tuple[datetime, ...] = ()
    hourly_temperatures_c: tuple[float | None, ...] = ()
    hourly_precipitation_probability_percent: tuple[
        float | None, ...
    ] = ()

    def __post_init__(self) -> None:
        """Validate timestamps and coordinates."""

        if not -90 <= self.latitude <= 90:
            raise ValueError("Invalid latitude.")

        if not -180 <= self.longitude <= 180:
            raise ValueError("Invalid longitude.")

        for name, value in (
            ("retrieved_at", self.retrieved_at),
            ("current_time", self.current_time),
        ):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(
                    f"{name} must be timezone-aware."
                )

        if (
            len(self.hourly_times)
            != len(self.hourly_temperatures_c)
            or len(self.hourly_times)
            != len(self.hourly_precipitation_probability_percent)
        ):
            raise ValueError(
                "Hourly forecast arrays must have equal lengths."
            )

    def is_fresh(
        self,
        now: datetime | None = None,
        max_age: timedelta = timedelta(hours=3),
    ) -> bool:
        """Check whether the current model data is sufficiently recent."""

        current = now or datetime.now(timezone.utc)

        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("now must be timezone-aware.")

        if max_age < timedelta(0):
            raise ValueError("max_age cannot be negative.")

        age = current.astimezone(timezone.utc) - (
            self.current_time.astimezone(timezone.utc)
        )

        return timedelta(0) <= age <= max_age
