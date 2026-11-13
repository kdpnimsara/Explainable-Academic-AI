
"""Coordinate official warnings and location-based forecasts."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from src.weather.cap_provider import (
    CAPWeatherProvider,
    WeatherProviderError,
)
from src.weather.evidence import (
    WeatherEvidence,
    alert_to_evidence,
    forecast_to_evidence,
)
from src.weather.forecast_provider import (
    ForecastProviderError,
    OpenMeteoForecastProvider,
)
from src.weather.models import WeatherAlert, WeatherForecast


class CAPProviderProtocol(Protocol):
    """Interface for an official CAP warning provider."""

    def fetch_current_alerts(
        self,
        now: datetime | None = None,
    ) -> list[WeatherAlert]:
        ...


class ForecastProviderProtocol(Protocol):
    """Interface for a location-based forecast provider."""

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
    ) -> WeatherForecast:
        ...


@dataclass(frozen=True, slots=True)
class WeatherProviderFailure:
    """Describe a failure from an external weather provider."""

    provider: str
    message: str


@dataclass(frozen=True, slots=True)
class WeatherInformationResult:
    """Combined result from weather information providers."""

    location: str | None
    evidence: tuple[WeatherEvidence, ...]
    errors: tuple[WeatherProviderFailure, ...]

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def is_partial(self) -> bool:
        return self.has_evidence and self.has_errors


class WeatherInformationService:
    """Coordinate weather warnings and forecast information."""

    def __init__(
        self,
        cap_provider: CAPProviderProtocol | None = None,
        forecast_provider: ForecastProviderProtocol | None = None,
        forecast_max_age: timedelta = timedelta(hours=3),
    ) -> None:

        if (
            not isinstance(forecast_max_age, timedelta)
            or forecast_max_age < timedelta(0)
        ):
            raise ValueError(
                "forecast_max_age must be a non-negative timedelta."
            )

        if cap_provider is not None and not callable(
            getattr(cap_provider, "fetch_current_alerts", None)
        ):
            raise TypeError(
                "cap_provider must provide fetch_current_alerts()."
            )

        if forecast_provider is not None and not callable(
            getattr(forecast_provider, "fetch_forecast", None)
        ):
            raise TypeError(
                "forecast_provider must provide fetch_forecast()."
            )

        self.cap_provider = (
            cap_provider
            if cap_provider is not None
            else CAPWeatherProvider()
        )

        self.forecast_provider = (
            forecast_provider
            if forecast_provider is not None
            else OpenMeteoForecastProvider()
        )

        self.forecast_max_age = forecast_max_age

    @staticmethod
    def _alert_matches_district(
        alert: WeatherAlert,
        district: str,
    ) -> bool:
        """Check whether an alert explicitly lists the requested district."""

        pattern = rf"(?<!\w){re.escape(district)}(?!\w)"

        return any(
            re.search(pattern, area, re.IGNORECASE)
            for area in alert.area_descriptions
        )

    def get_information(
        self,
        latitude: float,
        longitude: float,
        location_name: str | None = None,
        now: datetime | None = None,
        location_district: str | None = None,
    ) -> WeatherInformationResult:
        """Retrieve and combine location-specific weather evidence."""

        current_time = now or datetime.now(timezone.utc)

        if (
            current_time.tzinfo is None
            or current_time.utcoffset() is None
        ):
            raise ValueError("now must be timezone-aware.")

        if location_name is not None:
            if not isinstance(location_name, str):
                raise TypeError(
                    "location_name must be a string or None."
                )

            location_name = location_name.strip()

            if not location_name:
                raise ValueError(
                    "location_name cannot be empty."
                )

        if location_district is not None:
            if not isinstance(location_district, str):
                raise TypeError(
                    "location_district must be a string or None."
                )

            location_district = location_district.strip()

            if not location_district:
                raise ValueError(
                    "location_district cannot be empty."
                )

        evidence = []
        errors = []

        # Provider 1: Official Sri Lankan weather warnings.
        try:
            alerts = self.cap_provider.fetch_current_alerts(
                now=current_time
            )

            # Filter warnings when a district is provided.
            if location_district is not None:
                alerts = [
                    alert
                    for alert in alerts
                    if self._alert_matches_district(
                        alert,
                        location_district,
                    )
                ]

            for index, alert in enumerate(alerts, start=1):
                item = alert_to_evidence(
                    alert,
                    citation_id=f"[W{index}]",
                    now=current_time,
                )

                if item is not None:
                    evidence.append(item)

        except WeatherProviderError as exc:
            errors.append(
                WeatherProviderFailure(
                    provider="Sri Lanka Department of Meteorology",
                    message=str(exc),
                )
            )

        # Provider 2: Location-based weather forecast.
        try:
            forecast = self.forecast_provider.fetch_forecast(
                latitude,
                longitude,
            )

            forecast_item = forecast_to_evidence(
                forecast,
                citation_id=f"[W{len(evidence) + 1}]",
                location_name=location_name,
                now=current_time,
                max_age=self.forecast_max_age,
            )

            if forecast_item is not None:
                evidence.append(forecast_item)
            else:
                errors.append(
                    WeatherProviderFailure(
                        provider="Open-Meteo",
                        message=(
                            "Forecast evidence was rejected "
                            "because it was outdated or not yet valid."
                        ),
                    )
                )

        except ForecastProviderError as exc:
            errors.append(
                WeatherProviderFailure(
                    provider="Open-Meteo",
                    message=str(exc),
                )
            )

        return WeatherInformationResult(
            location=location_name,
            evidence=tuple(evidence),
            errors=tuple(errors),
        )
