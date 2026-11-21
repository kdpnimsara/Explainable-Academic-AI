
"""Open-Meteo provider for current conditions and hourly forecasts."""

import json
import math
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.weather.models import WeatherForecast


DEFAULT_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class ForecastProviderError(RuntimeError):
    """Raised when forecast information cannot be retrieved or parsed."""


class OpenMeteoForecastProvider:
    """Retrieve location-based weather information from Open-Meteo."""

    CURRENT_VARIABLES = (
        "temperature_2m,"
        "relative_humidity_2m,"
        "apparent_temperature,"
        "precipitation,"
        "weather_code,"
        "wind_speed_10m"
    )

    HOURLY_VARIABLES = (
        "temperature_2m,"
        "precipitation_probability"
    )

    def __init__(
        self,
        endpoint: str = DEFAULT_FORECAST_URL,
        timeout: int = 20,
        forecast_hours: int = 24,
    ) -> None:

        if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
            raise ValueError("A valid HTTPS endpoint is required.")

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive integer.")

        if (
            isinstance(forecast_hours, bool)
            or not isinstance(forecast_hours, int)
            or forecast_hours <= 0
        ):
            raise ValueError(
                "forecast_hours must be a positive integer."
            )

        self.endpoint = endpoint
        self.timeout = timeout
        self.forecast_hours = forecast_hours

    @staticmethod
    def _validate_coordinate(
        value: float,
        name: str,
        minimum: float,
        maximum: float,
    ) -> float:
        """Validate latitude or longitude."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if not minimum <= value <= maximum:
            raise ValueError(f"{name} is outside its valid range.")

        return value

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        """Convert a Unix timestamp into a timezone-aware UTC datetime."""

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ForecastProviderError(
                "Invalid timestamp returned by Open-Meteo."
            )

        try:
            return datetime.fromtimestamp(
                value,
                tz=timezone.utc,
            )
        except (OverflowError, OSError, ValueError) as exc:
            raise ForecastProviderError(
                "Unable to convert weather timestamp."
            ) from exc

    @staticmethod
    def _optional_number(
        value: object,
        field_name: str,
    ) -> float | None:
        """Validate an optional numeric weather value."""

        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ForecastProviderError(
                f"Invalid numeric value for {field_name}."
            )

        if not math.isfinite(value):
            raise ForecastProviderError(
                f"Non-finite value for {field_name}."
            )

        return float(value)

    @staticmethod
    def _required_mapping(
        data: dict,
        field_name: str,
    ) -> dict:
        """Get a required JSON object."""

        value = data.get(field_name)

        if not isinstance(value, dict):
            raise ForecastProviderError(
                f"Missing or invalid '{field_name}' data."
            )

        return value

    def _build_url(
        self,
        latitude: float,
        longitude: float,
    ) -> str:
        """Build the Open-Meteo forecast request URL."""

        parameters = {
            "latitude": latitude,
            "longitude": longitude,
            "current": self.CURRENT_VARIABLES,
            "hourly": self.HOURLY_VARIABLES,
            "forecast_hours": self.forecast_hours,
            "timezone": "auto",
            "timeformat": "unixtime",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "precipitation_unit": "mm",
        }

        return f"{self.endpoint}?{urlencode(parameters)}"

    def _download_json(self, url: str) -> dict:
        """Download and validate JSON from the API."""

        request = Request(
            url,
            headers={
                "User-Agent": "Explainable-Academic-AI/1.0",
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_data = response.read()

        except HTTPError as exc:
            raise ForecastProviderError(
                f"Open-Meteo returned HTTP {exc.code}."
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise ForecastProviderError(
                "Unable to connect to Open-Meteo."
            ) from exc

        try:
            data = json.loads(raw_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ForecastProviderError(
                "Open-Meteo returned invalid JSON."
            ) from exc

        if not isinstance(data, dict):
            raise ForecastProviderError(
                "Unexpected response format from Open-Meteo."
            )

        if data.get("error") is True:
            raise ForecastProviderError(
                str(data.get("reason", "Weather API request failed."))
            )

        return data

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
    ) -> WeatherForecast:
        """Fetch current conditions and hourly weather forecasts."""

        latitude = self._validate_coordinate(
            latitude,
            "latitude",
            -90,
            90,
        )

        longitude = self._validate_coordinate(
            longitude,
            "longitude",
            -180,
            180,
        )

        request_url = self._build_url(latitude, longitude)
        data = self._download_json(request_url)

        timezone_name = data.get("timezone")

        if not isinstance(timezone_name, str) or not timezone_name:
            raise ForecastProviderError(
                "Weather response does not contain a valid timezone."
            )

        current = self._required_mapping(data, "current")
        hourly = self._required_mapping(data, "hourly")

        current_time = self._parse_timestamp(
            current.get("time")
        )

        hourly_raw_times = hourly.get("time")
        hourly_raw_temperatures = hourly.get("temperature_2m")
        hourly_raw_precipitation = hourly.get(
            "precipitation_probability"
        )

        if not all(
            isinstance(values, list)
            for values in (
                hourly_raw_times,
                hourly_raw_temperatures,
                hourly_raw_precipitation,
            )
        ):
            raise ForecastProviderError(
                "Invalid hourly forecast arrays."
            )

        if not (
            len(hourly_raw_times)
            == len(hourly_raw_temperatures)
            == len(hourly_raw_precipitation)
        ):
            raise ForecastProviderError(
                "Hourly forecast arrays have different lengths."
            )

        hourly_times = tuple(
            self._parse_timestamp(value)
            for value in hourly_raw_times
        )

        hourly_temperatures = tuple(
            self._optional_number(value, "hourly temperature")
            for value in hourly_raw_temperatures
        )

        hourly_precipitation = tuple(
            self._optional_number(value, "precipitation probability")
            for value in hourly_raw_precipitation
        )

        weather_code = current.get("weather_code")

        if weather_code is not None:
            if (
                isinstance(weather_code, bool)
                or not isinstance(weather_code, int)
            ):
                raise ForecastProviderError(
                    "Invalid current weather code."
                )

        retrieved_at = datetime.now(timezone.utc)

        return WeatherForecast(
            latitude=latitude,
            longitude=longitude,
            timezone_name=timezone_name,
            source_url=request_url,
            retrieved_at=retrieved_at,
            current_time=current_time,
            temperature_c=self._optional_number(
                current.get("temperature_2m"),
                "temperature",
            ),
            relative_humidity_percent=self._optional_number(
                current.get("relative_humidity_2m"),
                "relative humidity",
            ),
            apparent_temperature_c=self._optional_number(
                current.get("apparent_temperature"),
                "apparent temperature",
            ),
            precipitation_mm=self._optional_number(
                current.get("precipitation"),
                "precipitation",
            ),
            weather_code=weather_code,
            wind_speed_kmh=self._optional_number(
                current.get("wind_speed_10m"),
                "wind speed",
            ),
            hourly_times=hourly_times,
            hourly_temperatures_c=hourly_temperatures,
            hourly_precipitation_probability_percent=(
                hourly_precipitation
            ),
        )
