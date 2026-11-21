
"""Resolve supported Sri Lankan locations into coordinates."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeatherLocation:
    """Represent a supported weather lookup location."""

    name: str
    latitude: float
    longitude: float
    district: str


LOCATION_CATALOG = {
    "colombo": WeatherLocation(
        "Colombo", 6.9271, 79.8612, "Colombo"
    ),
    "kandy": WeatherLocation(
        "Kandy", 7.2906, 80.6337, "Kandy"
    ),
    "galle": WeatherLocation(
        "Galle", 6.0535, 80.2210, "Galle"
    ),
    "jaffna": WeatherLocation(
        "Jaffna", 9.6615, 80.0255, "Jaffna"
    ),
    "negombo": WeatherLocation(
        "Negombo", 7.2083, 79.8358, "Gampaha"
    ),
    "matara": WeatherLocation(
        "Matara", 5.9549, 80.5550, "Matara"
    ),
    "nuwara eliya": WeatherLocation(
        "Nuwara Eliya", 6.9497, 80.7891, "Nuwara Eliya"
    ),
    "anuradhapura": WeatherLocation(
        "Anuradhapura", 8.3114, 80.4037, "Anuradhapura"
    ),
    "trincomalee": WeatherLocation(
        "Trincomalee", 8.5874, 81.2152, "Trincomalee"
    ),
    "batticaloa": WeatherLocation(
        "Batticaloa", 7.7310, 81.6747, "Batticaloa"
    ),
    "kurunegala": WeatherLocation(
        "Kurunegala", 7.4863, 80.3623, "Kurunegala"
    ),
    "ratnapura": WeatherLocation(
        "Ratnapura", 6.7056, 80.3847, "Ratnapura"
    ),
    "badulla": WeatherLocation(
        "Badulla", 6.9934, 81.0550, "Badulla"
    ),
    "hambantota": WeatherLocation(
        "Hambantota", 6.1429, 81.1212, "Hambantota"
    ),
    "puttalam": WeatherLocation(
        "Puttalam", 8.0362, 79.8283, "Puttalam"
    ),
    "kalutara": WeatherLocation(
        "Kalutara", 6.5854, 79.9607, "Kalutara"
    ),
    "ella": WeatherLocation(
        "Ella", 6.8667, 81.0466, "Badulla"
    ),
    "sigiriya": WeatherLocation(
        "Sigiriya", 7.9570, 80.7603, "Matale"
    ),
}


class LocationResolver:
    """Resolve explicitly named supported locations without guessing."""

    def resolve(
        self,
        location_name: str | None,
    ) -> WeatherLocation | None:
        """Resolve an exact supported location name."""

        if location_name is None:
            return None

        if not isinstance(location_name, str):
            raise TypeError(
                "location_name must be a string or None."
            )

        normalized = " ".join(location_name.casefold().split())

        if not normalized:
            return None

        return LOCATION_CATALOG.get(normalized)

    def resolve_from_question(
        self,
        question: str,
    ) -> WeatherLocation | None:
        """Resolve a question only when exactly one known location appears."""

        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        matches = []

        for key, location in LOCATION_CATALOG.items():
            pattern = rf"(?<!\w){re.escape(key)}(?!\w)"

            if re.search(pattern, question, re.IGNORECASE):
                matches.append(location)

        if len(matches) != 1:
            return None

        return matches[0]
