
"""Explainable tool selection for the academic AI agent."""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolSelection:
    """Explainable record of the tools selected for a question."""

    intent: str
    selected_tools: tuple[str, ...]
    rationale: str
    requires_weather: bool
    requires_academic_documents: bool
    location_name: str | None = None


class ExplainableToolSelector:
    """Select academic and weather tools based on query intent."""

    WEATHER_PATTERN = re.compile(
        r"\b("
        r"weather|forecast|temperature|rain|rainy|rainfall|"
        r"humidity|wind|windy|lightning|thunderstorm|"
        r"precipitation|flood warning|weather warning|"
        r"climate"
        r")\b",
        re.IGNORECASE,
    )

    ACADEMIC_PATTERN = re.compile(
        r"\b("
        r"exam|exams|examination|examinations|"
        r"academic|university|universities|institution|"
        r"regulation|regulations|policy|policies|"
        r"semester|lecture|lectures|class|classes|"
        r"campus|attendance|assessment|assessments|"
        r"postpone|postponed|postponement|"
        r"reschedule|cancel|cancelled|cancellation|"
        r"closure|deadline|assignment"
        r")\b",
        re.IGNORECASE,
    )

    KNOWN_LOCATIONS = (
        "Colombo",
        "Kandy",
        "Galle",
        "Jaffna",
        "Negombo",
        "Matara",
        "Nuwara Eliya",
        "Anuradhapura",
        "Trincomalee",
        "Batticaloa",
        "Kurunegala",
        "Ratnapura",
        "Badulla",
        "Hambantota",
        "Puttalam",
        "Kalutara",
        "Ella",
        "Sigiriya",
    )

    def select(
        self,
        question: str,
        location_name: str | None = None,
    ) -> ToolSelection:
        """Identify intent and select appropriate tools."""

        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

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

        normalized = question.casefold()

        requires_weather = bool(
            self.WEATHER_PATTERN.search(normalized)
        )

        requires_academic = bool(
            self.ACADEMIC_PATTERN.search(normalized)
        )

        if requires_weather and requires_academic:
            intent = "weather_academic"
            tools = (
                "weather_information",
                "academic_retrieval",
            )
            rationale = (
                "The question involves both weather conditions "
                "and academic administration. Weather information "
                "and academic document retrieval are selected."
            )

        elif requires_weather:
            intent = "weather"
            tools = ("weather_information",)
            rationale = (
                "The question requests weather information. "
                "The weather information tool is selected."
            )

        else:
            # Preserve the original academic retrieval behaviour
            # for questions without detected weather intent.
            intent = "academic"
            tools = ("academic_retrieval",)
            rationale = (
                "No weather requirement was detected. "
                "Academic document retrieval is selected."
            )

        if location_name is None and requires_weather:
            matches = [
                place
                for place in self.KNOWN_LOCATIONS
                if re.search(
                    rf"\b{re.escape(place)}\b",
                    question,
                    re.IGNORECASE,
                )
            ]

            # Do not guess when the question contains multiple places.
            if len(matches) == 1:
                location_name = matches[0]

        return ToolSelection(
            intent=intent,
            selected_tools=tools,
            rationale=rationale,
            requires_weather=requires_weather,
            requires_academic_documents=requires_academic,
            location_name=location_name,
        )
