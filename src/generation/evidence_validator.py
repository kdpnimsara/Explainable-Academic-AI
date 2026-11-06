"""Validate whether retrieved evidence is relevant and sufficient.

Includes a conservative policy gate for official institutional
decisions such as examination cancellations and postponements.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.generation.answer_generator import (
    AnswerGenerator,
    GeneratedAnswer,
)
from src.retrieval.hybrid_retriever import RetrievedChunk
from src.weather.evidence import WeatherEvidence


@dataclass(frozen=True, slots=True)
class EvidenceValidationResult:
    """Explain the result of an evidence relevance check."""

    sufficient: bool
    coverage: float
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    reason: str


class EvidenceValidator:
    """Validate query-term coverage and evidence sufficiency."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "before",
        "by", "can", "could", "did", "do", "does", "for",
        "from", "has", "have", "how", "in", "is", "it",
        "me", "of", "on", "or", "please", "should", "the",
        "their", "there", "this", "that", "to", "was",
        "were", "what", "when", "where", "which", "who",
        "why", "will", "with", "would", "tell", "show",
        "give", "find", "long", "many", "much", "exact",
        "amount", "because",
    }

    SYNONYMS = {
        "exam": "examination",
        "rule": "regulation",
        "fee": "fine",
    }

    DECISION_ACTION_PATTERN = re.compile(
        r"\b(?:cancel(?:led|ed|lation)?|"
        r"postpon(?:ed|ement)?|"
        r"reschedul(?:ed|ing)?|"
        r"suspend(?:ed|sion)?|"
        r"clos(?:ed|ure))\b",
        re.IGNORECASE,
    )

    EDUCATION_PATTERN = re.compile(
        r"\b(?:exam(?:ination)?s?|lectures?|classes?|"
        r"university|universities|institution|academic)\b",
        re.IGNORECASE,
    )

    STATUS_PATTERN = re.compile(
        r"\b(?:has|have|is|are|was|were|will|did|do|does|can)\b",
        re.IGNORECASE,
    )

    CURRENT_TIME_TERMS = {
        "today",
        "now",
        "currently",
        "current",
        "latest",
    }

    WEATHER_CAUSE_PATTERN = re.compile(
        r"\b(?:rain|rainfall|flood|flooding|storm|cyclone|"
        r"landslide|weather|wind)\b",
        re.IGNORECASE,
    )

    AUTHORITY_TERMS = (
        "academic governing council",
        "examinations unit",
        "examination registrar",
        "registrar",
        "vice chancellor",
        "vice-chancellor",
        "university administration",
        "official university announcement",
        "official university notice",
    )

    SUPPORTED_INSTITUTIONS = {
        "NSBM",
        "SLIIT",
        "UOM",
        "UOC",
        "USJ",
    }

    def __init__(
        self,
        min_coverage: float = 0.35,
        min_matched_terms: int = 2,
    ) -> None:
        if (
            isinstance(min_coverage, bool)
            or not isinstance(min_coverage, (int, float))
        ):
            raise TypeError("min_coverage must be a number.")

        if not 0 < min_coverage <= 1:
            raise ValueError("min_coverage must be between 0 and 1.")

        if (
            isinstance(min_matched_terms, bool)
            or not isinstance(min_matched_terms, int)
        ):
            raise TypeError("min_matched_terms must be an integer.")

        if min_matched_terms <= 0:
            raise ValueError("min_matched_terms must be positive.")

        self.min_coverage = float(min_coverage)
        self.min_matched_terms = min_matched_terms

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        """Apply lightweight token normalization."""

        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"

        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]

            if token.endswith("v"):
                token += "e"
            elif len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]

        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]

            if len(token) > 2 and token[-1] == token[-2]:
                token = token[:-1]

        elif (
            token.endswith("s")
            and len(token) > 3
            and not token.endswith("ss")
        ):
            token = token[:-1]

        return cls.SYNONYMS.get(token, token)

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        """Extract meaningful normalized terms."""

        raw_tokens = re.findall(r"[a-z0-9]+", text.lower())

        return {
            cls._normalize_token(token)
            for token in raw_tokens
            if token not in cls.STOP_WORDS
        }

    @staticmethod
    def _document_text(chunk: RetrievedChunk) -> str:
        """Collect document text and metadata."""

        record = chunk.record

        parts = [
            record.text,
            record.source,
        ]

        if record.heading:
            parts.append(record.heading)

        metadata = record.document_metadata

        if metadata is not None:
            parts.extend([
                metadata.document_title,
                metadata.document_category,
            ])

            if metadata.institution_id:
                parts.append(metadata.institution_id)

            if metadata.document_version:
                parts.append(metadata.document_version)

            if metadata.source_authority:
                parts.append(metadata.source_authority)

            if metadata.source_url:
                parts.append(metadata.source_url)

        return " ".join(parts)

    @staticmethod
    def _weather_text(item: WeatherEvidence) -> str:
        """Collect current weather evidence."""

        if not item.is_current():
            return ""

        return " ".join([
            item.content,
            item.source_name,
            item.evidence_type,
        ])

    @classmethod
    def _requires_official_decision_confirmation(
        cls,
        question: str,
    ) -> bool:
        """Identify questions asking about an institutional decision."""

        return (
            bool(cls.DECISION_ACTION_PATTERN.search(question))
            and bool(cls.EDUCATION_PATTERN.search(question))
            and bool(cls.STATUS_PATTERN.search(question))
        )

    @staticmethod
    def _contains_today_date(text: str) -> bool:
        """Check whether evidence explicitly contains today's date."""

        today = date.today()

        date_formats = (
            today.isoformat(),
            today.strftime("%d/%m/%Y"),
            today.strftime("%d-%m-%Y"),
            today.strftime("%m/%d/%Y"),
            today.strftime("%B %d, %Y"),
            today.strftime("%b %d, %Y"),
            today.strftime("%d %B %Y"),
            today.strftime("%d %b %Y"),
            f"{today.day} {today.strftime('%B')} {today.year}",
        )

        normalized_text = text.lower()

        return any(
            date_marker.lower() in normalized_text
            for date_marker in date_formats
        )

    def _has_authorized_decision_evidence(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> bool:
        """Require an authoritative, explicit and suitably dated decision."""

        question_lower = question.lower()

        requested_institutions = (
            set(
                re.findall(
                    r"\b(?:NSBM|SLIIT|UOM|UOC|USJ)\b",
                    question,
                    re.IGNORECASE,
                )
            )
        )

        requires_current_date = any(
            term in question_lower
            for term in self.CURRENT_TIME_TERMS
        )

        requires_weather_cause = bool(
            self.WEATHER_CAUSE_PATTERN.search(question)
        )

        for chunk in retrieved_chunks:

            metadata = chunk.record.document_metadata

            if metadata is None:
                continue

            if metadata.document_scope != "institution":
                continue

            if not metadata.institution_id:
                continue

            if (
                requested_institutions
                and metadata.institution_id.upper()
                not in requested_institutions
            ):
                continue

            evidence_text = self._document_text(chunk)

            if not self.DECISION_ACTION_PATTERN.search(evidence_text):
                continue

            if not self.EDUCATION_PATTERN.search(evidence_text):
                continue

            authority = metadata.source_authority

            authority_is_identifiable = bool(
                authority and authority.strip()
            ) or any(
                term in evidence_text.lower()
                for term in self.AUTHORITY_TERMS
            )

            if not authority_is_identifiable:
                continue

            if (
                requires_weather_cause
                and not self.WEATHER_CAUSE_PATTERN.search(evidence_text)
            ):
                continue

            if (
                requires_current_date
                and not self._contains_today_date(evidence_text)
            ):
                continue

            return True

        return False

    def validate(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk] = (),
        weather_evidence: Sequence[WeatherEvidence] = (),
    ) -> EvidenceValidationResult:
        """Check whether evidence adequately supports the question."""

        if not isinstance(question, str):
            raise TypeError("question must be a string.")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        if (
            not isinstance(retrieved_chunks, Sequence)
            or isinstance(retrieved_chunks, (str, bytes))
        ):
            raise TypeError("retrieved_chunks must be a sequence.")

        if (
            not isinstance(weather_evidence, Sequence)
            or isinstance(weather_evidence, (str, bytes))
        ):
            raise TypeError("weather_evidence must be a sequence.")

        evidence_parts = []

        for chunk in retrieved_chunks:
            if not isinstance(chunk, RetrievedChunk):
                raise TypeError("Invalid retrieved chunk.")

            evidence_parts.append(self._document_text(chunk))

        for item in weather_evidence:
            if not isinstance(item, WeatherEvidence):
                raise TypeError("Invalid weather evidence.")

            weather_text = self._weather_text(item)

            if weather_text:
                evidence_parts.append(weather_text)

        query_terms = self._tokens(question)

        if not query_terms:
            return EvidenceValidationResult(
                sufficient=False,
                coverage=0.0,
                matched_terms=(),
                missing_terms=(),
                reason="The question contains no identifiable key terms.",
            )

        if not evidence_parts:
            return EvidenceValidationResult(
                sufficient=False,
                coverage=0.0,
                matched_terms=(),
                missing_terms=tuple(sorted(query_terms)),
                reason="No usable supporting evidence was retrieved.",
            )

        evidence_terms = self._tokens(
            " ".join(evidence_parts)
        )

        matched_terms = tuple(
            sorted(query_terms & evidence_terms)
        )

        missing_terms = tuple(
            sorted(query_terms - evidence_terms)
        )

        coverage = len(matched_terms) / len(query_terms)

        # Official university decisions require authoritative evidence.
        # A weather forecast alone must never confirm a cancellation.
        if self._requires_official_decision_confirmation(question):

            if not self._has_authorized_decision_evidence(
                question,
                retrieved_chunks,
            ):
                return EvidenceValidationResult(
                    sufficient=False,
                    coverage=coverage,
                    matched_terms=matched_terms,
                    missing_terms=missing_terms,
                    reason=(
                        "No current, authoritative institutional "
                        "document explicitly confirms the requested "
                        "educational decision and its supporting conditions."
                    ),
                )

        requested_years = set(
            re.findall(r"\b(?:19|20)\d{2}\b", question)
        )

        available_years = set(
            re.findall(
                r"\b(?:19|20)\d{2}\b",
                " ".join(evidence_parts),
            )
        )

        missing_years = requested_years - available_years

        required_matches = min(
            self.min_matched_terms,
            len(query_terms),
        )

        sufficient = (
            len(matched_terms) >= required_matches
            and coverage >= self.min_coverage
            and not missing_years
        )

        if missing_years:
            reason = (
                "The requested year is not confirmed by the evidence: "
                + ", ".join(sorted(missing_years))
            )

        elif len(matched_terms) < required_matches:
            reason = (
                "Too few important question terms match the evidence."
            )

        elif coverage < self.min_coverage:
            reason = (
                "The evidence has insufficient question-term coverage."
            )

        else:
            reason = (
                "The evidence passed the initial relevance check."
            )

        return EvidenceValidationResult(
            sufficient=sufficient,
            coverage=coverage,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            reason=reason,
        )


class EvidenceValidatingGenerator:
    """Protect an existing generator with a pre-generation evidence gate."""

    def __init__(
        self,
        generator: Any,
        validator: EvidenceValidator | None = None,
    ) -> None:
        if not callable(getattr(generator, "generate_answer", None)):
            raise TypeError(
                "generator must provide generate_answer()."
            )

        self.generator = generator
        self.validator = validator or EvidenceValidator()

    @staticmethod
    def _abstain() -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=AnswerGenerator.ABSTENTION_MESSAGE,
            sources=(),
            external_sources=(),
        )

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> GeneratedAnswer:
        """Validate academic evidence before generation."""

        result = self.validator.validate(
            question,
            retrieved_chunks=retrieved_chunks,
        )

        if not result.sufficient:
            return self._abstain()

        return self.generator.generate_answer(
            question,
            retrieved_chunks,
        )

    def generate_answer_with_weather(
        self,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        weather_evidence: Sequence[WeatherEvidence],
    ) -> GeneratedAnswer:
        """Validate evidence and preserve safe weather-only answers.

        A weather forecast may be shown as context for an
        institution-related weather question. It must never be treated
        as proof that the institution has made an official decision.
        """

        method = getattr(
            self.generator,
            "generate_answer_with_weather",
            None,
        )

        if not callable(method):
            raise TypeError(
                "generator must provide generate_answer_with_weather()."
            )

        result = self.validator.validate(
            question,
            retrieved_chunks=retrieved_chunks,
            weather_evidence=weather_evidence,
        )

        is_weather_related_decision = (
            self.validator._requires_official_decision_confirmation(
                question
            )
            and bool(
                self.validator.WEATHER_CAUSE_PATTERN.search(question)
            )
        )

        current_weather_evidence = tuple(
            item
            for item in weather_evidence
            if isinstance(item, WeatherEvidence)
            and item.is_current()
        )

        # The official decision safeguard remains active.
        # However, current weather evidence can still be shown separately.
        if not result.sufficient:
            if (
                is_weather_related_decision
                and current_weather_evidence
            ):
                return method(
                    question,
                    (),
                    current_weather_evidence,
                )

            return self._abstain()

        return method(
            question,
            retrieved_chunks,
            weather_evidence,
        )