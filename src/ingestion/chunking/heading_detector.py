import re
from dataclasses import dataclass
from typing import Literal


HeadingKind = Literal[
    "section",
    "subsection",
]


@dataclass(frozen=True, slots=True)
class HeadingMatch:
    """Store information about a detected document heading."""

    heading: str
    level: int
    kind: HeadingKind
    line_number: int


class HeadingDetector:
    """Detect structured headings in disaster management documents."""

    MAX_HEADING_LENGTH = 160

    SECTION_PATTERN = re.compile(
        r"^(?P<number>\d+(?:\.\d+)*)"
        r"(?P<delimiter>[.)]?)"
        r"\s+(?P<title>[A-Z].*)$"
    )

    # Table of contents entries often contain a trailing
    # printed document page number.
    TOC_PAGE_SUFFIX_PATTERN = re.compile(
        r"(?:\s+-\s*\d+\s*-|\s+\d{1,3})$"
    )

    # These phrases commonly introduce numbered lists.
    LIST_CONTEXT_PATTERNS = (
        re.compile(
            r"\bnot limited to\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:are|is)\s+to\s*[;:]",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bas stated below\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bas follows\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\blisted below\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bfollowing\s+"
            r"(?:strategies|objectives|goals|items|"
            r"activities|measures|steps)\b",
            re.IGNORECASE,
        ),
    )

    @staticmethod
    def _compact(text: str) -> str:
        """Normalize whitespace in a detected heading."""

        return " ".join(text.split())

    def _is_numbered_list_context(
        self,
        preceding_text: str,
    ) -> bool:
        """Check whether preceding text introduces a numbered list."""

        recent_context = preceding_text[-2000:]

        return any(
            pattern.search(recent_context)
            for pattern in self.LIST_CONTEXT_PATTERNS
        )

    def _is_table_of_contents_entry(
        self,
        candidate: str,
        section_number: str,
    ) -> bool:
        """Identify numbered headings with trailing page references."""

        if "." not in section_number:
            return False

        return bool(
            self.TOC_PAGE_SUFFIX_PATTERN.search(candidate)
        )

    def detect(
        self,
        line: str,
        line_number: int = 1,
        preceding_text: str = "",
    ) -> HeadingMatch | None:
        """
        Detect structured numbered headings.

        Preceding text is used to distinguish section headings
        from numbered list items.
        """

        if not isinstance(line, str):
            raise TypeError(
                "Heading line must be a string."
            )

        if not isinstance(preceding_text, str):
            raise TypeError(
                "preceding_text must be a string."
            )

        if (
            isinstance(line_number, bool)
            or not isinstance(line_number, int)
            or line_number < 1
        ):
            raise ValueError(
                "line_number must be a positive integer."
            )

        candidate = line.strip()

        if not candidate:
            return None

        if len(candidate) > self.MAX_HEADING_LENGTH:
            return None

        section_match = self.SECTION_PATTERN.fullmatch(
            candidate
        )

        if section_match is None:
            return None

        section_number = section_match.group(
            "number"
        )

        delimiter = section_match.group(
            "delimiter"
        )

        # A single-level heading must have an explicit delimiter.
        if "." not in section_number and not delimiter:
            return None

        # Exclude table-of-contents entries that contain
        # printed page numbers at the end.
        if self._is_table_of_contents_entry(
            candidate,
            section_number,
        ):
            return None

        # Single-level numbered entries may be list items.
        # Nested section headings are still recognised.
        if (
            "." not in section_number
            and self._is_numbered_list_context(preceding_text)
        ):
            return None

        heading = self._compact(candidate)

        level = section_number.count(".") + 1

        return HeadingMatch(
            heading=heading,
            level=level,
            kind="section",
            line_number=line_number,
        )