"""Ground-truth source matching with a safe content-equivalence fallback.

Exact filename/page matching remains the primary criterion.  When a ground-truth
label points to a text snapshot but the retriever returns the equivalent PDF
(or vice versa), a conservative lexical comparison of the supplied expected
context and retrieved chunk text avoids a false mismatch caused only by names.
This is an evaluation helper; it does not change answer generation or evidence
validation thresholds.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_STOP_WORDS = {
    "the", "and", "for", "are", "was", "were", "with", "from", "that",
    "this", "there", "their", "have", "has", "had", "not", "but", "into",
    "onto", "about", "when", "where", "which", "what", "who", "will",
    "would", "should", "could", "can", "may", "must", "does", "did",
    "do", "how", "why", "its", "our", "your", "they", "them", "then",
    "than", "per", "all", "any", "each", "both", "also", "only", "such",
    "under", "over", "after", "before", "during", "between", "within",
    "into", "out", "not", "from", "based", "according", "generally",
}


def source_basename(value: Any) -> str:
    """Normalise Windows/POSIX paths to a case-folded basename."""
    return str(value or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()


def _normalise_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        token = token[:-3]
        if token.endswith("v"):
            token += "e"
        elif len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token
    if token.endswith("ed") and len(token) > 4:
        token = token[:-2]
        if len(token) > 2 and token[-1] == token[-2]:
            token = token[:-1]
        return token
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def meaningful_tokens(text: str) -> set[str]:
    return {
        _normalise_token(token)
        for token in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if token not in _STOP_WORDS and (len(token) >= 3 or token.isdigit())
    }


def _content_coverage(expected_text: str, actual_text: str) -> float:
    expected = meaningful_tokens(expected_text)
    if len(expected) < 5:
        return 0.0
    actual = meaningful_tokens(actual_text)
    return len(expected & actual) / len(expected)


def chunk_matches_expected_context(record: Any, expected: dict[str, Any]) -> bool:
    """True for exact source/page matches or a strong text-content match."""
    expected_name = source_basename(expected.get("document"))
    expected_pages = {int(page) for page in expected.get("pages", [])}
    actual_name = source_basename(getattr(record, "source", ""))
    actual_pages = {int(page) for page in (getattr(record, "page_numbers", ()) or ())}

    if expected_name and actual_name == expected_name:
        if not expected_pages or expected_pages.intersection(actual_pages):
            return True

    expected_text = str(expected.get("context") or "")
    if not expected_text:
        return False
    actual_text = " ".join(
        str(part or "")
        for part in (getattr(record, "heading", None), getattr(record, "text", ""))
    )
    return _content_coverage(expected_text, actual_text) >= 0.65


def retrieved_context_matches(retrieved: Iterable[Any], expected_contexts: list[dict[str, Any]]) -> tuple[int, int]:
    """Count expected contexts found in top-k retrieved chunks.

    A stricter merged-text fallback (75% expected-token recall) is used only
    when no single chunk carries enough of a context, to handle split passages.
    """
    items = list(retrieved)
    matched_count = 0
    for expected in expected_contexts:
        if any(chunk_matches_expected_context(item.record, expected) for item in items):
            matched_count += 1
            continue
        expected_text = str(expected.get("context") or "")
        combined = " ".join(
            " ".join((str(getattr(item.record, "heading", None) or ""), str(getattr(item.record, "text", "") or "")))
            for item in items
        )
        if expected_text and _content_coverage(expected_text, combined) >= 0.75:
            matched_count += 1
    return matched_count, len(expected_contexts)


def count_relevant_chunks(retrieved: Iterable[Any], expected_contexts: list[dict[str, Any]]) -> int:
    """Count retrieved chunks matching at least one gold context."""
    return sum(
        1
        for item in retrieved
        if any(chunk_matches_expected_context(item.record, context) for context in expected_contexts)
    )
