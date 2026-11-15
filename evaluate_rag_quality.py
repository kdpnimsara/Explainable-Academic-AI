
"""LLM-assisted quality evaluation for saved NSBM and SLIIT RAG answers.

Context relevance is measured against gold document/page annotations.
Groundedness and answer relevance are judged against gold supporting contexts
and the reference answer. LLM-as-a-judge scores are not objective proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.openai_compatible_model import OpenAICompatibleChatModel

DATASETS = ("nsbm", "sliit")


# ---------------------------------------------------------------------
# Input and scoring
# ---------------------------------------------------------------------

def context_relevance_score(item: dict[str, Any]) -> float | None:
    """Map gold source/page recall to a transparent 1-5 score."""
    if not item.get("expected_answerable", False):
        return None

    expected = int(item.get("expected_gold_contexts") or 0)
    matched = int(item.get("matched_gold_contexts") or 0)

    if expected <= 0:
        return None

    ratio = min(1.0, max(0.0, matched / expected))
    return round(1.0 + 4.0 * ratio, 2)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating surrounding Markdown code fences."""
    text = (text or "").strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if not match:
            raise ValueError("Judge response did not contain a JSON object.")

        value = json.loads(match.group(0))

    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object.")

    return value


def validate_judge_result(value: dict[str, Any]) -> dict[str, Any]:
    """Validate judge scores and keep concise textual rationales."""
    result: dict[str, Any] = {}

    for key in ("groundedness_score", "answer_relevance_score"):
        score = value.get(key)

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{key} must be a numeric score from 1 to 5.")

        if not 1 <= score <= 5 or float(score) != int(score):
            raise ValueError(f"{key} must be an integer from 1 to 5.")

        result[key] = int(score)

    for key in ("groundedness_reason", "answer_relevance_reason"):
        reason = value.get(key, "")

        if not isinstance(reason, str):
            raise ValueError(f"{key} must be text.")

        result[key] = reason.strip()[:1000]

    return result


# ---------------------------------------------------------------------
# LLM judge prompt
# ---------------------------------------------------------------------

def build_prompt(item: dict[str, Any]) -> str:
    """Create a bounded, evidence-only judging prompt."""
    contexts = item.get("supporting_contexts") or []

    context_text = "\n\n".join(
        f"[Gold context {index} | document={c.get('document')} "
        f"| pages={c.get('pages')}]\n{c.get('context', '')}"
        for index, c in enumerate(contexts, start=1)
    )

    if not context_text:
        context_text = (
            "[No gold supporting context: this question is unanswerable.]"
        )

    return f"""You are an independent evaluator of an academic-administration RAG system.
Evaluate ONLY the supplied question, reference answer, required facts, gold evidence,
and generated answer. Do not add outside facts. Do not reward confident unsupported claims.

QUESTION:
{item.get('question', '')}

EXPECTED ANSWERABLE:
{bool(item.get('expected_answerable'))}

REFERENCE ANSWER:
{item.get('expected_answer', '') or '[No reference answer: expected unanswerable]'}

REQUIRED FACTS:
{json.dumps(item.get('required_facts') or [], ensure_ascii=False)}

GOLD SUPPORTING EVIDENCE:
{context_text}

GENERATED SYSTEM ANSWER:
{item.get('generated_answer') or '[No answer generated]'}

Score both dimensions using integers 1-5:

Groundedness / Faithfulness:
5 = all material claims supported by supplied gold evidence; no material unsupported claims.
4 = substantially supported; only a minor omission or imprecision.
3 = mixed: some supported content but a notable unsupported claim or omission.
2 = mostly unsupported or materially inconsistent with evidence.
1 = fundamentally ungrounded, contradictory, or fabricates material facts.

For an unanswerable question, a clear abstention earns 5;
an unsupported substantive answer earns 1-2.

Answer Relevance:
5 = directly and completely addresses the question (or clearly abstains when unanswerable).
4 = relevant with a minor omission.
3 = partially addresses the question or includes notable irrelevant material.
2 = weakly related or misses the main request.
1 = does not answer the question.

Return ONLY one JSON object with exactly these keys:
{{
  "groundedness_score": 1,
  "groundedness_reason": "brief evidence-based explanation",
  "answer_relevance_score": 1,
  "answer_relevance_reason": "brief explanation"
}}
"""


# ---------------------------------------------------------------------
# Load source evaluation reports
# ---------------------------------------------------------------------

def load_input(dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = ROOT / "data" / "evaluation" / (
        f"{dataset}_batch_evaluation_report.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the existing {dataset.upper()} batch "
            "evaluator with --generate-answers first."
        )

    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report.get("question_results")

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} contains no question_results.")

    return report, rows


def make_base_row(
    dataset: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Create the standard evaluation output row."""
    return {
        "dataset": dataset.upper(),
        "id": item.get("id"),
        "question": item.get("question"),
        "expected_answerable": item.get("expected_answerable"),
        "context_relevance_score": context_relevance_score(item),
        "matched_gold_contexts": item.get("matched_gold_contexts"),
        "expected_gold_contexts": item.get("expected_gold_contexts"),
        "generated_answer": item.get("generated_answer"),
        "judge_status": "NOT_RUN",
    }


# ---------------------------------------------------------------------
# Retry handling
# ---------------------------------------------------------------------

def retry_wait_seconds(error: Exception, attempt: int) -> float:
    """Choose a wait duration, respecting Gemini retry-after messages."""
    message = str(error)

    match = re.search(
        r"retry in\s+([0-9.]+)\s*s",
        message,
        flags=re.IGNORECASE,
    )

    if match:
        return float(match.group(1)) + 1.0

    # Exponential fallback for temporary server/rate-limit errors.
    return min(60.0, 12.0 * (2 ** (attempt - 1)))


def is_retryable_error(error: Exception) -> bool:
    """Identify common temporary API errors."""
    message = str(error).lower()

    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "resource_exhausted",
        "rate limit",
        "rate_limit",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection error",
    )

    return any(marker in message for marker in retryable_markers)


def judge_with_retries(
    item: dict[str, Any],
    model: OpenAICompatibleChatModel,
    retries: int,
) -> dict[str, Any]:
    """Call the judge with bounded retries for temporary API failures."""
    prompt = build_prompt(item)

    for attempt in range(1, retries + 2):
        try:
            response = model.generate(prompt)
            judged = extract_json_object(response)
            return validate_judge_result(judged)

        except Exception as exc:
            if attempt > retries or not is_retryable_error(exc):
                raise

            wait = retry_wait_seconds(exc, attempt)

            print(
                f"  Temporary API error. Retry {attempt}/{retries} "
                f"after {wait:.1f} seconds."
            )

            time.sleep(wait)

    raise RuntimeError("Judge retry loop ended unexpectedly.")


# ---------------------------------------------------------------------
# Report output and checkpointing
# ---------------------------------------------------------------------

def calculate_metrics(
    dataset: str,
    rows: list[dict[str, Any]],
    source_created_at: Any,
) -> dict[str, Any]:
    """Calculate transparent dataset-level metrics."""
    completed = [
        row for row in rows
        if row.get("judge_status") == "COMPLETED"
    ]

    def average(key: str) -> float | None:
        values = [
            row[key]
            for row in completed
            if isinstance(row.get(key), int)
            and not isinstance(row.get(key), bool)
        ]

        return round(mean(values), 3) if values else None

    context_values = [
        row["context_relevance_score"]
        for row in rows
        if isinstance(row.get("context_relevance_score"), (int, float))
    ]

    return {
        "dataset": dataset.upper(),
        "questions_evaluated": len(rows),
        "judge_completed": len(completed),
        "judge_errors": sum(
            row.get("judge_status") == "ERROR" for row in rows
        ),
        "judge_skipped": sum(
            row.get("judge_status") == "SKIPPED_NO_GENERATED_ANSWER"
            for row in rows
        ),
        "judge_not_run": sum(
            row.get("judge_status") == "NOT_RUN" for row in rows
        ),
        "context_relevance_mean_1_to_5_answerable_only": (
            round(mean(context_values), 3) if context_values else None
        ),
        "groundedness_mean_1_to_5_judged_only": average(
            "groundedness_score"
        ),
        "answer_relevance_mean_1_to_5_judged_only": average(
            "answer_relevance_score"
        ),
        "context_relevance_definition": (
            "1 + 4 * (matched gold document/page contexts / "
            "expected gold contexts); unanswerable questions excluded."
        ),
        "judge_definition": (
            "OpenAI-compatible LLM-as-a-judge using the reference answer, "
            "required facts, gold supporting contexts and generated answer."
        ),
        "source_batch_report_created_at": source_created_at,
    }


def save_reports(
    output_dir: Path,
    all_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    judge_model: str,
) -> None:
    """Write JSON and CSV reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "evaluation_type": "LLM_AS_A_JUDGE_WITH_GOLD_CONTEXTS",
        "judge_model": judge_model,
        "rubric_version": "1.0",
        "limitations": [
            "LLM-judge scores are model-based assessments, not objective proof of correctness.",
            "The same configured model may be used for generation and judging, creating self-evaluation bias.",
            "Context relevance is a deterministic gold document/page match score, not a semantic assessment of every retrieved chunk.",
            "Human review and source-version verification remain necessary.",
            "Only saved generated answers are judged; this script does not regenerate answers.",
            "Temporary API failures may be retried, but final ERROR records remain visible.",
        ],
        "metrics": summaries,
        "question_results": all_rows,
    }

    json_path = output_dir / "rag_quality_judgement_report.json"
    csv_path = output_dir / "rag_quality_judgement_report.csv"

    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = sorted({
        key
        for row in all_rows
        for key in row
    })

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in all_rows:
            writer.writerow({
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            })


# ---------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------

def evaluate_dataset(
    dataset: str,
    source_rows: list[dict[str, Any]],
    model: OpenAICompatibleChatModel,
    previous_results: dict[tuple[str, str], dict[str, Any]],
    resume: bool,
    delay: float,
    retries: int,
    checkpoint: Callable[[], None],
) -> list[dict[str, Any]]:
    """Evaluate one dataset while preserving completed results."""
    output: list[dict[str, Any]] = []

    for index, item in enumerate(source_rows, start=1):
        question_id = str(item.get("id") or "")
        key = (dataset.upper(), question_id)

        previous = previous_results.get(key)

        # Preserve previously completed results without another API call.
        if (
            resume
            and previous
            and previous.get("judge_status") == "COMPLETED"
        ):
            row = dict(previous)
            output.append(row)

            print(
                f"{dataset.upper()} {index}/{len(source_rows)} "
                f"{question_id}: PRESERVED"
            )
            continue

        row = make_base_row(dataset, item)

        if item.get("generated_answer_error") or not item.get("generated_answer"):
            row["judge_status"] = "SKIPPED_NO_GENERATED_ANSWER"
            row["judge_error"] = (
                item.get("generated_answer_error")
                or "No generated answer."
            )

        else:
            try:
                judged = judge_with_retries(
                    item=item,
                    model=model,
                    retries=retries,
                )

                row.update(judged)
                row["judge_status"] = "COMPLETED"

            except Exception as exc:
                row["judge_status"] = "ERROR"
                row["judge_error"] = f"{type(exc).__name__}: {exc}"

        output.append(row)
        previous_results[key] = row

        print(
            f"{dataset.upper()} {index}/{len(source_rows)} "
            f"{question_id}: {row['judge_status']}"
        )

        # Save progress after every processed question.
        checkpoint()

        # Apply delay between question requests.
        if delay > 0 and index < len(source_rows):
            time.sleep(delay)

    return output


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved NSBM/SLIIT answers with an LLM judge, "
            "supporting resume, retry and checkpoints."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="Datasets to evaluate.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset limit for a smoke run.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds between questions (recommended: 6).",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries for temporary API failures (default: 2).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve COMPLETED rows from the existing JSON report.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "evaluation",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    if args.delay < 0:
        parser.error("--delay cannot be negative.")

    if args.retries < 0:
        parser.error("--retries cannot be negative.")

    output_dir = args.output_dir
    json_path = output_dir / "rag_quality_judgement_report.json"

    # Load previous report before overwriting it.
    previous_results: dict[
        tuple[str, str], dict[str, Any]
    ] = {}

    if args.resume and json_path.exists():
        old_report = json.loads(
            json_path.read_text(encoding="utf-8")
        )

        for row in old_report.get("question_results", []):
            key = (
                str(row.get("dataset", "")).upper(),
                str(row.get("id", "")),
            )

            if key[0] and key[1]:
                previous_results[key] = row

        print(
            f"Resume enabled: loaded {len(previous_results)} "
            "previous result rows."
        )

    elif args.resume:
        print(
            "Resume report not found. A new evaluation will be started."
        )

    judge_model = os.getenv("LLM_JUDGE_MODEL") or None

    model = OpenAICompatibleChatModel(
        model_name=judge_model,
        temperature=0.0,
    )

    model_label = judge_model or "LLM_MODEL from local environment"

    # Load source reports before starting evaluation.
    source_data: dict[
        str, tuple[dict[str, Any], list[dict[str, Any]]]
    ] = {}

    for dataset in args.datasets:
        source_data[dataset] = load_input(dataset)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    # The checkpoint always writes the latest results accumulated so far.
    current_rows: dict[tuple[str, str], dict[str, Any]] = dict(
        previous_results
    )

    def checkpoint() -> None:
        ordered_rows = [
            current_rows[key]
            for key in sorted(current_rows)
        ]

        current_summaries = []

        for dataset in args.datasets:
            dataset_rows = [
                row for row in ordered_rows
                if row.get("dataset") == dataset.upper()
            ]

            source_report = source_data[dataset][0]

            current_summaries.append(
                calculate_metrics(
                    dataset,
                    dataset_rows,
                    source_report.get("created_at"),
                )
            )

        save_reports(
            output_dir=output_dir,
            all_rows=ordered_rows,
            summaries=current_summaries,
            judge_model=model_label,
        )

    for dataset in args.datasets:
        source_report, source_rows = source_data[dataset]

        selected_rows = (
            source_rows[:args.limit]
            if args.limit is not None
            else source_rows
        )

        print(f"\n{'=' * 60}")
        print(f"Evaluating {dataset.upper()}: {len(selected_rows)} questions")
        print(f"{'=' * 60}")

        evaluated_rows = evaluate_dataset(
            dataset=dataset,
            source_rows=selected_rows,
            model=model,
            previous_results=current_rows,
            resume=args.resume,
            delay=args.delay,
            retries=args.retries,
            checkpoint=checkpoint,
        )

        for row in evaluated_rows:
            current_rows[
                (dataset.upper(), str(row.get("id") or ""))
            ] = row

        dataset_metrics = calculate_metrics(
            dataset,
            evaluated_rows,
            source_report.get("created_at"),
        )

        summaries.append(dataset_metrics)
        all_rows.extend(evaluated_rows)

        # Save after completing each dataset.
        checkpoint()

    # Final report contains all accumulated results.
    final_rows = [
        current_rows[key]
        for key in sorted(current_rows)
    ]

    final_summaries = []

    for dataset in args.datasets:
        dataset_rows = [
            row for row in final_rows
            if row.get("dataset") == dataset.upper()
        ]

        final_summaries.append(
            calculate_metrics(
                dataset,
                dataset_rows,
                source_data[dataset][0].get("created_at"),
            )
        )

    save_reports(
        output_dir=output_dir,
        all_rows=final_rows,
        summaries=final_summaries,
        judge_model=model_label,
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for summary in final_summaries:
        print(json.dumps(summary, indent=2))

    print(f"\nJSON: {json_path}")
    print(
        f"CSV:  {output_dir / 'rag_quality_judgement_report.csv'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
