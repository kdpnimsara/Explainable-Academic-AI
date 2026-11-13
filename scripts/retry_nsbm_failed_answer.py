
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_nsbm_rag

REPORT = ROOT / "data/evaluation/nsbm_batch_evaluation_report.json"
DATASET = ROOT / "data/evaluation/nsbm_ground_truth.json"

report = json.loads(REPORT.read_text(encoding="utf-8"))
dataset = json.loads(DATASET.read_text(encoding="utf-8"))

entry = next(r for r in report["question_results"] if r["id"] == "NSBM028")
question = next(q for q in dataset["questions"] if q["id"] == "NSBM028")

if not entry.get("generated_answer_error"):
    print("No failed answer found. Nothing to retry.")
    raise SystemExit(0)

run_nsbm_rag.HISTORY_PATH = ROOT / "data/history/nsbm_batch_evaluation_history.jsonl"
agent = run_nsbm_rag.build_agent()

response = None
for attempt in range(4):
    try:
        response = agent.ask(
            question=question["question"],
            institution_id="NSBM"
        )
        break
    except Exception as exc:
        message = str(exc)
        if not any(x in message for x in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")) or attempt == 3:
            raise
        wait = 30 * (attempt + 1)
        print(f"Temporary API error. Waiting {wait}s before retry.")
        time.sleep(wait)

if response is None or not str(getattr(response, "answer", "") or "").strip():
    raise RuntimeError("No answer received. Original report remains unchanged.")

answer = str(response.answer)
entry["generated_answer"] = answer
entry["generated_citations"] = [
    {
        "citation_id": getattr(s, "citation_id", None),
        "document": getattr(s, "document", None),
        "page_numbers": list(getattr(s, "page_numbers", ()) or ())
    }
    for s in (getattr(response, "sources", ()) or ())
]
entry["generated_answer_error"] = None

from evaluate_nsbm_batch import fact_coverage, ABSTENTION_MARKERS

coverage, covered, total = fact_coverage(question.get("required_facts", []), answer)
entry["answer_fact_coverage_proxy"] = coverage
entry["answer_facts_covered_proxy"] = covered
entry["answer_facts_total"] = total
entry["generated_answer_abstained_proxy"] = any(
    marker in answer.casefold() for marker in ABSTENTION_MARKERS
)

runs = [r for r in report["question_results"] if "generated_answer" in r]
errors = sum(bool(r.get("generated_answer_error")) for r in runs)
answerable = [r for r in runs if r.get("expected_answerable")]
coverages = [
    r["answer_fact_coverage_proxy"] for r in answerable
    if r.get("answer_fact_coverage_proxy") is not None
]
unsupported = [r for r in runs if not r.get("expected_answerable")]
abstained = sum(
    bool(r.get("generated_answer_abstained_proxy"))
    for r in unsupported if not r.get("generated_answer_error")
)

metrics = report["metrics"]
metrics["answer_generation"] = "COMPLETED_WITH_ERRORS" if errors else "COMPLETED"
metrics["generated_answer_count"] = len(runs) - errors
metrics["generated_answer_errors"] = errors
metrics["answer_fact_coverage_proxy_mean"] = (
    round(sum(coverages) / len(coverages), 4) if coverages else None
)
metrics["unanswerable_abstention_marker_rate_proxy"] = (
    round(abstained / (len(unsupported) - errors), 4)
    if len(unsupported) > errors else None
)

import shutil
from datetime import datetime

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy2(REPORT, REPORT.with_name(f"{REPORT.stem}_backup_{stamp}.json"))

csv_path = REPORT.with_suffix(".csv")
if csv_path.exists():
    shutil.copy2(csv_path, csv_path.with_name(f"{csv_path.stem}_backup_{stamp}.csv"))

report["created_at"] = datetime.now().astimezone().isoformat()
REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

import csv

fields = [
    "id", "category", "question", "expected_answerable", "predicted_answerable",
    "status", "validation_coverage", "validation_reason", "matched_gold_contexts",
    "expected_gold_contexts", "context_hit", "supporting_fact_coverage_proxy",
    "retrieved_sources", "expected_answer", "generated_answer", "generated_citations",
    "answer_fact_coverage_proxy", "generated_answer_abstained_proxy", "generated_answer_error"
]

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in report["question_results"]:
        writer.writerow({
            "id": r.get("id"),
            "category": r.get("category"),
            "question": r.get("question"),
            "expected_answerable": r.get("expected_answerable"),
            "predicted_answerable": r.get("predicted_answerable"),
            "status": r.get("status"),
            "validation_coverage": r.get("validation_coverage"),
            "validation_reason": r.get("validation_reason"),
            "matched_gold_contexts": r.get("matched_gold_contexts"),
            "expected_gold_contexts": r.get("expected_gold_contexts"),
            "context_hit": r.get("context_hit"),
            "supporting_fact_coverage_proxy": r.get("supporting_fact_coverage_proxy"),
            "retrieved_sources": json.dumps(r.get("retrieved_documents", []), ensure_ascii=False),
            "expected_answer": r.get("expected_answer"),
            "generated_answer": r.get("generated_answer", ""),
            "generated_citations": json.dumps(r.get("generated_citations", []), ensure_ascii=False),
            "answer_fact_coverage_proxy": r.get("answer_fact_coverage_proxy"),
            "generated_answer_abstained_proxy": r.get("generated_answer_abstained_proxy"),
            "generated_answer_error": r.get("generated_answer_error") or ""
        })

print("NSBM028 retry successful.")
print("Generated answers:", metrics["generated_answer_count"])
print("Remaining errors:", metrics["generated_answer_errors"])
print("Answer:", answer)
