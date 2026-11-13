"""Batch evaluation for the NSBM 60-question ground-truth dataset.

Default mode evaluates retrieval, evidence sufficiency, and source/page matching.
Pass --generate-answers to also run all 60 questions through the configured
NSBM RAG agent and save generated answers/citations plus a lexical fact-coverage
proxy. The latter is not a semantic/LLM-judge faithfulness score.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.embeddings.embedder import MultilingualE5Embedder
from src.generation.evidence_validator import EvidenceValidator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.vector_store import ChromaVectorStore

MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "multi_university_admin_chunks"
DATASET_PATH = ROOT / "data" / "evaluation" / "nsbm_ground_truth.json"
REPORT_JSON = ROOT / "data" / "evaluation" / "nsbm_batch_evaluation_report.json"
REPORT_CSV = ROOT / "data" / "evaluation" / "nsbm_batch_evaluation_report.csv"
INSTITUTION_ID = "NSBM"

ABSTENTION_MARKERS = (
    "insufficient information",
    "not enough information",
    "cannot determine",
    "can't determine",
    "cannot verify",
    "can't verify",
    "unable to verify",
    "not provided in the source",
    "not available in the provided",
    "not stated in the provided",
    "not found in the provided",
    "the documents do not",
    "the sources do not",
    "i do not have access to",
    "i don't have access to",
    "please contact the university",
    "please contact nsbm",
)


def source_basename(value: Any) -> str:
    """Normalise Windows/POSIX absolute source paths to a filename."""
    return str(value or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()


def load_records(store):
    raw = store._collection.get(include=["documents", "metadatas"])
    records = []
    for chunk_id, document, metadata in zip(
        raw["ids"], raw["documents"], raw["metadatas"]
    ):
        if document is None or metadata is None:
            continue
        records.append(
            store._create_record(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata,
            )
        )
    return records, raw


def context_matches(retrieved, expected_contexts):
    matched_count = 0
    for expected in expected_contexts:
        expected_name = source_basename(expected.get("document"))
        expected_pages = {int(p) for p in expected.get("pages", [])}
        found = False
        for item in retrieved:
            record = item.record
            actual_name = source_basename(record.source)
            actual_pages = {int(p) for p in (record.page_numbers or ())}
            if actual_name == expected_name and expected_pages.intersection(actual_pages):
                found = True
                break
        matched_count += int(found)
    return matched_count, len(expected_contexts)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").casefold()))


def fact_coverage(facts, text: str, threshold: float = 0.50):
    if not facts:
        return None, 0, 0
    answer_tokens = token_set(text)
    matched_count = 0
    scores = []
    for fact in facts:
        fact_tokens = token_set(fact)
        if not fact_tokens:
            continue
        score = len(fact_tokens & answer_tokens) / len(fact_tokens)
        scores.append(score)
        if score >= threshold:
            matched_count += 1
    if not scores:
        return None, 0, 0
    return sum(scores) / len(scores), matched_count, len(scores)


def safe_divide(a, b):
    return a / b if b else 0.0


def citation_data(response):
    output = []
    for source in getattr(response, "sources", ()) or ():
        output.append({
            "citation_id": getattr(source, "citation_id", None),
            "document": getattr(source, "document", None),
            "page_numbers": list(getattr(source, "page_numbers", ()) or ()),
        })
    return output


def run_generation(questions, results):
    """Generate answers with the same configured agent used by run_nsbm_rag.py."""
    import run_nsbm_rag

    # Keep evaluation conversations separate from a normal interactive session.
    run_nsbm_rag.HISTORY_PATH = ROOT / "data" / "history" / "nsbm_batch_evaluation_history.jsonl"
    agent = run_nsbm_rag.build_agent()

    result_by_id = {item["id"]: item for item in results}
    print("\nGENERATED ANSWER EVALUATION ENABLED")
    print("This sends one configured RAG request per question (60 requests).")

    for index, question in enumerate(questions, start=1):
        entry = result_by_id[question["id"]]
        # Stay below the configured Gemini Free Tier limit of 15 requests/minute.
        if index > 1:
            time.sleep(5.0)
        try:
            response = None
            for attempt in range(4):
                try:
                    response = agent.ask(
                        question=question["question"],
                        institution_id=INSTITUTION_ID,
                    )
                    break
                except Exception as request_error:
                    message = str(request_error)
                    is_rate_limited = (
                        "429" in message or "RESOURCE_EXHAUSTED" in message
                    )
                    if not is_rate_limited or attempt == 3:
                        raise
                    retry_match = re.search(
                        r"retry in\s+([0-9.]+)\s*s",
                        message,
                        re.IGNORECASE,
                    )
                    wait_seconds = (
                        max(5.0, float(retry_match.group(1)) + 1.0)
                        if retry_match else 20.0 * (attempt + 1)
                    )
                    print(
                        f"RATE LIMIT | {question['id']} | retrying in "
                        f"{wait_seconds:.1f}s | attempt {attempt + 2}/4"
                    )
                    time.sleep(wait_seconds)
            if response is None:
                raise RuntimeError("No response returned after retry attempts.")
            answer = str(getattr(response, "answer", "") or "")
            entry["generated_answer"] = answer
            entry["generated_citations"] = citation_data(response)
            entry["generated_answer_error"] = None
            entry["generated_answer_abstained_proxy"] = any(
                marker in answer.casefold() for marker in ABSTENTION_MARKERS
            )
            coverage, covered, count = fact_coverage(
                question.get("required_facts", []), answer
            )
            entry["answer_fact_coverage_proxy"] = coverage
            entry["answer_facts_covered_proxy"] = covered
            entry["answer_facts_total"] = count
            print(
                f"ANSWER {index:02d}/{len(questions)} | {question['id']} | "
                f"chars={len(answer)} | citations={len(entry['generated_citations'])}"
            )
        except Exception as exc:
            entry["generated_answer"] = ""
            entry["generated_citations"] = []
            entry["generated_answer_error"] = f"{type(exc).__name__}: {exc}"
            entry["generated_answer_abstained_proxy"] = None
            entry["answer_fact_coverage_proxy"] = None
            entry["answer_facts_covered_proxy"] = 0
            entry["answer_facts_total"] = len(question.get("required_facts", []))
            print(f"ANSWER ERROR | {question['id']} | {type(exc).__name__}: {exc}")


def evaluate(args):
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Install the provided 60-question dataset first."
        )

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    questions = dataset["questions"]
    if args.limit:
        questions = questions[:args.limit]

    print("=" * 72)
    print("NSBM BATCH GROUND-TRUTH EVALUATION")
    print("=" * 72)
    print("Dataset version:", dataset.get("version"))
    print("Questions:", len(questions))
    print("Answerable:", sum(bool(q["answerable"]) for q in questions))
    print("Unanswerable:", sum(not bool(q["answerable"]) for q in questions))

    store = ChromaVectorStore(
        persist_directory=ROOT / "data" / "vector_store",
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL_NAME,
    )
    print("Database chunks:", store.count())

    records, raw = load_records(store)
    nsbm_chunk_count = sum(
        1 for md in raw["metadatas"]
        if md and str(md.get("institution_id", "")).casefold() == "nsbm"
    )
    nsbm_document_ids = {
        md.get("document_id") for md in raw["metadatas"]
        if md and str(md.get("institution_id", "")).casefold() == "nsbm"
    }
    nsbm_document_ids.discard(None)
    print("NSBM chunks:", nsbm_chunk_count)
    print("NSBM indexed documents:", len(nsbm_document_ids))

    if nsbm_chunk_count == 0:
        raise RuntimeError("No NSBM chunks found in the vector database.")

    embedder = MultilingualE5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
        local_files_only=True,
    )
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        corpus_records=records,
    )
    validator = EvidenceValidator()

    results = []
    tp = tn = fp = fn = 0
    answerable_count = context_hits = 0
    relevant_chunks_total = answerable_chunks_total = 0
    fact_covered_total = fact_total = 0
    category_counts = Counter()

    for index, item in enumerate(questions, start=1):
        retrieved = retriever.retrieve(
            query=item["question"],
            top_k=args.top_k,
            institution_id=INSTITUTION_ID,
        )
        validation = validator.validate(
            item["question"],
            retrieved_chunks=retrieved,
        )
        predicted = bool(validation.sufficient)
        expected = bool(item["answerable"])

        if expected and predicted:
            tp += 1
            status = "CORRECTLY_ACCEPTED"
        elif not expected and not predicted:
            tn += 1
            status = "CORRECTLY_REJECTED"
        elif not expected and predicted:
            fp += 1
            status = "FALSE_ACCEPT"
        else:
            fn += 1
            status = "FALSE_REJECTION"

        matched, expected_context_count = context_matches(
            retrieved,
            item.get("supporting_contexts", []),
        )
        context_hit = (matched > 0) if expected else None

        if expected:
            answerable_count += 1
            context_hits += int(bool(context_hit))
            answerable_chunks_total += len(retrieved)
            relevant_chunks_total += sum(
                1 for candidate in retrieved
                if any(
                    source_basename(candidate.record.source) == source_basename(ctx["document"])
                    and bool(
                        {int(p) for p in (candidate.record.page_numbers or ())}
                        & {int(p) for p in ctx.get("pages", [])}
                    )
                    for ctx in item.get("supporting_contexts", [])
                )
            )
            evidence_text = " ".join(candidate.record.text for candidate in retrieved)
            cov, covered, count = fact_coverage(
                item.get("required_facts", []), evidence_text
            )
            if cov is not None:
                fact_covered_total += covered
                fact_total += count
        else:
            cov = None

        retrieved_docs = []
        for candidate in retrieved:
            md = getattr(candidate.record, "document_metadata", None)
            retrieved_docs.append({
                "source": candidate.record.source,
                "pages": list(candidate.record.page_numbers or ()),
                "document_category": getattr(md, "document_category", None),
                "document_title": getattr(md, "document_title", None),
                "dense_score": getattr(candidate, "dense_score", None),
                "tfidf_score": getattr(candidate, "tfidf_score", None),
                "hybrid_score": getattr(candidate, "hybrid_score", None),
            })

        result = {
            "id": item["id"],
            "category": item.get("category"),
            "question": item["question"],
            "expected_answerable": expected,
            "predicted_answerable": predicted,
            "status": status,
            "validation_coverage": round(float(validation.coverage), 4),
            "validation_reason": validation.reason,
            "matched_gold_contexts": matched,
            "expected_gold_contexts": expected_context_count,
            "context_hit": context_hit,
            "supporting_fact_coverage_proxy": round(cov, 4) if cov is not None else None,
            "retrieved_documents": retrieved_docs,
            "expected_answer": item.get("expected_answer", ""),
            "required_facts": item.get("required_facts", []),
            "supporting_contexts": item.get("supporting_contexts", []),
        }
        results.append(result)
        category_counts[item.get("category", "uncategorized")] += 1
        print(
            f"{index:02d}/{len(questions)} | {item['id']} | {status} | "
            f"context_hit={context_hit} | retrieved={len(retrieved)}"
        )

    total = len(questions)
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    unsupported_count = sum(not bool(q["answerable"]) for q in questions)

    metrics = {
        "total_questions": total,
        "answerable_questions": answerable_count,
        "unanswerable_questions": unsupported_count,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "answerability_accuracy": round(accuracy, 4),
        "answerability_precision": round(precision, 4),
        "answerability_recall": round(recall, 4),
        "answerability_f1": round(f1, 4),
        "unsupported_rejection_rate": round(safe_divide(tn, unsupported_count), 4),
        f"context_hit_rate_at_{args.top_k}": round(safe_divide(context_hits, answerable_count), 4),
        f"context_precision_at_{args.top_k}": round(safe_divide(relevant_chunks_total, answerable_chunks_total), 4),
        "supporting_fact_coverage_proxy": round(safe_divide(fact_covered_total, fact_total), 4),
        "answer_generation": "PENDING" if args.generate_answers else "NOT_RUN",
        "faithfulness_note": "Supporting-fact coverage is a lexical proxy, not a semantic or LLM-judge faithfulness score.",
    }

    if args.generate_answers:
        run_generation(questions, results)
        answer_runs = [r for r in results if "generated_answer" in r]
        errors = sum(bool(r.get("generated_answer_error")) for r in answer_runs)
        answerable_results = [r for r in answer_runs if r["expected_answerable"]]
        answer_covs = [
            r["answer_fact_coverage_proxy"]
            for r in answerable_results
            if r.get("answer_fact_coverage_proxy") is not None
        ]
        unsupported_results = [r for r in answer_runs if not r["expected_answerable"]]
        abstain_hits = sum(
            bool(r.get("generated_answer_abstained_proxy"))
            for r in unsupported_results
            if not r.get("generated_answer_error")
        )
        metrics["answer_generation"] = "COMPLETED_WITH_ERRORS" if errors else "COMPLETED"
        metrics["generated_answer_count"] = len(answer_runs) - errors
        metrics["generated_answer_errors"] = errors
        metrics["answer_fact_coverage_proxy_mean"] = (
            round(sum(answer_covs) / len(answer_covs), 4) if answer_covs else None
        )
        metrics["unanswerable_abstention_marker_rate_proxy"] = (
            round(safe_divide(abstain_hits, len(unsupported_results) - errors), 4)
            if len(unsupported_results) > errors else None
        )
        metrics["generated_answer_evaluation_note"] = (
            "Answer fact coverage is a lexical proxy, and abstention detection uses phrase markers. "
            "Neither is a semantic answer-quality or faithfulness judge. Human review is required."
        )

    report = {
        "dataset_name": dataset.get("dataset_name"),
        "dataset_version": dataset.get("version"),
        "institution_id": INSTITUTION_ID,
        "database_collection": COLLECTION_NAME,
        "database_total_chunks": store.count(),
        "nsbm_chunk_count": nsbm_chunk_count,
        "nsbm_unique_documents": len(nsbm_document_ids),
        "retrieval_top_k": args.top_k,
        "generated_answers_requested": bool(args.generate_answers),
        "category_counts": dict(category_counts),
        "metrics": metrics,
        "question_results": results,
        "created_at": datetime.now().astimezone().isoformat(),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    csv_fields = [
        "id", "category", "question", "expected_answerable", "predicted_answerable",
        "status", "validation_coverage", "validation_reason", "matched_gold_contexts",
        "expected_gold_contexts", "context_hit", "supporting_fact_coverage_proxy",
        "retrieved_sources", "expected_answer", "generated_answer", "generated_citations",
        "answer_fact_coverage_proxy", "generated_answer_abstained_proxy", "generated_answer_error",
    ]
    with REPORT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in results:
            writer.writerow({
                "id": item.get("id"),
                "category": item.get("category"),
                "question": item.get("question"),
                "expected_answerable": item.get("expected_answerable"),
                "predicted_answerable": item.get("predicted_answerable"),
                "status": item.get("status"),
                "validation_coverage": item.get("validation_coverage"),
                "validation_reason": item.get("validation_reason"),
                "matched_gold_contexts": item.get("matched_gold_contexts"),
                "expected_gold_contexts": item.get("expected_gold_contexts"),
                "context_hit": item.get("context_hit"),
                "supporting_fact_coverage_proxy": item.get("supporting_fact_coverage_proxy"),
                "retrieved_sources": json.dumps(item.get("retrieved_documents", []), ensure_ascii=False),
                "expected_answer": item.get("expected_answer"),
                "generated_answer": item.get("generated_answer", ""),
                "generated_citations": json.dumps(item.get("generated_citations", []), ensure_ascii=False),
                "answer_fact_coverage_proxy": item.get("answer_fact_coverage_proxy"),
                "generated_answer_abstained_proxy": item.get("generated_answer_abstained_proxy"),
                "generated_answer_error": item.get("generated_answer_error", ""),
            })

    print("\n" + "=" * 72)
    print("FINAL NSBM BATCH EVALUATION")
    print("=" * 72)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("JSON report:", REPORT_JSON)
    print("CSV report:", REPORT_CSV)
    print("\nNote: retrieval metrics evaluate retrieval/answerability. Generated-answer coverage is only a lexical proxy.")


def main():
    parser = argparse.ArgumentParser(description="Run the NSBM 60-question batch evaluator.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit; 0 runs all questions.")
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Also generate one model answer per question using the configured NSBM RAG agent (may incur API costs).",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    evaluate(args)


if __name__ == "__main__":
    main()
