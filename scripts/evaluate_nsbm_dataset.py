
"""Evaluate NSBM retrieval and evidence validation using ground truth."""

import json
from pathlib import Path

from src.embeddings.embedder import MultilingualE5Embedder
from src.generation.evidence_validator import EvidenceValidator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.vector_store import ChromaVectorStore


MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "multi_university_admin_chunks"

DATASET_PATH = Path("data/evaluation/nsbm_ground_truth.json")
REPORT_PATH = Path("data/evaluation/nsbm_evaluation_report.json")

INSTITUTION_ID = "NSBM"
TOP_K = 5
FACT_MATCH_THRESHOLD = 0.50


def load_records(store):
    """Load all documents from ChromaDB."""

    raw = store._collection.get(
        include=["documents", "metadatas"]
    )

    return [
        store._create_record(
            chunk_id=chunk_id,
            text=document,
            metadata=metadata,
        )
        for chunk_id, document, metadata in zip(
            raw["ids"],
            raw["documents"],
            raw["metadatas"],
        )
    ]


def check_supporting_context(retrieved, supporting_contexts):
    """Check exact document and page matches."""

    matched_contexts = 0

    for expected in supporting_contexts:
        expected_pages = set(expected["pages"])

        found = any(
            chunk.record.source == expected["document"]
            and bool(
                expected_pages.intersection(
                    chunk.record.page_numbers
                )
            )
            for chunk in retrieved
        )

        if found:
            matched_contexts += 1

    return matched_contexts, len(supporting_contexts)


def calculate_fact_coverage(
    required_facts,
    retrieved,
    validator,
):
    """Calculate lexical supporting-fact coverage.

    This is a proxy metric, not semantic fact verification.
    """

    if not required_facts:
        return 0.0, 0, 0

    evidence_text = " ".join(
        chunk.record.text
        for chunk in retrieved
    )

    evidence_terms = validator._tokens(evidence_text)

    fact_scores = []
    covered_facts = 0

    for fact in required_facts:
        fact_terms = validator._tokens(fact)

        if not fact_terms:
            continue

        matched = fact_terms.intersection(evidence_terms)
        score = len(matched) / len(fact_terms)

        fact_scores.append(score)

        if score >= FACT_MATCH_THRESHOLD:
            covered_facts += 1

    if not fact_scores:
        return 0.0, 0, 0

    average_coverage = sum(fact_scores) / len(fact_scores)

    return average_coverage, covered_facts, len(fact_scores)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():

    print("=" * 70)
    print("NSBM GROUND-TRUTH EVALUATION")
    print("=" * 70)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    questions = dataset["questions"]

    store = ChromaVectorStore(
        persist_directory=Path("data/vector_store"),
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL_NAME,
    )

    print("\nDatabase records:", store.count())

    records = load_records(store)

    if not records:
        raise RuntimeError("No vector database records found.")

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

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    answerable_count = 0
    context_hits = 0

    total_relevant_chunks = 0
    total_answerable_chunks = 0

    total_covered_facts = 0
    total_required_facts = 0

    for item in questions:

        question_id = item["id"]
        question = item["question"]
        expected_answerable = item["answerable"]

        retrieved = retriever.retrieve(
            query=question,
            top_k=TOP_K,
            institution_id=INSTITUTION_ID,
        )

        validation = validator.validate(
            question,
            retrieved_chunks=retrieved,
        )

        predicted_answerable = validation.sufficient

        # Answerability classification.
        if expected_answerable and predicted_answerable:
            tp += 1
            status = "CORRECTLY_ACCEPTED"

        elif not expected_answerable and not predicted_answerable:
            tn += 1
            status = "CORRECTLY_REJECTED"

        elif not expected_answerable and predicted_answerable:
            fp += 1
            status = "FALSE_ACCEPT"

        else:
            fn += 1
            status = "FALSE_REJECTION"

        # Retrieval relevance against the ground-truth sources.
        matched_contexts, expected_context_count = (
            check_supporting_context(
                retrieved,
                item["supporting_contexts"],
            )
        )

        context_hit = (
            matched_contexts > 0
            if expected_answerable
            else None
        )

        if expected_answerable:
            answerable_count += 1

            if context_hit:
                context_hits += 1

            total_answerable_chunks += len(retrieved)

            relevant_chunks = [
                chunk
                for chunk in retrieved
                if any(
                    chunk.record.source == context["document"]
                    and bool(
                        set(context["pages"]).intersection(
                            chunk.record.page_numbers
                        )
                    )
                    for context in item["supporting_contexts"]
                )
            ]

            total_relevant_chunks += len(relevant_chunks)

            fact_coverage, covered, fact_count = (
                calculate_fact_coverage(
                    item["required_facts"],
                    retrieved,
                    validator,
                )
            )

            total_covered_facts += covered
            total_required_facts += fact_count

        else:
            fact_coverage = None

        result = {
            "id": question_id,
            "question": question,
            "expected_answerable": expected_answerable,
            "predicted_answerable": predicted_answerable,
            "status": status,
            "validation_coverage": round(
                validation.coverage, 4
            ),
            "matched_terms": list(validation.matched_terms),
            "missing_terms": list(validation.missing_terms),
            "validation_reason": validation.reason,
            "retrieved_documents": [
                {
                    "source": chunk.record.source,
                    "pages": list(chunk.record.page_numbers),
                    "dense_score": chunk.dense_score,
                    "tfidf_score": chunk.tfidf_score,
                    "hybrid_score": chunk.hybrid_score,
                }
                for chunk in retrieved
            ],
            "matched_gold_contexts": matched_contexts,
            "expected_gold_contexts": expected_context_count,
            "context_hit": context_hit,
            "supporting_fact_coverage_proxy": (
                round(fact_coverage, 4)
                if fact_coverage is not None
                else None
            ),
        }

        results.append(result)

        print(
            f"\n{question_id} | {status}"
        )
        print("Question:", question)
        print(
            "Expected / Predicted:",
            expected_answerable,
            "/",
            predicted_answerable,
        )
        print(
            "Retrieved chunks:",
            len(retrieved),
        )
        print(
            "Validation coverage:",
            round(validation.coverage, 3),
        )

        if expected_answerable:
            print(
                "Gold context hit:",
                context_hit,
            )
            print(
                "Fact coverage proxy:",
                round(fact_coverage, 3),
            )

    # Calculate final metrics.
    total = len(questions)

    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    context_hit_rate = safe_divide(
        context_hits,
        answerable_count,
    )

    context_precision = safe_divide(
        total_relevant_chunks,
        total_answerable_chunks,
    )

    fact_coverage = safe_divide(
        total_covered_facts,
        total_required_facts,
    )

    unsupported_count = sum(
        not item["answerable"]
        for item in questions
    )

    unsupported_rejection_rate = safe_divide(
        tn,
        unsupported_count,
    )

    metrics = {
        "total_questions": total,
        "answerable_questions": answerable_count,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "answerability_accuracy": round(accuracy, 4),
        "answerability_precision": round(precision, 4),
        "answerability_recall": round(recall, 4),
        "answerability_f1": round(f1, 4),
        "unsupported_rejection_rate": round(
            unsupported_rejection_rate, 4
        ),
        "context_hit_rate_at_5": round(
            context_hit_rate, 4
        ),
        "context_precision_at_5": round(
            context_precision, 4
        ),
        "supporting_fact_coverage_proxy": round(
            fact_coverage, 4
        ),
        "answer_relevance": {
            "status": "NOT_EVALUATED",
            "reason": (
                "A production answer-generation model and "
                "generated-answer evaluation have not yet been "
                "integrated into this evaluation runner."
            ),
        },
        "faithfulness": {
            "status": "NOT_FULLY_EVALUATED",
            "reason": (
                "Supporting-fact coverage is a lexical proxy. "
                "Generated-answer faithfulness requires evaluation "
                "of actual model responses."
            ),
        },
    }

    report = {
        "dataset_name": dataset["dataset_name"],
        "dataset_version": dataset["version"],
        "institution_id": INSTITUTION_ID,
        "retrieval_top_k": TOP_K,
        "metrics": metrics,
        "question_results": results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nReport saved to:", REPORT_PATH)

    if fp or fn:
        print(
            "\nWARNING: Some answerability classifications "
            "did not match the ground truth."
        )
    else:
        print("\nALL ANSWERABILITY CLASSIFICATIONS MATCHED GROUND TRUTH")


if __name__ == "__main__":
    main()
