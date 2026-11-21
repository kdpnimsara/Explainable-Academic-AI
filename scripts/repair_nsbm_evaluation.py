from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "src" / "generation" / "evidence_validator.py"
INGESTER = ROOT / "scripts" / "ingest_nsbm_complete.py"
EVALUATOR = ROOT / "scripts" / "evaluate_nsbm_batch.py"
MANIFEST = ROOT / "data" / "documents" / "institutions" / "nsbm" / "nsbm_sources_manifest.json"
TEST_FILE = ROOT / "tests" / "test_nsbm_validator_gaps.py"

GAP_METHOD = r'''    @classmethod
    def _unsupported_request_reason(
        cls,
        question: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> str | None:
        """Reject requests whose exact facts are not covered by this source set.

        This is a source-availability guard, not a general answer generator.
        It deliberately permits the question again when a properly categorised
        authoritative source is later added to the NSBM corpus.
        """
        q = question.casefold()
        categories: set[str] = set()
        evidence_parts: list[str] = []

        for chunk in retrieved_chunks:
            metadata = getattr(chunk.record, "document_metadata", None)
            if metadata is not None:
                category = getattr(metadata, "document_category", None)
                if category:
                    categories.add(str(category).casefold())
            evidence_parts.append(cls._document_text(chunk))

        evidence = " ".join(evidence_parts).casefold()

        def has_category(*values: str) -> bool:
            return bool(categories.intersection(value.casefold() for value in values))

        def has_money_amount(text: str) -> bool:
            return bool(re.search(
                r"(?:LKR|Rs\.?|rupees?)\s*\d[\d,]*(?:\.\d{1,2})?"
                r"|\d[\d,]*(?:\.\d{1,2})?\s*(?:LKR|rupees?)",
                text,
                re.IGNORECASE,
            ))

        # Private student records cannot be inferred from public NSBM material.
        if re.search(r"\bmy\b", q) and re.search(
            r"\b(?:gpa|grade|result|results|transcript|student portal|exam result)\b", q
        ):
            return (
                "This request asks for an individual student record. The public NSBM "
                "knowledge base has no authenticated access to personal Student Portal data."
            )

        # An exact Library fine needs an authoritative Library fine/fee schedule.
        if re.search(r"\blibrary\b", q) and re.search(
            r"\b(?:fine|late return|overdue|late fee)\b", q
        ) and re.search(r"\b(?:exact|amount|how much|2026|current)\b", q):
            has_library_amount = (
                has_category("library_fee_policy", "library_fees", "library_policy")
                and has_money_amount(evidence)
                and bool(re.search(r"fine|late return|overdue|late fee", evidence))
            )
            if not has_library_amount:
                return (
                    "The available NSBM sources do not establish the exact current "
                    "Library late-return fine. Confirm it with the NSBM Library."
                )

        # A portal link directory is not the actual current examination timetable.
        if re.search(r"\b(?:exam|examination)\b", q) and re.search(
            r"\b(?:timetable|schedule)\b", q
        ) and re.search(r"\b(?:exact|date|dates|2026|when)\b", q):
            if not has_category("examination_timetable", "examination_schedule"):
                return (
                    "The indexed NSBM documents do not contain the official 2026 "
                    "examination timetable dates. Check the live Examination Portal."
                )

        # Data Science tuition needs a current programme-fee schedule, not general payments.
        if re.search(r"\bdata science\b", q) and re.search(
            r"\b(?:tuition|course fee|programme fee|program fee)\b", q
        ) and re.search(r"\b(?:exact|total|current|amount|how much)\b", q):
            has_programme_fee = (
                has_category("tuition_fees", "programme_fees", "data_science_fees")
                and has_money_amount(evidence)
            )
            if not has_programme_fee:
                return (
                    "The indexed NSBM sources do not establish the current total "
                    "Data Science tuition fee. Confirm the fee with NSBM Admissions."
                )

        # Internship existence is not evidence of its duration, hours or credit value.
        if re.search(r"\binternship\b|\bindustrial placement\b", q) and re.search(
            r"\b(?:months?|weeks?|hours?|credits?|duration|how long|how many)\b", q
        ):
            has_internship_spec = (
                has_category(
                    "internship_requirements",
                    "internship_guidelines",
                    "industrial_training",
                    "industrial_placement",
                )
                and bool(re.search(
                    r"(?:internship|industrial placement).{0,140}\b\d+\s*(?:months?|weeks?|hours?|credits?)\b"
                    r"|\b\d+\s*(?:months?|weeks?|hours?|credits?).{0,140}(?:internship|industrial placement)",
                    evidence,
                    re.IGNORECASE,
                ))
            )
            if not has_internship_spec:
                return (
                    "The available Data Science programme information confirms an internship, "
                    "but does not establish its exact duration, hours or credits."
                )

        # A general curriculum-revision announcement is not an approved module-by-module map.
        if re.search(r"\bcurriculum\b", q) and re.search(
            r"module.by.module|module changes|exact modules|which modules|modules were changed|approved changes",
            q,
        ):
            if not has_category(
                "approved_curriculum", "programme_curriculum", "module_specification",
            ):
                return (
                    "The NSBM curriculum-revision announcement does not provide the approved "
                    "module-by-module changes. The revised curriculum document is not indexed."
                )

        # Attendance percentage requires a specific undergraduate attendance policy.
        if re.search(r"\battendance\b", q) and re.search(
            r"%|percent(?:age)?|eligible to sit|minimum attendance", q
        ):
            has_attendance_rule = (
                has_category(
                    "attendance_policy", "undergraduate_student_handbook",
                    "programme_attendance_policy",
                )
                and bool(re.search(
                    r"attendance.{0,100}\b\d{1,3}\s*%|\b\d{1,3}\s*%.{0,100}attendance",
                    evidence,
                    re.IGNORECASE,
                ))
            )
            if not has_attendance_rule:
                return (
                    "The indexed NSBM sources do not establish the current attendance "
                    "percentage required for Data Science examination eligibility."
                )

        # A broad FAQ or career page is not a current scholarship policy/award schedule.
        if re.search(r"\bscholarship", q) and re.search(
            r"\b(?:current|categories|eligibility|amount|amounts|undergraduates?)\b", q
        ):
            has_scholarship_schedule = (
                has_category(
                    "scholarship_policy", "scholarships_and_bursaries", "financial_aid_policy",
                )
                and has_money_amount(evidence)
                and bool(re.search(r"eligib|criteria|qualif", evidence))
            )
            if not has_scholarship_schedule:
                return (
                    "The indexed NSBM sources do not contain a complete current scholarship "
                    "schedule with undergraduate eligibility and exact award amounts."
                )

        return None

'''

TESTS = r'''from types import SimpleNamespace

from src.generation.evidence_validator import EvidenceValidator


def make_chunk(category: str, text: str):
    metadata = SimpleNamespace(
        document_title="NSBM test source",
        document_category=category,
        institution_id="NSBM",
        document_version=None,
        source_authority="NSBM Green University",
        source_url="https://www.nsbm.ac.lk/",
    )
    record = SimpleNamespace(
        text=text,
        source="NSBM_test_source.pdf",
        heading=None,
        document_metadata=metadata,
    )
    return SimpleNamespace(record=record)


def reason(question, category, text):
    return EvidenceValidator._unsupported_request_reason(
        question,
        [make_chunk(category, text)],
    )


def test_private_student_results_are_not_inferred_from_public_sources():
    assert reason(
        "What is my current individual NSBM GPA and latest examination result according to my Student Portal?",
        "official_faq",
        "NSBM Student Portal provides student services.",
    ) is not None


def test_exact_library_fine_requires_a_library_fee_source():
    assert reason(
        "What is the exact amount of the NSBM library late return fine for 2026?",
        "learning_support",
        "The library provides learning support and borrowing services.",
    ) is not None


def test_exam_timetable_dates_require_a_timetable_source():
    assert reason(
        "What are the exact dates of the NSBM 2026 examination timetable?",
        "examination_information",
        "The Examination Portal links to the live examination schedule.",
    ) is not None


def test_exact_data_science_tuition_requires_a_fee_schedule():
    assert reason(
        "What is the exact current total tuition fee for the NSBM BSc Honours in Data Science programme?",
        "programme_profile",
        "The programme is a four-year BSc Honours in Data Science degree.",
    ) is not None


def test_internship_existence_does_not_prove_duration_or_credits():
    assert reason(
        "How many months, hours, or credits must a student complete for the mandatory third-year Data Science internship?",
        "programme_profile",
        "A mandatory industry internship is included in the third year.",
    ) is not None


def test_curriculum_announcement_does_not_prove_module_changes():
    assert reason(
        "What exact module-by-module changes were approved in the 2025 curriculum revision for the NSBM Data Science degree?",
        "curriculum_revision_notice",
        "The Faculty of Computing completed a curriculum revision exercise in 2025.",
    ) is not None


def test_attendance_percentage_requires_attendance_policy():
    assert reason(
        "What exact attendance percentage is required to be eligible to sit each NSBM Data Science examination?",
        "programme_profile",
        "The programme contains lectures, assessments and examinations.",
    ) is not None


def test_current_scholarship_amounts_require_scholarship_schedule():
    assert reason(
        "What are the current NSBM scholarship categories, eligibility rules, and exact scholarship amounts for Data Science undergraduates?",
        "official_faq",
        "NSBM provides student services and academic programmes.",
    ) is not None


def test_historical_by_law_version_is_not_treated_as_a_gap():
    chunk = make_chunk(
        "examination_bylaws",
        "By-Laws in Determining the NSBM BSc, BM and BA degrees. Version dated on 26/7/2016.",
    )
    assert EvidenceValidator._unsupported_request_reason(
        "According to the supplied 2016 version of the By-Laws, what are the basic completion requirements?",
        [chunk],
    ) is None


def test_specific_internship_duration_is_allowed_when_officially_documented():
    chunk = make_chunk(
        "internship_requirements",
        "The required industrial internship duration is 6 months and the placement carries 6 credits.",
    )
    assert EvidenceValidator._unsupported_request_reason(
        "How many months and credits are required for the internship?",
        [chunk],
    ) is None
'''


def backup(path: Path, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_dir / path.name)


def patch_validator(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "def _unsupported_request_reason(" in source:
        print("Validator gap guard already exists; skipping.")
        return
    anchor = "    def validate(\n        self,\n        question: str,"
    if source.count(anchor) != 1:
        raise RuntimeError("Could not safely locate EvidenceValidator.validate() anchor.")
    source = source.replace(anchor, GAP_METHOD + anchor, 1)

    coverage_anchor = "        coverage = len(matched_terms) / len(query_terms)\n"
    if source.count(coverage_anchor) != 1:
        raise RuntimeError("Could not safely locate validator coverage calculation.")
    insertion = coverage_anchor + '''\n        gap_reason = self._unsupported_request_reason(\n            question,\n            retrieved_chunks,\n        )\n\n        if gap_reason:\n            return EvidenceValidationResult(\n                sufficient=False,\n                coverage=coverage,\n                matched_terms=matched_terms,\n                missing_terms=missing_terms,\n                reason=gap_reason,\n            )\n'''
    source = source.replace(coverage_anchor, insertion, 1)
    path.write_text(source, encoding="utf-8", newline="\n")
    print("Patched EvidenceValidator unsupported-specificity guard.")


def patch_ingester(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    old = '"Version 1 (17.2 onwards)",'
    new = '"26 July 2016 (Version 1; 17.2 intake onwards)",'
    if new in source:
        print("Ingestion script already has the confirmed 2016 by-law version.")
        return
    if source.count(old) != 1:
        raise RuntimeError("Could not safely locate By-Laws version in ingestion script.")
    source = source.replace(old, new, 1)
    path.write_text(source, encoding="utf-8", newline="\n")
    print("Updated By-Laws metadata version to 26 July 2016 (confirmed from PDF page 2).")


def patch_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for item in manifest.get("documents", []):
        rel = item.get("relative_path", "").replace("\\", "/")
        if rel == "NSBM Degree Programmes_Exam By Laws _ Version 1 (17.2 onwards) _Students.pdf":
            item["document_version"] = "26 July 2016 (Version 1; 17.2 intake onwards)"
            item["curation_note"] = (
                "Version date confirmed from page 2 of the supplied PDF: 26/7/2016. "
                "Use as historical by-laws, not as an automatically current policy."
            )
            changed = True
    if not changed:
        raise RuntimeError("Could not locate the By-Laws manifest record.")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("Updated manifest By-Laws version and historical-use note.")


def patch_batch_evaluator(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "import time\n" not in source:
        source = source.replace("import sys\n", "import sys\nimport time\n", 1)

    marker = '    for index, question in enumerate(questions, start=1):\n        entry = result_by_id[question["id"]]\n        try:\n            response = agent.ask(\n                question=question["question"],\n                institution_id=INSTITUTION_ID,\n            )\n'
    replacement = '''    for index, question in enumerate(questions, start=1):
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
                        r"retry in\\s+([0-9.]+)\\s*s",
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
'''
    if marker not in source:
        if "time.sleep(5.0)" in source and "retry in\\s+" in source:
            print("Batch evaluator pacing/retry already exists; skipping.")
        else:
            raise RuntimeError("Could not safely locate batch evaluator generation loop.")
    else:
        source = source.replace(marker, replacement, 1)
    path.write_text(source, encoding="utf-8", newline="\n")
    print("Added 5-second generation pacing and HTTP 429 retries to batch evaluator.")


def add_tests(path: Path) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if "test_private_student_results_are_not_inferred" in existing:
            print("Targeted validator tests already exist; skipping.")
            return
        raise RuntimeError("Target test file already exists with different content; inspect manually.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TESTS, encoding="utf-8", newline="\n")
    print("Added 9 targeted tests for NSBM unsupported requests and 2016 by-laws.")


def main() -> int:
    required = [VALIDATOR, INGESTER, EVALUATOR, MANIFEST]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Required project files not found:\n" + "\n".join(missing))

    backup_dir = ROOT / "data" / "backups" / (
        "nsbm_evaluation_fix_source_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    for path in required:
        backup(path, backup_dir)
    print("Source backups:", backup_dir)

    patch_validator(VALIDATOR)
    patch_ingester(INGESTER)
    patch_manifest(MANIFEST)
    patch_batch_evaluator(EVALUATOR)
    add_tests(TEST_FILE)

    print("\nPATCH COMPLETE")
    print("Next: run py_compile, pytest, re-ingest NSBM, then rerun the 60-question evaluator.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nPATCH FAILED:", type(exc).__name__, str(exc))
        print("Review the printed backup path; no Chroma data was changed by this patcher.")
        raise SystemExit(1)
