from types import SimpleNamespace

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
