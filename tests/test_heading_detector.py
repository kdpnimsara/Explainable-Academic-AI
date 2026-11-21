import pytest

from src.ingestion.chunking.heading_detector import (
    HeadingDetector,
    HeadingMatch,
)


@pytest.fixture
def detector():
    return HeadingDetector()


def test_detect_numbered_section_heading(detector):

    result = detector.detect(
        "4. Tasks",
        line_number=25,
    )

    assert result == HeadingMatch(
        heading="4. Tasks",
        level=1,
        kind="section",
        line_number=25,
    )


def test_detect_nested_section_heading(detector):

    result = detector.detect("4.1 Introduction")

    assert result is not None
    assert result.heading == "4.1 Introduction"
    assert result.level == 2


def test_detect_deeply_nested_section_heading(detector):

    result = detector.detect("4.1.2 Details")

    assert result is not None
    assert result.level == 3


@pytest.mark.parametrize(
    "heading,expected_level",
    [
        ("2.2.1. Floods", 3),
        ("2.2.3. Landslides", 3),
        ("3.12.2. Early warnings", 3),
        ("3.13. Preparedness for response", 2),
        ("2.6. Hazard assessment and hazard mapping", 2),
    ],
)
def test_detect_disaster_management_headings(
    detector,
    heading,
    expected_level,
):

    result = detector.detect(heading)

    assert result is not None
    assert result.level == expected_level
    assert result.kind == "section"


def test_ignore_numbered_list_after_not_limited_to(detector):

    preceding = (
        "Risk assessments can include several aspects "
        "but are not limited to;"
    )

    result = detector.detect(
        "1. Prevention and mitigation",
        preceding_text=preceding,
    )

    assert result is None


def test_ignore_later_numbered_list_items(detector):

    preceding = (
        "Risk assessments are not limited to;\n"
        "1. Prevention and mitigation\n"
        "2. Planning for response and recovery\n"
    )

    result = detector.detect(
        "3. Enhance community level DRM",
        preceding_text=preceding,
    )

    assert result is None


def test_ignore_numbered_goals_list(detector):

    preceding = (
        "The major goals of the NAP are to;"
    )

    result = detector.detect(
        "1. Raise the adaptive capacity of individuals",
        preceding_text=preceding,
    )

    assert result is None


def test_ignore_numbered_strategies_list(detector):

    preceding = (
        "The objectives will be achieved through "
        "different strategies as stated below."
    )

    result = detector.detect(
        "2. Develop institutional level DRR strategies",
        preceding_text=preceding,
    )

    assert result is None


def test_detect_heading_without_list_context(detector):

    result = detector.detect(
        "4. Emergency Operations"
    )

    assert result is not None
    assert result.heading == "4. Emergency Operations"


@pytest.mark.parametrize(
    "text",
    [
        "2.3.9 National and International Search and Rescue 22",
        "4.8.2 Data Information for Assessment -41-",
    ],
)
def test_ignore_table_of_contents_entries(
    detector,
    text,
):

    assert detector.detect(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "This is a normal paragraph.",
        "class Node:",
        "A disaster preparedness document.",
        "7 True Found",
        "12 False Not Found",
        "25 False",
    ],
)
def test_ignore_non_heading_text(detector, text):

    assert detector.detect(text) is None


def test_ignore_heading_candidate_over_maximum_length(detector):

    text = "4. " + ("A" * 160)

    assert detector.detect(text) is None


@pytest.mark.parametrize(
    "line_number",
    [0, -1, True, 1.5, "1"],
)
def test_reject_invalid_line_number(detector, line_number):

    with pytest.raises(ValueError):
        detector.detect(
            "4. Tasks",
            line_number=line_number,
        )


def test_reject_non_string_line(detector):

    with pytest.raises(TypeError):
        detector.detect(123)


def test_reject_non_string_preceding_text(detector):

    with pytest.raises(TypeError):
        detector.detect(
            "4. Tasks",
            preceding_text=123,
        )


def test_heading_detector_does_not_modify_input(detector):

    original = "  2.2.1. Floods  "

    detector.detect(original)

    assert original == "  2.2.1. Floods  "