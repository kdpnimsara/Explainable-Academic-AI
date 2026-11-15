from src.ingestion.chunking.heading_detector import HeadingDetector
from src.retrieval.hybrid_retriever import HybridRetriever


def test_numbered_examination_rule_is_not_misclassified_as_heading():
    detector = HeadingDetector()
    clause = (
        '3.14 During the examination Candidates shall not communicate '
        'in any way with any person other than an'
    )
    assert detector.detect(clause, line_number=1) is None


def test_short_numbered_subsection_heading_is_still_detected():
    detector = HeadingDetector()
    result = detector.detect('2.1 Floods', line_number=1)
    assert result is not None
    assert result.heading == '2.1 Floods'


def test_nsbm_thesis_reference_query_gets_targeted_expansion():
    query = (
        'Which referencing styles may be used in an NSBM thesis '
        'according to the faculty specification?'
    )
    expanded = HybridRetriever._expand_tfidf_query(query, 'NSBM')
    lowered = expanded.casefold()
    assert all(term in lowered for term in ('vancouver', 'apa', 'ieee', 'faculty'))


def test_nsbm_data_science_option_2_query_gets_entry_expansion():
    query = "What additional requirements apply under Option 2 for entry to the NSBM BSc (Honours) in Data Science?"
    expanded = HybridRetriever._expand_tfidf_query(query, "NSBM").casefold()
    assert all(term in expanded for term in ("three simple passes", "gce a/l", "mathematics", "gce o/l", "bridging programme", "exemption"))

def test_nsbm_first_class_honours_query_gets_targeted_expansion():
    query = 'Under the supplied 2016 By-Laws, what are the stated academic requirements for First Class Honours in the BSc, BM, and BA degree framework?'
    expanded = HybridRetriever._expand_tfidf_query(query, 'NSBM').casefold()
    assert all(term in expanded for term in ('special note', 'examination offences', '70% attendance', 'valid reason', '120 credits', 'first attempt', 'grade a', 'half courses', 'four academic years'))

def test_non_nsbm_query_is_not_expanded():
    query = 'Which referencing styles may be used in a thesis?'
    assert HybridRetriever._expand_tfidf_query(query, 'OTHER') == query


def test_unrelated_nsbm_query_is_not_expanded():
    query = 'When can candidates enter the examination room?'
    assert HybridRetriever._expand_tfidf_query(query, 'NSBM') == query
