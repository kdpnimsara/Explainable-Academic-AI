"""Unit tests for the CampusMate AI server adapter and presentation layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.interfaces.campusmate_server import (
    EVALUATION_DIR,
    STATIC_DIR,
    CampusMateApp,
    build_evaluation_payload,
    serialize_agent_response,
)


class TestCampusMateStaticFiles:
    """Verify static asset routing and protection."""

    def setup_method(self) -> None:
        self.app = CampusMateApp(
            agents={},
            evaluation_dir=EVALUATION_DIR,
            static_dir=STATIC_DIR,
        )

    def test_root_path_serves_index_html(self) -> None:
        result = self.app.static_file("/")
        assert result is not None
        status, body, content_type = result
        assert status == 200
        assert "text/html" in content_type
        assert b"CampusMate AI" in body

    def test_empty_path_serves_index_html(self) -> None:
        result = self.app.static_file("")
        assert result is not None
        status, body, content_type = result
        assert status == 200
        assert "text/html" in content_type

    def test_index_html_path_serves_index_html(self) -> None:
        result = self.app.static_file("/index.html")
        assert result is not None
        status, body, content_type = result
        assert status == 200
        assert "text/html" in content_type

    def test_styles_css_serves_stylesheet(self) -> None:
        result = self.app.static_file("/styles.css")
        assert result is not None
        status, body, content_type = result
        assert status == 200
        assert "text/css" in content_type
        assert b"--accent" in body

    def test_app_js_serves_javascript(self) -> None:
        result = self.app.static_file("/app.js")
        assert result is not None
        status, body, content_type = result
        assert status == 200
        assert "javascript" in content_type
        assert b"CampusMate AI" in body

    def test_unknown_path_returns_none(self) -> None:
        result = self.app.static_file("/nonexistent.file")
        assert result is None


class TestCampusMateEvaluationPayload:
    """Verify evaluation reports are correctly parsed from real JSON files."""

    def test_build_evaluation_payload_nsbm_and_sliit(self) -> None:
        payload = build_evaluation_payload(EVALUATION_DIR)
        assert "institutions" in payload
        assert "limitations" in payload

        institutions = payload["institutions"]
        assert "NSBM" in institutions
        assert "SLIIT" in institutions

        # NSBM Quality
        nsbm_quality = institutions["NSBM"]["quality"]
        assert nsbm_quality["available"] is True
        assert nsbm_quality["error"] is None
        nsbm_q_metrics = nsbm_quality["metrics"]
        assert nsbm_q_metrics["questions_evaluated"] == 60
        assert nsbm_q_metrics["context_relevance_mean_1_to_5_answerable_only"] == 5.0
        assert nsbm_q_metrics["groundedness_mean_1_to_5_judged_only"] == 4.533
        assert nsbm_q_metrics["answer_relevance_mean_1_to_5_judged_only"] == 4.767

        # NSBM Batch
        nsbm_batch = institutions["NSBM"]["batch"]
        assert nsbm_batch["available"] is True
        nsbm_b_metrics = nsbm_batch["metrics"]
        assert nsbm_b_metrics["total_questions"] == 60
        assert nsbm_b_metrics["answerable_questions"] == 51
        assert nsbm_b_metrics["unanswerable_questions"] == 9
        assert nsbm_b_metrics["answerability_accuracy"] == 1.0

        # SLIIT Quality
        sliit_quality = institutions["SLIIT"]["quality"]
        assert sliit_quality["available"] is True
        assert sliit_quality["error"] is None
        sliit_q_metrics = sliit_quality["metrics"]
        assert sliit_q_metrics["questions_evaluated"] == 12
        assert sliit_q_metrics["context_relevance_mean_1_to_5_answerable_only"] == 5.0
        assert sliit_q_metrics["groundedness_mean_1_to_5_judged_only"] == 4.75
        assert sliit_q_metrics["answer_relevance_mean_1_to_5_judged_only"] == 4.917

        # SLIIT Batch
        sliit_batch = institutions["SLIIT"]["batch"]
        assert sliit_batch["available"] is True
        sliit_b_metrics = sliit_batch["metrics"]
        assert sliit_b_metrics["total_questions"] == 12
        assert sliit_b_metrics["answerable_questions"] == 10
        assert sliit_b_metrics["unanswerable_questions"] == 2
        assert sliit_b_metrics["answerability_accuracy"] == 1.0

        # Limitations
        assert len(payload["limitations"]) >= 5

    def test_app_evaluation_endpoint(self) -> None:
        app = CampusMateApp(agents={}, evaluation_dir=EVALUATION_DIR)
        status, payload = app.evaluation()
        assert status == 200
        assert "institutions" in payload


class TestCampusMateChatValidation:
    """Verify input validation and serialization for the chat endpoint."""

    def setup_method(self) -> None:
        self.mock_nsbm = MagicMock()
        self.mock_sliit = MagicMock()
        self.app = CampusMateApp(
            agents={"NSBM": self.mock_nsbm, "SLIIT": self.mock_sliit},
            evaluation_dir=EVALUATION_DIR,
            static_dir=STATIC_DIR,
        )

    def test_non_dict_payload_returns_400(self) -> None:
        status, response = self.app.chat("string payload")
        assert status == 400
        assert "error" in response

    def test_missing_or_invalid_institution_returns_400(self) -> None:
        status, response = self.app.chat({"institution_id": "INVALID", "question": "test"})
        assert status == 400
        assert "NSBM or SLIIT" in response["error"]

    def test_empty_question_returns_400(self) -> None:
        status, response = self.app.chat({"institution_id": "NSBM", "question": "   "})
        assert status == 400
        assert "question must be a non-empty string" in response["error"]

    def test_successful_chat_invocation(self) -> None:
        mock_response = MagicMock()
        mock_response.institution_id = "NSBM"
        mock_response.question = "When is the exam?"
        mock_response.answer = "Exams start at 9:00 AM."
        mock_response.created_at = "2026-11-17T00:00:00Z"
        mock_response.reasoning_plan.intent = "academic_inquiry"
        mock_response.reasoning_plan.retrieval_top_k = 5
        mock_response.reasoning_plan.rationale = "Retrieved exam schedule."
        mock_response.retrieval_attempts = 1
        mock_response.sources = ()
        mock_response.external_sources = ()
        mock_response.retrieved_chunks = ()
        mock_response.selected_tools = ()
        mock_response.tool_errors = ()
        mock_response.location_name = None

        self.mock_nsbm.ask.return_value = mock_response

        status, response = self.app.chat({"institution_id": "NSBM", "question": "When is the exam?"})
        assert status == 200
        assert response["institution_id"] == "NSBM"
        assert response["question"] == "When is the exam?"
        assert response["answer"] == "Exams start at 9:00 AM."
        assert response["reasoning"]["intent"] == "academic_inquiry"
        self.mock_nsbm.ask.assert_called_once_with(
            question="When is the exam?",
            institution_id="NSBM",
        )


class TestCampusMateHTTPServer:
    """End-to-end HTTP integration tests against the live local server."""

    @pytest.fixture(autouse=True)
    def run_server(self) -> None:
        import urllib.request
        from src.interfaces.campusmate_server import serve

        self.mock_nsbm = MagicMock()
        mock_response = MagicMock()
        mock_response.institution_id = "NSBM"
        mock_response.question = "Test Q"
        mock_response.answer = "Test Answer"
        mock_response.created_at = "2026-11-17T00:00:00Z"
        mock_response.reasoning_plan.intent = "academic_inquiry"
        mock_response.reasoning_plan.retrieval_top_k = 5
        mock_response.reasoning_plan.rationale = "Test rationale"
        mock_response.retrieval_attempts = 1
        mock_response.sources = ()
        mock_response.external_sources = ()
        mock_response.retrieved_chunks = ()
        mock_response.selected_tools = ()
        mock_response.tool_errors = ()
        mock_response.location_name = None
        self.mock_nsbm.ask.return_value = mock_response

        app = CampusMateApp(
            agents={"NSBM": self.mock_nsbm, "SLIIT": MagicMock()},
            evaluation_dir=EVALUATION_DIR,
            static_dir=STATIC_DIR,
        )
        # Bind to port 0 for an ephemeral free port
        server = serve(app, host="127.0.0.1", port=0)
        host, port = server.server_address[:2]
        self.base_url = f"http://{host}:{port}"
        yield
        server.shutdown()
        server.server_close()

    def test_http_get_index(self) -> None:
        import urllib.request

        req = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type")
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            body = resp.read().decode("utf-8")
            assert "CampusMate AI" in body

    def test_http_get_evaluation(self) -> None:
        import urllib.request

        req = urllib.request.Request(f"{self.base_url}/api/evaluation")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert "application/json" in resp.headers.get("Content-Type")
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            data = json.loads(resp.read().decode("utf-8"))
            assert "institutions" in data
            assert "NSBM" in data["institutions"]
            assert "SLIIT" in data["institutions"]

    def test_http_options_preflight(self) -> None:
        import urllib.request

        req = urllib.request.Request(f"{self.base_url}/api/chat", method="OPTIONS")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            assert "POST" in resp.headers.get("Access-Control-Allow-Methods")

    def test_http_post_chat(self) -> None:
        import urllib.request

        payload = json.dumps({"institution_id": "NSBM", "question": "Test Q"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
            data = json.loads(resp.read().decode("utf-8"))
            assert data["answer"] == "Test Answer"
            assert data["institution_id"] == "NSBM"
