"""Local HTTP adapter for the CampusMate AI presentation layer.

The RAG pipeline is used as-is. This module only accepts one question
at a time and reads existing evaluation reports.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "web" / "campusmate"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALLOWED_INSTITUTIONS = ("NSBM", "SLIIT")
MAX_BODY_BYTES = 1_000_000

QUALITY_REPORT = "rag_quality_judgement_report.json"
BATCH_REPORTS = {
    "NSBM": "nsbm_batch_evaluation_report.json",
    "SLIIT": "sliit_batch_evaluation_report.json",
}

QUALITY_FIELDS = (
    "questions_evaluated",
    "judge_completed",
    "judge_errors",
    "context_relevance_mean_1_to_5_answerable_only",
    "groundedness_mean_1_to_5_judged_only",
    "answer_relevance_mean_1_to_5_judged_only",
)

BATCH_COUNT_FIELDS = (
    "total_questions",
    "answerable_questions",
    "unanswerable_questions",
    "answerability_accuracy",
    "context_hit_rate_at_5",
    "context_precision_at_5",
    "supporting_fact_coverage_proxy",
    "true_positives",
    "true_negatives",
    "unsupported_rejection_rate",
)

STATIC_FILES = {
    "": "index.html",
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def load_agents() -> dict[str, Any]:
    """Build NSBM and SLIIT agents that share one loaded retriever."""

    from scripts.run_multi_university_rag import HISTORY_DIR, build_agent
    from src.agents.weather_aware_rag_agent import WeatherAwareRAGAgent

    nsbm_agent = build_agent("NSBM")
    sliit_agent = WeatherAwareRAGAgent(
        retriever=nsbm_agent.retriever,
        generator=nsbm_agent.generator,
        history_path=HISTORY_DIR / "sliit_agent.jsonl",
        weather_service=nsbm_agent.weather_service,
        tool_selector=nsbm_agent.tool_selector,
        location_resolver=nsbm_agent.location_resolver,
        evidence_validator=nsbm_agent.evidence_validator,
    )
    return {"NSBM": nsbm_agent, "SLIIT": sliit_agent}


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON object without creating or modifying the file."""

    if not path.is_file():
        return None, "Report file was not found."

    try:
        with path.open("r", encoding="utf-8") as report_file:
            value = json.load(report_file)
    except json.JSONDecodeError:
        return None, "Report file is not valid JSON."
    except OSError as exc:
        return None, str(exc)

    if not isinstance(value, dict):
        return None, "Report file does not contain a JSON object."

    return value, None


def _copy_fields(source: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy known fields and preserve missing keys as null."""

    copied: dict[str, Any] = {}
    for field in fields:
        if field in source:
            copied[field] = source[field]
        else:
            copied[field] = None
    return copied


def _quality_row(
    report: dict[str, Any] | None,
    institution_id: str,
) -> dict[str, Any] | None:
    """Find the judgement-report row for one institution."""

    if report is None:
        return None

    metrics = report.get("metrics")
    if not isinstance(metrics, list):
        return None

    for item in metrics:
        if not isinstance(item, dict):
            continue
        dataset = str(item.get("dataset", "")).strip().upper()
        if dataset == institution_id:
            return item

    return None


def build_evaluation_payload(report_dir: Path) -> dict[str, Any]:
    """Build a dashboard payload from the existing evaluation reports."""

    report_dir = Path(report_dir)
    quality_path = report_dir / QUALITY_REPORT
    quality_report, quality_error = _read_json_object(quality_path)

    limitations: list[str] = []
    if quality_report is not None and isinstance(quality_report.get("limitations"), list):
        limitations = [
            str(item)
            for item in quality_report["limitations"]
            if isinstance(item, str)
        ]

    institutions: dict[str, Any] = {}
    for institution_id, batch_name in BATCH_REPORTS.items():
        batch_path = report_dir / batch_name
        batch_report, batch_error = _read_json_object(batch_path)
        quality_item = _quality_row(quality_report, institution_id)

        batch_metrics = None
        if batch_report is not None and isinstance(batch_report.get("metrics"), dict):
            batch_metrics = batch_report["metrics"]

        institutions[institution_id] = {
            "institution_id": institution_id,
            "quality": {
                "available": quality_item is not None,
                "source": QUALITY_REPORT,
                "error": None if quality_item is not None else (
                    quality_error or "No metrics row was found for this institution."
                ),
                "metrics": _copy_fields(quality_item or {}, QUALITY_FIELDS),
            },
            "batch": {
                "available": batch_metrics is not None,
                "source": batch_name,
                "error": None if batch_metrics is not None else (
                    batch_error or "No metrics object was found in the batch report."
                ),
                "metrics": _copy_fields(batch_metrics or {}, BATCH_COUNT_FIELDS),
            },
        }

    return {
        "institutions": institutions,
        "limitations": limitations,
    }


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _excerpt(text: Any, limit: int = 700) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def serialize_agent_response(response: Any, institution_id: str) -> dict[str, Any]:
    """Convert an agent response into JSON for the browser."""

    plan = response.reasoning_plan
    sources = []
    for source in getattr(response, "sources", ()) or ():
        sources.append(
            {
                "citation_id": source.citation_id,
                "chunk_id": source.chunk_id,
                "document": source.document,
                "page_numbers": list(source.page_numbers),
            }
        )

    external_sources = []
    for source in getattr(response, "external_sources", ()) or ():
        external_sources.append(
            {
                "citation_id": source.citation_id,
                "evidence_type": source.evidence_type,
                "source_name": source.source_name,
                "source_url": source.source_url,
                "data_timestamp": source.data_timestamp,
                "valid_until": source.valid_until,
            }
        )

    retrieved_chunks = []
    for chunk in getattr(response, "retrieved_chunks", ()) or ():
        record = chunk.record
        retrieved_chunks.append(
            {
                "chunk_id": record.chunk_id,
                "document": record.source,
                "page_numbers": list(record.page_numbers),
                "heading": record.heading,
                "excerpt": _excerpt(record.text),
                "dense_score": _number(chunk.dense_score),
                "tfidf_score": _number(chunk.tfidf_score),
                "hybrid_score": _number(chunk.hybrid_score),
            }
        )

    selected_tools = list(getattr(response, "selected_tools", ()) or ())
    tool_errors = [str(item) for item in (getattr(response, "tool_errors", ()) or ())]

    return {
        "institution_id": getattr(response, "institution_id", None) or institution_id,
        "question": response.question,
        "answer": response.answer,
        "created_at": response.created_at,
        "reasoning": {
            "intent": plan.intent,
            "retrieval_top_k": plan.retrieval_top_k,
            "rationale": plan.rationale,
            "retrieval_attempts": response.retrieval_attempts,
        },
        "sources": sources,
        "external_sources": external_sources,
        "retrieved_chunks": retrieved_chunks,
        "selected_tools": selected_tools,
        "tool_errors": tool_errors,
        "location_name": getattr(response, "location_name", None),
    }


class CampusMateApp:
    """Request logic shared by the local server and the tests."""

    def __init__(
        self,
        agents: dict[str, Any],
        evaluation_dir: Path = EVALUATION_DIR,
        static_dir: Path = STATIC_DIR,
    ) -> None:
        self.agents = agents
        self.evaluation_dir = Path(evaluation_dir)
        self.static_dir = Path(static_dir)
        self._lock = threading.Lock()

    def chat(self, payload: Any) -> tuple[int, dict[str, Any]]:
        """Answer one question for exactly one campus."""

        if not isinstance(payload, dict):
            return 400, {"error": "Request body must be a JSON object."}

        institution_id = payload.get("institution_id")
        question = payload.get("question")

        if not isinstance(institution_id, str) or institution_id not in ALLOWED_INSTITUTIONS:
            return 400, {"error": "institution_id must be NSBM or SLIIT."}

        if not isinstance(question, str) or not question.strip():
            return 400, {"error": "question must be a non-empty string."}

        question = question.strip()
        agent = self.agents[institution_id]

        try:
            with self._lock:
                response = agent.ask(
                    question=question,
                    institution_id=institution_id,
                )
        except (TypeError, ValueError) as exc:
            return 400, {"error": str(exc)}
        except Exception as exc:
            return 500, {"error": f"{type(exc).__name__}: {exc}"}

        return 200, serialize_agent_response(response, institution_id)

    def evaluation(self) -> tuple[int, dict[str, Any]]:
        """Return evaluation figures read from the existing JSON reports."""

        return 200, build_evaluation_payload(self.evaluation_dir)

    def static_file(self, url_path: str) -> tuple[int, bytes, str] | None:
        """Return one file from the CampusMate web directory."""

        cleaned = url_path.rstrip("/")
        relative_name = (
            STATIC_FILES.get(url_path)
            or STATIC_FILES.get(cleaned)
            or (STATIC_FILES.get("/") if cleaned == "" else None)
        )
        if relative_name is None:
            return None

        file_path = (self.static_dir / relative_name).resolve()
        static_root = self.static_dir.resolve()
        if static_root not in file_path.parents and file_path != static_root:
            return None
        if not file_path.is_file():
            return None

        content_type = CONTENT_TYPES.get(file_path.suffix, "application/octet-stream")
        return 200, file_path.read_bytes(), content_type


def make_handler(app: CampusMateApp) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one application instance."""

    class CampusMateHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _read_json(self) -> Any:
            length_header = self.headers.get("Content-Length", "0")
            try:
                length = int(length_header)
            except (TypeError, ValueError):
                raise ValueError("Content-Length must be an integer.") from None

            if length < 0 or length > MAX_BODY_BYTES:
                raise ValueError("Request body is too large.")

            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Request body must be JSON.") from exc

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/evaluation":
                status, payload = app.evaluation()
                self._send_json(status, payload)
                return

            static = app.static_file(path)
            if static is None:
                self._send_json(404, {"error": "Not found."})
                return

            status, body, content_type = static
            self._send_bytes(status, body, content_type)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/chat":
                self._send_json(404, {"error": "Not found."})
                return

            try:
                payload = self._read_json()
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return

            status, response = app.chat(payload)
            self._send_json(status, response)

    return CampusMateHandler


def serve(
    app: CampusMateApp,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Start the local CampusMate server."""

    server = ThreadingHTTPServer((host, port), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> None:
    """Load the existing agents and serve the CampusMate UI."""

    host = "127.0.0.1"
    port = 8000
    print("Loading the existing academic knowledge base...")
    print("This can take a few minutes the first time.")
    agents = load_agents()
    app = CampusMateApp(agents)
    server = serve(app, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    print(f"CampusMate AI ready at http://{bound_host}:{bound_port}")
    print("Press Ctrl+C to stop.")
    try:
        # serve() already started a daemon thread. Block the main thread.
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping CampusMate AI.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
