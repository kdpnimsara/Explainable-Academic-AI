
"""Interactive NSBM academic and weather-aware RAG application."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agents.weather_aware_rag_agent import WeatherAwareRAGAgent
from src.embeddings.embedder import MultilingualE5Embedder
from src.generation.answer_generator import AnswerGenerator
from src.generation.openai_compatible_model import (
    OpenAICompatibleChatModel,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.storage.vector_store import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = PROJECT_ROOT / "data" / "vector_store"
COLLECTION_NAME = "multi_university_admin_chunks"
HISTORY_DIR = PROJECT_ROOT / 'data' / 'history'

MODEL_NAME = "intfloat/multilingual-e5-small"
INSTITUTION_ID = "NSBM"

# Sri Lanka Standard Time (UTC+05:30).
# A fixed offset avoids requiring an external timezone database.
SRI_LANKA_TIMEZONE = timezone(
    timedelta(hours=5, minutes=30),
    name="SLST",
)


def format_sri_lanka_timestamp(value: str | None) -> str | None:
    """Convert an ISO timestamp to Sri Lanka local time for display.

    The original UTC timestamps stored in source metadata are not changed.
    """

    if value is None:
        return None

    try:
        timestamp = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            return value

        return timestamp.astimezone(
            SRI_LANKA_TIMEZONE
        ).isoformat()

    except (TypeError, ValueError):
        # Preserve an unrecognised timestamp rather than hiding it.
        return value


def load_all_records(store):
    """Load document records from the persistent ChromaDB collection."""

    raw_data = store._collection.get(
        include=["documents", "metadatas"]
    )

    records = []

    for chunk_id, document, metadata in zip(
        raw_data["ids"],
        raw_data["documents"],
        raw_data["metadatas"],
    ):
        if document is None or metadata is None:
            continue

        record = store._create_record(
            chunk_id=chunk_id,
            text=document,
            metadata=metadata,
        )

        records.append(record)

    return records


def build_agent(institution_id: str = INSTITUTION_ID):
    """Initialize the selected university RAG workflow."""

    print('\nConnecting to the selected university knowledge base...')

    store = ChromaVectorStore(
        persist_directory=DATABASE_PATH,
        collection_name=COLLECTION_NAME,
        embedding_dimension=384,
        distance_metric="cosine",
        embedding_model=MODEL_NAME,
    )

    database_count = store.count()

    if database_count == 0:
        raise RuntimeError(
            "The selected university knowledge base is empty. "
            "Run scripts/ingest_multi_university.py first."
        )

    print("Database records:", database_count)

    records = load_all_records(store)

    if not records:
        raise RuntimeError(
            "No usable document records were found."
        )

    print("Loaded document records:", len(records))

    print("\nLoading multilingual embedding model...")

    embedder = MultilingualE5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
        batch_size=32,
        normalize_embeddings=True,
        local_files_only=True,
    )

    print("Initializing hybrid retriever...")

    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        corpus_records=records,
    )

    print("Connecting to the configured language model...")

    model = OpenAICompatibleChatModel()

    generator = AnswerGenerator(model)

    print("Initializing evidence validation and weather services...")

    agent = WeatherAwareRAGAgent(
        retriever=retriever,
        generator=generator,
        history_path=HISTORY_DIR / (institution_id.lower() + '_agent.jsonl'),
    )

    return agent


def display_response(response, institution_name, institution_id):
    """Display the answer and its explainability information."""

    print("\n" + "=" * 70)
    print("AI ANSWER")
    print("=" * 70)

    print(response.answer)

    print("\n" + "-" * 70)
    print("EXECUTION DETAILS")
    print("-" * 70)

    print('Institution:', institution_name, '(', institution_id, ')')
    print("Intent:", response.reasoning_plan.intent)

    print(
        "Retrieval depth:",
        response.reasoning_plan.retrieval_top_k,
    )

    print("Retrieval attempts:", response.retrieval_attempts)

    if hasattr(response, "selected_tools"):
        print("Selected tools:", response.selected_tools)

    print("\nReasoning:")
    print(response.reasoning_plan.rationale)

    print("\n" + "-" * 70)
    print("ACADEMIC SOURCE CITATIONS")
    print("-" * 70)

    if response.sources:
        for source in response.sources:
            pages = (
                ", ".join(str(page) for page in source.page_numbers)
                if source.page_numbers
                else "Unknown"
            )

            print(
                f"{source.citation_id} "
                f"{source.document} | Pages: {pages}"
            )
    else:
        print("No academic document citations were returned.")

    print("\n" + "-" * 70)
    print("WEATHER / EXTERNAL SOURCES")
    print("-" * 70)

    external_sources = getattr(
        response,
        "external_sources",
        (),
    )

    if external_sources:
        for source in external_sources:
            print(
                f"{source.citation_id} {source.source_name}"
            )

            print("Type:", source.evidence_type)

            local_timestamp = format_sri_lanka_timestamp(
                source.data_timestamp
            )

            print(
                "Timestamp (Sri Lanka time):",
                local_timestamp,
            )

            print("URL:", source.source_url)

            if source.valid_until:
                local_valid_until = format_sri_lanka_timestamp(
                    source.valid_until
                )

                print(
                    "Valid until (Sri Lanka time):",
                    local_valid_until,
                )

            print()

    else:
        print("No external weather sources were returned.")

    if response.retrieved_chunks:
        print("\n" + "-" * 70)
        print("RETRIEVED DOCUMENT EVIDENCE")
        print("-" * 70)

        for index, chunk in enumerate(
            response.retrieved_chunks,
            start=1,
        ):
            print(
                f"{index}. {chunk.record.source}"
                f" | Pages: {chunk.record.page_numbers}"
                f" | Hybrid score: {chunk.hybrid_score:.5f}"
            )

    tool_errors = getattr(response, "tool_errors", ())

    if tool_errors:
        print("\n" + "-" * 70)
        print("TOOL WARNINGS")
        print("-" * 70)

        for error in tool_errors:
            print("-", error)

    print("\n" + "=" * 70)


def main():
    """Run the interactive application."""

    print("=" * 70)
    print('MULTI-UNIVERSITY SMART ACADEMIC & WEATHER ASSISTANT')
    print("Explainable Agentic Retrieval-Augmented Generation System")
    print("=" * 70)

    print()
    print('Select your university:')
    choices = {
        '1': ('NSBM', 'NSBM Green University'),
        '2': ('SLIIT', 'SLIIT'),
    }
    while True:
        choice = input('Select university (1 or 2, or exit): ').strip()
        if choice.lower() in {'exit', 'quit'}:
            return
        selected = choices.get(choice)
        if selected:
            institution_id, institution_name = selected
            break
        print('Invalid choice. Please enter 1 or 2.')

    try:
        agent = build_agent(institution_id)

    except (ValueError, RuntimeError, OSError) as exc:
        print("\nAPPLICATION INITIALIZATION FAILED")
        print(str(exc))

        print(
            "\nCheck the local .env configuration, database, "
            "and model setup."
        )

        sys.exit(1)

    print("\n" + "=" * 70)
    print("APPLICATION READY")
    print("=" * 70)

    print('Institution:', institution_name, '(', institution_id, ')')
    print("Type a question to begin.")
    print("Type 'help' to see example questions.")
    print("Type 'exit' or 'quit' to close the application.")

    while True:
        try:
            question = input("\nYou: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nClosing the application.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print('Thank you for using the Multi-University Assistant.')
            break

        if question.lower() == "help":
            print(
                "\nExample questions:\n"
                "1. How long must a candidate wait before leaving "
                "the examination hall?\n"
                "2. What is the repeat coursework submission deadline?\n"
                "3. Are mobile phones allowed in the examination room?\n"
                "4. What is the weather in Colombo?\n"
                "5. Will heavy rain affect university examinations "
                "in Colombo?\n"
                '6. Ask about an academic service or regulation at your selected university.'

            )
            continue

        try:
            response = agent.ask(
                question=question,
                institution_id=institution_id,
            )

            display_response(response, institution_name, institution_id)

        except (RuntimeError, TypeError, ValueError) as exc:
            print("\nCould not complete the request:")
            print(str(exc))

        except Exception as exc:
            print("\nAn unexpected application error occurred:")
            print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
