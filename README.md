# Explainable-Academic-AI — CampusMate AI

> **Explainable Agentic RAG Assistant for University Academic Administration & Disaster-Aware Decision Support**
> Serving **NSBM Green University** and **SLIIT (Sri Lanka Institute of Information Technology)**.

---

## 📌 Executive Summary

**CampusMate AI** is an enterprise-grade, explainable Retrieval-Augmented Generation (RAG) assistant tailored for Sri Lankan higher education institutions. Combining institutional bylaws, examination guidelines, student handbooks, and official disaster management directives, CampusMate AI delivers verifiable, evidence-grounded answers to students, faculty, and administrative staff.

Unlike black-box conversational systems, CampusMate AI is built on **Explainable AI (XAI)** principles:
- Every factual claim is referenced with **verifiable document citations** and **exact page numbers**.
- Every retrieval step exposes **mathematical scoring transparency** (Hybrid RRF, Dense Cosine Similarity, and Sparse TF-IDF lexical scores).
- A transparent **LLM-as-a-Judge evaluation framework** quantifies Groundedness, Answer Relevance, and Context Precision across comprehensive ground-truth benchmarks.

---

## 🚀 Key Capabilities

- **Dual-University Support**: Seamlessly toggle between **NSBM Green University** and **SLIIT** with strict document partition and metadata isolation.
- **ChatGPT-Style Interface**: Modern, dark-themed UI with a fixed bottom composer, multiline auto-expanding textarea, responsive sidebar, and smooth scrollable conversation history.
- **Explainable Evidence Display**: Expandable inspection drawers showing query reasoning intent, top-$k$ depth, official citation cards, and retrieved chunk excerpts with mathematical similarity scores.
- **Disaster & Weather Awareness**: Integrates Sri Lanka Disaster Management Centre (DMC) guidelines and National Emergency Operation Plans (NEOP) to answer weather-related campus policy questions.
- **Comprehensive Evaluation Dashboard**: Interactive web dashboard presenting benchmark metrics, side-by-side comparison tables, and LLM-as-a-judge quality assessments directly from disk.
- **High-Reliability Local Server**: Fast startup with asynchronous background knowledge-base loading and dedicated readiness polling (`/api/status`).

---

## 🏛️ System Architecture

```
                                  USER INQUIRY
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │     CampusMate Web Interface     │
                     │  (HTML5 / Modern Dark CSS / JS)  │
                     └─────────────────┬────────────────┘
                                       │ POST /api/chat
                                       ▼
                     ┌──────────────────────────────────┐
                     │    Local HTTP Adapter Server     │
                     │ (src/interfaces/campusmate_server)│
                     └─────────────────┬────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │     WeatherAwareRAGAgent         │
                     │  (Observation → Reasoning → Act) │
                     └───────┬──────────────────┬───────┘
                             │                  │
           Institutional / Disaster Query       │ Weather Inquiries
                             │                  ▼
                             ▼          ┌─────────────────────────┐
                     ┌──────────────┐   │  WeatherService / CAP   │
                     │ ToolSelector │   │  (Location & Warnings)  │
                     └───────┬──────┘   └─────────────────────────┘
                             │
                             ▼
                     ┌──────────────────────────────────┐
                     │         HybridRetriever          │
                     │    Dense Cosine + Sparse TF-IDF  │
                     │     (Reciprocal Rank Fusion)     │
                     └───────┬──────────────────┬───────┘
                             │                  │
                Dense Vector Search       Sparse Term Matching
                             │                  │
                             ▼                  ▼
                   ┌──────────────────┐ ┌──────────────────┐
                   │  ChromaDB Store  │ │   TF-IDF Index   │
                   │ (Multilingual E5)│ │  (Corpus Lexicon)│
                   └──────────────────┘ └──────────────────┘
                             │                  │
                             └─────────┬────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │        EvidenceValidator         │
                     │  (Relevance & Fact Verification) │
                     └─────────────────┬────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │         AnswerGenerator          │
                     │   (OpenAI-Compatible LLM Model)  │
                     └─────────────────┬────────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │       Explainable Response       │
                     │ • Verified Answer & In-Text [S1] │
                     │ • Document & Page Citations      │
                     │ • Retrieval Scores (Dense/Sparse)│
                     └──────────────────────────────────┘
```

---

## 📊 Evaluation Benchmarks & Quantitative Results

CampusMate AI was validated using rigorous ground-truth evaluation datasets and an automated **LLM-as-a-Judge** (RAG Triad) assessment framework.

### 1. Cross-University Benchmark Summary

| Evaluation Dimension | NSBM Green University (60 Items) | SLIIT (12 Items) | Combined Target Benchmark |
| :--- | :---: | :---: | :---: |
| **Total Evaluated Questions** | 60 | 12 | **72 Questions** |
| **Answerable / Unanswerable** | 51 / 9 | 10 / 2 | Full Class Balance |
| **Answerability Accuracy** | **100.0%** (51/51 TP, 9/9 TN) | **100.0%** (10/10 TP, 2/2 TN) | **100.0% Target Achieved** |
| **Context Hit Rate @ 5** | **96.08%** | **100.0%** | $\ge 95.0\%$ Target |
| **Context Precision @ 5** | 38.43% | 30.00% | High-Density Top-5 |
| **Supporting Fact Coverage Proxy** | **85.71%** | **100.0%** | $\ge 85.0\%$ Coverage |
| **Unsupported Rejection Rate** | **100.0%** | **100.0%** | **Zero False Positives** |
| **Context Relevance Mean (1–5)** | **5.00 / 5.00** | **5.00 / 5.00** | **100% Gold Match** |
| **Groundedness / Faithfulness (1–5)** | **4.53 / 5.00** | **4.75 / 5.00** | High Evidentiary Rigor |
| **Answer Relevance Mean (1–5)** | **4.77 / 5.00** | **4.92 / 5.00** | Direct Query Alignment |

### 2. Metric Explanations

1. **Context Relevance (1.0 to 5.0)**: Measures whether the retrieved passages contain the exact gold supporting context pages specified in ground truth.
2. **Groundedness (1.0 to 5.0)**: Evaluates whether every statement in the generated response is strictly supported by retrieved facts without hallucination.
3. **Answer Relevance (1.0 to 5.0)**: Assesses how directly and completely the generated text addresses the user's inquiry.
4. **Unsupported Rejection Rate (100%)**: Assesses the agent's ability to abstain when answering questions whose facts do not exist in the official corpus (preventing false guidance).

---

## 📂 Project Structure

```
Explainable-Academic-AI/
├── .env.example                     # Sample LLM provider configuration
├── .gitignore                       # Production repository exclusion rules
├── requirements.txt                 # Pinned Python package dependencies
├── RUN_CAMPUSMATE_UI.cmd            # One-click Windows startup script
├── RUN_SLIIT_COLLECTION.cmd         # Helper script for SLIIT sources
├── evaluate_rag_quality.py          # LLM-as-a-judge evaluation harness
│
├── data/
│   ├── documents/                   # Official PDF source documents
│   │   ├── institutions/
│   │   │   ├── nsbm/                # NSBM bylaws, handbooks & guides
│   │   │   └── sliit/               # SLIIT booklets, guides & rules
│   │   └── safesteps/               # DMC Act, NEOP & emergency plans
│   ├── evaluation/                  # Benchmark datasets and reports
│   │   ├── nsbm_ground_truth.json   # 60-question NSBM ground-truth pack
│   │   ├── sliit_ground_truth.json  # 12-question SLIIT ground-truth pack
│   │   ├── nsbm_batch_evaluation_report.json
│   │   ├── sliit_batch_evaluation_report.json
│   │   └── rag_quality_judgement_report.json
│   └── vector_store/                # Persistent ChromaDB collection
│
├── docs/
│   ├── METADATA_DESIGN.md           # Metadata taxonomy and chunk schemas
│   ├── PROJECT_SCOPE.md             # Functional requirements & objectives
│   └── SYSTEM_ARCHITECTURE.md       # Detailed technical design document
│
├── scripts/
│   ├── evaluate_nsbm_batch.py       # NSBM batch evaluation script
│   ├── evaluate_sliit_batch.py      # SLIIT batch evaluation script
│   ├── ingest_multi_university.py   # Multi-institution PDF ingestion pipeline
│   ├── run_multi_university_rag.py  # Interactive CLI for dual-campus RAG
│   └── verify_evidence_validation.py # Verification harness for citations
│
├── src/
│   ├── agents/                      # Agentic orchestration & tool selection
│   │   ├── rag_agent.py             # Base RAG agent
│   │   ├── tool_selector.py         # Query intent & tool routing logic
│   │   └── weather_aware_rag_agent.py # Weather and campus multi-agent
│   ├── embeddings/
│   │   └── embedder.py              # Multilingual E5-Small embedder
│   ├── generation/
│   │   ├── answer_generator.py      # Prompt construction & citation builder
│   │   ├── evidence_validator.py    # Fact grounding & abstention logic
│   │   └── openai_compatible_model.py # HTTP LLM provider interface
│   ├── ingestion/                   # Document extraction & normalization
│   │   ├── pdf_loader.py            # PDF parser with page tracking
│   │   └── chunking/                # Recursive splitter & heading detection
│   ├── interfaces/
│   │   └── campusmate_server.py     # Local HTTP server & API adapter
│   ├── metadata/
│   │   └── document_metadata.py     # Metadata validation & models
│   ├── retrieval/
│   │   └── hybrid_retriever.py      # Dense E5 + Sparse TF-IDF + RRF
│   ├── storage/
│   │   └── vector_store.py          # ChromaDB collection wrapper
│   └── weather/                     # Meteorological data providers
│
├── tests/                           # Comprehensive test suite (30+ test files)
│   ├── test_campusmate_server.py    # Server endpoints, routing & API tests
│   ├── test_institution_filtering.py # Multi-tenant isolation verification
│   ├── test_retrieval.py            # Hybrid retrieval & RRF scoring tests
│   └── test_heading_detector.py     # Layout & heading extraction tests
│
└── web/
    └── campusmate/                  # CampusMate AI Web UI
        ├── index.html               # Semantic HTML5 layout
        ├── styles.css               # Modern dark theme with blue accents
        └── app.js                   # ChatGPT-style client logic & charts
```

---

## ⚙️ Installation & Setup

### Prerequisites

- **Python**: Version `3.10` to `3.13` (tested and verified on Python `3.13.5`).
- **Operating System**: Windows 10/11, Linux, or macOS.
- **Hardware**: 4 GB RAM minimum (8 GB recommended for local embedding model).

### Step 1: Clone Repository & Create Virtual Environment

```bash
# Clone the repository
git clone https://github.com/kdpnimsara/Explainable-Academic-AI.git
cd Explainable-Academic-AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd):
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
```

Configure your OpenAI-compatible model credentials inside `.env`:

```ini
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-api-key-here
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.2
```

*(Note: The server, static web UI, and Evaluation Dashboard run without an API key. An active LLM key is only needed when generating new live chat answers).*

---

## 🖥️ Running CampusMate AI

### Option A: One-Click Startup (Windows)

Double-click `RUN_CAMPUSMATE_UI.cmd` or run in terminal:

```cmd
RUN_CAMPUSMATE_UI.cmd
```

This immediately launches the local server at `http://127.0.0.1:8000` and automatically opens your default web browser.

### Option B: Command Line

```bash
.venv\Scripts\python src/interfaces/campusmate_server.py
```

Open `http://127.0.0.1:8000` in any modern web browser.

---

## 📡 API Reference

CampusMate AI exposes a lightweight, CORS-enabled HTTP API on port `8000`:

### 1. `GET /api/status`
Returns server readiness and background knowledge-base loading status.

**Response (200 OK)**:
```json
{
  "status": "ready",
  "institutions": ["NSBM", "SLIIT"],
  "agents_loaded": true
}
```

### 2. `GET /api/evaluation`
Returns complete benchmark and evaluation metrics parsed directly from disk.

**Response (200 OK)**:
```json
{
  "institutions": {
    "NSBM": {
      "institution_id": "NSBM",
      "quality": {
        "available": true,
        "metrics": {
          "context_relevance_mean_1_to_5_answerable_only": 5.0,
          "groundedness_mean_1_to_5_judged_only": 4.533,
          "answer_relevance_mean_1_to_5_judged_only": 4.767
        }
      },
      "batch": {
        "available": true,
        "metrics": {
          "total_questions": 60,
          "answerability_accuracy": 1.0,
          "context_hit_rate_at_5": 0.9608,
          "unsupported_rejection_rate": 1.0
        }
      }
    },
    "SLIIT": { ... }
  },
  "limitations": [ ... ]
}
```

### 3. `POST /api/chat`
Answers a question for a specific institution with citations and evidence scores.

**Request Payload**:
```json
{
  "institution_id": "NSBM",
  "question": "When can candidates enter the examination room?"
}
```

**Response (200 OK)**:
```json
{
  "institution_id": "NSBM",
  "question": "When can candidates enter the examination room?",
  "answer": "Candidates may enter the examination room 15 minutes before the scheduled start time [S1].",
  "reasoning": {
    "intent": "examination_inquiry",
    "retrieval_top_k": 5,
    "rationale": "Retrieved candidate instructions for examination entry rules.",
    "retrieval_attempts": 1
  },
  "sources": [
    {
      "citation_id": "S1",
      "chunk_id": "nsbm_exam_regulations_p3_c1",
      "document": "Examination Regulations_Instructions To Candidates.pdf",
      "page_numbers": [3]
    }
  ],
  "retrieved_chunks": [
    {
      "chunk_id": "nsbm_exam_regulations_p3_c1",
      "document": "Examination Regulations_Instructions To Candidates.pdf",
      "page_numbers": [3],
      "heading": "Admission to the Examination Hall",
      "excerpt": "Candidates will be permitted to enter the examination room fifteen minutes before the commencement of the examination...",
      "dense_score": 0.8842,
      "tfidf_score": 0.7410,
      "hybrid_score": 0.0325
    }
  ],
  "created_at": "2026-11-20T04:15:00Z"
}
```

---

## 🧪 Testing & Verification

Automated unit, integration, and server tests are managed with `pytest`:

```bash
# Run server endpoint and presentation layer tests
.venv\Scripts\python -m pytest tests/test_campusmate_server.py -v

# Run core metadata and chunking tests
.venv\Scripts\python -m pytest tests/test_document_metadata.py tests/test_heading_detector.py tests/test_chunking_normalizer.py -v

# Run full repository test suite
.venv\Scripts\python -m pytest
```

---

## 🛡️ Safety & Ethical Guidelines

1. **Decision-Support Scope**: CampusMate AI provides administrative guidance and evidence navigation. It does not possess authority to independently cancel exams, issue academic sanctions, or modify degree requirements.
2. **Abstention on Unanswerable Queries**: When an inquiry asks for facts not present in the indexed university documents, the agent explicitly abstains rather than speculating.
3. **Temporal Freshness**: Weather alerts and emergency advisories contain explicit validity timestamps to prevent expired directives from misleading students.
4. **Official Verification**: Students are always advised to verify high-stakes deadlines and extenuating circumstances claims directly with their faculty office or examination division.

---

## 👥 Contributors & Academic Context

- **Author**: K. D. P. Nimsara ('kdpnimsara@students.nsbm.ac.lk') / C.H.B.Amarathunga ('chbamarathunga@students.nsbm.ac.lk')
- **Institution**: NSBM Green University
- **Project**: Explainable-Academic-AI — Final Year Research & System Implementation
- **License**: Academic & Educational Use Only

