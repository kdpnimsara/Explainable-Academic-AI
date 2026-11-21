# System Architecture Design

## Project Title

Explainable Agentic RAG System for Weather-Aware Educational Administration

## 1. Architecture Overview

The proposed system uses a modular, multi-institution Explainable Agentic Retrieval-Augmented Generation (RAG) architecture.

It combines educational administration documents, disaster preparedness guidelines, and official weather information to support evidence-based administrative decision-making.

The architecture reuses the existing PDF ingestion, advanced chunking, embedding generation, ChromaDB storage, hybrid retrieval, answer generation, and RAG agent components.

## 2. Main Architectural Components

### 2.1 Document Ingestion Layer

Responsible for processing official PDF documents.

Responsibilities:

* PDF text extraction.
* Page number preservation.
* Document metadata extraction.
* Text normalization.
* Heading detection.
* Advanced recursive chunking.
* Chunk ID generation.

### 2.2 Embedding Layer

Uses the existing multilingual E5 embedding model.

Responsibilities:

* Generate document embeddings.
* Generate query embeddings.
* Normalize embedding vectors.
* Preserve document and chunk relationships.

### 2.3 Knowledge Storage Layer

Uses persistent ChromaDB storage.

Knowledge is logically organized into:

1. Common Disaster Knowledge.
2. Institution-Specific Educational Knowledge.

Institution-specific records are identified using metadata such as:

* institution_id
* document_id
* category
* document_version
* source_authority
* document_scope

### 2.4 Hybrid Retrieval Layer

Combines dense semantic retrieval and sparse TF-IDF retrieval.

The system uses Reciprocal Rank Fusion (RRF) to combine retrieval rankings.

The retrieval process should expose:

* TF-IDF similarity scores.
* Dense retrieval distances and similarity scores.
* Vector normalization information.
* RRF ranking contributions.
* Final hybrid scores.

Institution-specific retrieval must apply the appropriate institution filters. Common documents must remain available to all supported institutions.

### 2.5 Weather Information Layer

Provides access to current weather information through an external weather information provider.

Responsibilities:

* Retrieve weather information for a specified location.
* Identify the observation or forecast timestamp.
* Preserve the information source.
* Handle unavailable or outdated information.
* Distinguish weather forecasts from official disaster warnings.

Weather information is treated as time-sensitive data and is not assumed to remain valid indefinitely in the document vector database.

### 2.6 Agentic Decision-Support Layer

The agent coordinates retrieval, external information access, reasoning decisions, and response generation.

It selects the required tools according to the user's question.

For example:

* An examination regulation question requires institutional document retrieval.
* A current weather question requires weather information.
* A weather-related examination question may require weather information, common disaster documents, and institution-specific examination regulations.

## 3. Agentic Workflow

The system follows:

Observation -> Reasoning -> Action -> Persistence

### Observation

* Receive and validate the user question.
* Identify the requested institution, if applicable.
* Identify location and time requirements.
* Identify relevant query information.

### Reasoning

* Determine the query intent.
* Identify the knowledge sources required.
* Select appropriate retrieval tools.
* Prepare institution and document filters.
* Determine the retrieval depth.

The system records a structured explanation of its decisions.

### Action

* Retrieve relevant institutional documents.
* Retrieve common disaster preparedness documents.
* Request weather information when required.
* Evaluate retrieved evidence.
* Retry retrieval when appropriate.
* Generate an evidence-grounded response.

### Persistence

Store a structured execution trace containing:

* User question.
* Query intent.
* Selected tools.
* Institution identifier.
* Retrieved source references.
* Retrieval scores.
* Execution timestamp.
* Final response.
* Tool errors, where applicable.

## 4. Answer Generation

The answer generator uses retrieved evidence and external information returned by the selected tools.

Requirements:

* Use identifiable supporting sources.
* Include document citations and page numbers.
* Include weather information timestamps.
* Avoid unsupported claims.
* Clearly identify insufficient evidence.
* Distinguish official information from generated summaries.

The system provides decision support and does not independently make official university decisions.

## 5. Multi-Institution Architecture

The system supports multiple universities and educational institutions.

Each institution has a unique institution identifier.

Institution-specific documents must only be retrieved for the selected institution.

Common national disaster management documents may be accessed across institutions.

The system must prevent information belonging to one institution from being incorrectly presented as another institution's policy.

## 6. Mathematical Explainability

The system exposes the mathematical principles used by the retrieval pipeline.

The main concepts include:

* Term Frequency-Inverse Document Frequency (TF-IDF).
* Vector representations.
* Dense embeddings.
* Cosine similarity.
* Dot products.
* Vector magnitudes.
* Normalization.
* Reciprocal Rank Fusion.

The system records relevant retrieval scores and ranking contributions to make its retrieval decisions traceable.

## 7. Evaluation Architecture

A custom ground-truth dataset will contain at least ten question, answer, and supporting-context examples.

The system will evaluate:

1. Context Relevance.
2. Groundedness (Faithfulness).
3. Answer Relevance.

Additional tests will examine:

* Institution-level document isolation.
* Source citation correctness.
* Weather information freshness.
* Insufficient evidence handling.
* Agent tool selection.
* Retrieval failures.

## 8. Reliability and Safety

The system must:

* Prefer identifiable official information sources.
* Record information timestamps.
* Avoid presenting expired weather information as current.
* Clearly communicate retrieval failures.
* Avoid inventing institutional regulations.
* Never independently cancel examinations or close institutions.
* Provide traceable evidence for administrative decision support.

## 9. Existing Components to Reuse

The following components will be retained and adapted:

* PDFLoader.
* TextChunker.
* MultilingualE5Embedder.
* ChromaVectorStore.
* HybridRetriever.
* AnswerGenerator.
* RAGAgent.

The existing automated tests will be retained as regression tests while new features are developed.

## 10. Planned Development Order

1. Introduce document and institution metadata.
2. Adapt PDF ingestion and chunking.
3. Extend persistent storage and retrieval filters.
4. Organize common and institution-specific knowledge.
5. Integrate an official weather information provider.
6. Improve agent tool selection and evidence validation.
7. Improve mathematical explainability.
8. Develop the evaluation framework.
9. Conduct integration and regression testing.
10. Prepare technical documentation and demonstration.
