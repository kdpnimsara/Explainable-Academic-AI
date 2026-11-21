# Project Scope

## Project Title

Explainable Agentic RAG System for Weather-Aware Educational Administration

## 1. Project Overview

This project aims to develop an Explainable Agentic Retrieval-Augmented Generation (RAG) System to support educational institutions during weather-related disruptions and disaster situations.

The system combines official weather information, disaster preparedness guidelines, and institution-specific academic and administrative documents to provide evidence-based information and decision support.

The proposed architecture is designed to support multiple universities and educational institutions rather than focusing on a single university.

## 2. Problem Statement

Universities and educational institutions may experience disruptions to examinations, lectures, registration activities, and other administrative operations due to severe weather conditions and natural disasters.

Relevant information is often distributed across weather reports, disaster management guidelines, emergency response documents, and institutional policies.

Traditional question-answering systems may fail to connect these different sources reliably or explain why particular information was retrieved.

Therefore, an explainable and agentic RAG system is required to retrieve relevant evidence from multiple sources, process complex questions, and provide traceable decision support.

## 3. Main Objectives

* Develop a multi-institution educational administration RAG system.
* Integrate official weather and disaster-related information.
* Retrieve relevant institutional policies and common disaster management documents.
* Implement an explainable agentic workflow.
* Expose mathematical retrieval mechanisms such as TF-IDF, cosine similarity, vector similarity, and Reciprocal Rank Fusion.
* Provide evidence-grounded answers with document citations.
* Evaluate the system using a custom ground-truth dataset.

## 4. Main Functional Scope

### 4.1 Weather Information

Retrieve relevant official weather information for a specified location and period.

### 4.2 Disaster Preparedness

Provide information from official disaster management guidelines, including flood and landslide preparedness.

### 4.3 Educational Administration

Retrieve institution-specific information relating to examinations, lectures, academic regulations, and administrative procedures.

### 4.4 Educational Continuity Support

Connect weather-related information with relevant educational administration policies to support institutional continuity planning.

### 4.5 Explainable Agentic RAG

Implement the following workflow:

Observation → Reasoning → Action → Persistence

The system should provide understandable explanations of its retrieval decisions, evidence sources, and actions.

## 5. Multi-Institution Support

The architecture will support multiple universities and educational institutions.

Institution-specific documents will be identified using metadata such as institution ID, document ID, category, and document version.

Common national disaster management information may be shared across institutions, while institution-specific policies will remain separately identifiable.

## 6. Non-Functional Requirements

* Modular Object-Oriented Python architecture.
* Dependency Injection.
* Persistent vector storage.
* Explainable retrieval calculations.
* Evidence-grounded answer generation.
* Reproducible software environment.
* Automated testing and evaluation.
* Traceable Git development history.

## 7. Project Boundaries

The system provides decision support rather than making official university decisions.

It does not independently cancel examinations, close educational institutions, or issue official weather or disaster warnings.

Weather and hazard information must be obtained from identifiable and reliable sources.

The system must clearly communicate uncertainty and insufficient evidence.

## 8. Evaluation

The system will be evaluated using a custom ground-truth dataset containing at least ten questions, expected answers, and supporting contexts.

Evaluation metrics will include:

* Context Relevance
* Groundedness (Faithfulness)
* Answer Relevance

## 9. Expected Outcome

A reusable and explainable Agentic RAG system capable of connecting weather and disaster information with educational administration requirements across multiple institutions.
