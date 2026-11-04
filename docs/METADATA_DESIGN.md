# Multi-Institution Metadata Design

## 1. Purpose

This document defines the metadata structure required to support multiple universities and educational institutions within the Explainable Agentic RAG System.

The metadata design ensures that institutional documents are correctly identified, stored, and retrieved without mixing information between institutions.

## 2. Document Metadata Fields

Each document will contain the following metadata:

| Field             | Description                          |
| ----------------- | ------------------------------------ |
| document_id       | Unique document identifier           |
| institution_id    | Identifier of the owning institution |
| document_scope    | Common or institution-specific       |
| document_category | Document category                    |
| document_title    | Official document title              |
| document_version  | Document version                     |
| source_authority  | Issuing authority                    |
| source_url        | Original document location           |

## 3. Document Scope

Two document scopes will be supported.

### 3.1 Common Documents

Common documents contain information that can be shared across supported institutions.

Examples:

* National disaster management guidelines.
* Flood preparedness guidelines.
* Landslide preparedness guidelines.
* Common emergency management policies.

Metadata:

* document_scope: common
* institution_id: None

#### ChromaDB Metadata Handling

For common documents, institution_id may be represented as None in the application layer.

However, None values will not be stored in ChromaDB metadata.

The institution_id field will be omitted when storing common documents.

The document_scope field will be stored as common.

When an institution-specific query is performed, the retrieval filter must include:

1. Common documents.
2. Documents belonging to the selected institution.

Documents belonging to other institutions must be excluded.

### 3.2 Institution-Specific Documents

These documents contain rules and policies belonging to a particular educational institution.

Examples:

* Examination regulations.
* Student handbooks.
* Academic policies.
* Registration guidelines.
* Institutional emergency response plans.

Metadata:

* document_scope: institution
* institution_id: Unique institution identifier.

Every institution-specific document must have a valid institution identifier.

## 4. Institution Identification

Each supported institution will receive a unique identifier.

Examples:

* UNI001
* UNI002
* UNI003

These identifiers are examples and will be assigned through an institution registry.

## 5. Document Categories

The initial document categories will include:

* examination
* academic_regulations
* student_administration
* registration
* attendance
* disaster_management
* emergency_preparedness
* educational_continuity

Additional categories may be introduced when required.

## 6. Metadata Preservation

Metadata must remain available throughout the complete document processing pipeline.

The pipeline includes:

PDF Loading -> Text Chunking -> Embedding Generation -> Vector Storage -> Hybrid Retrieval -> Answer Generation

Every retrieved chunk must retain its document and institution identification.

## 7. Retrieval Isolation

The retrieval system must support two types of queries.

### 7.1 Institution-Specific Queries

When an institution is selected, the system may retrieve:

1. Common documents.
2. Documents belonging to the selected institution.

Documents belonging to other institutions must be excluded.

Both Dense Retrieval and TF-IDF Retrieval must apply consistent institution-level filtering.

### 7.2 Common Queries

Queries that do not require institution-specific information may retrieve relevant common documents.

The system must not present institution-specific regulations as universally applicable.

If an institution-specific question is received without a valid institution identifier, the system must request the required institution context or clearly state that institution-specific information cannot be verified.

## 8. Backward Compatibility

Existing SafeSteps LK documents and previously developed RAG components must remain functional.

New metadata fields should use backward-compatible defaults where appropriate.

Existing automated tests must continue to pass unless a documented behaviour change i
