import pytest

from src.metadata.document_metadata import DocumentMetadata


def test_create_common_document_metadata():

    metadata = DocumentMetadata(
        document_id="DOC001",
        document_scope="common",
        document_category="disaster_management",
        document_title="Flood Preparedness Guidelines",
    )

    assert metadata.document_scope == "common"
    assert metadata.institution_id is None


def test_common_metadata_excludes_institution_id():

    metadata = DocumentMetadata(
        document_id="DOC001",
        document_scope="common",
        document_category="disaster_management",
        document_title="Flood Guidelines",
    )

    stored_metadata = metadata.to_chroma_metadata()

    assert "institution_id" not in stored_metadata
    assert stored_metadata["document_scope"] == "common"


def test_create_institution_metadata():

    metadata = DocumentMetadata(
        document_id="DOC002",
        document_scope="institution",
        document_category="examination",
        document_title="Examination Regulations",
        institution_id="UNI001",
    )

    stored_metadata = metadata.to_chroma_metadata()

    assert stored_metadata["institution_id"] == "UNI001"


def test_institution_document_requires_institution_id():

    with pytest.raises(ValueError):

        DocumentMetadata(
            document_id="DOC003",
            document_scope="institution",
            document_category="examination",
            document_title="Examination Regulations",
        )


def test_common_document_rejects_institution_id():

    with pytest.raises(ValueError):

        DocumentMetadata(
            document_id="DOC004",
            document_scope="common",
            document_category="disaster_management",
            document_title="Flood Guidelines",
            institution_id="UNI001",
        )


def test_invalid_document_scope():

    with pytest.raises(ValueError):

        DocumentMetadata(
            document_id="DOC005",
            document_scope="invalid",
            document_category="examination",
            document_title="Examination Regulations",
        )


def test_empty_document_id():

    with pytest.raises(ValueError):

        DocumentMetadata(
            document_id="",
            document_scope="common",
            document_category="disaster_management",
            document_title="Flood Guidelines",
        )