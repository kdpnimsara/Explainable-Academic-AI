from dataclasses import dataclass
from typing import Literal


DocumentScope = Literal["common", "institution"]


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Store metadata associated with an official document."""

    document_id: str
    document_scope: DocumentScope
    document_category: str
    document_title: str

    institution_id: str | None = None
    document_version: str | None = None
    source_authority: str | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        """Validate document metadata."""

        required_fields = {
            "document_id": self.document_id,
            "document_category": self.document_category,
            "document_title": self.document_title,
        }

        for field_name, value in required_fields.items():

            if not isinstance(value, str):
                raise TypeError(
                    f"{field_name} must be a string."
                )

            if not value.strip():
                raise ValueError(
                    f"{field_name} cannot be empty."
                )

            object.__setattr__(
                self,
                field_name,
                value.strip(),
            )

        if self.document_scope not in (
            "common",
            "institution",
        ):
            raise ValueError(
                "document_scope must be common or institution."
            )

        if self.document_scope == "common":

            if self.institution_id is not None:
                raise ValueError(
                    "Common documents cannot have an institution_id."
                )

        else:

            if (
                not isinstance(self.institution_id, str)
                or not self.institution_id.strip()
            ):
                raise ValueError(
                    "Institution-specific documents require "
                    "a valid institution_id."
                )

            object.__setattr__(
                self,
                "institution_id",
                self.institution_id.strip(),
            )

        optional_fields = {
            "document_version": self.document_version,
            "source_authority": self.source_authority,
            "source_url": self.source_url,
        }

        for field_name, value in optional_fields.items():

            if value is not None:

                if not isinstance(value, str):
                    raise TypeError(
                        f"{field_name} must be a string or None."
                    )

                if not value.strip():
                    raise ValueError(
                        f"{field_name} cannot be empty."
                    )

                object.__setattr__(
                    self,
                    field_name,
                    value.strip(),
                )

    def to_chroma_metadata(self) -> dict[str, str]:
        """
        Convert document metadata into ChromaDB-compatible values.

        None values are omitted because ChromaDB metadata
        does not accept None values.
        """

        metadata = {
            "document_id": self.document_id,
            "document_scope": self.document_scope,
            "document_category": self.document_category,
            "document_title": self.document_title,
        }

        if self.institution_id is not None:
            metadata["institution_id"] = self.institution_id

        if self.document_version is not None:
            metadata["document_version"] = self.document_version

        if self.source_authority is not None:
            metadata["source_authority"] = self.source_authority

        if self.source_url is not None:
            metadata["source_url"] = self.source_url

        return metadata