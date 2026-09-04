from dataclasses import dataclass

from app.schemas.ai_result import ExtractedField
from app.schemas.document import (
    DocumentContent,
    SourceLocation,
)


@dataclass
class EvidenceMatch:
    """
    Evidence found in the original document.
    """

    page: int
    text: str


class EvidenceResolver:
    """
    Resolves AI-extracted values back to the original document.

    The resolver does not ask the LLM for page numbers. Instead,
    it searches the original page-level evidence produced by the
    document processor.

    This prevents the AI model from inventing source locations.
    """

    NOT_STATED = "Not stated"

    def resolve_field(
        self,
        field: ExtractedField,
        document: DocumentContent,
    ) -> ExtractedField:
        """
        Attach source evidence to an extracted field when the
        value can be located in the original document.

        Missing values remain source-less.
        """

        if self._is_not_stated(field.value):
            return field.model_copy(
                update={"source": None}
            )

        match = self._find_exact_value(
            value=field.value,
            document=document,
        )

        if match is None:
            return field.model_copy(
                update={"source": None}
            )

        return field.model_copy(
            update={
                "source": SourceLocation(
                    source_type="pdf",
                    source_id=document.document_id,
                    page=match.page,
                    text=match.text,
                )
            }
        )

    def _find_exact_value(
        self,
        value: str,
        document: DocumentContent,
    ) -> EvidenceMatch | None:
        """
        Search page-level text for an exact extracted value.

        Matching is case-insensitive and whitespace-normalized.
        """

        normalized_value = self._normalize(value)

        if not normalized_value:
            return None

        for page in document.pages:
            page_text = page.text.strip()

            if not page_text:
                continue

            normalized_page = self._normalize(
                page_text
            )

            if normalized_value in normalized_page:
                return EvidenceMatch(
                    page=page.page_number,
                    text=self._find_evidence_line(
                        value=value,
                        page_text=page_text,
                    ),
                )

        return None

    @staticmethod
    def _find_evidence_line(
        value: str,
        page_text: str,
    ) -> str:
        """
        Return the most relevant source line containing the
        extracted value.

        If a matching line cannot be isolated, the full page
        text is retained as evidence.
        """

        normalized_value = (
            EvidenceResolver._normalize(value)
        )

        for line in page_text.splitlines():
            if normalized_value in EvidenceResolver._normalize(
                line
            ):
                return line.strip()

        return page_text

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize whitespace and case for comparison.
        """

        return " ".join(text.lower().split())

    @classmethod
    def _is_not_stated(
        cls,
        value: str,
    ) -> bool:
        return (
            not value.strip()
            or value.strip().lower()
            == cls.NOT_STATED.lower()
        )