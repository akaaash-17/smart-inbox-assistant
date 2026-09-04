from copy import deepcopy

from app.schemas.ai_result import AIAnalysisResult
from app.schemas.email import EmailMessage
from app.repositories.interface import (
    InboxRepository,
    RepositoryError,
)


class InMemoryInboxRepository(InboxRepository):
    """
    In-memory repository used for development and testing.

    The implementation mirrors the repository contract that will
    later be backed by Oracle.
    """

    def __init__(self):
        self._emails: dict[
            str,
            EmailMessage,
        ] = {}

        self._analyses: dict[
            str,
            AIAnalysisResult,
        ] = {}

    def save_email(
        self,
        email: EmailMessage,
    ) -> None:
        """
        Persist an email.

        Existing IDs are replaced intentionally so that an
        ingestion retry remains idempotent.
        """

        if not email.id.strip():
            raise RepositoryError(
                "Email ID cannot be empty."
            )

        self._emails[email.id] = deepcopy(
            email
        )

    def get_email(
        self,
        email_id: str,
    ) -> EmailMessage | None:
        """
        Retrieve a copy of an email by ID.
        """

        email = self._emails.get(
            email_id
        )

        if email is None:
            return None

        return deepcopy(email)

    def save_analysis(
        self,
        analysis: AIAnalysisResult,
    ) -> None:
        """
        Persist an AI analysis result.

        Existing document IDs are replaced intentionally to support
        deterministic reprocessing.
        """

        if not analysis.document_id.strip():
            raise RepositoryError(
                "Document ID cannot be empty."
            )

        self._analyses[
            analysis.document_id
        ] = deepcopy(analysis)

    def get_analysis(
        self,
        document_id: str,
    ) -> AIAnalysisResult | None:
        """
        Retrieve a copy of an analysis by document ID.
        """

        analysis = self._analyses.get(
            document_id
        )

        if analysis is None:
            return None

        return deepcopy(analysis)

    def list_emails(
        self,
    ) -> list[EmailMessage]:
        """
        Return all persisted emails.
        """

        return [
            deepcopy(email)
            for email in self._emails.values()
        ]

    def list_analyses(
        self,
    ) -> list[AIAnalysisResult]:
        """
        Return all persisted analyses.
        """

        return [
            deepcopy(analysis)
            for analysis in self._analyses.values()
        ]

    def clear(self) -> None:
        """
        Clear all stored records.

        Primarily useful for tests and local development.
        """

        self._emails.clear()
        self._analyses.clear()