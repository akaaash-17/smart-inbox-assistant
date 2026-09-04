from abc import ABC, abstractmethod

from app.schemas.ai_result import AIAnalysisResult
from app.schemas.email import EmailMessage


class RepositoryError(Exception):
    """
    Base exception for persistence-related failures.
    """


class InboxRepository(ABC):
    """
    Abstract persistence interface for the Smart Inbox Assistant.

    Business services depend on this interface rather than a
    specific database implementation.
    """

    @abstractmethod
    def save_email(
        self,
        email: EmailMessage,
    ) -> None:
        """
        Persist an incoming email.
        """
        raise NotImplementedError

    @abstractmethod
    def get_email(
        self,
        email_id: str,
    ) -> EmailMessage | None:
        """
        Retrieve an email by ID.
        """
        raise NotImplementedError

    @abstractmethod
    def save_analysis(
        self,
        analysis: AIAnalysisResult,
    ) -> None:
        """
        Persist an AI analysis result.
        """
        raise NotImplementedError

    @abstractmethod
    def get_analysis(
        self,
        document_id: str,
    ) -> AIAnalysisResult | None:
        """
        Retrieve an AI analysis by document ID.
        """
        raise NotImplementedError

    @abstractmethod
    def list_emails(
        self,
    ) -> list[EmailMessage]:
        """
        Return all persisted emails.
        """
        raise NotImplementedError

    @abstractmethod
    def list_analyses(
        self,
    ) -> list[AIAnalysisResult]:
        """
        Return all persisted analyses.
        """
        raise NotImplementedError