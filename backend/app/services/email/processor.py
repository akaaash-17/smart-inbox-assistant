from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.repositories.interface import InboxRepository
from app.schemas.ai_result import AIAnalysisResult
from app.schemas.document import DocumentContent
from app.schemas.email import EmailAttachment, EmailMessage
from app.services.ai.analysis import AIAnalysisService
from app.services.pdf_processor import PDFProcessor


@dataclass
class AttachmentProcessingResult:
    """
    Result of processing one email attachment.
    """

    attachment_id: str
    filename: str
    is_pdf: bool
    status: str
    stored_path: str | None = None
    document: DocumentContent | None = None
    analysis: AIAnalysisResult | None = None
    error: str | None = None


@dataclass
class EmailProcessingResult:
    """
    Result of processing all attachments belonging to one email.
    """

    email_id: str
    attachments: list[AttachmentProcessingResult]


class EmailProcessor:
    """
    Orchestrates email persistence, attachment processing,
    PDF analysis, and AI result persistence.

    Responsibilities:
    - persist the incoming email
    - process every attachment
    - persist PDF attachments
    - process PDF documents
    - run AI analysis
    - persist AI analysis results
    - skip non-PDF attachments
    - isolate attachment failures
    """

    def __init__(
        self,
        pdf_processor: PDFProcessor,
        ai_analysis_service: AIAnalysisService,
        repository: InboxRepository,
        storage_directory: str | Path = "data/pdfs",
        file_writer: Callable[[Path, bytes], None] | None = None,
    ):
        self.pdf_processor = pdf_processor
        self.ai_analysis_service = ai_analysis_service
        self.repository = repository

        self.storage_directory = Path(
            storage_directory
        )

        self.file_writer = (
            file_writer
            or self._write_file
        )

    def process_email(
        self,
        email: EmailMessage,
        attachment_bytes: dict[str, bytes],
    ) -> EmailProcessingResult:
        """
        Persist and process every attachment associated
        with an email.

        attachment_bytes is keyed by EmailAttachment.id.
        """

        self.repository.save_email(email)

        results: list[
            AttachmentProcessingResult
        ] = []

        for attachment in email.attachments:
            result = self._process_attachment(
                email=email,
                attachment=attachment,
                attachment_bytes=attachment_bytes.get(
                    attachment.id
                ),
            )

            results.append(result)

        return EmailProcessingResult(
            email_id=email.id,
            attachments=results,
        )

    def _process_attachment(
        self,
        email: EmailMessage,
        attachment: EmailAttachment,
        attachment_bytes: bytes | None,
    ) -> AttachmentProcessingResult:
        """
        Process one attachment independently.

        An error here is converted into a failed result rather
        than stopping processing of other attachments.
        """

        if not attachment.is_pdf:
            return AttachmentProcessingResult(
                attachment_id=attachment.id,
                filename=attachment.filename,
                is_pdf=False,
                status="SKIPPED",
            )

        if attachment_bytes is None:
            return AttachmentProcessingResult(
                attachment_id=attachment.id,
                filename=attachment.filename,
                is_pdf=True,
                status="FAILED",
                error=(
                    "Attachment bytes were not provided."
                ),
            )

        try:
            stored_path = self._store_attachment(
                email_id=email.id,
                attachment=attachment,
                content=attachment_bytes,
            )

            document = self.pdf_processor.process(
                file_path=stored_path,
                document_id=attachment.id,
            )

            analysis = self.ai_analysis_service.analyze(
                document
            )

            self.repository.save_analysis(
                analysis
            )

            return AttachmentProcessingResult(
                attachment_id=attachment.id,
                filename=attachment.filename,
                is_pdf=True,
                status="PROCESSED",
                stored_path=str(stored_path),
                document=document,
                analysis=analysis,
            )

        except Exception as exc:
            return AttachmentProcessingResult(
                attachment_id=attachment.id,
                filename=attachment.filename,
                is_pdf=True,
                status="FAILED",
                error=str(exc),
            )

    def _store_attachment(
        self,
        email_id: str,
        attachment: EmailAttachment,
        content: bytes,
    ) -> Path:
        self.storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_filename = self._sanitize_filename(
            attachment.filename
        )

        destination = (
            self.storage_directory
            / f"{email_id}_{attachment.id}_{safe_filename}"
        )

        self.file_writer(
            destination,
            content,
        )

        return destination

    @staticmethod
    def _write_file(
        path: Path,
        content: bytes,
    ) -> None:
        path.write_bytes(content)

    @staticmethod
    def _sanitize_filename(
        filename: str,
    ) -> str:
        return Path(filename).name