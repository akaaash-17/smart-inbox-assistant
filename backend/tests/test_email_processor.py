from datetime import datetime, timezone
from pathlib import Path

from app.schemas.ai_result import (
    AIAnalysisResult,
    ClassificationResult,
)
from app.schemas.document import (
    DocumentContent,
    DocumentPage,
)
from app.schemas.email import (
    EmailAttachment,
    EmailMessage,
)
from app.services.ai.analysis import AIAnalysisService
from app.services.email.processor import EmailProcessor


class FakePDFProcessor:
    """
    Deterministic PDF processor for unit tests.
    """

    def __init__(self):
        self.processed_paths: list[str] = []

    def process(
        self,
        file_path,
        document_id=None,
    ):
        self.processed_paths.append(
            str(file_path)
        )

        return DocumentContent(
            document_id=document_id or "test-document",
            filename=Path(file_path).name,
            pdf_type="digital",
            text="Synthetic safety report text.",
            pages=[
                DocumentPage(
                    page_number=1,
                    text="Synthetic safety report text.",
                    confidence=1.0,
                )
            ],
        )


class FakeAIAnalysisService:
    """
    Deterministic AI analysis service for unit tests.
    """

    def __init__(self):
        self.analyzed_documents: list[str] = []

    def analyze(
        self,
        document,
    ):
        self.analyzed_documents.append(
            document.document_id
        )

        return AIAnalysisResult(
            document_id=document.document_id,
            classifications=[
                ClassificationResult(
                    category="SAFETY_REPORT",
                    confidence=0.95,
                    reason=(
                        "Synthetic safety report."
                    ),
                )
            ],
            relevant=True,
            relevance_reason=(
                "Synthetic safety report."
            ),
            summary=(
                "Synthetic summary for testing."
            ),
        )


def create_email() -> EmailMessage:
    return EmailMessage(
        id="email-processor-001",
        sender="reporter@example.com",
        subject="Safety report",
        received_at=datetime(
            2026,
            9,
            1,
            10,
            30,
            tzinfo=timezone.utc,
        ),
        body="Please review the attached report.",
        attachments=[
            EmailAttachment(
                id="attachment-001",
                filename="safety_report.pdf",
                content_type="application/pdf",
                size_bytes=100,
                is_pdf=True,
            ),
            EmailAttachment(
                id="attachment-002",
                filename="notes.txt",
                content_type="text/plain",
                size_bytes=50,
                is_pdf=False,
            ),
        ],
    )


def test_process_email_processes_pdf_and_skips_non_pdf(
    tmp_path,
):
    pdf_processor = FakePDFProcessor()
    ai_service = FakeAIAnalysisService()

    processor = EmailProcessor(
        pdf_processor=pdf_processor,
        ai_analysis_service=ai_service,
        storage_directory=tmp_path,
    )

    email = create_email()

    result = processor.process_email(
        email=email,
        attachment_bytes={
            "attachment-001": b"synthetic pdf content",
        },
    )

    assert (
        result.email_id
        == "email-processor-001"
    )

    assert len(result.attachments) == 2

    pdf_result = result.attachments[0]

    assert (
        pdf_result.status
        == "PROCESSED"
    )

    assert pdf_result.is_pdf is True
    assert pdf_result.stored_path is not None
    assert pdf_result.document is not None
    assert pdf_result.analysis is not None

    assert (
        Path(
            pdf_result.stored_path
        ).exists()
    )

    assert (
        pdf_result.document.document_id
        == "attachment-001"
    )

    assert (
        ai_service.analyzed_documents
        == ["attachment-001"]
    )

    non_pdf_result = result.attachments[1]

    assert (
        non_pdf_result.status
        == "SKIPPED"
    )

    assert (
        non_pdf_result.is_pdf
        is False
    )


def test_missing_pdf_bytes_is_failed(
    tmp_path,
):
    pdf_processor = FakePDFProcessor()
    ai_service = FakeAIAnalysisService()

    processor = EmailProcessor(
        pdf_processor=pdf_processor,
        ai_analysis_service=ai_service,
        storage_directory=tmp_path,
    )

    email = create_email()

    result = processor.process_email(
        email=email,
        attachment_bytes={},
    )

    pdf_result = result.attachments[0]

    assert (
        pdf_result.status
        == "FAILED"
    )

    assert (
        pdf_result.error
        == "Attachment bytes were not provided."
    )

    assert (
        ai_service.analyzed_documents
        == []
    )


def test_one_failed_attachment_does_not_stop_other_attachments(
    tmp_path,
):
    class SelectivePDFProcessor:
        def process(
            self,
            file_path,
            document_id=None,
        ):
            if (
                document_id
                == "attachment-001"
            ):
                raise RuntimeError(
                    "Synthetic PDF processing failure."
                )

            return DocumentContent(
                document_id=document_id,
                filename=Path(file_path).name,
                pdf_type="digital",
                text="Valid document",
                pages=[
                    DocumentPage(
                        page_number=1,
                        text="Valid document",
                        confidence=1.0,
                    )
                ],
            )

    email = EmailMessage(
        id="email-failure-isolation",
        sender="reporter@example.com",
        subject="Multiple reports",
        received_at=datetime.now(
            timezone.utc
        ),
        body="Two PDF reports.",
        attachments=[
            EmailAttachment(
                id="attachment-001",
                filename="first.pdf",
                content_type="application/pdf",
                is_pdf=True,
            ),
            EmailAttachment(
                id="attachment-002",
                filename="second.pdf",
                content_type="application/pdf",
                is_pdf=True,
            ),
        ],
    )

    ai_service = FakeAIAnalysisService()

    processor = EmailProcessor(
        pdf_processor=SelectivePDFProcessor(),
        ai_analysis_service=ai_service,
        storage_directory=tmp_path,
    )

    result = processor.process_email(
        email=email,
        attachment_bytes={
            "attachment-001": b"first",
            "attachment-002": b"second",
        },
    )

    assert (
        result.attachments[0].status
        == "FAILED"
    )

    assert (
        "Synthetic PDF processing failure."
        in result.attachments[0].error
    )

    assert (
        result.attachments[1].status
        == "PROCESSED"
    )

    assert (
        ai_service.analyzed_documents
        == ["attachment-002"]
    )


def test_attachment_filename_cannot_escape_storage_directory(
    tmp_path,
):
    pdf_processor = FakePDFProcessor()
    ai_service = FakeAIAnalysisService()

    processor = EmailProcessor(
        pdf_processor=pdf_processor,
        ai_analysis_service=ai_service,
        storage_directory=tmp_path,
    )

    email = EmailMessage(
        id="email-safe-path",
        sender="reporter@example.com",
        subject="Safety report",
        received_at=datetime.now(
            timezone.utc
        ),
        body="Attached report.",
        attachments=[
            EmailAttachment(
                id="attachment-safe",
                filename="../../outside.pdf",
                content_type="application/pdf",
                is_pdf=True,
            )
        ],
    )

    result = processor.process_email(
        email=email,
        attachment_bytes={
            "attachment-safe": b"pdf",
        },
    )

    assert (
        result.attachments[0].status
        == "PROCESSED"
    )

    stored_path = Path(
        result.attachments[0].stored_path
    )

    assert (
        stored_path.parent.resolve()
        == tmp_path.resolve()
    )

    assert (
        stored_path.name
        == (
            "email-safe-path_"
            "attachment-safe_outside.pdf"
        )
    )