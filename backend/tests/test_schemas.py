from datetime import datetime, timezone

from app.schemas.ai_result import (
    AIAnalysisResult,
    ClassificationResult,
    ExtractedField,
)
from app.schemas.document import (
    DocumentContent,
    SourceLocation,
)
from app.schemas.email import (
    EmailAttachment,
    EmailMessage,
)


def test_email_schema():
    email = EmailMessage(
        id="email-001",
        sender="doctor@example.com",
        subject="Patient reaction report",
        received_at=datetime.now(timezone.utc),
        body="Patient experienced a rash.",
        attachments=[
            EmailAttachment(
                id="attachment-001",
                filename="report.pdf",
                content_type="application/pdf",
                size_bytes=1024,
                is_pdf=True,
            )
        ],
    )

    assert email.id == "email-001"
    assert len(email.attachments) == 1
    assert email.attachments[0].is_pdf is True


def test_document_schema():
    document = DocumentContent(
        document_id="doc-001",
        filename="report.pdf",
        pdf_type="digital",
        language="en",
        text="54-year-old male experienced rash.",
    )

    assert document.document_id == "doc-001"
    assert document.pdf_type == "digital"


def test_source_location():
    source = SourceLocation(
        source_type="pdf",
        source_id="doc-001",
        page=2,
        text="54-year-old male",
    )

    assert source.page == 2
    assert source.text == "54-year-old male"


def test_classification_schema():
    result = ClassificationResult(
        category="SAFETY_REPORT",
        confidence=0.94,
        reason="Patient, product and adverse reaction are described.",
    )

    assert result.category == "SAFETY_REPORT"
    assert result.confidence == 0.94


def test_extracted_field_defaults_to_not_stated():
    field = ExtractedField(
        confidence=0.0
    )

    assert field.value == "Not stated"


def test_ai_analysis_schema():
    result = AIAnalysisResult(
        document_id="doc-001",
        classifications=[
            ClassificationResult(
                category="SAFETY_REPORT",
                confidence=0.95,
                reason="A patient experienced a reported reaction.",
            )
        ],
        relevant=True,
        relevance_reason="The document contains a patient safety case.",
        summary="The document describes a fictional patient safety event.",
    )

    assert result.document_id == "doc-001"
    assert len(result.classifications) == 1
    assert result.safety_report is None