from datetime import timezone
from email.message import EmailMessage as MIMEEmailMessage

import pytest

from app.services.email.parser import EmailParser


def build_email(
    include_pdf: bool = True,
    include_text_file: bool = True,
) -> bytes:
    """
    Build a deterministic synthetic email for testing.
    """

    message = MIMEEmailMessage()

    message["From"] = "reporter@example.com"
    message["To"] = "inbox@example.com"
    message["Subject"] = "Safety report - MedX"
    message["Date"] = (
        "Tue, 1 Sep 2026 10:30:00 +0530"
    )

    message.set_content(
        "A patient developed a skin rash after "
        "taking MedX 10 mg."
    )

    if include_pdf:
        message.add_attachment(
            b"%PDF-1.4 synthetic pdf content",
            maintype="application",
            subtype="pdf",
            filename="safety_report.pdf",
        )

    if include_text_file:
        message.add_attachment(
            b"administrative notes",
            maintype="text",
            subtype="plain",
            filename="notes.txt",
        )

    return message.as_bytes()


def test_parse_email_metadata():
    parser = EmailParser()

    result = parser.parse(
        raw_message=build_email(
            include_pdf=False,
            include_text_file=False,
        ),
        message_id="email-001",
    )

    assert result.id == "email-001"
    assert (
        result.sender
        == "reporter@example.com"
    )
    assert (
        result.subject
        == "Safety report - MedX"
    )

    assert (
        result.received_at.year
        == 2026
    )

    assert (
        result.received_at.month
        == 9
    )

    assert (
        result.received_at.day
        == 1
    )

    assert (
        result.received_at.tzinfo
        is not None
    )

    assert (
        result.received_at.astimezone(
            timezone.utc
        ).hour
        == 5
    )

    assert (
        "patient developed a skin rash"
        in result.body
    )


def test_parse_pdf_attachment():
    parser = EmailParser()

    result = parser.parse(
        raw_message=build_email(),
        message_id="email-002",
    )

    assert len(result.attachments) == 2

    pdf_attachment = next(
        attachment
        for attachment in result.attachments
        if attachment.filename
        == "safety_report.pdf"
    )

    assert (
        pdf_attachment.content_type
        == "application/pdf"
    )

    assert (
        pdf_attachment.is_pdf
        is True
    )

    assert (
        pdf_attachment.size_bytes
        > 0
    )


def test_non_pdf_attachment_is_logged():
    parser = EmailParser()

    result = parser.parse(
        raw_message=build_email(
            include_pdf=False,
            include_text_file=True,
        ),
        message_id="email-003",
    )

    assert len(result.attachments) == 1

    attachment = result.attachments[0]

    assert (
        attachment.filename
        == "notes.txt"
    )

    assert (
        attachment.is_pdf
        is False
    )

    assert (
        attachment.content_type
        == "text/plain"
    )


def test_parse_empty_email_rejected():
    parser = EmailParser()

    with pytest.raises(ValueError):
        parser.parse(
            raw_message=b"",
            message_id="email-empty",
        )


def test_multipart_email_prefers_plain_text():
    parser = EmailParser()

    message = MIMEEmailMessage()

    message["From"] = "doctor@example.com"
    message["Subject"] = "Medical information request"
    message["Date"] = (
        "Tue, 1 Sep 2026 12:00:00 +0530"
    )

    message.set_content(
        "Plain text medical information request."
    )

    message.add_alternative(
        """
        <html>
            <body>
                <p>HTML medical information request.</p>
            </body>
        </html>
        """,
        subtype="html",
    )

    result = parser.parse(
        raw_message=message.as_bytes(),
        message_id="email-004",
    )

    assert (
        "Plain text medical information request."
        in result.body
    )