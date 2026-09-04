from datetime import datetime

from pydantic import BaseModel, Field


class EmailAttachment(BaseModel):
    """
    Metadata and storage information for an email attachment.
    """

    id: str
    filename: str
    content_type: str
    size_bytes: int = 0
    is_pdf: bool = False

    stored_path: str | None = None


class EmailMessage(BaseModel):
    """
    Canonical representation of an incoming email.
    """

    id: str
    sender: str
    subject: str
    received_at: datetime
    body: str

    attachments: list[EmailAttachment] = Field(
        default_factory=list
    )