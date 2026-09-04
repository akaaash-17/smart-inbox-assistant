from datetime import datetime

from pydantic import BaseModel, Field


class EmailAttachment(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int = 0
    is_pdf: bool = False


class EmailMessage(BaseModel):
    id: str
    sender: str
    subject: str
    received_at: datetime
    body: str

    attachments: list[EmailAttachment] = Field(default_factory=list)