from typing import Literal

from pydantic import BaseModel, Field


PDFType = Literal[
    "digital",
    "scanned",
    "handwritten",
    "article",
    "non_english",
    "unknown",
]


class SourceLocation(BaseModel):
    """
    Identifies exactly where extracted information originated.
    """

    source_type: Literal["email", "pdf"]
    source_id: str

    page: int | None = None

    text: str | None = None


class DocumentPage(BaseModel):
    page_number: int

    text: str = ""

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class ExtractedTable(BaseModel):
    name: str | None = None

    columns: list[str] = Field(
        default_factory=list
    )

    rows: list[dict[str, str]] = Field(
        default_factory=list
    )

    source: SourceLocation | None = None


class ExtractedImage(BaseModel):
    description: str

    requires_review: bool = True

    source: SourceLocation | None = None


class DocumentContent(BaseModel):
    document_id: str
    filename: str

    pdf_type: PDFType = "unknown"

    language: str | None = None

    text: str = ""

    pages: list[DocumentPage] = Field(
        default_factory=list
    )

    tables: list[ExtractedTable] = Field(
        default_factory=list
    )

    images: list[ExtractedImage] = Field(
        default_factory=list
    )