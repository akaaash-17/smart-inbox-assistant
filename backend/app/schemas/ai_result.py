from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.document import SourceLocation


Category = Literal[
    "SAFETY_REPORT",
    "QUALITY_COMPLAINT",
    "INFO_REQUEST",
    "NOT_RELEVANT",
]


class ClassificationResult(BaseModel):
    category: Category

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str


class ExtractedField(BaseModel):
    """
    A single AI-extracted value together with
    confidence and source evidence.
    """

    value: str = "Not stated"

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    source: SourceLocation | None = None


class PatientInformation(BaseModel):
    age: ExtractedField
    sex: ExtractedField
    weight: ExtractedField
    height: ExtractedField
    relevant_history: ExtractedField


class ReporterInformation(BaseModel):
    name: ExtractedField
    role: ExtractedField
    country: ExtractedField


class ProductInformation(BaseModel):
    name: ExtractedField
    dose: ExtractedField
    route: ExtractedField
    start_date: ExtractedField
    stop_date: ExtractedField


class ReactionInformation(BaseModel):
    description: ExtractedField
    onset: ExtractedField
    outcome: ExtractedField


class SeverityInformation(BaseModel):
    serious: ExtractedField
    death: ExtractedField
    hospitalization: ExtractedField
    life_threatening: ExtractedField


class SafetyReportExtraction(BaseModel):
    patient: PatientInformation
    reporter: ReporterInformation
    product: ProductInformation
    reaction: ReactionInformation
    severity: SeverityInformation
    narrative: ExtractedField


class QualityComplaintExtraction(BaseModel):
    product: ExtractedField
    batch_or_lot_number: ExtractedField
    issue: ExtractedField
    photo_mentioned: ExtractedField


class InfoRequestExtraction(BaseModel):
    questions: ExtractedField
    product_or_topic: ExtractedField


class AIAnalysisResult(BaseModel):
    document_id: str

    classifications: list[ClassificationResult] = Field(
        default_factory=list
    )

    relevant: bool

    relevance_reason: str

    summary: str

    safety_report: SafetyReportExtraction | None = None

    quality_complaint: QualityComplaintExtraction | None = None

    info_request: InfoRequestExtraction | None = None