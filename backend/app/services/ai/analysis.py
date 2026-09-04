from app.schemas.ai_result import (
    AIAnalysisResult,
    ExtractedField,
    InfoRequestExtraction,
    QualityComplaintExtraction,
    SafetyReportExtraction,
)
from app.schemas.document import DocumentContent
from app.services.ai.classifier import DocumentClassifier
from app.services.ai.evidence_resolver import EvidenceResolver
from app.services.ai.extractor import DocumentExtractor


class AIAnalysisService:
    """
    Orchestrates classification, domain extraction, and
    evidence resolution for a processed document.
    """

    def __init__(
        self,
        classifier: DocumentClassifier,
        extractor: DocumentExtractor,
        evidence_resolver: EvidenceResolver,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.evidence_resolver = evidence_resolver

    def analyze(
        self,
        document: DocumentContent,
    ) -> AIAnalysisResult:
        """
        Perform complete AI analysis on a processed document.
        """

        if not document.text.strip():
            raise ValueError(
                "Cannot analyze a document without text."
            )

        classification_result = self.classifier.classify(
            document_id=document.document_id,
            text=document.text,
        )

        extraction_result = self.extractor.extract(
            document_id=document.document_id,
            text=document.text,
        )

        safety_report = self._resolve_safety_report(
            extraction_result.safety_report,
            document,
        )

        quality_complaint = self._resolve_quality_complaint(
            extraction_result.quality_complaint,
            document,
        )

        info_request = self._resolve_info_request(
            extraction_result.info_request,
            document,
        )

        relevant = any(
            classification.category != "NOT_RELEVANT"
            for classification in (
                classification_result.classifications
            )
        )

        relevance_reason = self._build_relevance_reason(
            classification_result.classifications
        )

        return AIAnalysisResult(
            document_id=document.document_id,
            classifications=classification_result.classifications,
            relevant=relevant,
            relevance_reason=relevance_reason,
            summary="Not generated yet.",
            safety_report=safety_report,
            quality_complaint=quality_complaint,
            info_request=info_request,
        )

    def _resolve_safety_report(
        self,
        extraction: SafetyReportExtraction | None,
        document: DocumentContent,
    ) -> SafetyReportExtraction | None:
        if extraction is None:
            return None

        extraction.patient.age = self._resolve_field(
            extraction.patient.age,
            document,
        )

        extraction.patient.sex = self._resolve_field(
            extraction.patient.sex,
            document,
        )

        extraction.patient.weight = self._resolve_field(
            extraction.patient.weight,
            document,
        )

        extraction.patient.height = self._resolve_field(
            extraction.patient.height,
            document,
        )

        extraction.patient.relevant_history = (
            self._resolve_field(
                extraction.patient.relevant_history,
                document,
            )
        )

        extraction.reporter.name = self._resolve_field(
            extraction.reporter.name,
            document,
        )

        extraction.reporter.role = self._resolve_field(
            extraction.reporter.role,
            document,
        )

        extraction.reporter.country = self._resolve_field(
            extraction.reporter.country,
            document,
        )

        extraction.product.name = self._resolve_field(
            extraction.product.name,
            document,
        )

        extraction.product.dose = self._resolve_field(
            extraction.product.dose,
            document,
        )

        extraction.product.route = self._resolve_field(
            extraction.product.route,
            document,
        )

        extraction.product.start_date = self._resolve_field(
            extraction.product.start_date,
            document,
        )

        extraction.product.stop_date = self._resolve_field(
            extraction.product.stop_date,
            document,
        )

        extraction.reaction.description = self._resolve_field(
            extraction.reaction.description,
            document,
        )

        extraction.reaction.onset = self._resolve_field(
            extraction.reaction.onset,
            document,
        )

        extraction.reaction.outcome = self._resolve_field(
            extraction.reaction.outcome,
            document,
        )

        extraction.severity.serious = self._resolve_field(
            extraction.severity.serious,
            document,
        )

        extraction.severity.death = self._resolve_field(
            extraction.severity.death,
            document,
        )

        extraction.severity.hospitalization = (
            self._resolve_field(
                extraction.severity.hospitalization,
                document,
            )
        )

        extraction.severity.life_threatening = (
            self._resolve_field(
                extraction.severity.life_threatening,
                document,
            )
        )

        extraction.narrative = self._resolve_field(
            extraction.narrative,
            document,
        )

        return extraction

    def _resolve_quality_complaint(
        self,
        extraction: QualityComplaintExtraction | None,
        document: DocumentContent,
    ) -> QualityComplaintExtraction | None:
        if extraction is None:
            return None

        extraction.product = self._resolve_field(
            extraction.product,
            document,
        )

        extraction.batch_or_lot_number = (
            self._resolve_field(
                extraction.batch_or_lot_number,
                document,
            )
        )

        extraction.issue = self._resolve_field(
            extraction.issue,
            document,
        )

        extraction.photo_mentioned = (
            self._resolve_field(
                extraction.photo_mentioned,
                document,
            )
        )

        return extraction

    def _resolve_info_request(
        self,
        extraction: InfoRequestExtraction | None,
        document: DocumentContent,
    ) -> InfoRequestExtraction | None:
        if extraction is None:
            return None

        extraction.questions = self._resolve_field(
            extraction.questions,
            document,
        )

        extraction.product_or_topic = self._resolve_field(
            extraction.product_or_topic,
            document,
        )

        return extraction

    def _resolve_field(
        self,
        field: ExtractedField,
        document: DocumentContent,
    ) -> ExtractedField:
        return self.evidence_resolver.resolve_field(
            field=field,
            document=document,
        )

    @staticmethod
    def _build_relevance_reason(
        classifications,
    ) -> str:
        if not classifications:
            return "No classification was produced."

        if len(classifications) == 1:
            return classifications[0].reason

        return " ".join(
            classification.reason
            for classification in classifications
        )