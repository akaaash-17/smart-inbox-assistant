from app.schemas.document import (
    DocumentContent,
    DocumentPage,
)
from app.services.ai.analysis import AIAnalysisService
from app.services.ai.classifier import DocumentClassifier
from app.services.ai.evidence_resolver import EvidenceResolver
from app.services.ai.extractor import DocumentExtractor
from app.services.ai.provider import AIProvider
from app.services.ai.summarizer import DocumentSummarizer


class FakeAIProvider(AIProvider):
    """
    Deterministic provider that returns:
    1. classification JSON
    2. extraction JSON
    3. summary text
    """

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1

        if self.calls == 1:
            return """
            {
              "classifications": [
                {
                  "category": "SAFETY_REPORT",
                  "confidence": 0.96,
                  "reason": "A specific patient experienced an adverse reaction."
                }
              ]
            }
            """

        if self.calls == 2:
            return """
            {
              "safety_report": {
                "patient": {
                  "age": {
                    "value": "54",
                    "confidence": 0.99
                  },
                  "sex": {
                    "value": "Male",
                    "confidence": 0.99
                  },
                  "weight": {
                    "value": "Not stated",
                    "confidence": 1.0
                  },
                  "height": {
                    "value": "Not stated",
                    "confidence": 1.0
                  },
                  "relevant_history": {
                    "value": "Not stated",
                    "confidence": 1.0
                  }
                },
                "reporter": {
                  "name": {
                    "value": "Dr. John Smith",
                    "confidence": 0.98
                  },
                  "role": {
                    "value": "Physician",
                    "confidence": 0.98
                  },
                  "country": {
                    "value": "India",
                    "confidence": 0.97
                  }
                },
                "product": {
                  "name": {
                    "value": "MedX 10 mg",
                    "confidence": 0.99
                  },
                  "dose": {
                    "value": "10 mg",
                    "confidence": 0.98
                  },
                  "route": {
                    "value": "Oral",
                    "confidence": 0.98
                  },
                  "start_date": {
                    "value": "Not stated",
                    "confidence": 1.0
                  },
                  "stop_date": {
                    "value": "Not stated",
                    "confidence": 1.0
                  }
                },
                "reaction": {
                  "description": {
                    "value": "Skin rash",
                    "confidence": 0.99
                  },
                  "onset": {
                    "value": "2 days after starting",
                    "confidence": 0.96
                  },
                  "outcome": {
                    "value": "Recovered",
                    "confidence": 0.98
                  }
                },
                "severity": {
                  "serious": {
                    "value": "Not stated",
                    "confidence": 1.0
                  },
                  "death": {
                    "value": "Not stated",
                    "confidence": 1.0
                  },
                  "hospitalization": {
                    "value": "Not stated",
                    "confidence": 1.0
                  },
                  "life_threatening": {
                    "value": "Not stated",
                    "confidence": 1.0
                  }
                },
                "narrative": {
                  "value": "A 54-year-old male developed a skin rash after starting MedX 10 mg orally.",
                  "confidence": 0.95
                }
              },
              "quality_complaint": null,
              "info_request": null
            }
            """

        return (
            "The document reports a safety event involving a specific patient. "
            "The patient is a 54-year-old male. "
            "The reported product is MedX 10 mg. "
            "The product was administered orally. "
            "The reported dose was 10 mg. "
            "The patient developed a skin rash after starting the product. "
            "The reaction reportedly began two days after treatment started. "
            "The reported outcome was recovery. "
            "The reporter is identified as Dr. John Smith. "
            "The reporter is described as a physician. "
            "The reporter is associated with India. "
            "The document contains information relevant to a potential safety report."
        )


def create_document() -> DocumentContent:
    text = (
        "SAFETY REPORT\n"
        "Patient Age: 54\n"
        "Patient Sex: Male\n"
        "Product: MedX 10 mg\n"
        "Dose: 10 mg\n"
        "Route: Oral\n"
        "Reaction: Skin rash\n"
        "Onset: 2 days after starting\n"
        "Outcome: Recovered\n"
        "Reporter: Dr. John Smith\n"
        "Reporter Role: Physician\n"
        "Country: India"
    )

    return DocumentContent(
        document_id="doc-analysis-001",
        filename="safety_report.pdf",
        pdf_type="digital",
        text=text,
        pages=[
            DocumentPage(
                page_number=1,
                text=text,
                confidence=1.0,
            )
        ],
    )


def create_service(
    provider: FakeAIProvider,
) -> AIAnalysisService:
    return AIAnalysisService(
        classifier=DocumentClassifier(provider),
        extractor=DocumentExtractor(provider),
        evidence_resolver=EvidenceResolver(),
        summarizer=DocumentSummarizer(provider),
    )


def test_integrated_ai_analysis():
    provider = FakeAIProvider()

    service = create_service(provider)

    document = create_document()

    result = service.analyze(document)

    assert result.document_id == "doc-analysis-001"
    assert result.relevant is True

    assert len(result.classifications) == 1

    assert (
        result.classifications[0].category
        == "SAFETY_REPORT"
    )

    assert result.safety_report is not None

    safety = result.safety_report

    assert safety.patient.age.value == "54"
    assert safety.patient.age.source is not None
    assert safety.patient.age.source.page == 1
    assert (
        safety.patient.age.source.text
        == "Patient Age: 54"
    )

    assert safety.patient.sex.value == "Male"
    assert safety.patient.sex.source is not None

    assert safety.product.name.value == "MedX 10 mg"
    assert safety.product.name.source is not None
    assert (
        safety.product.name.source.text
        == "Product: MedX 10 mg"
    )

    assert safety.reaction.description.value == "Skin rash"
    assert safety.reaction.description.source is not None

    assert (
        safety.patient.weight.value
        == "Not stated"
    )
    assert safety.patient.weight.source is None

    assert result.quality_complaint is None
    assert result.info_request is None

    assert (
        result.relevance_reason
        == "A specific patient experienced an adverse reaction."
    )

    assert result.summary != "Not generated yet."
    assert result.summary.strip()

    sentence_count = (
        DocumentSummarizer._count_sentences(
            result.summary
        )
    )

    assert (
        DocumentSummarizer.MIN_SENTENCES
        <= sentence_count
        <= DocumentSummarizer.MAX_SENTENCES
    )

    assert provider.calls == 3


def test_analysis_rejects_empty_document():
    provider = FakeAIProvider()

    service = create_service(provider)

    empty_document = DocumentContent(
        document_id="doc-empty",
        filename="empty.pdf",
        pdf_type="digital",
        text="",
        pages=[],
    )

    try:
        service.analyze(empty_document)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert (
            str(exc)
            == "Cannot analyze a document without text."
        )