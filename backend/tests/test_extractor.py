import pytest

from app.services.ai.extractor import (
    AIExtractionError,
    DocumentExtractor,
)
from app.services.ai.provider import AIProvider


class FakeAIProvider(AIProvider):
    """
    Deterministic provider used for extraction tests.
    """

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_extract_safety_report():
    provider = FakeAIProvider(
        """
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
                "value": "10 mg once daily",
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
              "value": "A 54-year-old male developed a skin rash two days after starting MedX 10 mg orally and subsequently recovered.",
              "confidence": 0.95
            }
          },
          "quality_complaint": null,
          "info_request": null
        }
        """
    )

    extractor = DocumentExtractor(provider)

    result = extractor.extract(
        document_id="doc-safety-001",
        text=(
            "A 54-year-old male developed a skin rash "
            "two days after starting MedX 10 mg orally. "
            "The patient recovered. Reporter: Dr. John Smith, "
            "Physician, India."
        ),
    )

    assert result.safety_report is not None

    safety = result.safety_report

    assert safety.patient.age.value == "54"
    assert safety.patient.sex.value == "Male"

    assert (
        safety.patient.weight.value
        == "Not stated"
    )

    assert (
        safety.patient.relevant_history.value
        == "Not stated"
    )

    assert (
        safety.reporter.name.value
        == "Dr. John Smith"
    )

    assert safety.reporter.role.value == "Physician"
    assert safety.reporter.country.value == "India"

    assert (
        safety.product.name.value
        == "MedX 10 mg"
    )

    assert safety.product.route.value == "Oral"

    assert (
        safety.reaction.description.value
        == "Skin rash"
    )

    assert (
        safety.reaction.onset.value
        == "2 days after starting"
    )

    assert (
        safety.reaction.outcome.value
        == "Recovered"
    )

    assert (
        safety.severity.death.value
        == "Not stated"
    )

    assert (
        safety.severity.hospitalization.value
        == "Not stated"
    )

    assert result.quality_complaint is None
    assert result.info_request is None


def test_extract_quality_complaint():
    provider = FakeAIProvider(
        """
        {
          "safety_report": null,
          "quality_complaint": {
            "product": {
              "value": "MedX 20 mg",
              "confidence": 0.98
            },
            "batch_or_lot_number": {
              "value": "BATCH-1234",
              "confidence": 0.96
            },
            "issue": {
              "value": "Broken seal",
              "confidence": 0.99
            },
            "photo_mentioned": {
              "value": "Yes",
              "confidence": 0.95
            }
          },
          "info_request": null
        }
        """
    )

    extractor = DocumentExtractor(provider)

    result = extractor.extract(
        document_id="doc-pqc-001",
        text=(
            "The customer received MedX 20 mg from batch "
            "BATCH-1234 with a broken seal. A photo is attached."
        ),
    )

    assert result.safety_report is None
    assert result.info_request is None

    assert result.quality_complaint is not None

    complaint = result.quality_complaint

    assert complaint.product.value == "MedX 20 mg"
    assert (
        complaint.batch_or_lot_number.value
        == "BATCH-1234"
    )
    assert complaint.issue.value == "Broken seal"
    assert complaint.photo_mentioned.value == "Yes"


def test_extract_info_request():
    provider = FakeAIProvider(
        """
        {
          "safety_report": null,
          "quality_complaint": null,
          "info_request": {
            "questions": {
              "value": "Can MedX be taken with ibuprofen?",
              "confidence": 0.98
            },
            "product_or_topic": {
              "value": "MedX and ibuprofen interaction",
              "confidence": 0.96
            }
          }
        }
        """
    )

    extractor = DocumentExtractor(provider)

    result = extractor.extract(
        document_id="doc-mi-001",
        text=(
            "Can MedX be taken with ibuprofen? "
            "Please advise regarding interactions."
        ),
    )

    assert result.safety_report is None
    assert result.quality_complaint is None
    assert result.info_request is not None

    info_request = result.info_request

    assert (
        info_request.questions.value
        == "Can MedX be taken with ibuprofen?"
    )

    assert (
        info_request.product_or_topic.value
        == "MedX and ibuprofen interaction"
    )


def test_extraction_accepts_markdown_json():
    provider = FakeAIProvider(
        """
        ```json
        {
          "safety_report": null,
          "quality_complaint": null,
          "info_request": {
            "questions": {
              "value": "How should MedX be taken?",
              "confidence": 0.94
            },
            "product_or_topic": {
              "value": "MedX dosing",
              "confidence": 0.93
            }
          }
        }
        ```
        """
    )

    extractor = DocumentExtractor(provider)

    result = extractor.extract(
        document_id="doc-mi-002",
        text="How should MedX be taken?",
    )

    assert result.info_request is not None
    assert (
        result.info_request.questions.value
        == "How should MedX be taken?"
    )


def test_extraction_rejects_invalid_json():
    provider = FakeAIProvider(
        "This is not JSON."
    )

    extractor = DocumentExtractor(provider)

    with pytest.raises(AIExtractionError):
        extractor.extract(
            document_id="doc-invalid-001",
            text="Patient developed headache.",
        )


def test_extraction_rejects_empty_document():
    provider = FakeAIProvider(
        "{}"
    )

    extractor = DocumentExtractor(provider)

    with pytest.raises(AIExtractionError):
        extractor.extract(
            document_id="doc-empty-001",
            text="   ",
        )