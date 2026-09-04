import pytest

from app.services.ai.classifier import (
    AIClassificationError,
    DocumentClassifier,
)
from app.services.ai.provider import AIProvider


class FakeAIProvider(AIProvider):
    """
    Deterministic AI provider for unit testing.

    This prevents tests from depending on Ollama being installed
    or running.
    """

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_classify_safety_report():
    provider = FakeAIProvider(
        """
        {
            "classifications": [
                {
                    "category": "SAFETY_REPORT",
                    "confidence": 0.96,
                    "reason": "The document describes a specific patient experiencing a reaction."
                }
            ]
        }
        """
    )

    classifier = DocumentClassifier(provider)

    result = classifier.classify(
        document_id="doc-001",
        text=(
            "Patient 54 years old developed a skin rash "
            "after taking MedX."
        ),
    )

    assert len(result.classifications) == 1

    classification = result.classifications[0]

    assert classification.category == "SAFETY_REPORT"
    assert classification.confidence == 0.96
    assert (
        classification.reason
        == "The document describes a specific patient experiencing a reaction."
    )


def test_classify_multiple_categories():
    provider = FakeAIProvider(
        """
        {
            "classifications": [
                {
                    "category": "SAFETY_REPORT",
                    "confidence": 0.94,
                    "reason": "A specific patient experienced an adverse reaction."
                },
                {
                    "category": "QUALITY_COMPLAINT",
                    "confidence": 0.87,
                    "reason": "The message also reports damaged packaging."
                }
            ]
        }
        """
    )

    classifier = DocumentClassifier(provider)

    result = classifier.classify(
        document_id="doc-002",
        text=(
            "The patient developed nausea after taking MedX. "
            "The package was also damaged."
        ),
    )

    assert len(result.classifications) == 2

    categories = {
        classification.category
        for classification in result.classifications
    }

    assert categories == {
        "SAFETY_REPORT",
        "QUALITY_COMPLAINT",
    }


def test_classify_not_relevant():
    provider = FakeAIProvider(
        """
        {
            "classifications": [
                {
                    "category": "NOT_RELEVANT",
                    "confidence": 0.99,
                    "reason": "The message is unrelated marketing content."
                }
            ]
        }
        """
    )

    classifier = DocumentClassifier(provider)

    result = classifier.classify(
        document_id="doc-003",
        text="Special promotional offer available this week.",
    )

    assert len(result.classifications) == 1
    assert (
        result.classifications[0].category
        == "NOT_RELEVANT"
    )


def test_classifier_accepts_markdown_json_fence():
    provider = FakeAIProvider(
        """
        ```json
        {
            "classifications": [
                {
                    "category": "INFO_REQUEST",
                    "confidence": 0.91,
                    "reason": "The message asks how the product should be taken."
                }
            ]
        }
        ```
        """
    )

    classifier = DocumentClassifier(provider)

    result = classifier.classify(
        document_id="doc-004",
        text="How should I take MedX?",
    )

    assert (
        result.classifications[0].category
        == "INFO_REQUEST"
    )


def test_classifier_rejects_invalid_json():
    provider = FakeAIProvider(
        "This is not valid JSON."
    )

    classifier = DocumentClassifier(provider)

    with pytest.raises(AIClassificationError):
        classifier.classify(
            document_id="doc-005",
            text="Patient developed headache.",
        )


def test_classifier_rejects_empty_document():
    provider = FakeAIProvider(
        """
        {
            "classifications": [
                {
                    "category": "NOT_RELEVANT",
                    "confidence": 1.0,
                    "reason": "No content."
                }
            ]
        }
        """
    )

    classifier = DocumentClassifier(provider)

    with pytest.raises(AIClassificationError):
        classifier.classify(
            document_id="doc-006",
            text="   ",
        )