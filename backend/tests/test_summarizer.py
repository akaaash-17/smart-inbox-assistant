import pytest

from app.services.ai.provider import AIProvider
from app.services.ai.summarizer import (
    AISummaryError,
    DocumentSummarizer,
)


class FakeAIProvider(AIProvider):
    """
    Deterministic provider for summary tests.
    """

    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


VALID_SUMMARY = (
    "The document reports an event involving a specific patient. "
    "The patient is identified as a 54-year-old male. "
    "The reported product is MedX 10 mg. "
    "The product was administered orally. "
    "The patient developed a skin rash after starting treatment. "
    "The reaction reportedly began two days after treatment started. "
    "The patient subsequently recovered from the reaction. "
    "The report identifies a physician as the reporter. "
    "The reporter is associated with India. "
    "The document contains information relevant to a potential safety report. "
    "The available information should be reviewed by a human reviewer. "
    "Fields not explicitly stated in the source should not be inferred."
)


def test_generate_valid_summary():
    provider = FakeAIProvider(
        VALID_SUMMARY
    )

    summarizer = DocumentSummarizer(provider)

    result = summarizer.summarize(
        "A 54-year-old male developed a skin rash "
        "after taking MedX 10 mg orally."
    )

    assert result.summary == VALID_SUMMARY

    sentence_count = (
        summarizer._count_sentences(
            result.summary
        )
    )

    assert 10 <= sentence_count <= 15


def test_summary_rejects_too_few_sentences():
    provider = FakeAIProvider(
        "This is only one sentence."
    )

    summarizer = DocumentSummarizer(provider)

    with pytest.raises(AISummaryError):
        summarizer.summarize(
            "A document containing some information."
        )


def test_summary_rejects_too_many_sentences():
    provider = FakeAIProvider(
        (
            "One sentence. "
            "Two sentence. "
            "Three sentence. "
            "Four sentence. "
            "Five sentence. "
            "Six sentence. "
            "Seven sentence. "
            "Eight sentence. "
            "Nine sentence. "
            "Ten sentence. "
            "Eleven sentence. "
            "Twelve sentence. "
            "Thirteen sentence. "
            "Fourteen sentence. "
            "Fifteen sentence. "
            "Sixteen sentence."
        )
    )

    summarizer = DocumentSummarizer(provider)

    with pytest.raises(AISummaryError):
        summarizer.summarize(
            "A document containing some information."
        )


def test_summary_rejects_empty_document():
    provider = FakeAIProvider(
        VALID_SUMMARY
    )

    summarizer = DocumentSummarizer(provider)

    with pytest.raises(AISummaryError):
        summarizer.summarize("   ")


def test_summary_accepts_question_mark_and_exclamation():
    provider = FakeAIProvider(
        (
            "The document contains a product question? "
            "The requester asks about the product. "
            "The question concerns how the product should be used. "
            "No adverse reaction is reported. "
            "No physical product defect is reported. "
            "The request is therefore information-focused. "
            "The product name is explicitly stated. "
            "The available information should be reviewed. "
            "The reviewer should consider the stated question. "
            "No missing information should be inferred. "
            "The document may be relevant to medical information handling!"
        )
    )

    summarizer = DocumentSummarizer(provider)

    result = summarizer.summarize(
        "How should the product be used?"
    )

    assert (
        summarizer._count_sentences(
            result.summary
        )
        == 11
    )