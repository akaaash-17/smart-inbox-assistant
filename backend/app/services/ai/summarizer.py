import re

from pydantic import BaseModel, Field

from app.services.ai.provider import AIProvider


class AISummaryError(Exception):
    """
    Raised when summary generation fails validation.
    """


class SummaryResponse(BaseModel):
    """
    Structured summary returned by the AI summarizer.
    """

    summary: str = Field(min_length=1)


class DocumentSummarizer:
    """
    Generates a concise factual summary of a processed document.

    The summary is deliberately kept separate from classification
    and extraction so each AI responsibility remains independently
    testable.
    """

    MIN_SENTENCES = 10
    MAX_SENTENCES = 15

    def __init__(self, provider: AIProvider):
        self.provider = provider

    def summarize(
        self,
        text: str,
    ) -> SummaryResponse:
        """
        Generate and validate a 10-15 sentence document summary.
        """

        if not text.strip():
            raise AISummaryError(
                "Cannot summarize an empty document."
            )

        prompt = self._build_prompt(text)

        raw_response = self.provider.generate(prompt)

        summary = raw_response.strip()

        sentence_count = self._count_sentences(summary)

        if not (
            self.MIN_SENTENCES
            <= sentence_count
            <= self.MAX_SENTENCES
        ):
            raise AISummaryError(
                "AI summary must contain between "
                f"{self.MIN_SENTENCES} and "
                f"{self.MAX_SENTENCES} sentences. "
                f"Received {sentence_count}."
            )

        return SummaryResponse(
            summary=summary
        )

    @staticmethod
    def _build_prompt(text: str) -> str:
        """
        Build a constrained factual summarization prompt.
        """

        return f"""
You are a healthcare document summarization assistant.

Summarize the following email or document for a human reviewer.

Requirements:
- Write between 10 and 15 sentences.
- Summarize only information explicitly present in the document.
- Do not invent or infer facts.
- Explain what the document is about.
- Mention important patient, reporter, product, reaction,
  complaint, or information-request details when present.
- Explain why the document may be relevant to the healthcare
  reporting workflow.
- If information is missing, do not make assumptions.
- Keep the language factual and suitable for human review.
- Do not use bullet points.
- Return only the summary text.
- Do not include a heading.
- Do not include Markdown.

DOCUMENT:
{text}
""".strip()

    @staticmethod
    def _count_sentences(text: str) -> int:
        """
        Count sentence-like segments using terminal punctuation.

        This is intentionally simple and deterministic. It is a
        validation guard rather than a linguistic parser.
        """

        sentences = re.findall(
            r"[^.!?]+[.!?]+",
            text,
        )

        return len(sentences)