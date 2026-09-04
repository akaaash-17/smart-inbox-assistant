import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.schemas.ai_result import (
    Category,
    ClassificationResult,
)
from app.services.ai.provider import AIProvider


class ClassificationResponse(BaseModel):
    """
    Structured response returned by the classifier.
    """

    classifications: list[ClassificationResult] = Field(
        default_factory=list
    )


class AIClassificationError(Exception):
    """
    Raised when the AI classification response cannot be
    converted into the required structured format.
    """


class DocumentClassifier:
    """
    Classifies document text into one or more of the
    assignment-defined healthcare inbox categories.
    """

    ALLOWED_CATEGORIES: set[str] = {
        "SAFETY_REPORT",
        "QUALITY_COMPLAINT",
        "INFO_REQUEST",
        "NOT_RELEVANT",
    }

    def __init__(self, provider: AIProvider):
        self.provider = provider

    def classify(
        self,
        document_id: str,
        text: str,
    ) -> ClassificationResponse:
        """
        Classify a document using the configured AI provider.
        """

        if not text.strip():
            raise AIClassificationError(
                "Cannot classify an empty document."
            )

        prompt = self._build_prompt(text)

        raw_response = self.provider.generate(prompt)

        return self._parse_response(
            document_id=document_id,
            raw_response=raw_response,
        )

    @classmethod
    def _build_prompt(
        cls,
        text: str,
    ) -> str:
        """
        Build a constrained classification prompt.

        The model is explicitly instructed to return JSON so that
        its output can be validated by Pydantic.
        """

        return f"""
You are a healthcare inbox classification assistant.

Classify the following email or document into one or more
of these categories:

1. SAFETY_REPORT
   A specific patient, reporter, product/drug, and adverse
   outcome/reaction are described.

2. QUALITY_COMPLAINT
   A physical product issue is described, such as broken seal,
   wrong color, contamination, damaged packaging, or counterfeit
   product.

3. INFO_REQUEST
   The message asks about dosing, how to take a product,
   interactions, or another product-related information topic,
   without reporting an adverse reaction or physical product defect.

4. NOT_RELEVANT
   Marketing, spam, administrative material, or content unrelated
   to the healthcare reporting categories.

A document may have multiple categories.

Return ONLY valid JSON using exactly this structure:

{{
  "classifications": [
    {{
      "category": "SAFETY_REPORT",
      "confidence": 0.95,
      "reason": "Short one-line explanation."
    }}
  ]
}}

Rules:
- category must be one of the four allowed categories.
- confidence must be between 0.0 and 1.0.
- reason must be one concise sentence.
- Do not invent facts.
- Return every applicable category.
- If none of the healthcare reporting categories apply,
  use NOT_RELEVANT.
- Return JSON only. Do not include markdown fences.

DOCUMENT:
{text}
""".strip()

    @classmethod
    def _parse_response(
        cls,
        document_id: str,
        raw_response: str,
    ) -> ClassificationResponse:
        """
        Parse and validate the model's JSON response.
        """

        cleaned_response = cls._clean_json_response(
            raw_response
        )

        try:
            parsed: Any = json.loads(cleaned_response)
        except json.JSONDecodeError as exc:
            raise AIClassificationError(
                "AI provider returned invalid JSON."
            ) from exc

        try:
            result = ClassificationResponse.model_validate(
                parsed
            )
        except ValidationError as exc:
            raise AIClassificationError(
                f"AI classification failed schema validation: {exc}"
            ) from exc

        if not result.classifications:
            raise AIClassificationError(
                "AI provider returned no classifications."
            )

        for classification in result.classifications:
            if (
                classification.category
                not in cls.ALLOWED_CATEGORIES
            ):
                raise AIClassificationError(
                    "AI provider returned an unsupported category."
                )

        return result

    @staticmethod
    def _clean_json_response(
        response: str,
    ) -> str:
        """
        Remove accidental Markdown JSON fences from a model
        response while keeping the actual JSON unchanged.
        """

        cleaned = response.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]

        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()