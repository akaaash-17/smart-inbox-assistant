import json
from typing import Any

from pydantic import BaseModel, ValidationError

from app.schemas.ai_result import (
    InfoRequestExtraction,
    QualityComplaintExtraction,
    SafetyReportExtraction,
)
from app.services.ai.provider import AIProvider


class DomainExtractionResponse(BaseModel):
    """
    Structured domain extraction returned by the AI layer.

    Only the extraction sections relevant to the document need
    to be populated. Missing sections remain None.
    """

    safety_report: SafetyReportExtraction | None = None
    quality_complaint: QualityComplaintExtraction | None = None
    info_request: InfoRequestExtraction | None = None


class AIExtractionError(Exception):
    """
    Raised when AI domain extraction cannot be validated.
    """


class DocumentExtractor:
    """
    Extracts healthcare-domain information from document text.

    The extractor is deliberately provider-agnostic. It receives
    an AIProvider and converts the model response into the
    application's canonical Pydantic schemas.
    """

    def __init__(self, provider: AIProvider):
        self.provider = provider

    def extract(
        self,
        document_id: str,
        text: str,
    ) -> DomainExtractionResponse:
        """
        Extract domain-specific information from a document.
        """

        if not text.strip():
            raise AIExtractionError(
                "Cannot extract information from an empty document."
            )

        prompt = self._build_prompt(text)

        raw_response = self.provider.generate(prompt)

        return self._parse_response(raw_response)

    @staticmethod
    def _build_prompt(text: str) -> str:
        """
        Build a strict extraction prompt.

        The prompt explicitly prevents the model from filling
        missing healthcare information with assumptions.
        """

        return f"""
You are a healthcare document extraction assistant.

Extract only information explicitly stated in the document.

IMPORTANT RULES:
- Never guess or infer missing information.
- If a requested field is not explicitly stated, use exactly:
  "Not stated"
- Every extracted field must contain:
  - value
  - confidence between 0.0 and 1.0
- Use high confidence only when the information is clearly stated.
- Preserve the meaning of the source text.
- Do not invent names, dates, doses, reactions, medical history,
  outcomes, or severity.
- Return ONLY valid JSON.
- Do not use Markdown code fences.

For SAFETY_REPORT extract:

Patient:
- age
- sex
- weight
- height
- relevant history

Reporter:
- name
- role
- country

Product:
- name
- dose
- route
- start date
- stop date

Reaction:
- description
- onset
- outcome

Severity:
- serious
- death
- hospitalization
- life-threatening

Narrative:
- concise factual narrative of the reported case

For QUALITY_COMPLAINT extract:
- product
- batch or lot number
- issue
- whether a photo is mentioned

For INFO_REQUEST extract:
- actual questions
- product or topic

Use this exact JSON structure:

{{
  "safety_report": {{
    "patient": {{
      "age": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "sex": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "weight": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "height": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "relevant_history": {{
        "value": "Not stated",
        "confidence": 1.0
      }}
    }},
    "reporter": {{
      "name": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "role": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "country": {{
        "value": "Not stated",
        "confidence": 1.0
      }}
    }},
    "product": {{
      "name": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "dose": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "route": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "start_date": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "stop_date": {{
        "value": "Not stated",
        "confidence": 1.0
      }}
    }},
    "reaction": {{
      "description": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "onset": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "outcome": {{
        "value": "Not stated",
        "confidence": 1.0
      }}
    }},
    "severity": {{
      "serious": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "death": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "hospitalization": {{
        "value": "Not stated",
        "confidence": 1.0
      }},
      "life_threatening": {{
        "value": "Not stated",
        "confidence": 1.0
      }}
    }},
    "narrative": {{
      "value": "Not stated",
      "confidence": 1.0
    }}
  }},
  "quality_complaint": null,
  "info_request": null
}}

DOCUMENT:
{text}
""".strip()

    @staticmethod
    def _parse_response(
        raw_response: str,
    ) -> DomainExtractionResponse:
        """
        Parse and validate model JSON against the canonical
        extraction schemas.
        """

        cleaned_response = (
            DocumentExtractor._clean_json_response(
                raw_response
            )
        )

        try:
            parsed: Any = json.loads(cleaned_response)
        except json.JSONDecodeError as exc:
            raise AIExtractionError(
                "AI provider returned invalid JSON."
            ) from exc

        try:
            result = DomainExtractionResponse.model_validate(
                parsed
            )
        except ValidationError as exc:
            raise AIExtractionError(
                f"AI extraction failed schema validation: {exc}"
            ) from exc

        return result

    @staticmethod
    def _clean_json_response(
        response: str,
    ) -> str:
        """
        Remove accidental Markdown JSON fences.
        """

        cleaned = response.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return cleaned.strip()