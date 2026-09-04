from dataclasses import dataclass

import easyocr
import numpy as np


@dataclass
class OCRResult:
    """
    Result returned by the OCR processor.
    """

    text: str
    confidence: float


class OCRProcessor:
    """
    Local OCR processor using EasyOCR.

    The OCR reader is initialized once and reused across
    multiple pages/documents to avoid repeatedly loading
    the OCR models into memory.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
    ):
        self.languages = languages or ["en"]

        self.reader = easyocr.Reader(
            self.languages,
            gpu=gpu,
        )

    def process_image(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        """
        Run OCR against an image.

        Args:
            image: Image represented as a NumPy array.

        Returns:
            OCRResult containing extracted text and
            aggregated confidence.
        """

        results = self.reader.readtext(image)

        if not results:
            return OCRResult(
                text="",
                confidence=0.0,
            )

        text_parts: list[str] = []
        confidences: list[float] = []

        for _, detected_text, confidence in results:
            cleaned_text = detected_text.strip()

            if not cleaned_text:
                continue

            text_parts.append(cleaned_text)
            confidences.append(float(confidence))

        if not text_parts:
            return OCRResult(
                text="",
                confidence=0.0,
            )

        average_confidence = (
            sum(confidences) / len(confidences)
        )

        return OCRResult(
            text="\n".join(text_parts),
            confidence=average_confidence,
        )