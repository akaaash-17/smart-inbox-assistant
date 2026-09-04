from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw

from app.services.ocr_processor import OCRResult
from app.services.pdf_processor import PDFProcessor


FIXTURE_DIR = (
    Path(__file__).parent
    / "fixtures"
    / "pdfs"
)

DIGITAL_PDF = (
    FIXTURE_DIR
    / "digital_safety_report.pdf"
)

SCANNED_PDF = (
    FIXTURE_DIR
    / "scanned_document.pdf"
)


class FakeOCRProcessor:
    """
    Deterministic OCR implementation used for PDFProcessor
    tests.

    This avoids loading the real EasyOCR model every time
    the PDF tests run.
    """

    def process_image(
        self,
        image: np.ndarray,
    ) -> OCRResult:
        return OCRResult(
            text=(
                "Patient Age: 42\n"
                "Reaction: Headache"
            ),
            confidence=0.93,
        )


@pytest.fixture(scope="module", autouse=True)
def create_scanned_pdf():
    """
    Create a synthetic image-only PDF fixture.

    The page contains an image rather than a PDF text layer.
    """

    doc = pymupdf.open()

    page = doc.new_page()

    image = Image.new(
        "RGB",
        (1200, 1600),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (100, 100),
        "SCANNED SAFETY REPORT",
        fill="black",
    )

    draw.text(
        (100, 180),
        "Patient Age: 42",
        fill="black",
    )

    draw.text(
        (100, 240),
        "Reaction: Headache",
        fill="black",
    )

    import io

    image_buffer = io.BytesIO()

    image.save(
        image_buffer,
        format="PNG",
    )

    page.insert_image(
        page.rect,
        stream=image_buffer.getvalue(),
    )

    doc.save(SCANNED_PDF)
    doc.close()

    yield

    if SCANNED_PDF.exists():
        SCANNED_PDF.unlink()


def test_process_digital_pdf():
    processor = PDFProcessor()

    document = processor.process(
        DIGITAL_PDF,
        document_id="doc-safety-001",
    )

    assert document.document_id == "doc-safety-001"
    assert document.filename == "digital_safety_report.pdf"
    assert document.pdf_type == "digital"

    assert len(document.pages) == 1

    assert "SAFETY REPORT" in document.text
    assert "Patient Age: 54" in document.text
    assert "Patient Sex: Male" in document.text
    assert "Product: MedX 10 mg" in document.text
    assert "Reaction: Patient developed a skin rash." in document.text

    assert document.pages[0].page_number == 1
    assert document.pages[0].confidence == 1.0


def test_process_scanned_pdf_with_ocr():
    processor = PDFProcessor(
        ocr_processor=FakeOCRProcessor()
    )

    document = processor.process(
        SCANNED_PDF,
        document_id="doc-scanned-001",
    )

    assert document.document_id == "doc-scanned-001"
    assert document.filename == "scanned_document.pdf"

    assert document.pdf_type == "scanned"

    assert len(document.pages) == 1

    assert "Patient Age: 42" in document.pages[0].text
    assert "Reaction: Headache" in document.pages[0].text

    assert "Patient Age: 42" in document.text
    assert "Reaction: Headache" in document.text

    assert document.pages[0].confidence == 0.93


def test_scanned_pdf_without_ocr_processor():
    processor = PDFProcessor()

    document = processor.process(
        SCANNED_PDF,
        document_id="doc-scanned-002",
    )

    assert document.pdf_type == "scanned"
    assert document.pages[0].text == ""
    assert document.pages[0].confidence == 0.0
    assert document.text == ""


def test_process_missing_pdf():
    processor = PDFProcessor()

    missing_file = (
        FIXTURE_DIR
        / "does_not_exist.pdf"
    )

    with pytest.raises(FileNotFoundError):
        processor.process(missing_file)


def test_process_non_pdf_file():
    processor = PDFProcessor()

    non_pdf_file = (
        Path(__file__).parent
        / "fixtures"
        / "not_a_pdf.txt"
    )

    non_pdf_file.write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        processor.process(non_pdf_file)

    non_pdf_file.unlink()