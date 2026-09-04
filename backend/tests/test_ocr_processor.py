import numpy as np
from PIL import Image, ImageDraw

from app.services.ocr_processor import OCRProcessor


def create_test_image() -> np.ndarray:
    """
    Create a simple synthetic image containing
    machine-readable text.
    """

    image = Image.new(
        "RGB",
        (1200, 400),
        "white",
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (100, 100),
        "Patient Age: 42",
        fill="black",
    )

    draw.text(
        (100, 180),
        "Reaction: Headache",
        fill="black",
    )

    return np.array(image)


def test_ocr_processor():
    processor = OCRProcessor(
        languages=["en"],
        gpu=False,
    )

    image = create_test_image()

    result = processor.process_image(image)

    assert result.text
    assert result.confidence > 0.0

    assert "Patient" in result.text
    assert "42" in result.text
    assert "Headache" in result.text