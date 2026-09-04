from pathlib import Path

import numpy as np
import pdfplumber
import pymupdf

from app.schemas.document import (
    DocumentContent,
    DocumentPage,
    ExtractedImage,
    ExtractedTable,
    SourceLocation,
)
from app.services.ocr_processor import OCRProcessor


class PDFProcessor:
    """
    Processes PDF documents and converts them into the
    canonical DocumentContent schema.

    Current capabilities:
    - PDF validation
    - Page-by-page text extraction
    - Digital vs scanned detection
    - OCR for image-only pages
    - Structured table extraction
    - Embedded image detection
    - Page-level confidence
    - Page-level traceability

    Future capabilities:
    - Handwriting/vision analysis
    - Image description
    - Article detection
    - Language detection
    - Translation
    """

    def __init__(
        self,
        ocr_processor: OCRProcessor | None = None,
    ):
        """
        Initialize the PDF processor.

        OCR is dependency-injected so that:
        - the OCR model can be reused
        - tests can use a mock OCR processor
        - another OCR implementation can be plugged in later
        """

        self.ocr_processor = ocr_processor

    def process(
        self,
        file_path: str | Path,
        document_id: str | None = None,
    ) -> DocumentContent:
        """
        Process a PDF and return canonical document content.
        """

        path = Path(file_path)

        self._validate_file(path)

        document_id = document_id or path.stem

        with pymupdf.open(path) as pdf:
            pages = self._extract_pages(pdf)

        pdf_type = self._detect_pdf_type(pages)

        if pdf_type == "scanned":
            pages = self._run_ocr(path, pages)

        tables = self._extract_tables(
            path,
            document_id,
        )

        images = self._extract_images(
            path,
            document_id,
        )

        combined_text = "\n\n".join(
            page.text
            for page in pages
            if page.text
        )

        return DocumentContent(
            document_id=document_id,
            filename=path.name,
            pdf_type=pdf_type,
            language="en" if combined_text else None,
            text=combined_text,
            pages=pages,
            tables=tables,
            images=images,
        )

    @staticmethod
    def _validate_file(path: Path) -> None:
        """
        Validate that the input exists and is a PDF.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Expected a file, received: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, received: {path.suffix}"
            )

    @staticmethod
    def _extract_pages(
        pdf: pymupdf.Document,
    ) -> list[DocumentPage]:
        """
        Extract text from every page while preserving
        page numbers and initial extraction confidence.
        """

        pages: list[DocumentPage] = []

        for page_number, page in enumerate(
            pdf,
            start=1,
        ):
            text = page.get_text("text").strip()

            pages.append(
                DocumentPage(
                    page_number=page_number,
                    text=text,
                    confidence=1.0 if text else 0.0,
                )
            )

        return pages

    @staticmethod
    def _detect_pdf_type(
        pages: list[DocumentPage],
    ) -> str:
        """
        Detect whether a PDF contains extractable text.

        A PDF with at least one text-bearing page is currently
        classified as digital.

        A PDF with no extractable text is classified as scanned.
        """

        if not pages:
            return "unknown"

        pages_with_text = sum(
            1
            for page in pages
            if page.text.strip()
        )

        if pages_with_text > 0:
            return "digital"

        return "scanned"

    def _run_ocr(
        self,
        file_path: Path,
        pages: list[DocumentPage],
    ) -> list[DocumentPage]:
        """
        Run OCR against pages that have no extractable text.

        The original page number is preserved.
        """

        if self.ocr_processor is None:
            return pages

        updated_pages: list[DocumentPage] = []

        with pymupdf.open(file_path) as pdf:
            for page_data in pages:
                if page_data.text.strip():
                    updated_pages.append(page_data)
                    continue

                page = pdf[
                    page_data.page_number - 1
                ]

                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )

                image = np.frombuffer(
                    pixmap.samples,
                    dtype=np.uint8,
                )

                image = image.reshape(
                    pixmap.height,
                    pixmap.width,
                    pixmap.n,
                )

                if pixmap.n == 4:
                    image = image[:, :, :3]

                ocr_result = (
                    self.ocr_processor.process_image(
                        image
                    )
                )

                updated_pages.append(
                    DocumentPage(
                        page_number=page_data.page_number,
                        text=ocr_result.text,
                        confidence=ocr_result.confidence,
                    )
                )

        return updated_pages

    @staticmethod
    def _extract_tables(
        file_path: Path,
        document_id: str,
    ) -> list[ExtractedTable]:
        """
        Extract tables from a PDF using pdfplumber.

        Each table is converted into:
        - column names
        - structured row dictionaries
        - source page information
        """

        extracted_tables: list[ExtractedTable] = []

        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):
                tables = page.extract_tables()

                for table_index, table in enumerate(
                    tables,
                    start=1,
                ):
                    if not table:
                        continue

                    cleaned_rows = [
                        [
                            (cell or "").strip()
                            for cell in row
                        ]
                        for row in table
                        if row
                    ]

                    if not cleaned_rows:
                        continue

                    columns = cleaned_rows[0]

                    if not any(columns):
                        continue

                    rows: list[dict[str, str]] = []

                    for row in cleaned_rows[1:]:
                        normalized_row = (
                            row[:len(columns)]
                            + [""] * (
                                len(columns)
                                - len(row)
                            )
                        )

                        rows.append(
                            {
                                columns[index]: normalized_row[index]
                                for index in range(
                                    len(columns)
                                )
                                if columns[index]
                            }
                        )

                    extracted_tables.append(
                        ExtractedTable(
                            name=f"Table {table_index}",
                            columns=columns,
                            rows=rows,
                            source=SourceLocation(
                                source_type="pdf",
                                source_id=document_id,
                                page=page_number,
                            ),
                        )
                    )

        return extracted_tables

    @staticmethod
    def _extract_images(
        file_path: Path,
        document_id: str,
    ) -> list[ExtractedImage]:
        """
        Detect embedded images in a PDF.

        This stage intentionally does not attempt to interpret
        the visual content. It only establishes that an image
        exists and records its exact PDF page.

        Visual interpretation will be handled later by the
        vision/AI layer.

        Every detected image is flagged for human review because
        its meaning has not yet been determined.
        """

        extracted_images: list[ExtractedImage] = []

        with pymupdf.open(file_path) as pdf:
            for page_number, page in enumerate(
                pdf,
                start=1,
            ):
                images = page.get_images(
                    full=True
                )

                for image_index, _image in enumerate(
                    images,
                    start=1,
                ):
                    extracted_images.append(
                        ExtractedImage(
                            description=(
                                "Embedded image detected; "
                                "visual analysis required."
                            ),
                            requires_review=True,
                            source=SourceLocation(
                                source_type="pdf",
                                source_id=document_id,
                                page=page_number,
                                text=(
                                    f"Embedded image "
                                    f"{image_index} detected "
                                    f"on page {page_number}."
                                ),
                            ),
                        )
                    )

        return extracted_images