from app.schemas.ai_result import ExtractedField
from app.schemas.document import (
    DocumentContent,
    DocumentPage,
)
from app.services.ai.evidence_resolver import (
    EvidenceResolver,
)


def create_document() -> DocumentContent:
    return DocumentContent(
        document_id="doc-evidence-001",
        filename="safety_report.pdf",
        pdf_type="digital",
        text=(
            "Patient Age: 54\n"
            "Patient Sex: Male\n"
            "Product: MedX 10 mg\n"
            "Reaction: Patient developed a skin rash."
        ),
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "Patient Age: 54\n"
                    "Patient Sex: Male\n"
                    "Product: MedX 10 mg\n"
                    "Reaction: Patient developed a skin rash."
                ),
                confidence=1.0,
            )
        ],
    )


def test_resolve_exact_value():
    resolver = EvidenceResolver()
    document = create_document()

    field = ExtractedField(
        value="54",
        confidence=0.99,
    )

    resolved = resolver.resolve_field(
        field,
        document,
    )

    assert resolved.value == "54"
    assert resolved.confidence == 0.99

    assert resolved.source is not None
    assert resolved.source.source_type == "pdf"
    assert resolved.source.source_id == "doc-evidence-001"
    assert resolved.source.page == 1
    assert resolved.source.text == "Patient Age: 54"


def test_resolve_case_insensitive_value():
    resolver = EvidenceResolver()
    document = create_document()

    field = ExtractedField(
        value="male",
        confidence=0.98,
    )

    resolved = resolver.resolve_field(
        field,
        document,
    )

    assert resolved.source is not None
    assert resolved.source.page == 1
    assert resolved.source.text == "Patient Sex: Male"


def test_resolve_value_inside_longer_text():
    resolver = EvidenceResolver()
    document = create_document()

    field = ExtractedField(
        value="skin rash",
        confidence=0.97,
    )

    resolved = resolver.resolve_field(
        field,
        document,
    )

    assert resolved.source is not None
    assert resolved.source.page == 1
    assert (
        resolved.source.text
        == "Reaction: Patient developed a skin rash."
    )


def test_not_stated_has_no_source():
    resolver = EvidenceResolver()
    document = create_document()

    field = ExtractedField(
        value="Not stated",
        confidence=1.0,
    )

    resolved = resolver.resolve_field(
        field,
        document,
    )

    assert resolved.value == "Not stated"
    assert resolved.source is None


def test_missing_value_has_no_source():
    resolver = EvidenceResolver()
    document = create_document()

    field = ExtractedField(
        value="Dr. Unknown",
        confidence=0.50,
    )

    resolved = resolver.resolve_field(
        field,
        document,
    )

    assert resolved.source is None


def test_resolve_product_value():
    resolver = EvidenceResolver()
    document = create_document()

    field = ExtractedField(
        value="MedX 10 mg",
        confidence=0.99,
    )

    resolved = resolver.resolve_field(
        field,
        document,
    )

    assert resolved.source is not None
    assert resolved.source.page == 1
    assert (
        resolved.source.text
        == "Product: MedX 10 mg"
    )