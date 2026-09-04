from datetime import datetime, timezone

import pytest

from app.repositories.interface import (
    RepositoryError,
)
from app.repositories.memory import (
    InMemoryInboxRepository,
)
from app.schemas.ai_result import (
    AIAnalysisResult,
    ClassificationResult,
)
from app.schemas.email import (
    EmailMessage,
)


def create_email(
    email_id: str = "email-001",
) -> EmailMessage:
    return EmailMessage(
        id=email_id,
        sender="reporter@example.com",
        subject="Safety report",
        received_at=datetime(
            2026,
            9,
            1,
            10,
            30,
            tzinfo=timezone.utc,
        ),
        body="Patient developed a reaction.",
    )


def create_analysis(
    document_id: str = "document-001",
) -> AIAnalysisResult:
    return AIAnalysisResult(
        document_id=document_id,
        classifications=[
            ClassificationResult(
                category="SAFETY_REPORT",
                confidence=0.96,
                reason=(
                    "A specific patient experienced "
                    "an adverse reaction."
                ),
            )
        ],
        relevant=True,
        relevance_reason=(
            "A specific patient experienced "
            "an adverse reaction."
        ),
        summary=(
            "The document contains a safety report."
        ),
    )


def test_save_and_get_email():
    repository = InMemoryInboxRepository()

    email = create_email()

    repository.save_email(email)

    result = repository.get_email(
        "email-001"
    )

    assert result is not None
    assert result.id == "email-001"
    assert (
        result.sender
        == "reporter@example.com"
    )
    assert (
        result.subject
        == "Safety report"
    )


def test_get_missing_email_returns_none():
    repository = InMemoryInboxRepository()

    result = repository.get_email(
        "does-not-exist"
    )

    assert result is None


def test_save_and_get_analysis():
    repository = InMemoryInboxRepository()

    analysis = create_analysis()

    repository.save_analysis(
        analysis
    )

    result = repository.get_analysis(
        "document-001"
    )

    assert result is not None
    assert (
        result.document_id
        == "document-001"
    )

    assert len(
        result.classifications
    ) == 1

    assert (
        result.classifications[0].category
        == "SAFETY_REPORT"
    )

    assert result.relevant is True


def test_get_missing_analysis_returns_none():
    repository = InMemoryInboxRepository()

    result = repository.get_analysis(
        "does-not-exist"
    )

    assert result is None


def test_list_emails():
    repository = InMemoryInboxRepository()

    repository.save_email(
        create_email("email-001")
    )

    repository.save_email(
        create_email("email-002")
    )

    emails = repository.list_emails()

    assert len(emails) == 2

    assert {
        email.id
        for email in emails
    } == {
        "email-001",
        "email-002",
    }


def test_list_analyses():
    repository = InMemoryInboxRepository()

    repository.save_analysis(
        create_analysis("document-001")
    )

    repository.save_analysis(
        create_analysis("document-002")
    )

    analyses = repository.list_analyses()

    assert len(analyses) == 2

    assert {
        analysis.document_id
        for analysis in analyses
    } == {
        "document-001",
        "document-002",
    }


def test_saving_existing_email_id_is_idempotent():
    repository = InMemoryInboxRepository()

    first_email = create_email()

    repository.save_email(
        first_email
    )

    updated_email = create_email()

    updated_email.subject = (
        "Updated safety report"
    )

    repository.save_email(
        updated_email
    )

    result = repository.get_email(
        "email-001"
    )

    assert result is not None
    assert (
        result.subject
        == "Updated safety report"
    )

    assert len(
        repository.list_emails()
    ) == 1


def test_saving_existing_analysis_id_is_idempotent():
    repository = InMemoryInboxRepository()

    first_analysis = create_analysis()

    repository.save_analysis(
        first_analysis
    )

    updated_analysis = create_analysis()

    updated_analysis.summary = (
        "Updated analysis summary."
    )

    repository.save_analysis(
        updated_analysis
    )

    result = repository.get_analysis(
        "document-001"
    )

    assert result is not None
    assert (
        result.summary
        == "Updated analysis summary."
    )

    assert len(
        repository.list_analyses()
    ) == 1


def test_repository_returns_copies():
    repository = InMemoryInboxRepository()

    email = create_email()

    repository.save_email(email)

    retrieved = repository.get_email(
        "email-001"
    )

    assert retrieved is not None

    retrieved.subject = "Modified outside repository"

    stored_again = repository.get_email(
        "email-001"
    )

    assert stored_again is not None

    assert (
        stored_again.subject
        == "Safety report"
    )


def test_empty_email_id_is_rejected():
    repository = InMemoryInboxRepository()

    email = create_email("")

    with pytest.raises(
        RepositoryError
    ) as exc_info:
        repository.save_email(email)

    assert (
        str(exc_info.value)
        == "Email ID cannot be empty."
    )


def test_empty_document_id_is_rejected():
    repository = InMemoryInboxRepository()

    analysis = create_analysis("")

    with pytest.raises(
        RepositoryError
    ) as exc_info:
        repository.save_analysis(
            analysis
        )

    assert (
        str(exc_info.value)
        == "Document ID cannot be empty."
    )


def test_clear_removes_all_records():
    repository = InMemoryInboxRepository()

    repository.save_email(
        create_email()
    )

    repository.save_analysis(
        create_analysis()
    )

    repository.clear()

    assert (
        repository.list_emails()
        == []
    )

    assert (
        repository.list_analyses()
        == []
    )