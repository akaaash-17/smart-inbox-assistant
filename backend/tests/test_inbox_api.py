from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.routes.inbox import (
    get_repository,
)
from app.main import app
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


client = TestClient(app)


def create_email(
    email_id: str = "email-api-001",
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
    document_id: str = "document-api-001",
) -> AIAnalysisResult:
    return AIAnalysisResult(
        document_id=document_id,
        classifications=[
            ClassificationResult(
                category="SAFETY_REPORT",
                confidence=0.95,
                reason="Patient reaction reported.",
            )
        ],
        relevant=True,
        relevance_reason="Patient reaction reported.",
        summary="A synthetic safety report.",
    )


def setup_repository():
    repository = InMemoryInboxRepository()

    repository.save_email(
        create_email()
    )

    repository.save_email(
        create_email("email-api-002")
    )

    repository.save_analysis(
        create_analysis()
    )

    repository.save_analysis(
        create_analysis(
            "document-api-002"
        )
    )

    app.dependency_overrides[
        get_repository
    ] = lambda: repository

    return repository


def teardown_repository():
    app.dependency_overrides.clear()


def test_list_emails():
    setup_repository()

    try:
        response = client.get(
            "/api/emails"
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 2

        assert {
            email["id"]
            for email in data
        } == {
            "email-api-001",
            "email-api-002",
        }

    finally:
        teardown_repository()


def test_get_email():
    setup_repository()

    try:
        response = client.get(
            "/api/emails/email-api-001"
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data["id"]
            == "email-api-001"
        )

        assert (
            data["sender"]
            == "reporter@example.com"
        )

        assert (
            data["subject"]
            == "Safety report"
        )

    finally:
        teardown_repository()


def test_get_missing_email():
    setup_repository()

    try:
        response = client.get(
            "/api/emails/missing-email"
        )

        assert response.status_code == 404

        assert (
            response.json()["detail"]
            == "Email not found."
        )

    finally:
        teardown_repository()


def test_list_analyses():
    setup_repository()

    try:
        response = client.get(
            "/api/analyses"
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data) == 2

        assert {
            analysis["document_id"]
            for analysis in data
        } == {
            "document-api-001",
            "document-api-002",
        }

    finally:
        teardown_repository()


def test_get_analysis():
    setup_repository()

    try:
        response = client.get(
            "/api/analyses/document-api-001"
        )

        assert response.status_code == 200

        data = response.json()

        assert (
            data["document_id"]
            == "document-api-001"
        )

        assert (
            data["relevant"]
            is True
        )

        assert (
            data["classifications"][0][
                "category"
            ]
            == "SAFETY_REPORT"
        )

        assert (
            data["classifications"][0][
                "confidence"
            ]
            == 0.95
        )

    finally:
        teardown_repository()


def test_get_missing_analysis():
    setup_repository()

    try:
        response = client.get(
            "/api/analyses/missing-document"
        )

        assert response.status_code == 404

        assert (
            response.json()["detail"]
            == "Analysis not found."
        )

    finally:
        teardown_repository()