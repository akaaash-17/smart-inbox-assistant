from fastapi import APIRouter, Depends, HTTPException

from app.repositories.interface import InboxRepository
from app.repositories.memory import InMemoryInboxRepository
from app.schemas.ai_result import AIAnalysisResult
from app.schemas.email import EmailMessage


router = APIRouter(
    prefix="/api",
    tags=["Inbox"],
)


_repository = InMemoryInboxRepository()


def get_repository() -> InboxRepository:
    """
    Provide the repository used by the API.

    Keeping repository creation behind a dependency makes it
    possible to replace the in-memory implementation with Oracle
    later without changing the route logic.
    """

    return _repository


@router.get(
    "/emails",
    response_model=list[EmailMessage],
)
def list_emails(
    repository: InboxRepository = Depends(
        get_repository
    ),
):
    """
    Return all persisted emails.
    """

    return repository.list_emails()


@router.get(
    "/emails/{email_id}",
    response_model=EmailMessage,
)
def get_email(
    email_id: str,
    repository: InboxRepository = Depends(
        get_repository
    ),
):
    """
    Return one persisted email by ID.
    """

    email = repository.get_email(
        email_id
    )

    if email is None:
        raise HTTPException(
            status_code=404,
            detail="Email not found.",
        )

    return email


@router.get(
    "/analyses",
    response_model=list[AIAnalysisResult],
)
def list_analyses(
    repository: InboxRepository = Depends(
        get_repository
    ),
):
    """
    Return all persisted AI analyses.
    """

    return repository.list_analyses()


@router.get(
    "/analyses/{document_id}",
    response_model=AIAnalysisResult,
)
def get_analysis(
    document_id: str,
    repository: InboxRepository = Depends(
        get_repository
    ),
):
    """
    Return one AI analysis by document ID.
    """

    analysis = repository.get_analysis(
        document_id
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    return analysis