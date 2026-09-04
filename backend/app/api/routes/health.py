from fastapi import APIRouter

from app.core.config import settings


router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    }