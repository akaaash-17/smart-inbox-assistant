from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    description="AI-powered healthcare email and document intelligence platform.",
    version=settings.version,
)


app.include_router(health_router)


@app.get("/", tags=["Root"])
def root():
    return {
        "application": settings.app_name,
        "version": settings.version,
        "status": "running",
    }