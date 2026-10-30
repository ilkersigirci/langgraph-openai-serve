from fastapi import APIRouter

from langgraph_openai_serve.core.version import VERSION

router = APIRouter()


@router.get("/health")
def health_check() -> None:
    """Return 200 while the app is serving requests."""


@router.get("/version")
def version() -> dict[str, str]:
    """Return API version."""
    return {"version": VERSION}
