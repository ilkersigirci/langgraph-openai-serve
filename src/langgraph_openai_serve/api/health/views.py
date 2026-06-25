from fastapi import APIRouter

from langgraph_openai_serve.core.version import get_version

router = APIRouter()


@router.get("/health")
def health_check() -> None:
    """Checks the health of a project.

    It returns 200 if the project is healthy.
    """


@router.get("/version")
def version() -> dict[str, str]:
    """Return API version."""
    return {"version": get_version()}
