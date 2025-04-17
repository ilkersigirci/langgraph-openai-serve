from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health_check() -> None:
    """Checks the health of a project.

    It returns 200 if the project is healthy.
    """
