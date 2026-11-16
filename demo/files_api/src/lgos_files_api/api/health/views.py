from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return 200 while the app is serving requests."""
    return {"status": "ok"}
