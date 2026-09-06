"""Errors shared by Responses adaptation and interrupt execution."""


class InvalidResumeRequestError(ValueError):
    """Raised when a protocol request is not a valid interrupt resume."""

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.param = param


class InvalidInterruptPayloadError(ValueError):
    """Raised when graph-authored interrupt data is not JSON-compatible."""


__all__ = ["InvalidInterruptPayloadError", "InvalidResumeRequestError"]
