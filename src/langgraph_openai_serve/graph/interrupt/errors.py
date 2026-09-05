"""Errors shared by interrupt codecs and graph execution."""


class InvalidResumeRequestError(ValueError):
    """Raised when a protocol request is not a valid interrupt resume."""


class InvalidInterruptPayloadError(ValueError):
    """Raised when graph-authored interrupt data is not JSON-compatible."""


__all__ = ["InvalidInterruptPayloadError", "InvalidResumeRequestError"]
