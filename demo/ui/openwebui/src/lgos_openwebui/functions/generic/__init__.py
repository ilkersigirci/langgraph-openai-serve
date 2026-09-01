"""Modular source for the Generic Open WebUI Function."""

from .contracts import (
    ASK_USER_REJECTED_OUTPUT,
    INTERRUPT_CANCELLED_MESSAGE,
    LIMITED_FUNCTIONALITY_MESSAGE,
)
from .pipe import Pipe

__all__ = [
    "ASK_USER_REJECTED_OUTPUT",
    "INTERRUPT_CANCELLED_MESSAGE",
    "LIMITED_FUNCTIONALITY_MESSAGE",
    "Pipe",
]
