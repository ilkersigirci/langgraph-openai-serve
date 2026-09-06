"""Validation constraints shared by OpenAI request metadata fields."""

from typing import Annotated

from pydantic import StringConstraints

OPENAI_METADATA_MAX_PAIRS = 16
OPENAI_METADATA_KEY_MAX_LENGTH = 64
OPENAI_METADATA_VALUE_MAX_LENGTH = 512

MetadataKey = Annotated[
    str,
    StringConstraints(min_length=1, max_length=OPENAI_METADATA_KEY_MAX_LENGTH),
]
MetadataValue = Annotated[
    str,
    StringConstraints(max_length=OPENAI_METADATA_VALUE_MAX_LENGTH),
]

__all__ = [
    "OPENAI_METADATA_KEY_MAX_LENGTH",
    "OPENAI_METADATA_MAX_PAIRS",
    "OPENAI_METADATA_VALUE_MAX_LENGTH",
    "MetadataKey",
    "MetadataValue",
]
