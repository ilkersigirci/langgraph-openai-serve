"""Validation constraints shared by OpenAI request metadata and end-user fields."""

from typing import Annotated

from pydantic import StringConstraints

OPENAI_METADATA_MAX_PAIRS = 16
OPENAI_METADATA_KEY_MAX_LENGTH = 64
OPENAI_METADATA_VALUE_MAX_LENGTH = 512
OPENAI_SAFETY_IDENTIFIER_MAX_LENGTH = 128

MetadataKey = Annotated[
    str,
    StringConstraints(min_length=1, max_length=OPENAI_METADATA_KEY_MAX_LENGTH),
]
MetadataValue = Annotated[
    str,
    StringConstraints(max_length=OPENAI_METADATA_VALUE_MAX_LENGTH),
]
SafetyIdentifier = Annotated[
    str,
    StringConstraints(max_length=OPENAI_SAFETY_IDENTIFIER_MAX_LENGTH),
]
