"""Rendering settings shared by the generated Python and TOML files."""

from jinja2 import Environment
from jinja2.ext import Extension


class UnicodeJSONExtension(Extension):
    """Keep non-BMP characters literal; JSON surrogate escapes are invalid TOML."""

    def __init__(self, environment: Environment) -> None:
        super().__init__(environment)
        environment.policies["json.dumps_kwargs"] = {
            **environment.policies["json.dumps_kwargs"],
            "ensure_ascii": False,
        }
