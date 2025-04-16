"""{{cookiecutter.project_name}} package."""

from importlib.metadata import version

# Fetches the version of the package as defined in pyproject.toml
__version__ = version("{{cookiecutter.project_slug}}")
