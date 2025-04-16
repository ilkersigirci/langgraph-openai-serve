import re
import sys


def validate_name(name, regex, error_message):
    if not re.match(regex, name):
        print(f"ERROR: {error_message}")
        sys.exit(1)


project_name = "{{cookiecutter.project_name}}"
PROJECT_NAME_REGEX = r"^[-a-zA-Z][-a-zA-Z0-9]+$"
project_name_error = (
    f"The project name {project_name} is not a valid Python module name. "
    "Please do not use a _ and use - instead"
)

project_slug = "{{cookiecutter.project_slug}}"
PROJECT_SLUG_REGEX = r"^[_a-zA-Z][_a-zA-Z0-9]+$"
project_slug_error = (
    f"The project slug {project_slug} is not a valid Python module name. "
    "Please do not use a - and use _ instead"
)

validate_name(
    name=project_name, regex=PROJECT_NAME_REGEX, error_message=project_name_error
)
validate_name(
    name=project_slug, regex=PROJECT_SLUG_REGEX, error_message=project_slug_error
)
