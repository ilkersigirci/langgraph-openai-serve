"""Validate identifiers before they become package, command, and image names."""

import keyword
import re

project_name = {{ cookiecutter.project_name | tojson }}
project_slug = {{ cookiecutter.project_slug | tojson }}

if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", project_name):
    raise SystemExit("project_name must contain lowercase words separated by hyphens.")
if not re.fullmatch(r"[a-z][a-z0-9_]*", project_slug) or keyword.iskeyword(project_slug):
    raise SystemExit("project_slug must be a lowercase Python identifier, not a keyword.")
