"""Apply the license choice without installing anything or starting services."""

from pathlib import Path

if {{ cookiecutter.license | tojson }} == "Proprietary":
    Path("LICENSE").unlink()

print("Created {{ cookiecutter.project_name }}. Next: cd {{ cookiecutter.project_name }} && just setup")
