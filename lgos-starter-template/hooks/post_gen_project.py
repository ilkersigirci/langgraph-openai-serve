"""Apply the license and CI choices without installing anything or starting services."""

import shutil
from pathlib import Path

if {{ cookiecutter.license | tojson }} == "Proprietary":
    Path("LICENSE").unlink()
if {{ cookiecutter.ci | tojson }} == "github":
    Path(".gitlab-ci.yml").unlink()
else:
    shutil.rmtree(".github")

print("Created {{ cookiecutter.project_name }}. Next: cd {{ cookiecutter.project_name }} && just setup")
