import shutil
from pathlib import Path

PROJECT_DIRECTORY = Path.cwd()


def remove_file(filepath: str) -> None:
    (PROJECT_DIRECTORY / filepath).unlink()


def remove_dir(filepath: str) -> None:
    shutil.rmtree(PROJECT_DIRECTORY / filepath)


if __name__ == "__main__":
    if "{{cookiecutter.git_remote_location}}" == "gitlab":
        remove_dir(".github")
    elif "{{cookiecutter.git_remote_location}}" == "github":
        remove_file(".gitlab-ci.yml")

    if "{{cookiecutter.devcontainer}}" != "y":
        remove_dir(".devcontainer")

    if "{{cookiecutter.docs}}" != "y":
        remove_dir("docs")
        remove_file("mkdocs.yml")

        if "{{cookiecutter.git_remote_location}}" == "github":
            remove_file(".github/workflows/doc.yml")
