# Python Template

<p align="center">
  <img width="600" src="https://raw.githubusercontent.com/ilkersigirci/python-template/main/docs/static/logo.jpg">
</p style = "margin-bottom: 2rem;">
---

[![Build status](https://img.shields.io/github/actions/workflow/status/ilkersigirci/python-template/main.yml?branch=main)](https://github.com/ilkersigirci/python-template/actions/workflows/main.yml?query=branch%3Amain)
[![Supported Python versions](https://img.shields.io/badge/python-3.11_%7C_3.12_%7C_3.13-blue?labelColor=grey&color=blue)](https://github.com/ilkersigirci/python-template/blob/main/pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-gh--pages-blue)](https://ilkersigirci.github.io/python-template/)
[![License](https://img.shields.io/github/license/ilkersigirci/python-template)](https://img.shields.io/github/license/ilkersigirci/python-template)

## Project Structure

- It uses [cookiecutter](https://github.com/cookiecutter/cookiecutter) for project templating.
- It uses [uv](https://github.com/astral-sh/uv) for python dependency operations and virtual environment management.
- It uses `src` layout, which is the recommended layout for python projects to avoid common [pitfalls](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure).
- It uses `project.toml` instead of `setup.py` and `setup.cfg`. The reasoning is following:
    - As [official setuptools guide](https://github.com/pypa/setuptools/blob/main/docs/userguide/quickstart.rst) says, " configuring new projects via setup.py is discouraged"
    - One of the biggest problems with setuptools is that the use of an executable file (i.e. the setup.py) cannot be executed without knowing its dependencies. And there is really no way to know what these dependencies are unless you actually execute the file that contains the information related to package dependencies.
    - The pyproject.toml file is supposed to solve the build-tool dependency chicken and egg problem since pip itself can read pyproject.yoml along with the version of setuptools or wheel the project requires.
    - The pyproject.toml file was introduced in PEP-518 (2016) as a way of separating configuration of the build system from a specific, optional library (setuptools) and also enabling setuptools to install itself without already being installed. Subsequently PEP-621 (2020) introduces the idea that the pyproject.toml file be used for wider project configuration and PEP-660 (2021) proposes finally doing away with the need for setup.py for editable installation using pip.

## Steps

- Install uv system-wide

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- Create the project using cookiecutter
```bash
uvx cookiecutter https://github.com/ilkersigirci/python-template.git
```
