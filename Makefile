# Oneshell means one can run multiple lines in a recipe in the same shell, so one doesn't have to
# chain commands together with semicolon
.ONESHELL:
SHELL=/bin/bash

.PHONY: install
.DEFAULT_GOAL=help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

install-uv: ## Install uv
	! command -v uv &> /dev/null && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="~/.local/bin" sh

update-uv: ## Update uv to the latest version
	@uv self update

upgrade-dependencies: ## Updates the lockfiles and installs the latest version of the dependencies
	@uv sync -U

install: ## Installs the development version of the package
	$(MAKE) install-uv
	$(MAKE) update-uv
	@uv sync --frozen

doc-build: ## Test whether documentation can be built
	@uv run mkdocs build -s

doc-serve: ## Build and serve the documentation
	@uv run mkdocs serve

doc-deploy-github: ## Build documentation with mkdocs and deploy to github pages
	@uv run mkdocs gh-deploy --force

create-example-project: ## Create and test a new project with the cookiecutter template
	@rm -rf python-template-example || true
	@uv run cookiecutter --no-input . --overwrite-if-exists \
		author="İlker SIĞIRCI" \
		email="sigirci.ilker@gmail.com" \
		github_author_handle=ilkersigirci \
		project_name=python-template-example \
		project_slug=python_template_example \
		python_version=3.12 \
		git_remote_location=github \
		devcontainer=n \
		docs=n \
		mypy=n \
		open_source_license="MIT license"
