SHELL=/bin/bash
PACKAGE=src/langgraph_openai_serve
TEST_DIRS=./tests ./demo/tests
LINT_TARGETS=$(PACKAGE) $(TEST_DIRS)
TEST_PATH=tests
PRECOMMIT_FILE_PATHS=$(PACKAGE)/__init__.py

.PHONY: help install test clean build-sdist build-wheel publish pre-commit format lint
.DEFAULT_GOAL=help

help:
	@grep -hE '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

# If .env file exists, include it and export its variables
ifeq ($(shell test -f .env && echo 1),1)
    include .env
    export
endif

install-uv: ## Install uv
	! command -v uv &> /dev/null && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$$HOME/.local/bin" sh

update-uv: ## Update uv to the latest version
	uv self update

install-base: ## Installs only package dependencies
	uv sync --frozen --no-dev --no-install-project

install: ## Installs the development version of the package
	$(MAKE) install-uv
	$(MAKE) update-uv
	uv sync --frozen
	$(MAKE) install-precommit

install-no-cache: ## Installs the development version of the package without cache
	$(MAKE) install-uv
	$(MAKE) update-uv
	uv sync --frozen --no-cache
	$(MAKE) install-precommit

install-precommit: ## Install pre-commit hooks
	uv run prek install

update-dependencies: ## Updates the lockfiles and installs dependencies. Dependencies are updated if necessary
	uv sync

upgrade-dependencies: ## Updates the lockfiles and installs the latest version of the dependencies
	uv sync -U

test-one: ## Run a pytest path/node ID with TEST_PATH=<selector>, default is `tests`
	uv lock --locked
	uv run --module pytest ${TEST_PATH}

test-one-parallel: ## Run a pytest path/node ID in parallel with TEST_PATH=<selector>
	uv lock --locked
	uv run --module pytest -n auto ${TEST_PATH}

test-all: ## Run all tests
	uv lock --locked
	uv run --module pytest

test-all-parallel: ## Run all tests with parallelization
	uv lock --locked
	uv run --module pytest -n auto

test-coverage: ## Run all tests with coverage
	uv lock --locked
	uv run --module pytest --cov=${PACKAGE} --cov-report=html:coverage

test-coverage-parallel:
	uv lock --locked
	uv run --module pytest -n auto --cov=${PACKAGE} --cov-report=html:coverage

test: clean-test test-all ## Cleans and runs all tests
test-parallel: clean-test test-all-parallel ## Cleans and runs all tests with parallelization

clean-build: ## Clean build dist and egg directories left after install
	rm -rf ./build ./dist */*.egg-info *.egg-info
	find . -type f -iname "*.so" -delete
	find . -type f -iname '*.pyc' -delete
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} \;
	find . -type d -name '__pycache__' -prune -exec rm -rf {} \;
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} \;

clean-test: ## Clean test related files left after test
	rm -rf ./coverage ./htmlcov
	find . -type f -name '.coverage*' ! -name '.coveragerc' -delete
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} \;

clean: clean-build clean-test ## Cleans build and test related files

build-sdist: ## Make Python source distribution
	$(MAKE) clean-build
	uv build --sdist --out-dir dist

build-wheel: ## Make Python wheel distribution
	$(MAKE) clean-build
	uv build --wheel --out-dir dist

publish: ## Builds the project and publish the package to Pypi
	# $(MAKE) build-sdist
	uv publish dist/*
	# uv publish --publish-url https://test.pypi.org/legacy/ --username DUMMY --password DUMMY dist/*

doc-build: ## Test whether documentation can be built
	uv run zensical build --clean --strict

doc-serve: ## Build and serve the documentation
	uv run zensical serve --dev-addr 0.0.0.0:7999

pre-commit-one: ## Run pre-commit with specific files
	uv lock --locked
	uv run prek run --files ${PRECOMMIT_FILE_PATHS}

pre-commit: ## Run pre-commit for all package files
	uv lock --locked
	uv run prek run --all-files

pre-commit-clean: ## Clean pre-commit cache
	uv run prek clean

lint: ## Lint code with ruff
	uv lock --locked
	uv run --module ruff format ${LINT_TARGETS} --check --diff
	uv run --module ruff check ${LINT_TARGETS}

lint-report: ## Lint report for gitlab
	uv lock --locked
	uv run --module ruff format ${LINT_TARGETS} --check --diff
	uv run --module ruff check ${LINT_TARGETS} --output-format gitlab > gl-code-quality-report.json

format: ## Format package and test files. CHANGES CODE
	uv lock --locked
	uv run --module ruff format ${LINT_TARGETS}
	uv run --module ruff check ${LINT_TARGETS} --fix --show-fixes

type-check: ## Run ty for type checking
	uv run ty check

format-unsafe: ## Unsafely format package and test files. CHANGES CODE
	uv lock --locked
	uv run --module ruff format ${LINT_TARGETS}
	uv run --module ruff check ${LINT_TARGETS} --fix --show-fixes --unsafe-fixes

setup-demo-checkpointer: ## Initialize or migrate the demo checkpoint schema
	uv run --module demo.api.setup_checkpointer

setup-demo-chainlit: ## Initialize or migrate the demo Chainlit persistence schema
	uv run --module demo.ui.chainlit_ui.setup_database

run-demo-api: setup-demo-checkpointer # ## Run the demo api in development mode
	uv run --module demo.api.app

run-demo-ui-chainlit: setup-demo-chainlit ## Run the persistent Chainlit UI
	uv run uvicorn demo.ui.chainlit_ui.main:app --host 0.0.0.0 --port 5000 --no-access-log

run-demo-ui-chainlit-hitl: setup-demo-chainlit ## Run the persistent Chainlit human-in-the-loop UI
	DEMO_CHAINLIT_UI_FILE=hitl uv run uvicorn demo.ui.chainlit_ui.main:app --host 0.0.0.0 --port 5000 --no-access-log
