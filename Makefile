SHELL=/bin/bash
PACKAGE=src/langgraph_openai_serve
TEST_DIRS=./tests
LINT_TARGETS=$(PACKAGE) $(TEST_DIRS)
TEST_PATH=tests
PRECOMMIT_FILE_PATHS=$(PACKAGE)/__init__.py
DEMO_DIR=demo

.PHONY: help install test test-demo test-demo-local check-demo clean build-sdist build-wheel publish pre-commit format lint
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

test-bifrost: DEMO_TEST_BIFROST_BASE_URL ?= http://localhost:8081/openai_passthrough/v1
test-bifrost: DEMO_TEST_DIRECT_BASE_URL ?= http://localhost:8000/v1
test-bifrost: ## Run the optional Bifrost proxy integration test
	DEMO_TEST_BIFROST_BASE_URL=$(DEMO_TEST_BIFROST_BASE_URL) \
		DEMO_TEST_DIRECT_BASE_URL=$(DEMO_TEST_DIRECT_BASE_URL) \
		uv run --directory $(DEMO_DIR)/api --locked --with-editable ../.. \
		pytest -m integration tests/integration/test_bifrost_proxy.py

test-demo: ## Test all demo projects against their locked dependencies
	$(MAKE) -C $(DEMO_DIR) test

test-demo-local: ## Test this checkout through the API plus both standalone UIs
	uv run --directory $(DEMO_DIR)/api --locked --with-editable ../.. pytest
	uv run --directory $(DEMO_DIR)/ui/chainlit_ui --locked pytest
	uv run --directory $(DEMO_DIR)/ui/openwebui --locked pytest

check-demo: ## Run every standalone demo validation
	$(MAKE) -C $(DEMO_DIR) check

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
