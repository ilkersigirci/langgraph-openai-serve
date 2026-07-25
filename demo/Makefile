SHELL := /bin/sh
LOCAL_API_RUN := uv run --directory api --locked --with-editable ../..

.PHONY: help sync test lint check compose-config setup-api run-api setup-chainlit run-chainlit sync-openwebui compose compose-dev
.DEFAULT_GOAL := help

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

sync: ## Synchronize all three projects from their lockfiles
	uv --directory api sync --locked
	uv --directory ui/chainlit_ui sync --locked
	uv --directory ui/openwebui sync --locked

test: ## Run all standalone project tests from their lockfiles
	$(LOCAL_API_RUN) pytest
	uv run --directory ui/chainlit_ui --locked pytest
	uv run --directory ui/openwebui --locked pytest

lint: ## Check lint and formatting for all standalone projects
	uv run --directory api --locked ruff check src tests
	uv run --directory api --locked ruff format --check src tests
	uv run --directory ui/chainlit_ui --locked ruff check src tests
	uv run --directory ui/chainlit_ui --locked ruff format --check src tests
	uv run --directory ui/openwebui --locked ruff check src tests
	uv run --directory ui/openwebui --locked ruff format --check src tests

compose-config: ## Validate published and development Compose models
	docker compose -f compose.yaml config --quiet
	docker compose -f compose.yaml -f compose.dev.yaml config --quiet

check: test lint compose-config ## Run every standalone validation

setup-api: ## Initialize or migrate the API checkpoint database
	$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api-setup

run-api: setup-api ## Run the demo API locally
	$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api

setup-chainlit: ## Initialize or migrate the Chainlit database
	uv run --directory ui/chainlit_ui --locked --env-file ../../.env lgos-chainlit-setup

run-chainlit: setup-chainlit ## Run the Chainlit UI locally
	uv run --directory ui/chainlit_ui --locked --env-file ../../.env lgos-chainlit

sync-openwebui: ## Create or update the bundled Open WebUI Functions
	uv run --directory ui/openwebui --locked --env-file ../../.env lgos-openwebui-sync

compose: ## Run the stack from published images
	docker compose -f compose.yaml up

compose-dev: ## Build the API and Chainlit images and run the stack
	docker compose -f compose.yaml -f compose.dev.yaml up --build
