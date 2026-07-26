SHELL := /bin/sh
COMPOSE := docker compose -f compose.yaml
LOCAL_API_RUN := uv run --directory api --locked --with-editable ../..
LGOS_CHAINLIT_PORT ?= 3002
LGOS_A_PORT ?= 3004
LGOS_B_PORT ?= 3005

.PHONY: help sync test lint check compose-config \
	run-api-local run-api-a-local run-api-b-local run-chainlit-local \
	run-api run-api-a run-api-b run-chainlit \
	sync-openwebui compose compose-dev
.DEFAULT_GOAL := help

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

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
	$(COMPOSE) config --quiet
	$(COMPOSE) -f compose.dev.yaml config --quiet

check: test lint compose-config ## Run every standalone validation

run-api-local: run-api-a-local ## Run lgos-a locally

run-api-a-local: ## Run lgos-a locally
	$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api-setup
	$(LOCAL_API_RUN) --env-file ../.env uvicorn lgos_demo_api.app:app \
		--host 0.0.0.0 --port $(LGOS_A_PORT) --reload

run-api-b-local: ## Run lgos-b locally
	$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api-setup
	$(LOCAL_API_RUN) --env-file ../.env uvicorn lgos_demo_api.app:app \
		--host 0.0.0.0 --port $(LGOS_B_PORT) --reload

run-chainlit-local: ## Run the Chainlit UI locally
	uv run --directory ui/chainlit_ui --locked --env-file ../../.env lgos-chainlit-setup
	uv run --directory ui/chainlit_ui --locked --env-file ../../.env \
		uvicorn lgos_chainlit.main:app --host 0.0.0.0 \
		--port $(LGOS_CHAINLIT_PORT)

run-api: run-api-a ## Run the lgos-a container

run-api-a: ## Run the lgos-a container
	$(COMPOSE) up lgos-demo-api-a

run-api-b: ## Run the lgos-b container
	$(COMPOSE) up lgos-demo-api-b

run-chainlit: ## Run the Chainlit container and its dependencies
	$(COMPOSE) up lgos-chainlit

sync-openwebui: ## Create or update the bundled Open WebUI Functions
	uv run --directory ui/openwebui --locked --env-file ../../.env lgos-openwebui-sync

compose: ## Run the stack from published images
	$(COMPOSE) up

compose-dev: ## Build the API and Chainlit images and run the stack
	$(COMPOSE) -f compose.dev.yaml up --build
