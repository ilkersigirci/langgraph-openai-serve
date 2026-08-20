SHELL := /bin/sh
COMPOSE := docker compose --env-file .env -f docker/compose/demo.yml
LOCAL_API_RUN := uv run --directory api --locked --with-editable ../..
LGOS_CHAINLIT_PORT ?= 3002
LGOS_A_PORT ?= 3004
LGOS_B_PORT ?= 3005
UP_ARGS ?=

.PHONY: help sync test test-postgres lint check compose-config \
	run-api-local run-api-a-local run-api-b-local run-chainlit-local \
	run-api run-api-a run-api-b run-chainlit \
	sync-openwebui compose compose-down compose-dev compose-otel
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

DEMO_API_TEST_POSTGRES_URI ?= postgresql://lgos:lgos@localhost:3001/lgos
test-postgres: ## Test interrupt restart and coordination against PostgreSQL
	DEMO_API_TEST_POSTGRES_URI=$(DEMO_API_TEST_POSTGRES_URI) \
		$(LOCAL_API_RUN) pytest -m integration \
		tests/integration/test_postgres_interrupt_runtime.py

lint: ## Check lint and formatting for all standalone projects
	uv run --directory api --locked ruff check src tests
	uv run --directory api --locked ruff format --check src tests
	uv run --directory ui/chainlit_ui --locked ruff check src tests
	uv run --directory ui/chainlit_ui --locked ruff format --check src tests
	uv run --directory ui/openwebui --locked ruff check src tests
	uv run --directory ui/openwebui --locked ruff format --check src tests

type-check-ui: ## Run type checking for the UI projects
	uv run --directory ui/chainlit_ui --locked ty check src
	uv run --directory ui/openwebui --locked ty check src

type-check: type-check-ui ## Run type checking for all standalone projects
	uv run --directory api --locked ty check src

type-check-local: type-check-ui ## Run type checking for all standalone projects with local API source
	$(LOCAL_API_RUN) ty check src --extra-search-path ../../src

compose-config: ## Validate published and development Compose models
	$(COMPOSE) config --quiet
	$(COMPOSE) -f docker/compose/development.yml config --quiet
	OTEL_COLLECTOR_GATEWAY_ENDPOINT=$${OTEL_COLLECTOR_GATEWAY_ENDPOINT:-https://otel-gateway.example.internal} \
		$(COMPOSE) -f docker/compose/development.yml \
		-f docker/compose/otel.yml config --quiet

check: test lint type-check compose-config ## Run every standalone validation

run-api-local: run-api-a-local ## Run lgos-a locally

run-api-a-local: ## Run lgos-a locally
	$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api-setup
	DEMO_API_PORT=$(LGOS_A_PORT) \
		$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api

run-api-b-local: ## Run lgos-b locally
	$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api-setup
	DEMO_API_PORT=$(LGOS_B_PORT) \
		$(LOCAL_API_RUN) --env-file ../.env lgos-demo-api

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

sync-openwebui: ## Sync the Open WebUI Functions and LGOS Workspace Models
	uv run --directory ui/openwebui --locked --env-file ../../.env lgos-openwebui-sync

compose: ## Run the stack from published images
	$(COMPOSE) up $(UP_ARGS)

compose-down: ## Stop and remove any demo stack variant
	$(COMPOSE) down --remove-orphans

compose-dev: ## Build the API and Chainlit images and run the stack
	$(COMPOSE) -f docker/compose/development.yml up --build $(UP_ARGS)

compose-otel: ## Build this checkout and run it with the OpenTelemetry Collector overlay
	$(COMPOSE) -f docker/compose/development.yml \
		-f docker/compose/otel.yml up --build $(UP_ARGS)
