# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.13-slim-trixie@sha256:57b29a7c050c4e679e4bea572e14b3965cf4b5839de2da49d55e49ffc4024f39
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336
FROM scratch AS lgos
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS base
ARG PUID=1000
ARG PGID=1000
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 LGOS_REGISTRY=lgos_api_coding_agent.registry:create_registry
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl git ripgrep \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid "$PGID" app \
    && useradd --uid "$PUID" --gid "$PGID" --create-home --shell /bin/bash app \
    && mkdir /workspace \
    && chown app:app /workspace
COPY --from=uv /uv /bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock README.md /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups
COPY src /app/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups
CMD ["lgos", "serve"]

FROM base AS development
COPY --from=lgos pyproject.toml README.md LICENSE /opt/langgraph-openai-serve/
COPY --from=lgos src /opt/langgraph-openai-serve/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python --editable "/opt/langgraph-openai-serve[server]"
USER app

FROM base AS production
USER app
