# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.13-slim-trixie@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded
FROM scratch AS lgos
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS base
ARG PUID=1000
ARG PGID=1000
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
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
CMD ["lgos-api-coding-agent"]

FROM base AS development
COPY --from=lgos pyproject.toml README.md LICENSE /opt/langgraph-openai-serve/
COPY --from=lgos src /opt/langgraph-openai-serve/src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python --editable /opt/langgraph-openai-serve
USER app

FROM base AS production
USER app
