# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.13-slim-trixie@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc

ARG PUID=1000
ARG PGID=1000

# The development Compose model overrides this fallback with the repository
# checkout as the "lgos" named build context.
FROM scratch AS lgos

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS python

ARG PUID
ARG PGID

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid "$PGID" app \
    && useradd --uid "$PUID" --gid "$PGID" --create-home \
        --shell /usr/sbin/nologin app

CMD ["uvicorn", "lgos_demo_api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM python AS dependencies

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project \
        --no-install-package langgraph-openai-serve \
        --no-default-groups

FROM dependencies AS development

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups \
        --no-install-package langgraph-openai-serve

COPY --from=lgos pyproject.toml README.md LICENSE /workspace/langgraph-openai-serve/
COPY --from=lgos src /workspace/langgraph-openai-serve/src

# The checkout intentionally keeps its local-only 0.0.1 version. CI verifies
# the editable source path instead of comparing it with the published range.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        --python /app/.venv/bin/python \
        --editable "/workspace/langgraph-openai-serve[tracing]"

USER app

FROM dependencies AS production-build

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable \
    && uv pip check --python /app/.venv/bin/python

FROM python AS production

COPY --from=production-build /app/.venv /app/.venv

USER app
