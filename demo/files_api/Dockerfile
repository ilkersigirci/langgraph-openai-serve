# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.13-slim-trixie@sha256:2d44738ebf4ce5f63b9d9ea871e9c7cfdc0426f7b0c97cac915fd88ded1d86bb
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336

ARG PUID=1000
ARG PGID=1000

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

CMD ["lgos-files-api"]

FROM python AS dependencies

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-default-groups

FROM dependencies AS development

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

USER app

FROM dependencies AS production-build

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable \
    && uv pip check --python /app/.venv/bin/python

FROM python AS production

COPY --from=production-build /app/.venv /app/.venv

USER app
