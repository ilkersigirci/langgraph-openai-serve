ARG DEV_IMAGE=ghcr.io/astral-sh/uv:bookworm-slim
ARG PROD_IMAGE=ghcr.io/astral-sh/uv:python3.11-bookworm-slim
ARG UID=1000
ARG GID=1000

#
# Stage: staging
#
FROM ${DEV_IMAGE} AS staging

WORKDIR /app

RUN \
    --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt/archives \
    apt-get update -y \
    && apt-get install -y --no-install-recommends \
    # curl \
    build-essential

ENV \
    PATH="/app/.venv/bin:$PATH"

#
# Stage: development
#
FROM staging AS development

WORKDIR /app

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_LINK_MODE=copy \
    UV_FROZEN=1

# Install the project's dependencies.
RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    # --mount=type=bind,source=.python-version,target=.python-version \
    # --mount=type=bind,source=README.md,target=README.md \
    # --mount=type=bind,source=src,target=src \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

COPY . .

# Install the project.
RUN \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync

RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]


#
# Stage: build
#
FROM staging AS build

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=src,target=src \
    uv build --wheel --out-dir dist

#
# Stage: production
#
# FROM staging AS production
ARG PROD_IMAGE
FROM ${PROD_IMAGE} as production

ARG UID
ARG GID

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CONFIG=1 \
    UV_NO_CACHE=1 \
    # UV_CACHE_DIR=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 \
    UV_FROZEN=1

WORKDIR /app

COPY --from=build /app/dist/*.whl ./

COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    # --mount=type=cache,target=/root/.cache/uv \
    uv export --no-emit-project --format requirements-txt > requirements.txt  \
    && uv pip install ./*.whl --constraint requirements.txt \
    && rm -rf requirements.txt

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN chown -R appuser:appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

ENTRYPOINT ["/app/entrypoint.sh"]
