# The chainlit-utils Compose override replaces this fallback with the sibling
# checkout as the "chainlit_utils" named build context.
FROM scratch AS chainlit_utils

FROM ghcr.io/astral-sh/uv:0.12.1-python3.13-trixie-slim@sha256:8db423175bfff42bd1c81f77280bc92f10ef9cf03161803bd5cb6e15d86c3d10 AS base

ARG PUID=1000
ARG PGID=1000

WORKDIR /app

ENV CHAINLIT_APP_ROOT=/app/src/lgos_chainlit \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-default-groups

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

RUN groupadd --gid "$PGID" app \
    && useradd --uid "$PUID" --gid "$PGID" --create-home \
        --shell /usr/sbin/nologin app

USER app

CMD ["lgos-chainlit"]

FROM base AS local-utils

USER root

COPY --from=chainlit_utils pyproject.toml README.md LICENSE /workspace/chainlit-utils/
COPY --from=chainlit_utils src /workspace/chainlit-utils/src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python \
        --editable "/workspace/chainlit-utils[audio,sso]"

USER app

# The default build uses only the project and its locked PyPI dependencies.
FROM base AS production
