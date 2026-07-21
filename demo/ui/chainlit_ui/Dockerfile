FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

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
