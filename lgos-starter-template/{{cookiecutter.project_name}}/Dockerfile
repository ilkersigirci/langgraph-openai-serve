# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:{{ cookiecutter.python_version }}-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.23

FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable

FROM ${PYTHON_IMAGE} AS production
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LGOS_HOST=0.0.0.0 \
    LGOS_REGISTRY={{ cookiecutter.project_slug }}.registry:create_registry
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/v1/health', timeout=3)"
CMD ["lgos", "serve"]
