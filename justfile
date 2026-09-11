set default-list
set dotenv-load
set indentation := "    "
set minimum-version := "1.58.0"
set positional-arguments
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

package := "src/langgraph_openai_serve"
lint_targets := package + " tests"

# Install the locked development environment and Git hooks.
[arg('no_cache', long='no-cache', value='true')]
[group('setup')]
install no_cache='false':
    uv sync --frozen {{ if no_cache == "true" { "--no-cache" } else { "" } }}
    uv run prek install

# Upgrade dependencies and refresh the lockfile.
[group('setup')]
upgrade *args:
    uv sync --upgrade "$@"

# Run package tests, optionally followed by pytest paths and arguments.
[group('test')]
test *args:
    uv run --locked --module pytest "$@"

# Remove package build, coverage, and Python cache artifacts.
[group('clean')]
clean:
    rm -rf ./build ./dist ./coverage ./htmlcov ./.coverage ./.coverage.* ./.pytest_cache ./.ruff_cache ./src/*.egg-info ./*.egg-info
    find src tests -type f -iname "*.so" -delete
    find src tests -type f -iname "*.pyc" -delete
    find src tests -type d -name "*.egg-info" -prune -exec rm -rf {} +
    find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +
    find src tests -type d -name ".ruff_cache" -prune -exec rm -rf {} +

# Build distributions; pass --sdist or --wheel to restrict the formats.
[group('package')]
build *args: clean
    uv build --out-dir dist "$@"

# Publish distributions from dist, followed by optional uv arguments.
[group('package')]
publish *args:
    uv publish "$@" dist/*

# Build documentation strictly, or serve it locally with --serve.
[arg('address', long)]
[arg('serve', long, value='true')]
[group('docs')]
docs serve='false' address='0.0.0.0:7999':
    {{ if serve == "true" { "uv run zensical serve --dev-addr " + quote(address) } else { "uv run zensical build --clean --strict" } }}

# Run Git hooks; defaults to every tracked file.
[group('quality')]
hooks *args='--all-files':
    uv run --locked prek run "$@"

# Check Just formatting and Ruff against selected or default paths.
[group('quality')]
[script('bash')]
lint *targets:
    set -euo pipefail
    just --fmt --check
    if (( $# == 0 )); then
        set -- {{ lint_targets }}
    fi
    uv run --locked --module ruff format "$@" --check --diff
    uv run --locked --module ruff check "$@"

# Run type checking against a path.
[arg('path', long)]
[group('quality')]
type-check path='src':
    uv run --locked ty check "$1"

# Run every static package check.
[group('quality')]
check: lint type-check

# Format and fix selected or default paths; add --unsafe for unsafe Ruff fixes.
[arg('unsafe', long, value='true')]
[group('quality')]
[script('bash')]
format unsafe='false' *targets:
    set -euo pipefail
    shift
    if (( $# == 0 )); then
        set -- {{ lint_targets }}
    fi
    uv run --locked --module ruff format "$@"
    uv run --locked --module ruff check "$@" --fix --show-fixes {{ if unsafe == "true" { "--unsafe-fixes" } else { "" } }}
