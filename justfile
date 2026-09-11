set default-list
set dotenv-load
set indentation := "    "
set minimum-version := "1.58.0"
set positional-arguments
set script-interpreter := ["bash", "-euo", "pipefail"]
# Preserve the caller's environment when Bash is launched over SSH.
set shell := ["bash", "--norc", "-euo", "pipefail", "-c"]

# Install the locked development environment and Git hooks; accepts uv sync flags.
[group('setup')]
install *args:
    uv sync --locked "$@"
    uv run --locked prek install

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
    find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +

# Build fresh distributions; pass --sdist or --wheel to restrict the formats.
[group('package')]
build *args:
    uv build --clear "$@"

# Publish distributions from dist, followed by optional uv arguments.
[group('package')]
publish *args:
    uv publish "$@" dist/*

# Build documentation strictly, or serve it locally with --serve.
[arg('address', long)]
[arg('serve', long, value='true')]
[group('docs')]
docs serve='false' address='0.0.0.0:7999' *args:
    uv run --locked zensical {{ if serve == "true" { "serve --dev-addr " + quote(address) } else { "build --clean --strict" } }} "${@:3}"

# Run Git hooks; defaults to every tracked file.
[group('quality')]
hooks *args='--all-files':
    uv run --locked prek run "$@"

# Check Just formatting and Ruff against selected or default paths.
[group('quality')]
[script]
lint *targets:
    just --fmt --check
    if (( $# == 0 )); then
        set -- src tests
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
[script]
format unsafe='false' *targets:
    just --fmt
    shift
    if (( $# == 0 )); then
        set -- src tests
    fi
    uv run --locked --module ruff format "$@"
    uv run --locked --module ruff check "$@" --fix --show-fixes {{ if unsafe == "true" { "--unsafe-fixes" } else { "" } }}
