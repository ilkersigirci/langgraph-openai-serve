"""Flatten modular Open WebUI Function sources for database synchronization."""

import ast
from pathlib import Path

GENERIC_BUNDLE = (
    "contracts.py",
    "api.py",
    "metadata.py",
    "events.py",
    "interrupts.py",
    "pipe.py",
)


def extract_frontmatter_source(path: Path) -> str:
    """Return the Open WebUI frontmatter block from a Function entrypoint."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != '"""':
        msg = f"Open WebUI Function entrypoint must start with frontmatter: {path}"
        raise ValueError(msg)

    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n").strip() == '"""':
            return "".join(lines[: index + 1]).rstrip("\r\n")

    msg = f"Open WebUI Function frontmatter is not closed: {path}"
    raise ValueError(msg)


def _flatten_module(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.level > 0)
    ]
    return ast.unparse(tree).strip()


def bundle_function(function_dir: Path) -> str:
    """Build one executable source string from a modular Function directory."""
    entrypoint = function_dir / "function.py"
    parts = [extract_frontmatter_source(entrypoint)]

    for module_name in GENERIC_BUNDLE:
        module_path = function_dir / module_name
        module_source = _flatten_module(module_path)
        parts.append(
            f"# ===== BEGIN {module_name} =====\n"
            f"{module_source}\n"
            f"# ===== END {module_name} ====="
        )

    content = "\n\n".join(parts) + "\n"
    compile(content, f"<openwebui:{function_dir.name}>", "exec")
    return content
