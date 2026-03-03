"""Action implementations for the partial observability harness.

Each function takes a codebase directory and action arguments, and returns
the action output as a string.  Tool semantics are FIXED:

  LIST      → filenames only, no contents, no recursion
  OPEN      → full file contents
  SEARCH    → filepaths + line numbers only, NO content snippets
  INSPECT   → type signature + docstring only, no function body
  DONE      → empty string (no-op)
"""

from __future__ import annotations

import ast
from pathlib import Path

# Files that are metadata, not part of the codebase the agent explores.
_HIDDEN_FILES = {"ground_truth.json"}


def _is_visible(path: Path, codebase_dir: Path) -> bool:
    """Return True if *path* should be visible to the agent."""
    rel = path.relative_to(codebase_dir)
    if rel.name in _HIDDEN_FILES:
        return False
    if "__pycache__" in rel.parts:
        return False
    return True


# ── LIST ────────────────────────────────────────────────────────────


def action_list(codebase_dir: Path, directory: str) -> str:
    """LIST(directory) → filenames only, no contents, no recursion."""
    target = codebase_dir / directory if directory else codebase_dir
    if not target.is_dir():
        return f"Error: '{directory}' is not a directory."
    entries = sorted(
        p.name
        for p in target.iterdir()
        if _is_visible(p, codebase_dir) and p.name != "__pycache__"
    )
    if not entries:
        return "(empty directory)"
    return "\n".join(entries)


# ── OPEN ────────────────────────────────────────────────────────────


def action_open(codebase_dir: Path, filepath: str) -> str:
    """OPEN(filepath) → full file contents as string."""
    target = codebase_dir / filepath
    if not target.is_file() or not _is_visible(target, codebase_dir):
        return f"Error: '{filepath}' not found."
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{filepath}' is not a text file."


# ── SEARCH ──────────────────────────────────────────────────────────


def action_search(codebase_dir: Path, query: str, max_results: int = 10) -> str:
    """SEARCH(query) → filepaths + line numbers only, NO content.

    Plain substring match.  Returns at most *max_results* file entries.
    """
    if not query:
        return "Error: empty search query."
    results: list[str] = []
    for p in sorted(codebase_dir.rglob("*")):
        if not p.is_file() or not _is_visible(p, codebase_dir):
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        matching = [i + 1 for i, line in enumerate(lines) if query in line]
        if matching:
            rel = str(p.relative_to(codebase_dir))
            results.append(f"{rel}:{','.join(str(n) for n in matching)}")
        if len(results) >= max_results:
            break
    if not results:
        return "No matches found."
    return "\n".join(results)


# ── INSPECT ─────────────────────────────────────────────────────────


def action_inspect(codebase_dir: Path, filepath: str, symbol: str) -> str:
    """INSPECT(filepath, symbol) → signature + docstring, no body."""
    target = codebase_dir / filepath
    if not target.is_file() or not _is_visible(target, codebase_dir):
        return f"Error: '{filepath}' not found."
    try:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError) as e:
        return f"Error: could not parse '{filepath}': {e}"

    source_lines = source.splitlines()

    # Search top-level definitions and class members.
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                return _format_function(node, source_lines)
        if isinstance(node, ast.ClassDef):
            if node.name == symbol:
                return _format_class(node, source_lines)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == symbol:
                        return _format_function(
                            item, source_lines, class_name=node.name
                        )
    return f"Symbol '{symbol}' not found in '{filepath}'."


def _extract_signature_lines(node: ast.FunctionDef, source_lines: list[str]) -> str:
    """Extract the 'def ...:' lines from source (handles multi-line sigs)."""
    start = node.lineno - 1  # 0-indexed
    sig_parts: list[str] = []
    paren_depth = 0
    for i in range(start, len(source_lines)):
        line = source_lines[i]
        sig_parts.append(line)
        for ch in line:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
            elif ch == ":" and paren_depth == 0:
                return "\n".join(sig_parts)
    return "\n".join(sig_parts)


def _format_function(
    node: ast.FunctionDef,
    source_lines: list[str],
    class_name: str | None = None,
) -> str:
    """Format a function as signature + docstring (no body)."""
    parts: list[str] = []

    # Decorators
    for dec in node.decorator_list:
        parts.append(f"@{ast.unparse(dec)}")

    # Signature from source (preserves original formatting)
    sig = _extract_signature_lines(node, source_lines)
    parts.append(sig)

    # Docstring
    docstring = ast.get_docstring(node)
    if docstring:
        parts.append(f'    """{docstring}"""')

    prefix = f"({class_name})" if class_name else ""
    if prefix:
        parts.insert(0, f"# Member of class {class_name}")

    return "\n".join(parts)


def _format_class(node: ast.ClassDef, source_lines: list[str]) -> str:
    """Format a class as its definition line, docstring, and method signatures."""
    parts: list[str] = []

    # Decorators
    for dec in node.decorator_list:
        parts.append(f"@{ast.unparse(dec)}")

    # Class definition line
    bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
    if bases:
        parts.append(f"class {node.name}({bases}):")
    else:
        parts.append(f"class {node.name}:")

    # Class docstring
    docstring = ast.get_docstring(node)
    if docstring:
        parts.append(f'    """{docstring}"""')

    # Method signatures (name + params only, no bodies)
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _extract_signature_lines(item, source_lines)
            parts.append("")
            for dec in item.decorator_list:
                parts.append(f"    @{ast.unparse(dec)}")
            parts.append(f"    {sig.strip()}")
            method_doc = ast.get_docstring(item)
            if method_doc:
                parts.append(f'        """{method_doc}"""')

    return "\n".join(parts)
