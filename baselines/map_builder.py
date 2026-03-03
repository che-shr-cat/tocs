"""Utility to build CognitiveMaps from observed file contents.

Shared by all rule-based baselines. Extracts IMPORTS edges via ast,
and optionally REGISTRY_WIRES edges from config file parsing.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import PurePosixPath

from models import (
    BeliefEdge,
    BeliefExport,
    CognitiveMap,
    ComponentBelief,
    EdgeType,
    FunctionSignature,
    ModuleStatus,
    ParameterSpec,
)


def build_map(
    opened_files: dict[str, str],
    listed_files: set[str],
    step: int,
    registry_edges: dict[str, list[str]] | None = None,
) -> CognitiveMap:
    """Build a CognitiveMap from current observations.

    Args:
        opened_files: filepath → file contents for every OPEN'd file.
        listed_files: all filenames discovered via LIST.
        step: current action step number.
        registry_edges: source → [targets] for REGISTRY_WIRES edges
            (populated by config-aware baseline).
    """
    registry_edges = registry_edges or {}
    components: dict[str, ComponentBelief] = {}

    # Observed components (opened .py files)
    for filepath, contents in opened_files.items():
        if not filepath.endswith(".py"):
            continue
        edges = _extract_imports(filepath, contents, listed_files)
        if filepath in registry_edges:
            for target in registry_edges[filepath]:
                edges.append(BeliefEdge(
                    target=target, type=EdgeType.REGISTRY_WIRES, confidence=0.7,
                ))
        exports = _extract_exports(contents)
        components[filepath] = ComponentBelief(
            filepath=filepath,
            status=ModuleStatus.OBSERVED,
            purpose="",
            edges=edges,
            exports=exports,
            confidence=0.9,
        )

    # Inferred components (listed but not opened)
    for filepath in listed_files:
        if filepath not in components and filepath.endswith(".py"):
            comp = ComponentBelief(
                filepath=filepath,
                status=ModuleStatus.INFERRED,
                confidence=0.2,
            )
            # Add registry edges even for unopened files
            if filepath in registry_edges:
                for target in registry_edges[filepath]:
                    comp.edges.append(BeliefEdge(
                        target=target, type=EdgeType.REGISTRY_WIRES, confidence=0.5,
                    ))
            components[filepath] = comp

    unexplored = sorted(
        f for f in listed_files
        if f not in opened_files and not f.startswith("__")
    )

    return CognitiveMap(
        step=step,
        components=components,
        unexplored=unexplored,
    )


def _extract_imports(
    filepath: str, source: str, known_files: set[str]
) -> list[BeliefEdge]:
    """Parse import statements and resolve to known files.

    Handles both flat (small codebase) and nested (medium codebase) paths.
    For relative imports, resolves against the importing file's package.
    """
    edges: list[BeliefEdge] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return edges

    # Build a lookup: module_name → full_path for faster matching
    name_to_paths = _build_module_lookup(known_files)

    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            level = node.level or 0  # number of dots in relative import
            resolved = _resolve_relative_import(
                filepath, node.module, level, known_files, name_to_paths,
            )
            for target in resolved:
                if target not in seen:
                    seen.add(target)
                    edges.append(BeliefEdge(
                        target=target,
                        type=EdgeType.IMPORTS,
                        confidence=0.95,
                    ))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name.split(".")[-1]
                for target in _match_module(mod_name, known_files, name_to_paths):
                    if target not in seen:
                        seen.add(target)
                        edges.append(BeliefEdge(
                            target=target,
                            type=EdgeType.IMPORTS,
                            confidence=0.95,
                        ))
    return edges


def _build_module_lookup(known_files: set[str]) -> dict[str, list[str]]:
    """Build module_name → [full_paths] lookup from known files."""
    lookup: dict[str, list[str]] = {}
    for f in known_files:
        if f.endswith(".py"):
            mod_name = PurePosixPath(f).stem
            lookup.setdefault(mod_name, []).append(f)
    return lookup


def _resolve_relative_import(
    from_file: str,
    module: str,
    level: int,
    known_files: set[str],
    name_to_paths: dict[str, list[str]],
) -> list[str]:
    """Resolve a relative import to known file paths.

    E.g., from a file at 'pkg/stages/mod_a.py':
      - 'from ..models import X' (level=2, module='models') → 'pkg/models.py'
      - 'from .base import Y' (level=1, module='base') → 'pkg/stages/base.py'
    """
    parts = module.split(".")
    target_name = parts[-1]

    if level > 0:
        # Relative import: resolve based on the importing file's location
        from_parts = PurePosixPath(from_file).parts
        if len(from_parts) > level:
            # Go up 'level' directories from the file's directory
            base_parts = from_parts[:-level]  # -1 for file, -(level-1) for dots
            # Build candidate path
            sub_parts = parts[:-1] if len(parts) > 1 else []
            candidate_dir = "/".join(list(base_parts) + sub_parts)
            candidate = f"{candidate_dir}/{target_name}.py" if candidate_dir else f"{target_name}.py"
            if candidate in known_files:
                return [candidate]

    # Fallback: match by module name anywhere in known files
    return _match_module(target_name, known_files, name_to_paths)


def _match_module(
    mod_name: str,
    known_files: set[str],
    name_to_paths: dict[str, list[str]],
) -> list[str]:
    """Match a module name to known file paths."""
    # First try direct filename match (flat structure)
    simple = mod_name + ".py"
    if simple in known_files:
        return [simple]

    # Then try from lookup (nested structure)
    return name_to_paths.get(mod_name, [])


def _extract_exports(source: str) -> list[BeliefExport]:
    """Extract top-level function and class definitions as exports."""
    exports: list[BeliefExport] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return exports

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            sig = _extract_signature(node)
            exports.append(BeliefExport(
                name=node.name, signature=sig, confidence=0.9,
            ))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            exports.append(BeliefExport(
                name=node.name, confidence=0.9,
            ))
    return exports


def _extract_signature(node: ast.FunctionDef) -> FunctionSignature:
    """Extract a structured signature from a FunctionDef AST node."""
    params: list[ParameterSpec] = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue
        type_hint = ast.unparse(arg.annotation) if arg.annotation else ""
        params.append(ParameterSpec(name=arg.arg, type_hint=type_hint))

    # Mark defaults
    n_defaults = len(node.args.defaults)
    if n_defaults:
        for p in params[-n_defaults:]:
            p.has_default = True

    return_type = ast.unparse(node.returns) if node.returns else ""
    return FunctionSignature(params=params, return_type=return_type)


def parse_config_references(
    config_content: str, known_files: set[str]
) -> list[str]:
    """Extract Python module references from a JSON config file.

    Looks for string values that match known .py filenames (without extension).
    """
    try:
        data = json.loads(config_content)
    except json.JSONDecodeError:
        return []

    refs: list[str] = []
    _walk_json(data, known_files, refs)
    return refs


def _walk_json(obj: object, known_files: set[str], refs: list[str]) -> None:
    """Recursively walk JSON and collect module references."""
    if isinstance(obj, str):
        # Check if string matches a known module name (flat or nested)
        candidate = obj + ".py"
        if candidate in known_files and candidate not in refs:
            refs.append(candidate)
        else:
            # Try matching against nested paths by module stem
            for f in known_files:
                if f.endswith(".py"):
                    stem = PurePosixPath(f).stem
                    if stem == obj and f not in refs:
                        refs.append(f)
                        break
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_json(v, known_files, refs)
    elif isinstance(obj, list):
        for item in obj:
            _walk_json(item, known_files, refs)
