"""Config-aware baseline: open config/registry files first, then BFS imports."""

from __future__ import annotations

from collections import deque

from harness.environment import Environment
from models import ActionType, AgentAction, CognitiveMap

from .discovery import discover_files
from .map_builder import build_map, parse_config_references


def run(
    env: Environment,
    probe_interval: int = 3,
) -> list[CognitiveMap]:
    """Explore config files first, then follow imports BFS."""
    maps: list[CognitiveMap] = []
    opened: dict[str, str] = {}
    registry_edges: dict[str, list[str]] = {}
    queue: deque[str] = deque()

    # 1. Discover all files (recursive LIST)
    listed = discover_files(env)
    _maybe_probe(env, maps, opened, listed, registry_edges, probe_interval)

    # 2. Identify and open config/registry files first
    config_files = sorted(
        f for f in listed
        if _is_config_file(f) and f not in opened
    )
    for cf in config_files:
        if env.remaining_budget <= 0:
            break
        result = env.step(AgentAction(type=ActionType.OPEN, argument=cf))
        if result.output.startswith("Error"):
            continue
        opened[cf] = result.output

        # Parse module references from config
        refs = parse_config_references(result.output, listed)
        if refs:
            # Record registry wiring edges from the config-reading module
            registry_files = [
                f for f in listed
                if f.endswith(".py") and _might_read_config(f, cf, listed)
            ]
            for rf in registry_files:
                registry_edges.setdefault(rf, []).extend(refs)
            # Also enqueue referenced modules for opening
            for ref in refs:
                if ref not in opened:
                    queue.append(ref)

        _maybe_probe(env, maps, opened, listed, registry_edges, probe_interval)

    # 3. Open registry/config Python files (files with "registry" or "config" in name)
    registry_py = sorted(
        f for f in listed
        if f.endswith(".py")
        and ("registry" in f.lower() or "config" in f.lower())
        and f not in opened
    )
    for rf in registry_py:
        if env.remaining_budget <= 0:
            break
        result = env.step(AgentAction(type=ActionType.OPEN, argument=rf))
        if result.output.startswith("Error"):
            continue
        opened[rf] = result.output

        # Check if this file reads a config/json file
        for cf in config_files:
            cf_base = cf.rsplit("/", 1)[-1].replace(".json", "").replace(".yaml", "")
            if cf_base in result.output:
                refs = parse_config_references(
                    opened.get(cf, ""), listed
                )
                if refs:
                    registry_edges.setdefault(rf, []).extend(refs)

        _maybe_probe(env, maps, opened, listed, registry_edges, probe_interval)

    # 4. BFS remaining from queue + all .py files
    for f in sorted(listed):
        if f.endswith(".py") and f not in opened and f not in queue:
            queue.append(f)

    while queue and env.remaining_budget > 0:
        filepath = queue.popleft()
        if filepath in opened or filepath not in listed:
            continue

        result = env.step(AgentAction(type=ActionType.OPEN, argument=filepath))
        if result.output.startswith("Error"):
            continue
        opened[filepath] = result.output

        # Follow imports
        for name in listed:
            if name.endswith(".py") and name not in opened and name not in queue:
                mod_name = name.rsplit("/", 1)[-1].replace(".py", "")
                if f"from .{mod_name}" in result.output or \
                   f"import {mod_name}" in result.output or \
                   f"from ..{mod_name}" in result.output:
                    queue.append(name)

        _maybe_probe(env, maps, opened, listed, registry_edges, probe_interval)

    # Final map
    final = build_map(opened, listed, env.actions_taken, registry_edges)
    maps.append(final)
    return maps


def _is_config_file(name: str) -> bool:
    """Is this likely a config/registry file?"""
    lower = name.lower()
    basename = lower.rsplit("/", 1)[-1]
    if basename.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")):
        if basename != "ground_truth.json":
            return True
    if "config" in basename or "registry" in basename:
        return True
    return False


def _might_read_config(py_file: str, config_file: str, listed: set[str]) -> bool:
    """Heuristic: does this .py file likely read the config file?"""
    base = py_file.rsplit("/", 1)[-1].replace(".py", "").lower()
    return "registry" in base or "config" in base or "runner" in base


def _maybe_probe(
    env: Environment,
    maps: list[CognitiveMap],
    opened: dict[str, str],
    listed: set[str],
    registry_edges: dict[str, list[str]],
    interval: int,
) -> None:
    if env.actions_taken > 0 and env.actions_taken % interval == 0:
        maps.append(build_map(opened, listed, env.actions_taken, registry_edges))
