"""BFS-Import baseline: follow import statements breadth-first."""

from __future__ import annotations

from collections import deque

from harness.environment import Environment
from models import ActionType, AgentAction, CognitiveMap

from .discovery import discover_files
from .map_builder import build_map


def run(
    env: Environment,
    probe_interval: int = 3,
) -> list[CognitiveMap]:
    """Explore by following imports BFS. Returns maps every *probe_interval* actions."""
    maps: list[CognitiveMap] = []
    opened: dict[str, str] = {}
    queue: deque[str] = deque()

    # 1. Discover all files (recursive LIST)
    listed = discover_files(env)
    _maybe_probe(env, maps, opened, listed, probe_interval)

    # Seed queue with all .py files (skip __init__.py)
    for f in sorted(listed):
        if f.endswith(".py") and not f.endswith("__init__.py"):
            queue.append(f)

    # 2. BFS: open files and follow imports
    while queue and env.remaining_budget > 0:
        filepath = queue.popleft()
        if filepath in opened:
            continue
        if filepath not in listed:
            continue

        result = env.step(AgentAction(type=ActionType.OPEN, argument=filepath))
        if result.output.startswith("Error"):
            continue
        opened[filepath] = result.output

        # Parse imports and enqueue new targets
        for name in listed:
            if name.endswith(".py") and name not in opened and name not in queue:
                mod_name = name.rsplit("/", 1)[-1].replace(".py", "")
                if f"from .{mod_name}" in result.output or \
                   f"import {mod_name}" in result.output or \
                   f"from ..{mod_name}" in result.output:
                    queue.append(name)

        _maybe_probe(env, maps, opened, listed, probe_interval)

    # Final map
    final = build_map(opened, listed, env.actions_taken)
    maps.append(final)
    return maps


def _maybe_probe(
    env: Environment,
    maps: list[CognitiveMap],
    opened: dict[str, str],
    listed: set[str],
    interval: int,
) -> None:
    if env.actions_taken > 0 and env.actions_taken % interval == 0:
        maps.append(build_map(opened, listed, env.actions_taken))
