"""Random baseline: open files uniformly at random."""

from __future__ import annotations

import random as _random

from harness.environment import Environment
from models import ActionType, AgentAction, CognitiveMap

from .discovery import discover_files
from .map_builder import build_map


def run(
    env: Environment,
    probe_interval: int = 3,
    seed: int | None = None,
) -> list[CognitiveMap]:
    """Open files at random until budget exhausted."""
    rng = _random.Random(seed)
    maps: list[CognitiveMap] = []
    opened: dict[str, str] = {}

    # 1. Discover all files (recursive LIST)
    listed = discover_files(env)
    _maybe_probe(env, maps, opened, listed, probe_interval)

    # 2. Shuffle and open
    candidates = sorted(f for f in listed if f.endswith(".py") and not f.endswith("__init__.py"))
    rng.shuffle(candidates)

    for filepath in candidates:
        if env.remaining_budget <= 0:
            break
        if filepath in opened:
            continue
        result = env.step(AgentAction(type=ActionType.OPEN, argument=filepath))
        if not result.output.startswith("Error"):
            opened[filepath] = result.output
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
