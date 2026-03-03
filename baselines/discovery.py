"""Shared file discovery for baselines.

Recursively LISTs directories to discover all files in a codebase,
handling both flat (small) and nested (medium) directory structures.
"""

from __future__ import annotations

from collections import deque

from harness.environment import Environment
from models import ActionType, AgentAction


def discover_files(env: Environment) -> set[str]:
    """Recursively LIST directories to discover all files.

    Uses BFS over directories. Returns full relative paths
    (e.g., 'text_processor/stages/mod_a.py').

    Each LIST call costs 1 action from the budget.
    """
    listed: set[str] = set()
    dir_queue: deque[str] = deque([""])

    while dir_queue and env.remaining_budget > 0:
        directory = dir_queue.popleft()
        result = env.step(AgentAction(type=ActionType.LIST, argument=directory))

        for name in result.output.splitlines():
            name = name.strip()
            if not name or name.startswith("("):
                continue

            full_path = f"{directory}/{name}" if directory else name

            # Heuristic: if the name has no extension and doesn't look like a
            # file, it's likely a directory. LIST it too.
            if "." not in name and name != "__pycache__":
                dir_queue.append(full_path)
            else:
                listed.add(full_path)

    return listed
