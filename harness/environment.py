"""Partial observability environment for codebase exploration.

Wraps a generated codebase directory and enforces:
- Action budget (default 20)
- Information hiding (no file contents leak except through OPEN)
- Action logging and opened-file tracking
"""

from __future__ import annotations

from pathlib import Path

from models import ActionResult, ActionType, AgentAction

from .actions import action_inspect, action_list, action_open, action_search


class BudgetExhausted(Exception):
    """Raised when the agent exceeds its action budget."""


class Environment:
    """Partial observability wrapper around a codebase directory.

    The agent interacts with the codebase exclusively through the
    ``step`` method, which dispatches to LIST / OPEN / SEARCH / INSPECT / DONE.
    """

    def __init__(self, codebase_dir: str | Path, budget: int = 20) -> None:
        self.codebase_dir = Path(codebase_dir).resolve()
        if not self.codebase_dir.is_dir():
            raise ValueError(f"Codebase directory does not exist: {self.codebase_dir}")
        self.budget = budget
        self.actions_taken: int = 0
        self.opened_files: list[str] = []
        self.action_log: list[ActionResult] = []
        self.terminated: bool = False

    # ── Public API ──────────────────────────────────────────────────

    @property
    def remaining_budget(self) -> int:
        return max(0, self.budget - self.actions_taken)

    def step(self, action: AgentAction) -> ActionResult:
        """Execute *action* and return the result.

        Raises ``BudgetExhausted`` if the budget has been spent.
        """
        if self.terminated:
            raise BudgetExhausted("Environment already terminated via DONE.")

        # DONE is free and terminates.
        if action.type == ActionType.DONE:
            self.terminated = True
            result = ActionResult(
                action=action, output="Exploration terminated.", step=self.actions_taken
            )
            self.action_log.append(result)
            return result

        # All other actions cost 1.
        if self.actions_taken >= self.budget:
            raise BudgetExhausted(
                f"Budget exhausted ({self.budget} actions). "
                f"Use DONE to terminate."
            )

        self.actions_taken += 1
        output = self._dispatch(action)

        result = ActionResult(
            action=action, output=output, step=self.actions_taken
        )
        self.action_log.append(result)
        return result

    # ── Dispatch ────────────────────────────────────────────────────

    def _dispatch(self, action: AgentAction) -> str:
        match action.type:
            case ActionType.LIST:
                return action_list(
                    self.codebase_dir, action.argument or ""
                )
            case ActionType.OPEN:
                if not action.argument:
                    return "Error: OPEN requires a filepath argument."
                self.opened_files.append(action.argument)
                return action_open(self.codebase_dir, action.argument)
            case ActionType.SEARCH:
                if not action.argument:
                    return "Error: SEARCH requires a query argument."
                return action_search(self.codebase_dir, action.argument)
            case ActionType.INSPECT:
                if not action.argument or not action.secondary_argument:
                    return "Error: INSPECT requires filepath and symbol arguments."
                return action_inspect(
                    self.codebase_dir,
                    action.argument,
                    action.secondary_argument,
                )
            case _:
                return f"Error: unknown action type '{action.type}'."
