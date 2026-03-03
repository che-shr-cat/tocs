"""Base adapter and utilities for model evaluation.

Provides:
- parse_action() — extract AgentAction from model output text
- load_prompt() — load prompt templates from evaluation/prompts/
- BaseAdapter — abstract base with shared exploration loop
- AdapterResult — dataclass for exploration results
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from harness.environment import BudgetExhausted, Environment
from harness.probing import parse_cognitive_map
from models import ActionType, AgentAction, ActionResult, CognitiveMap

logger = logging.getLogger(__name__)

# ── Prompt loading ──────────────────────────────────────────────────

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Load a prompt template from evaluation/prompts/."""
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()


# ── Action parsing ──────────────────────────────────────────────────

# Primary regex: ACTION(argument) — handles single-line args and paths
_ACTION_RE = re.compile(
    r"\b(LIST|OPEN|SEARCH|INSPECT|DONE)\s*\(\s*(.*?)\s*\)",
    re.IGNORECASE | re.DOTALL,
)

# Fallback: markdown-formatted actions like **OPEN**(file.py) or `OPEN(file.py)`
_ACTION_RE_MARKDOWN = re.compile(
    r"(?:\*\*|`)(LIST|OPEN|SEARCH|INSPECT|DONE)(?:\*\*|`)\s*\(\s*(.*?)\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def parse_action(text: str) -> AgentAction | None:
    """Parse a model output into an AgentAction.

    Handles various model output formats:
    - Plain: OPEN(file.py)
    - With preamble: "Let me open the file. OPEN(file.py)"
    - Quoted args: OPEN("file.py") or OPEN('file.py')
    - Backtick-wrapped args: OPEN(`file.py`)
    - Markdown bold: **OPEN**(file.py)
    - Subdirectory paths: OPEN(pkg/stages/mod_a.py)
    - INSPECT with two args: INSPECT(file.py, ClassName)

    Returns None if no valid action is found.
    """
    match = _ACTION_RE.search(text)
    if not match:
        match = _ACTION_RE_MARKDOWN.search(text)
    if not match:
        return None

    action_name = match.group(1).upper()
    raw_args = match.group(2).strip()

    action_type = ActionType(action_name.lower())

    if action_type == ActionType.DONE:
        return AgentAction(type=action_type)

    if action_type == ActionType.INSPECT:
        # INSPECT takes two arguments: filepath, symbol
        parts = [p.strip().strip("'\"`") for p in raw_args.split(",", 1)]
        if len(parts) >= 2 and parts[1]:
            return AgentAction(
                type=action_type,
                argument=parts[0],
                secondary_argument=parts[1],
            )
        elif len(parts) == 1 and parts[0]:
            return AgentAction(type=action_type, argument=parts[0])
        return AgentAction(type=action_type)

    # Strip quotes, backticks, and whitespace from argument
    arg = raw_args.strip("'\"`") if raw_args else ""
    # Collapse any interior newlines/whitespace (model might split path across lines)
    arg = " ".join(arg.split())
    return AgentAction(type=action_type, argument=arg)


# ── Result type ─────────────────────────────────────────────────────


@dataclass
class AdapterResult:
    """Result of running a model through the exploration loop."""

    maps: list[CognitiveMap] = field(default_factory=list)
    action_log: list[ActionResult] = field(default_factory=list)
    conversation: list[dict] = field(default_factory=list)


# ── Base adapter ────────────────────────────────────────────────────

MAX_PARSE_RETRIES = 2
# Safety cap: maximum API calls per exploration run (prevents runaway loops)
MAX_TURNS = 100


class BaseAdapter(ABC):
    """Base class for model adapters.

    Subclasses implement _call_model() for their specific API.
    The exploration loop is shared.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        probe_prompt: str | None = None,
        action_prompt: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model = model
        self._system_prompt = system_prompt or load_prompt("system.txt")
        self._probe_prompt = probe_prompt or load_prompt("probe.txt")
        self._action_prompt = action_prompt or load_prompt("action.txt")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def _call_model(self, messages: list[dict], system: str) -> str:
        """Call the model API and return the response text."""
        ...

    def _call_with_retry(self, messages: list[dict], system: str) -> str:
        """Call model with exponential-backoff retry for transient errors."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._call_model(messages, system)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError(
            f"Model call failed after {self.max_retries} retries: {last_error}"
        ) from last_error

    def _format_action_prompt(self, env: Environment) -> str:
        """Format the action prompt with current state."""
        opened = ", ".join(env.opened_files) if env.opened_files else "(none)"
        return self._action_prompt.format(
            remaining_budget=env.remaining_budget,
            total_budget=env.budget,
            opened_files=opened,
        )

    def run_exploration(
        self,
        env: Environment,
        probe_interval: int = 3,
        track: str = "probe_as_scratchpad",
        initial_message: str | None = None,
    ) -> AdapterResult:
        """Run the exploration loop: action -> result -> probe -> repeat.

        Args:
            env: The partial observability environment.
            probe_interval: Probe for cognitive map every N actions.
            track: One of "probe_as_scratchpad", "probe_only", "no_probe".
            initial_message: Optional context to prepend to the first action prompt
                (e.g. evidence from REVISE phase).

        Returns:
            AdapterResult with maps, action log, and conversation.
        """
        maps: list[CognitiveMap] = []
        messages: list[dict] = []
        self._last_messages = messages  # Expose for crash recovery
        total_api_calls = 0
        parse_failures = 0

        system = self._system_prompt.format(probe_interval=probe_interval)

        # Initial action prompt (optionally prefixed with context)
        first_prompt = self._format_action_prompt(env)
        if initial_message:
            first_prompt = f"{initial_message}\n\n{first_prompt}"
        messages.append({
            "role": "user",
            "content": first_prompt,
        })

        logger.info(
            "Starting exploration: model=%s budget=%d probe_interval=%d track=%s",
            self.model, env.budget, probe_interval, track,
        )

        while True:
            # Safety cap: prevent runaway API calls
            if total_api_calls >= MAX_TURNS:
                logger.warning(
                    "Safety cap reached (%d API calls). Forcing termination.",
                    total_api_calls,
                )
                break

            # Get model response
            response_text = self._call_with_retry(messages, system)
            total_api_calls += 1
            messages.append({"role": "assistant", "content": response_text})

            # Parse action (with retries for parse failures)
            action = parse_action(response_text)
            retries = 0
            while action is None and retries < MAX_PARSE_RETRIES:
                retries += 1
                parse_failures += 1
                logger.debug(
                    "Parse failure %d (retry %d/%d): %s",
                    parse_failures,
                    retries,
                    MAX_PARSE_RETRIES,
                    response_text[:200],
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "Could not parse your action. Please respond with exactly "
                        "ONE action in the format: ACTION(argument)\n"
                        "Examples: LIST() | OPEN(file.py) | SEARCH(query) | "
                        "INSPECT(file.py, symbol) | DONE()"
                    ),
                })
                response_text = self._call_with_retry(messages, system)
                total_api_calls += 1
                messages.append({"role": "assistant", "content": response_text})
                action = parse_action(response_text)

            if action is None:
                logger.warning(
                    "Giving up after %d parse retries. Terminating exploration.",
                    MAX_PARSE_RETRIES,
                )
                break

            logger.info(
                "Step %d/%d: %s",
                env.actions_taken + 1,
                env.budget,
                _format_action_str(action),
            )

            # Execute action
            try:
                result = env.step(action)
            except BudgetExhausted:
                logger.info("Budget exhausted at step %d.", env.actions_taken)
                break

            if action.type == ActionType.DONE:
                logger.info("Agent called DONE at step %d.", env.actions_taken)
                break

            # Format the result message
            action_str = _format_action_str(action)
            result_text = f"Result of {action_str}:\n{result.output}"

            # Check if we should probe
            should_probe = (
                track != "no_probe"
                and env.actions_taken > 0
                and env.actions_taken % probe_interval == 0
            )

            if should_probe:
                # Send result + probe prompt
                messages.append({
                    "role": "user",
                    "content": f"{result_text}\n\n{self._probe_prompt}",
                })

                # Get probe response
                probe_response = self._call_with_retry(messages, system)
                total_api_calls += 1
                messages.append({"role": "assistant", "content": probe_response})

                # Parse cognitive map
                probe_parse = parse_cognitive_map(
                    probe_response, env.actions_taken
                )
                if probe_parse.success and probe_parse.map is not None:
                    maps.append(probe_parse.map)
                    logger.info(
                        "Probe at step %d: %d components, %d invariants",
                        env.actions_taken,
                        len(probe_parse.map.components),
                        len(probe_parse.map.invariants),
                    )
                else:
                    logger.warning(
                        "Probe parse failed at step %d: %s",
                        env.actions_taken,
                        probe_parse.error,
                    )

                if track == "probe_only":
                    # Strip probe Q+A from conversation, keep result
                    messages = messages[:-2]
                    messages.append({
                        "role": "user",
                        "content": (
                            f"{result_text}\n\n"
                            f"{self._format_action_prompt(env)}"
                        ),
                    })
                else:
                    # probe_as_scratchpad — add action prompt after probe
                    messages.append({
                        "role": "user",
                        "content": self._format_action_prompt(env),
                    })
            else:
                # Regular: send result + action prompt
                messages.append({
                    "role": "user",
                    "content": (
                        f"{result_text}\n\n"
                        f"{self._format_action_prompt(env)}"
                    ),
                })

        # Final probe (always, unless no actions were taken)
        if env.actions_taken > 0:
            needs_final = not maps or maps[-1].step != env.actions_taken
            if needs_final:
                logger.info("Collecting final probe at step %d.", env.actions_taken)
                messages.append({
                    "role": "user",
                    "content": self._probe_prompt,
                })
                probe_response = self._call_with_retry(messages, system)
                total_api_calls += 1
                messages.append({
                    "role": "assistant",
                    "content": probe_response,
                })
                probe_parse = parse_cognitive_map(
                    probe_response, env.actions_taken
                )
                if probe_parse.success and probe_parse.map is not None:
                    maps.append(probe_parse.map)
                else:
                    logger.warning(
                        "Final probe parse failed: %s", probe_parse.error,
                    )

        logger.info(
            "Exploration complete: %d actions, %d maps, %d API calls, "
            "%d parse failures",
            env.actions_taken,
            len(maps),
            total_api_calls,
            parse_failures,
        )

        return AdapterResult(
            maps=maps,
            action_log=env.action_log,
            conversation=messages,
        )


def _format_action_str(action: AgentAction) -> str:
    """Format an action as a human-readable string."""
    name = action.type.value.upper()
    if action.type == ActionType.INSPECT:
        return f"{name}({action.argument}, {action.secondary_argument})"
    elif action.argument:
        return f"{name}({action.argument})"
    return f"{name}()"
