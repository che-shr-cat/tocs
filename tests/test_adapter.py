"""Tests for Task 6: Model adapter and evaluation prompts.

Validates:
- Action parsing works for all 5 action types
- Prompt templates render without errors
- Mock API test: given canned responses, adapter produces valid CognitiveMaps
- Adapter classes can be instantiated (without API keys)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.model_adapters.base import (
    AdapterResult,
    BaseAdapter,
    load_prompt,
    parse_action,
)
from evaluation.model_adapters.anthropic import AnthropicAdapter
from evaluation.model_adapters.litellm_adapter import LiteLLMAdapter
from evaluation.model_adapters.openai import OpenAIAdapter
from harness.environment import Environment
from models import ActionType, CognitiveMap

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample_pipeline"


# ── Action parsing ──────────────────────────────────────────────────


class TestParseAction:
    def test_list_empty(self) -> None:
        action = parse_action("LIST()")
        assert action is not None
        assert action.type == ActionType.LIST
        assert action.argument == ""

    def test_list_with_dir(self) -> None:
        action = parse_action("LIST(src)")
        assert action is not None
        assert action.type == ActionType.LIST
        assert action.argument == "src"

    def test_open(self) -> None:
        action = parse_action("OPEN(runner.py)")
        assert action is not None
        assert action.type == ActionType.OPEN
        assert action.argument == "runner.py"

    def test_search(self) -> None:
        action = parse_action("SEARCH(import)")
        assert action is not None
        assert action.type == ActionType.SEARCH
        assert action.argument == "import"

    def test_inspect_two_args(self) -> None:
        action = parse_action("INSPECT(mod_a.py, IngestStage)")
        assert action is not None
        assert action.type == ActionType.INSPECT
        assert action.argument == "mod_a.py"
        assert action.secondary_argument == "IngestStage"

    def test_done(self) -> None:
        action = parse_action("DONE()")
        assert action is not None
        assert action.type == ActionType.DONE

    def test_action_in_sentence(self) -> None:
        action = parse_action("I'll start by listing the root directory. LIST()")
        assert action is not None
        assert action.type == ActionType.LIST

    def test_no_action(self) -> None:
        action = parse_action("I'm thinking about what to do next.")
        assert action is None

    def test_case_insensitive(self) -> None:
        action = parse_action("open(runner.py)")
        assert action is not None
        assert action.type == ActionType.OPEN
        assert action.argument == "runner.py"

    def test_quoted_argument(self) -> None:
        action = parse_action('OPEN("runner.py")')
        assert action is not None
        assert action.argument == "runner.py"

    def test_search_with_spaces(self) -> None:
        action = parse_action("SEARCH(from . import)")
        assert action is not None
        assert action.type == ActionType.SEARCH
        assert action.argument == "from . import"

    def test_inspect_without_symbol(self) -> None:
        """INSPECT with only filepath still parses (though env may reject it)."""
        action = parse_action("INSPECT(mod_a.py)")
        assert action is not None
        assert action.type == ActionType.INSPECT
        assert action.argument == "mod_a.py"

    # ── Task 13: Additional robustness tests for real model output ────

    def test_subdirectory_path(self) -> None:
        """OPEN with subdirectory paths (medium codebases)."""
        action = parse_action("OPEN(text_processor/stages/mod_a.py)")
        assert action is not None
        assert action.type == ActionType.OPEN
        assert action.argument == "text_processor/stages/mod_a.py"

    def test_markdown_bold_action(self) -> None:
        """Models sometimes bold the action name."""
        action = parse_action("**OPEN**(runner.py)")
        assert action is not None
        assert action.type == ActionType.OPEN
        assert action.argument == "runner.py"

    def test_backtick_wrapped_action(self) -> None:
        """`OPEN(file.py)` wrapped in backticks."""
        action = parse_action("`OPEN`(runner.py)")
        assert action is not None
        assert action.type == ActionType.OPEN
        assert action.argument == "runner.py"

    def test_backtick_wrapped_argument(self) -> None:
        """Backticks around the argument."""
        action = parse_action("OPEN(`runner.py`)")
        assert action is not None
        assert action.argument == "runner.py"

    def test_verbose_preamble(self) -> None:
        """Models often add reasoning before the action."""
        text = (
            "Based on my understanding of the codebase so far, I should look "
            "at the registry module to understand how stages are wired together. "
            "Let me open it.\n\nOPEN(text_processor/registry.py)"
        )
        action = parse_action(text)
        assert action is not None
        assert action.type == ActionType.OPEN
        assert action.argument == "text_processor/registry.py"

    def test_inspect_subdirectory_and_symbol(self) -> None:
        """INSPECT with subdirectory path and class name."""
        action = parse_action("INSPECT(text_processor/stages/mod_a.py, IngestStage)")
        assert action is not None
        assert action.type == ActionType.INSPECT
        assert action.argument == "text_processor/stages/mod_a.py"
        assert action.secondary_argument == "IngestStage"

    def test_list_subdirectory(self) -> None:
        """LIST with a subdirectory."""
        action = parse_action("LIST(text_processor/stages)")
        assert action is not None
        assert action.type == ActionType.LIST
        assert action.argument == "text_processor/stages"

    def test_search_complex_query(self) -> None:
        """SEARCH with a complex query string."""
        action = parse_action("SEARCH(StageBase)")
        assert action is not None
        assert action.type == ActionType.SEARCH
        assert action.argument == "StageBase"


# ── Prompt templates ────────────────────────────────────────────────


class TestPrompts:
    def test_system_prompt_loads(self) -> None:
        prompt = load_prompt("system.txt")
        assert len(prompt) > 100
        assert "LIST" in prompt
        assert "OPEN" in prompt
        assert "SEARCH" in prompt
        assert "INSPECT" in prompt
        assert "DONE" in prompt

    def test_probe_prompt_loads(self) -> None:
        prompt = load_prompt("probe.txt")
        assert len(prompt) > 100
        assert "IMPORTS" in prompt
        assert "CALLS_API" in prompt
        assert "DATA_FLOWS_TO" in prompt
        assert "REGISTRY_WIRES" in prompt

    def test_action_prompt_loads(self) -> None:
        prompt = load_prompt("action.txt")
        assert "action" in prompt.lower() or "ACTION" in prompt

    def test_system_prompt_formats(self) -> None:
        prompt = load_prompt("system.txt")
        formatted = prompt.format(probe_interval=3)
        assert "3" in formatted
        # Should not have remaining format placeholders
        assert "{probe_interval}" not in formatted

    def test_action_prompt_formats(self) -> None:
        prompt = load_prompt("action.txt")
        formatted = prompt.format(
            remaining_budget=15,
            total_budget=20,
            opened_files="mod_a.py, mod_b.py",
        )
        assert "15" in formatted
        assert "20" in formatted
        assert "mod_a.py" in formatted

    def test_nonexistent_prompt_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent.txt")

    def test_probe_prompt_has_schema(self) -> None:
        prompt = load_prompt("probe.txt")
        assert '"components"' in prompt
        assert '"invariants"' in prompt
        assert '"unexplored"' in prompt

    def test_probe_prompt_has_example(self) -> None:
        prompt = load_prompt("probe.txt")
        assert "models.py" in prompt
        assert "registry.py" in prompt


# ── Mock adapter ────────────────────────────────────────────────────


class MockAdapter(BaseAdapter):
    """Test adapter that returns canned responses instead of calling an API."""

    def __init__(self, responses: list[str], **kwargs) -> None:
        super().__init__(model="mock-model", **kwargs)
        self._responses = list(responses)
        self._call_index = 0

    def _call_model(self, messages: list[dict], system: str) -> str:
        if self._call_index >= len(self._responses):
            return "DONE()"
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


MOCK_PROBE_JSON = json.dumps({
    "step": 3,
    "components": {
        "models.py": {
            "filepath": "models.py",
            "status": "observed",
            "purpose": "Data models for the pipeline",
            "edges": [],
            "exports": [],
            "confidence": 0.9,
        },
        "runner.py": {
            "filepath": "runner.py",
            "status": "observed",
            "purpose": "Orchestrates the pipeline",
            "edges": [
                {"target": "models.py", "type": "IMPORTS", "confidence": 0.9}
            ],
            "exports": [],
            "confidence": 0.8,
        },
    },
    "invariants": [],
    "unexplored": ["helpers.py", "legacy.py"],
    "uncertainty_summary": "Haven't explored all files.",
})


class TestMockExploration:
    def test_basic_exploration(self) -> None:
        """Mock adapter produces valid maps through the exploration loop."""
        responses = [
            "Let me start by listing files. LIST()",  # Action 1: LIST
            "OPEN(models.py)",                         # Action 2: OPEN
            "OPEN(runner.py)",                         # Action 3: OPEN
            MOCK_PROBE_JSON,                           # Probe at step 3
            "DONE()",                                  # Terminate
        ]

        adapter = MockAdapter(responses)
        env = Environment(FIXTURE_DIR, budget=10)
        result = adapter.run_exploration(env, probe_interval=3)

        assert isinstance(result, AdapterResult)
        assert len(result.maps) >= 1
        assert all(isinstance(m, CognitiveMap) for m in result.maps)
        assert len(result.conversation) > 0
        assert len(result.action_log) > 0

    def test_maps_have_components(self) -> None:
        """Parsed cognitive maps contain the expected components."""
        responses = [
            "LIST()",
            "OPEN(models.py)",
            "OPEN(runner.py)",
            MOCK_PROBE_JSON,
            "DONE()",
        ]

        adapter = MockAdapter(responses)
        env = Environment(FIXTURE_DIR, budget=10)
        result = adapter.run_exploration(env, probe_interval=3)

        assert len(result.maps) >= 1
        cmap = result.maps[0]
        assert "models.py" in cmap.components
        assert "runner.py" in cmap.components

    def test_respects_budget(self) -> None:
        """Exploration stops when budget is exhausted."""
        # Generate more actions than budget allows, with probes interspersed
        responses: list[str] = []
        files = ["models.py", "runner.py", "mod_a.py", "mod_b.py",
                 "mod_c.py", "mod_d.py", "base.py", "registry.py"]
        for i, f in enumerate(files * 3):
            responses.append(f"OPEN({f})")
            if (i + 1) % 3 == 0:
                responses.append(MOCK_PROBE_JSON)
        responses.append(MOCK_PROBE_JSON)  # Extra for final probe

        adapter = MockAdapter(responses)
        env = Environment(FIXTURE_DIR, budget=5)
        result = adapter.run_exploration(env, probe_interval=3)

        assert env.actions_taken <= 5

    def test_no_probe_track(self) -> None:
        """no_probe track skips periodic probes, only collects final."""
        responses = [
            "LIST()",
            "OPEN(models.py)",
            "OPEN(runner.py)",
            "DONE()",
            MOCK_PROBE_JSON,  # Final probe response
        ]

        adapter = MockAdapter(responses)
        env = Environment(FIXTURE_DIR, budget=10)
        result = adapter.run_exploration(env, probe_interval=3, track="no_probe")

        # Should have at most 1 map (the final probe)
        assert len(result.maps) <= 1

    def test_probe_only_track(self) -> None:
        """probe_only track collects maps but strips them from conversation."""
        responses = [
            "LIST()",
            "OPEN(models.py)",
            "OPEN(runner.py)",
            MOCK_PROBE_JSON,  # Probe at step 3
            "DONE()",
        ]

        adapter = MockAdapter(responses)
        env = Environment(FIXTURE_DIR, budget=10)
        result = adapter.run_exploration(env, probe_interval=3, track="probe_only")

        # Map should still be collected
        assert len(result.maps) >= 1
        # Probe JSON should NOT appear in conversation (stripped)
        probe_in_conv = any(
            MOCK_PROBE_JSON in msg.get("content", "")
            for msg in result.conversation
            if msg["role"] == "assistant"
        )
        assert not probe_in_conv

    def test_parse_retry_on_bad_response(self) -> None:
        """Adapter retries when model output doesn't contain a valid action."""
        responses = [
            "Hmm, let me think about this...",  # Bad response → retry
            "LIST()",                             # Retry succeeds
            "OPEN(models.py)",
            "OPEN(runner.py)",
            MOCK_PROBE_JSON,                      # Probe at step 3
            "DONE()",
        ]

        adapter = MockAdapter(responses)
        env = Environment(FIXTURE_DIR, budget=10)
        result = adapter.run_exploration(env, probe_interval=3)

        # Should have recovered and produced results
        assert len(result.action_log) >= 1
        assert len(result.maps) >= 1

    def test_conversation_alternates_roles(self) -> None:
        """Conversation messages alternate between user and assistant."""
        responses = [
            "LIST()",
            "OPEN(models.py)",
            "DONE()",
            MOCK_PROBE_JSON,  # Final probe
        ]

        adapter = MockAdapter(responses)
        env = Environment(FIXTURE_DIR, budget=10)
        result = adapter.run_exploration(env, probe_interval=3)

        for i, msg in enumerate(result.conversation):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role, (
                f"Message {i} has role '{msg['role']}', expected '{expected_role}'"
            )


# ── Adapter instantiation (without real API) ────────────────────────


class TestAdapterInit:
    def test_anthropic_adapter_init(self) -> None:
        """AnthropicAdapter can be constructed without calling the API."""
        adapter = AnthropicAdapter(model="claude-sonnet-4-5-20250929")
        assert adapter.model == "claude-sonnet-4-5-20250929"
        # Client is lazy — not created until needed
        assert adapter._client is None

    def test_openai_adapter_init(self) -> None:
        """OpenAIAdapter can be constructed without calling the API."""
        adapter = OpenAIAdapter(model="gpt-4.1")
        assert adapter.model == "gpt-4.1"
        assert adapter._client is None

    def test_anthropic_default_model(self) -> None:
        adapter = AnthropicAdapter()
        assert adapter.model == "claude-sonnet-4-5-20250929"

    def test_openai_default_model(self) -> None:
        adapter = OpenAIAdapter()
        assert adapter.model == "gpt-4.1"

    def test_anthropic_adapter_has_timeout(self) -> None:
        """AnthropicAdapter accepts a timeout parameter."""
        adapter = AnthropicAdapter(timeout=30.0)
        assert adapter.timeout == 30.0

    def test_anthropic_adapter_default_timeout(self) -> None:
        """AnthropicAdapter has a reasonable default timeout."""
        adapter = AnthropicAdapter()
        assert adapter.timeout == 120.0

    def test_litellm_adapter_init(self) -> None:
        """LiteLLMAdapter can be constructed without calling any API."""
        adapter = LiteLLMAdapter(model="gemini/gemini-2.0-flash")
        assert adapter.model == "gemini/gemini-2.0-flash"
        assert adapter.max_tokens == 65536
        assert adapter.temperature == 0.0
        assert adapter.timeout == 120.0

    def test_litellm_adapter_custom_params(self) -> None:
        """LiteLLMAdapter accepts custom parameters."""
        adapter = LiteLLMAdapter(
            model="anthropic/claude-sonnet-4-5-20250929",
            max_tokens=2048,
            temperature=0.5,
            timeout=60.0,
        )
        assert adapter.model == "anthropic/claude-sonnet-4-5-20250929"
        assert adapter.max_tokens == 2048
        assert adapter.temperature == 0.5
        assert adapter.timeout == 60.0

    def test_litellm_adapter_bare_model_name(self) -> None:
        """LiteLLMAdapter accepts bare model names (no provider prefix)."""
        adapter = LiteLLMAdapter(model="gpt-4.1")
        assert adapter.model == "gpt-4.1"


# ── Medium codebase exploration (mock) ────────────────────────────


class TestMediumCodebaseExploration:
    """Tests that the adapter handles medium codebase paths correctly."""

    def test_subdirectory_open_actions(self) -> None:
        """Exploration handles OPEN with subdirectory paths."""
        from generator.export import export_from_blueprint
        from generator.grammar import PipelineTemplate
        from models import ComplexityTier

        template = PipelineTemplate()
        bp = template.generate(complexity=ComplexityTier.MEDIUM, seed=42)

        # Use a temp directory for this test
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            gt = export_from_blueprint(bp, Path(tmpdir) / "codebase")
            codebase_dir = Path(tmpdir) / "codebase"

            # Pick some real filepaths from the ground truth
            filepaths = list(gt.modules.keys())[:3]

            # Build mock responses that open these files
            responses = ["LIST()"]
            for fp in filepaths:
                responses.append(f"OPEN({fp})")
            responses.append("DONE()")
            # Add probe response for final probe
            responses.append(json.dumps({
                "step": len(filepaths) + 1,
                "components": {},
                "invariants": [],
                "unexplored": [],
                "uncertainty_summary": "test",
            }))

            adapter = MockAdapter(responses)
            env = Environment(codebase_dir, budget=20)
            result = adapter.run_exploration(env, probe_interval=20)

            # Verify files were actually opened
            assert env.actions_taken == len(filepaths) + 1  # +1 for LIST
            assert len(env.opened_files) == len(filepaths)
