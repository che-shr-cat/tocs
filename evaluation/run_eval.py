"""End-to-end evaluation pipeline for ToCS.

CLI usage:
  python -m evaluation.run_eval evaluate \
    --model bfs-import --codebase ./data/test --mode active --output ./results/

  python -m evaluation.run_eval evaluate \
    --model claude-sonnet-4-5-20250929 --codebase ./data/test --phase revise --output ./results/

  python -m evaluation.run_eval batch \
    --models bfs-import,config-aware --codebases ./data/ --modes active --output ./results/
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env file (API keys)

logger = logging.getLogger(__name__)

import typer

import baselines.bfs_import as _bfs
import baselines.config_aware as _cfg
import baselines.oracle as _oracle
import baselines.random_explorer as _rnd
from evaluation.model_adapters.base import BaseAdapter
from evaluation.model_adapters.litellm_adapter import LiteLLMAdapter
from generator.export import list_codebase_files, load_ground_truth
from harness.environment import Environment
from harness.mutations import (
    MutationEngine,
    MutationResult,
    ShamMutationEngine,
    score_revision,
)
from harness.probing import parse_cognitive_map
from metrics.map_accuracy import score_exploration, score_map
from models import (
    ActionResult,
    ActionType,
    AgentAction,
    BeliefRevisionMetrics,
    CodebaseGroundTruth,
    CognitiveMap,
    EvalResult,
    ExplorationMetrics,
    MapAccuracyMetrics,
    MutationType,
)

# ── Baseline registry ───────────────────────────────────────────────

BASELINES = {
    "bfs-import": _bfs.run,
    "config-aware": _cfg.run,
    "random": _rnd.run,
}


# ── Adapter factory ─────────────────────────────────────────────────


def get_adapter(model: str, **kwargs) -> BaseAdapter:
    """Create a model adapter by name.

    Uses LiteLLM for universal model support. Model names follow LiteLLM
    conventions — prefix with provider if needed:
    - gemini/gemini-2.0-flash, gemini/gemini-2.5-flash
    - anthropic/claude-sonnet-4-5-20250929
    - openai/gpt-4.1
    - Or bare names (LiteLLM auto-detects provider)
    """
    return LiteLLMAdapter(model=model, **kwargs)


# ── Core pipeline ───────────────────────────────────────────────────


def run_single(
    model: str,
    codebase: Path,
    mode: str = "active",
    track: str = "probe_as_scratchpad",
    budget: int = 20,
    probe_interval: int = 3,
    output: Path | None = None,
    replay_log: Path | None = None,
) -> EvalResult:
    """Run a single evaluation and return an EvalResult.

    Supports baselines (bfs-import, config-aware, random, oracle)
    and model adapters (claude-*, gpt-*) in active/passive modes.
    """
    codebase = Path(codebase)
    gt = load_ground_truth(codebase)

    maps: list[CognitiveMap] = []
    action_log: list[ActionResult] = []
    conversation: list[dict] = []

    if model == "oracle":
        maps = _oracle.run(gt)
    elif model in BASELINES:
        env = Environment(codebase, budget=budget)
        baseline_fn = BASELINES[model]
        if model == "random":
            maps = baseline_fn(env, probe_interval=probe_interval, seed=42)
        else:
            maps = baseline_fn(env, probe_interval=probe_interval)
        action_log = env.action_log
    elif mode == "active":
        env = Environment(codebase, budget=budget)
        adapter = get_adapter(model)
        try:
            result = adapter.run_exploration(
                env, probe_interval=probe_interval, track=track
            )
            maps = result.maps
            action_log = result.action_log
            conversation = result.conversation
        except Exception as exc:
            # Save partial results before re-raising
            logger.error("Exploration crashed: %s. Saving partial results.", exc)
            action_log = env.action_log
            # Recover conversation from adapter if possible
            if hasattr(adapter, '_last_messages'):
                conversation = adapter._last_messages
            if output:
                _save_partial(
                    model, gt.codebase_id, mode, track,
                    maps, action_log, conversation, gt, Path(output), exc,
                )
            raise
    elif mode == "passive-full":
        adapter = get_adapter(model)
        try:
            maps, action_log = _run_passive_full(adapter, codebase, gt)
        except Exception as exc:
            logger.error("passive-full crashed: %s. Saving partial results.", exc)
            if hasattr(adapter, '_last_messages'):
                conversation = adapter._last_messages
            if output:
                _save_partial(
                    model, gt.codebase_id, mode, track,
                    maps, action_log, conversation, gt, Path(output), exc,
                )
            raise
    elif mode == "passive-oracle":
        adapter = get_adapter(model)
        try:
            maps, action_log = _run_passive_oracle(
                adapter, codebase, gt, budget, probe_interval, track
            )
        except Exception as exc:
            logger.error("passive-oracle crashed: %s. Saving partial results.", exc)
            if hasattr(adapter, '_last_messages'):
                conversation = adapter._last_messages
            if output:
                _save_partial(
                    model, gt.codebase_id, mode, track,
                    maps, action_log, conversation, gt, Path(output), exc,
                )
            raise
    elif mode == "passive-replay":
        if replay_log is None:
            raise ValueError("passive-replay mode requires --replay-log path")
        adapter = get_adapter(model)
        try:
            maps, action_log = _run_passive_replay(
                adapter, replay_log, probe_interval, track
            )
        except Exception as exc:
            logger.error("passive-replay crashed: %s. Saving partial results.", exc)
            if hasattr(adapter, '_last_messages'):
                conversation = adapter._last_messages
            if output:
                _save_partial(
                    model, gt.codebase_id, mode, track,
                    maps, action_log, conversation, gt, Path(output), exc,
                )
            raise
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ── Score (protected) ──────────────────────────────────────
    if output is None:
        output = Path("./results/")

    zeroed_map = MapAccuracyMetrics(
        dependency_precision=0.0,
        dependency_recall=0.0,
        dependency_f1=0.0,
        invariant_precision=0.0,
        invariant_recall=0.0,
        invariant_f1=0.0,
        confidence_ece=0.0,
    )

    try:
        if maps:
            map_metrics = score_map(maps[-1], gt)
        else:
            map_metrics = zeroed_map
    except Exception as exc:
        logger.error("score_map failed: %s — using zeroed metrics", exc)
        map_metrics = zeroed_map

    try:
        exploration_metrics = score_exploration(maps, action_log, gt)
    except Exception as exc:
        logger.error("score_exploration failed: %s — using defaults", exc)
        exploration_metrics = ExplorationMetrics(
            steps_taken=len(action_log),
            unique_files_opened=0,
            action_auc=0.0,
            observation_auc=0.0,
        )

    # Parse mode/condition
    if "-" in mode:
        eval_mode, condition = mode.split("-", 1)
    else:
        eval_mode = mode
        condition = None

    eval_result = EvalResult(
        model_name=model,
        codebase_id=gt.codebase_id,
        mode=eval_mode,
        passive_condition=condition,
        track=track,
        exploration=exploration_metrics,
        map_accuracy=map_metrics,
        cognitive_maps=maps,
    )

    # Save
    save_result(eval_result, Path(output), action_log, conversation)

    return eval_result


# ── Revise phase ───────────────────────────────────────────────────


def run_revise(
    model: str,
    codebase: Path,
    mutation_type: MutationType = MutationType.BOUNDARY_BREACH,
    is_sham: bool = False,
    construct_budget: int = 15,
    revise_budget: int = 5,
    probe_interval: int = 3,
    track: str = "probe_as_scratchpad",
    output: Path | None = None,
) -> EvalResult:
    """Run the full REVISE phase: CONSTRUCT → mutate → re-explore → score.

    1. CONSTRUCT: agent builds initial cognitive map
    2. Snapshot pre-mutation belief
    3. Apply mutation (or sham)
    4. Present evidence to agent
    5. Agent re-explores with limited budget
    6. Probe new belief state
    7. Score against post-mutation ground truth
    """
    codebase = Path(codebase)
    gt = load_ground_truth(codebase)

    # --- Phase 1: CONSTRUCT ---
    construct_result = run_single(
        model=model,
        codebase=codebase,
        mode="active",
        track=track,
        budget=construct_budget,
        probe_interval=probe_interval,
    )

    # Snapshot pre-mutation belief (last map from construct phase)
    if not construct_result.cognitive_maps:
        raise ValueError("CONSTRUCT phase produced no cognitive maps")
    pre_belief = construct_result.cognitive_maps[-1]

    # --- Phase 2: Apply mutation ---
    # Work on a copy so we don't destroy the original codebase
    workdir = codebase.parent / f".tocs_revise_{codebase.name}"
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(codebase, workdir)

    try:
        workdir_gt = load_ground_truth(workdir)

        if is_sham:
            engine = ShamMutationEngine(workdir, workdir_gt)
            mutation_result = engine.generate_sham(mutation_type)
        else:
            engine = MutationEngine(workdir, workdir_gt)
            mutation_result = engine.apply(mutation_type)

        # --- Phase 3: Present evidence + re-explore ---
        if model in BASELINES or model == "oracle":
            # Baselines: just re-run on mutated codebase
            env = Environment(workdir, budget=revise_budget)
            if model == "oracle":
                post_maps = _oracle.run(mutation_result.post_gt)
            elif model == "random":
                post_maps = BASELINES[model](
                    env, probe_interval=probe_interval, seed=42
                )
            else:
                post_maps = BASELINES[model](
                    env, probe_interval=probe_interval
                )
        else:
            env = Environment(workdir, budget=revise_budget)
            adapter = get_adapter(model)

            # Seed the conversation with evidence
            evidence_msg = (
                "A change has been made to the codebase. "
                "Here is the test output after the change:\n\n"
                f"{mutation_result.evidence}\n\n"
                "You may re-explore the codebase to update your understanding. "
                f"You have {revise_budget} actions remaining."
            )

            result = adapter.run_exploration(
                env,
                probe_interval=probe_interval,
                track=track,
                initial_message=evidence_msg,
            )
            post_maps = result.maps

        # Post-mutation belief
        post_belief = post_maps[-1] if post_maps else pre_belief

        # --- Phase 4: Score ---
        revision_metrics = score_revision(
            pre_belief, post_belief, mutation_result, gt
        )

        # Score map accuracy against post-mutation ground truth
        map_metrics = score_map(post_belief, mutation_result.post_gt)
        exploration_metrics = construct_result.exploration

        eval_result = EvalResult(
            model_name=model,
            codebase_id=gt.codebase_id,
            mode="active",
            passive_condition=None,
            track=track,
            exploration=exploration_metrics,
            map_accuracy=map_metrics,
            belief_revision=revision_metrics,
            cognitive_maps=construct_result.cognitive_maps + post_maps,
        )

        if output:
            save_result(eval_result, Path(output))

        return eval_result

    finally:
        # Clean up working copy
        if workdir.exists():
            shutil.rmtree(workdir)


# ── Passive modes ───────────────────────────────────────────────────


def _run_passive_full(
    adapter: BaseAdapter,
    codebase: Path,
    gt: CodebaseGroundTruth,
) -> tuple[list[CognitiveMap], list[ActionResult]]:
    """Present the entire codebase at once, then probe once."""
    files = list_codebase_files(codebase)
    content_parts: list[str] = []
    for f in files:
        try:
            text = (codebase / f).read_text()
            content_parts.append(f"=== {f} ===\n{text}")
        except Exception:
            continue

    all_content = "\n\n".join(content_parts)
    system = adapter._system_prompt.format(probe_interval=1)
    messages = [
        {
            "role": "user",
            "content": (
                "Here is the complete codebase. All files are shown below.\n\n"
                f"{all_content}\n\n"
                f"{adapter._probe_prompt}"
            ),
        },
    ]

    response = adapter._call_with_retry(messages, system)
    parse_result = parse_cognitive_map(response, step=0)
    maps = [parse_result.map] if parse_result.success and parse_result.map else []

    # Synthetic action log (all files "opened" at once)
    action_log = [
        ActionResult(
            action=AgentAction(type=ActionType.OPEN, argument=f),
            output="(passive-full)",
            step=i + 1,
        )
        for i, f in enumerate(files)
    ]
    return maps, action_log


def _run_passive_oracle(
    adapter: BaseAdapter,
    codebase: Path,
    gt: CodebaseGroundTruth,
    budget: int,
    probe_interval: int,
    track: str,
) -> tuple[list[CognitiveMap], list[ActionResult]]:
    """Reveal oracle-selected files one at a time, probe every K steps."""
    # Rank modules by edge participation (most connected first)
    edge_counts: dict[str, int] = {}
    for edge in gt.dependency_edges:
        for key in ("source", "target"):
            name = edge.get(key, "")
            if name:
                edge_counts[name] = edge_counts.get(name, 0) + 1

    all_modules = list(gt.modules.keys())
    ranked = sorted(
        all_modules, key=lambda f: edge_counts.get(f, 0), reverse=True
    )
    selected = ranked[:budget]

    system = adapter._system_prompt.format(probe_interval=probe_interval)
    messages: list[dict] = []
    maps: list[CognitiveMap] = []
    action_log: list[ActionResult] = []

    for i, filepath in enumerate(selected):
        step = i + 1
        try:
            content = (codebase / filepath).read_text()
        except Exception:
            content = "(file not found)"

        action_log.append(
            ActionResult(
                action=AgentAction(type=ActionType.OPEN, argument=filepath),
                output=content,
                step=step,
            )
        )

        should_probe = (
            track != "no_probe" and step % probe_interval == 0
        )

        if should_probe:
            messages.append({
                "role": "user",
                "content": (
                    f"File revealed ({step}/{len(selected)}): {filepath}\n\n"
                    f"{content}\n\n{adapter._probe_prompt}"
                ),
            })
            response = adapter._call_with_retry(messages, system)
            messages.append({"role": "assistant", "content": response})

            parse_result = parse_cognitive_map(response, step)
            if parse_result.success and parse_result.map:
                maps.append(parse_result.map)

            if track == "probe_only":
                # Strip probe Q+A, keep file content
                messages = messages[:-2]
                messages.append({
                    "role": "user",
                    "content": (
                        f"File revealed ({step}/{len(selected)}): {filepath}\n\n"
                        f"{content}\n\nNote any architectural observations."
                    ),
                })
                messages.append({
                    "role": "assistant",
                    "content": "Noted.",
                })
        else:
            messages.append({
                "role": "user",
                "content": (
                    f"File revealed ({step}/{len(selected)}): {filepath}\n\n"
                    f"{content}\n\nNote any architectural observations."
                ),
            })
            response = adapter._call_with_retry(messages, system)
            messages.append({"role": "assistant", "content": response})

    # Final probe
    if action_log and (not maps or maps[-1].step != len(action_log)):
        messages.append({"role": "user", "content": adapter._probe_prompt})
        response = adapter._call_with_retry(messages, system)
        messages.append({"role": "assistant", "content": response})
        parse_result = parse_cognitive_map(response, len(action_log))
        if parse_result.success and parse_result.map:
            maps.append(parse_result.map)

    return maps, action_log


def _run_passive_replay(
    adapter: BaseAdapter,
    replay_log_path: Path,
    probe_interval: int,
    track: str,
) -> tuple[list[CognitiveMap], list[ActionResult]]:
    """Replay a prior run's full observation trace."""
    with open(replay_log_path) as f:
        raw_log = json.load(f)

    system = adapter._system_prompt.format(probe_interval=probe_interval)
    messages: list[dict] = []
    maps: list[CognitiveMap] = []
    action_log: list[ActionResult] = []

    for i, entry in enumerate(raw_log):
        step = i + 1
        action_type = entry.get("action", {}).get("type", "open")
        argument = entry.get("action", {}).get("argument", "")
        output = entry.get("output", "")

        action = AgentAction(
            type=ActionType(action_type),
            argument=argument,
            secondary_argument=entry.get("action", {}).get("secondary_argument"),
        )
        result = ActionResult(action=action, output=output, step=step)
        action_log.append(result)

        action_str = f"{action_type.upper()}({argument})"
        msg = f"Result of {action_str}:\n{output}"

        should_probe = track != "no_probe" and step % probe_interval == 0

        if should_probe:
            messages.append({
                "role": "user",
                "content": f"{msg}\n\n{adapter._probe_prompt}",
            })
            response = adapter._call_with_retry(messages, system)
            messages.append({"role": "assistant", "content": response})

            parse_result = parse_cognitive_map(response, step)
            if parse_result.success and parse_result.map:
                maps.append(parse_result.map)

            if track == "probe_only":
                messages = messages[:-2]
                messages.append({"role": "user", "content": msg})
                messages.append({
                    "role": "assistant",
                    "content": "Understood.",
                })
        else:
            messages.append({"role": "user", "content": msg})
            response = adapter._call_with_retry(messages, system)
            messages.append({"role": "assistant", "content": response})

    # Final probe
    if action_log and (not maps or maps[-1].step != len(action_log)):
        messages.append({"role": "user", "content": adapter._probe_prompt})
        response = adapter._call_with_retry(messages, system)
        messages.append({"role": "assistant", "content": response})
        parse_result = parse_cognitive_map(response, len(action_log))
        if parse_result.success and parse_result.map:
            maps.append(parse_result.map)

    return maps, action_log


# ── Filename helper ──────────────────────────────────────────────────


def _expected_filename(model: str, codebase_id: str, mode: str) -> str:
    """Compute the canonical result filename for a given run configuration.

    Must stay in sync with save_result().
    """
    if "-" in mode:
        eval_mode, condition = mode.split("-", 1)
        condition = f"-{condition}"
    else:
        eval_mode = mode
        condition = ""
    filename = f"{model}_{codebase_id}_{eval_mode}{condition}.json"
    return filename.replace("/", "_").replace("\\", "_")


# ── Save ────────────────────────────────────────────────────────────


def _save_partial(
    model: str,
    codebase_id: str,
    mode: str,
    track: str,
    maps: list,
    action_log: list,
    conversation: list[dict],
    gt,
    output_dir: Path,
    error: Exception,
) -> None:
    """Save partial results when a run crashes mid-way.

    Ensures we never lose expensive API call data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_").replace("\\", "_")
    prefix = f"{safe_model}_{codebase_id}_{mode}_PARTIAL"

    # Save conversation (most valuable — can re-score from this)
    if conversation:
        conv_path = output_dir / f"{prefix}_conversation.json"
        conv_path.write_text(json.dumps(conversation, indent=2))
        logger.info("Saved partial conversation to %s", conv_path)

    # Save action log
    if action_log:
        log_path = output_dir / f"{prefix}_action_log.json"
        log_data = [
            {
                "action": {
                    "type": a.action.type.value,
                    "argument": a.action.argument,
                    "secondary_argument": a.action.secondary_argument,
                },
                "output": a.output,
                "step": a.step,
            }
            for a in action_log
        ]
        log_path.write_text(json.dumps(log_data, indent=2))
        logger.info("Saved partial action log to %s", log_path)

    # Save error info
    err_path = output_dir / f"{prefix}_error.txt"
    err_path.write_text(
        f"Model: {model}\n"
        f"Codebase: {codebase_id}\n"
        f"Steps completed: {len(action_log)}\n"
        f"Maps collected: {len(maps)}\n"
        f"Error: {error}\n"
    )
    logger.info("Saved partial error info to %s", err_path)


def save_result(
    result: EvalResult,
    output_dir: Path,
    action_log: list[ActionResult] | None = None,
    conversation: list[dict] | None = None,
) -> Path:
    """Save EvalResult as JSON. Optionally saves action log and conversation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition = f"-{result.passive_condition}" if result.passive_condition else ""
    filename = (
        f"{result.model_name}_{result.codebase_id}_{result.mode}{condition}.json"
    )
    filename = filename.replace("/", "_").replace("\\", "_")

    path = output_dir / filename
    path.write_text(result.model_dump_json(indent=2))

    # Save action log for passive-replay
    if action_log:
        log_path = path.with_name(path.stem + "_action_log.json")
        log_data = [
            {
                "action": {
                    "type": a.action.type.value,
                    "argument": a.action.argument,
                    "secondary_argument": a.action.secondary_argument,
                },
                "output": a.output,
                "step": a.step,
            }
            for a in action_log
        ]
        log_path.write_text(json.dumps(log_data, indent=2))

    # Save full conversation log (model dialogue)
    if conversation:
        conv_path = path.with_name(path.stem + "_conversation.json")
        conv_path.write_text(json.dumps(conversation, indent=2))

    return path


# ── CLI ─────────────────────────────────────────────────────────────

app = typer.Typer(help="ToCS evaluation pipeline")


@app.command()
def evaluate(
    model: str = typer.Option(
        ..., help="Model name or baseline (bfs-import, config-aware, random, oracle)"
    ),
    codebase: Path = typer.Option(..., help="Path to codebase directory"),
    mode: str = typer.Option(
        "active", help="active | passive-full | passive-oracle | passive-replay"
    ),
    phase: str = typer.Option(
        "construct", help="construct | revise"
    ),
    track: str = typer.Option(
        "probe_as_scratchpad",
        help="no_probe | probe_only | probe_as_scratchpad",
    ),
    budget: int = typer.Option(20, help="Action budget"),
    revise_budget: int = typer.Option(
        5, "--revise-budget", help="Re-exploration budget for revise phase"
    ),
    mutation_type: str = typer.Option(
        "BOUNDARY_BREACH",
        "--mutation-type",
        help="INTERFACE_BREAK | DEPENDENCY_SHIFT | BOUNDARY_BREACH",
    ),
    sham: bool = typer.Option(False, "--sham", help="Use sham mutation (control)"),
    probe_interval: int = typer.Option(
        3, "--probe-interval", help="Probe every N actions"
    ),
    output: Path = typer.Option(Path("./results/"), help="Output directory"),
    replay_log: Optional[Path] = typer.Option(
        None, "--replay-log", help="Action log for passive-replay"
    ),
) -> None:
    """Run evaluation on a single model + codebase."""
    if phase == "revise":
        result = run_revise(
            model=model,
            codebase=codebase,
            mutation_type=MutationType(mutation_type),
            is_sham=sham,
            construct_budget=budget,
            revise_budget=revise_budget,
            probe_interval=probe_interval,
            track=track,
            output=output,
        )
    else:
        result = run_single(
            model=model,
            codebase=codebase,
            mode=mode,
            track=track,
            budget=budget,
            probe_interval=probe_interval,
            output=output,
            replay_log=replay_log,
        )

    typer.echo(f"Model:      {result.model_name}")
    typer.echo(f"Codebase:   {result.codebase_id}")
    typer.echo(
        f"Mode:       {result.mode}"
        + (f" ({result.passive_condition})" if result.passive_condition else "")
    )
    typer.echo(f"Maps:       {len(result.cognitive_maps)}")
    typer.echo(f"Dep F1:     {result.map_accuracy.dependency_f1:.3f}")
    typer.echo(f"Inv F1:     {result.map_accuracy.invariant_f1:.3f}")
    typer.echo(f"Action AUC: {result.exploration.action_auc:.3f}")
    typer.echo(f"Steps:      {result.exploration.steps_taken}")
    if result.belief_revision:
        typer.echo(f"BRS:        {result.belief_revision.revision_score:.3f}")
        typer.echo(f"Inertia:    {result.belief_revision.inertia_proper:.3f}")
        typer.echo(f"Gullibility:{result.belief_revision.gullibility_rate:.3f}")


@app.command("batch")
def batch_run(
    models: str = typer.Option(..., help="Comma-separated model names"),
    codebases: Path = typer.Option(
        ..., help="Directory containing codebase subdirectories"
    ),
    modes: str = typer.Option("active", help="Comma-separated modes"),
    track: str = typer.Option("probe_as_scratchpad"),
    budget: int = typer.Option(20),
    probe_interval: int = typer.Option(3, "--probe-interval"),
    output: Path = typer.Option(Path("./results/")),
    resume: bool = typer.Option(True, help="Skip already-completed runs"),
    retry_failed: bool = typer.Option(
        False, "--retry-failed",
        help="Retry runs that have PARTIAL crash artifacts",
    ),
) -> None:
    """Run evaluation on multiple models x codebases x modes."""
    model_list = [m.strip() for m in models.split(",")]
    mode_list = [m.strip() for m in modes.split(",")]

    # Find codebase directories with ground_truth.json
    codebase_dirs = sorted(
        d
        for d in codebases.iterdir()
        if d.is_dir() and (d / "ground_truth.json").exists()
    )

    if not codebase_dirs:
        typer.echo(f"No codebases found in {codebases}")
        raise typer.Exit(1)

    # ── Pre-flight validation ────────────────────────────────
    output.mkdir(parents=True, exist_ok=True)

    # Load codebase IDs upfront (cheap — just one JSON field each)
    codebase_ids: dict[Path, str] = {}
    for cb in codebase_dirs:
        try:
            gt = load_ground_truth(cb)
            codebase_ids[cb] = gt.codebase_id
        except Exception as exc:
            typer.echo(f"WARNING: Cannot load ground_truth.json from {cb}: {exc}")

    codebase_dirs = [cb for cb in codebase_dirs if cb in codebase_ids]

    if not codebase_dirs:
        typer.echo("No valid codebases after pre-flight check")
        raise typer.Exit(1)

    # Validate LLM adapters early (catches missing API keys)
    llm_models = [m for m in model_list if m not in BASELINES and m != "oracle"]
    for m in llm_models:
        try:
            get_adapter(m)
        except Exception as exc:
            typer.echo(f"ERROR: Cannot create adapter for '{m}': {exc}")
            typer.echo("  Check that the required API key is set.")
            raise typer.Exit(1)

    n = len(model_list) * len(codebase_dirs) * len(mode_list)
    typer.echo(
        f"Running {len(model_list)} models x "
        f"{len(codebase_dirs)} codebases x "
        f"{len(mode_list)} modes = {n} evaluations"
    )
    if resume:
        typer.echo("Resume mode ON — will skip completed runs")

    # ── Run loop ─────────────────────────────────────────────
    results: list[EvalResult] = []
    skipped: list[str] = []
    failed: list[str] = []

    for m in model_list:
        for cb in codebase_dirs:
            codebase_id = codebase_ids[cb]
            for md in mode_list:
                run_label = f"{m} / {cb.name} / {md}"
                typer.echo(f"\n--- {run_label} ---")

                expected_name = _expected_filename(m, codebase_id, md)
                expected_path = output / expected_name

                # Check for existing completed result
                if resume and expected_path.exists():
                    try:
                        existing = EvalResult.model_validate_json(
                            expected_path.read_text()
                        )
                        results.append(existing)
                        skipped.append(run_label)
                        typer.echo(f"  SKIP (exists: {expected_name})")
                        continue
                    except Exception:
                        typer.echo(
                            f"  WARNING: {expected_name} exists but is "
                            "unreadable — will re-run"
                        )

                # Check for partial crash artifacts
                safe_model = m.replace("/", "_").replace("\\", "_")
                partial_prefix = f"{safe_model}_{codebase_id}_{md}_PARTIAL"
                partial_files = list(output.glob(f"{partial_prefix}*"))
                if partial_files and not retry_failed:
                    skipped.append(run_label)
                    typer.echo(
                        f"  SKIP (partial crash files exist; "
                        "use --retry-failed to retry)"
                    )
                    continue
                elif partial_files and retry_failed:
                    for pf in partial_files:
                        pf.unlink()
                    typer.echo(
                        f"  Cleaned {len(partial_files)} partial files, retrying"
                    )

                try:
                    r = run_single(
                        model=m,
                        codebase=cb,
                        mode=md,
                        track=track,
                        budget=budget,
                        probe_interval=probe_interval,
                        output=output,
                    )
                    results.append(r)
                    typer.echo(f"  Dep F1: {r.map_accuracy.dependency_f1:.3f}")
                except Exception as e:
                    failed.append(run_label)
                    typer.echo(f"  FAILED: {e}")

    # ── Summary ──────────────────────────────────────────────
    completed = len(results) - len(skipped)
    typer.echo("\n" + "=" * 60)
    typer.echo("BATCH SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"  Total:     {n}")
    typer.echo(f"  Completed: {completed}")
    typer.echo(f"  Skipped:   {len(skipped)}")
    typer.echo(f"  Failed:    {len(failed)}")
    if failed:
        typer.echo("  Failed runs:")
        for name in failed:
            typer.echo(f"    - {name}")
    typer.echo(f"  Output:    {output}")
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
