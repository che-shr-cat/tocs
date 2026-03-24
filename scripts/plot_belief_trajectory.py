#!/usr/bin/env python
"""Generate belief trajectory figure: Dependency F1 vs probe step for all 6 LLM agents.

Shows how cognitive map quality evolves across exploration,
highlighting contrasting stability patterns (stable accumulation, monotonic,
catastrophic collapse, recency bias).

Usage:
    .venv/bin/python scripts/plot_belief_trajectory.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import CognitiveMap, CodebaseGroundTruth
from metrics.map_accuracy import edge_prf


# ── Configuration ──────────────────────────────────────────────────

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Codebase ID -> data directory mapping
CODEBASE_MAP = {
    "gen-pipeline-medium-f3784e0c": "codebase_42",
    "gen-pipeline-medium-c2777a58": "codebase_123",
    "gen-pipeline-medium-fd3fe909": "codebase_999",
}

CODEBASES = list(CODEBASE_MAP.keys())

# Models to plot (result file prefix -> display config)
MODELS = {
    "openai_gpt-5.3-codex": {
        "label": "GPT-5.3-Codex",
        "color": "#17becf",
        "marker": "s",
    },
    "anthropic_claude-sonnet-4-6": {
        "label": "Claude Sonnet 4.6",
        "color": "#d62728",
        "marker": "o",
    },
    "gemini_gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "color": "#9467bd",
        "marker": "^",
    },
    "gemini_gemini-3.1-pro-preview": {
        "label": "Gemini 3.1 Pro",
        "color": "#2ca02c",
        "marker": "P",
    },
    "gemini_gemini-2.5-pro": {
        "label": "Gemini 2.5 Pro",
        "color": "#e377c2",
        "marker": "D",
    },
    "gemini_gemini-3-flash-preview": {
        "label": "Gemini 3 Flash",
        "color": "#8c564b",
        "marker": "v",
    },
}

OUTPUT_PATH = PROJECT_ROOT / "paper" / "figures" / "belief_trajectory.png"

PROBE_STEPS = [3, 6, 9, 12, 15, 18, 20]


# ── Data loading ───────────────────────────────────────────────────


def load_ground_truth(codebase_id: str) -> CodebaseGroundTruth:
    """Load ground truth for a codebase."""
    data_dir = DATA_DIR / CODEBASE_MAP[codebase_id]
    gt_path = data_dir / "ground_truth.json"
    with open(gt_path) as f:
        return CodebaseGroundTruth(**json.load(f))


def load_result(model_prefix: str, codebase_id: str) -> dict | None:
    """Load a result JSON file."""
    filename = f"{model_prefix}_{codebase_id}_active.json"
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  WARNING: Missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


# ── Scoring ────────────────────────────────────────────────────────


def score_trajectory(
    result: dict, gt: CodebaseGroundTruth
) -> dict[int, float]:
    """Score each cognitive map in a result against ground truth.

    Returns {step: dependency_f1}.
    """
    step_f1: dict[int, float] = {}
    for map_dict in result.get("cognitive_maps", []):
        cmap = CognitiveMap(**map_dict)
        _, _, f1 = edge_prf(cmap, gt)
        step_f1[cmap.step] = f1
    return step_f1


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    # Pre-load ground truths
    ground_truths: dict[str, CodebaseGroundTruth] = {}
    for cid in CODEBASES:
        ground_truths[cid] = load_ground_truth(cid)
        print(f"Loaded GT for {cid}: {len(ground_truths[cid].dependency_edges)} edges")

    # Score all model x codebase combinations
    # model_prefix -> {step -> [f1 values across codebases]}
    all_trajectories: dict[str, dict[int, list[float]]] = {}

    for model_prefix, config in MODELS.items():
        print(f"\nScoring {config['label']}...")
        step_values: dict[int, list[float]] = defaultdict(list)

        for cid in CODEBASES:
            result = load_result(model_prefix, cid)
            if result is None:
                continue
            gt = ground_truths[cid]
            step_f1 = score_trajectory(result, gt)
            print(f"  {cid}: steps={sorted(step_f1.keys())}, "
                  f"F1s={[f'{step_f1[s]:.3f}' for s in sorted(step_f1.keys())]}")
            for step, f1 in step_f1.items():
                step_values[step].append(f1)

        all_trajectories[model_prefix] = step_values

    # ── Plot ───────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model_prefix, config in MODELS.items():
        step_values = all_trajectories[model_prefix]
        if not step_values:
            continue

        # Sort steps and compute mean F1 at each
        steps_sorted = sorted(step_values.keys())
        mean_f1 = [np.mean(step_values[s]) for s in steps_sorted]
        std_f1 = [np.std(step_values[s]) for s in steps_sorted]

        # Plot mean line with markers
        ax.plot(
            steps_sorted,
            mean_f1,
            color=config["color"],
            marker=config["marker"],
            markersize=6,
            linewidth=2,
            label=config["label"],
            zorder=3,
        )

        # Shaded std band
        mean_arr = np.array(mean_f1)
        std_arr = np.array(std_f1)
        ax.fill_between(
            steps_sorted,
            np.clip(mean_arr - std_arr, 0, 1),
            np.clip(mean_arr + std_arr, 0, 1),
            color=config["color"],
            alpha=0.12,
            zorder=1,
        )

    ax.set_xlabel("Exploration step", fontsize=12)
    ax.set_ylabel("Dependency F1", fontsize=12)
    ax.set_xticks(PROBE_STEPS)
    ax.set_xlim(PROBE_STEPS[0] - 0.5, PROBE_STEPS[-1] + 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
