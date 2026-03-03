"""Figure generation for ToCS paper.

Generates 3 figure types:
1. F1 vs Steps line plot: all methods overlaid, per-codebase + average
2. Method comparison bar chart: final F1, Precision, Recall averaged across codebases
3. Edge type discovery heatmap: which methods discover which edge types

CLI: python -m analysis.figures --results ./results/paper/ --results ./results/gemini_full/ --output ./paper/figures/
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from metrics.gap_analysis import load_results
from models import EvalResult


# ── Color palette & display config ────────────────────────────────

MODEL_COLORS = {
    "oracle": "#2ca02c",
    "config-aware": "#1f77b4",
    "random": "#ff7f0e",
    "bfs-import": "#d62728",
    "anthropic/claude-sonnet-4-6": "#e377c2",
    "openai/gpt-5.3-codex": "#17becf",
    "gemini/gemini-2.5-flash": "#9467bd",
    "gemini/gemini-3.1-pro-preview": "#bcbd22",
    "gemini/gemini-3-flash-preview": "#8c564b",
}

MODEL_ORDER = [
    "oracle",
    "config-aware",
    "random",
    "bfs-import",
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.3-codex",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3-flash-preview",
]

MODEL_LABELS = {
    "oracle": "Oracle",
    "config-aware": "Config-Aware",
    "random": "Random",
    "bfs-import": "BFS-Import",
    "anthropic/claude-sonnet-4-6": "Claude Sonnet 4.6",
    "openai/gpt-5.3-codex": "GPT-5.3-Codex",
    "gemini/gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini/gemini-3-flash-preview": "Gemini 3 Flash",
}

# Models that are LLM agents (vs rule-based baselines)
LLM_MODELS = {
    "anthropic/claude-sonnet-4-6",
    "openai/gpt-5.3-codex",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3-flash-preview",
}


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#999999")


# ── 1. F1 vs Steps line plot ─────────────────────────────────────


def plot_f1_vs_steps(
    results: list[EvalResult],
    output: Path,
) -> Path:
    """F1 vs exploration steps for each baseline.

    Uses the action_efficiency_curve stored in each result (real per-step
    F1 values). Thin lines for individual codebases, bold for average.
    Oracle is shown as a horizontal dashed line at 1.0.
    """
    active_results = [r for r in results if r.mode == "active"]
    if not active_results:
        return _empty_figure(output / "f1_vs_steps.png", "No active results")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by model
    by_model: dict[str, list[EvalResult]] = defaultdict(list)
    for r in active_results:
        by_model[r.model_name].append(r)

    for model_name in MODEL_ORDER:
        if model_name not in by_model:
            continue
        runs = by_model[model_name]
        color = _color(model_name)
        is_llm = model_name in LLM_MODELS

        if model_name == "oracle":
            ax.axhline(y=1.0, color=color, linestyle="--", linewidth=2,
                        label=f"{_label(model_name)} (F1 = 1.0)", alpha=0.8)
            continue

        curves = []
        for r in runs:
            curve = r.exploration.action_efficiency_curve
            if not curve:
                continue
            steps = list(range(len(curve)))
            ax.plot(steps, curve, color=color, alpha=0.25, linewidth=1)
            curves.append(curve)

        # Bold average (dashed for LLMs, solid for baselines)
        if curves:
            max_len = max(len(c) for c in curves)
            padded = []
            for c in curves:
                padded.append(c + [c[-1]] * (max_len - len(c)))
            avg = np.mean(padded, axis=0)
            steps = list(range(len(avg)))
            linestyle = "--" if is_llm else "-"
            ax.plot(steps, avg, color=color, linewidth=2.5,
                    linestyle=linestyle,
                    label=f"{_label(model_name)} (F1 = {avg[-1]:.2f})")

    ax.set_xlabel("Exploration Step (action count)", fontsize=11)
    ax.set_ylabel("Dependency F1", fontsize=11)
    ax.set_title("Architectural Discovery vs. Exploration Budget", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output / "f1_vs_steps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 2. Baseline comparison bar chart ─────────────────────────────


def plot_baseline_comparison(
    results: list[EvalResult],
    output: Path,
) -> Path:
    """Bar chart of final Dep F1 and Action AUC per baseline.

    Averaged across codebases with std error bars.
    """
    active_results = [r for r in results if r.mode == "active"]
    if not active_results:
        return _empty_figure(output / "baseline_comparison.png", "No active results")

    by_model: dict[str, list[EvalResult]] = defaultdict(list)
    for r in active_results:
        by_model[r.model_name].append(r)

    models = [m for m in MODEL_ORDER if m in by_model]
    f1_means, f1_stds = [], []
    auc_means, auc_stds = [], []
    prec_means, rec_means = [], []

    for m in models:
        runs = by_model[m]
        f1s = [r.map_accuracy.dependency_f1 for r in runs]
        aucs = [r.exploration.action_auc for r in runs]
        precs = [r.map_accuracy.dependency_precision for r in runs]
        recs = [r.map_accuracy.dependency_recall for r in runs]
        f1_means.append(np.mean(f1s))
        f1_stds.append(np.std(f1s))
        auc_means.append(np.mean(aucs))
        auc_stds.append(np.std(aucs))
        prec_means.append(np.mean(precs))
        rec_means.append(np.mean(recs))

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_f1 = ax.bar(x - width, f1_means, width, yerr=f1_stds, label="Dependency F1",
           color="#1f77b4", capsize=4)
    bars_prec = ax.bar(x, prec_means, width, label="Precision",
           color="#2ca02c", capsize=4)
    bars_rec = ax.bar(x + width, rec_means, width, label="Recall",
           color="#ff7f0e", capsize=4)

    # Visual separator between baselines and LLMs
    baseline_count = sum(1 for m in models if m not in LLM_MODELS)
    llm_count = sum(1 for m in models if m in LLM_MODELS)
    if baseline_count > 0 and llm_count > 0:
        sep_x = baseline_count - 0.5
        ax.axvline(x=sep_x, color="#cccccc", linestyle=":", linewidth=1.5)
        ax.text(sep_x - 0.15, 1.08, "Baselines", ha="right", fontsize=8, color="#888")
        ax.text(sep_x + 0.15, 1.08, "LLM Agents", ha="left", fontsize=8, color="#888")

    ax.set_xlabel("Exploration Method", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Method Performance on Medium Codebases (n=3)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(m) for m in models], fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    # Add AUC annotation
    for i, (m, auc) in enumerate(zip(models, auc_means)):
        if auc > 0:
            ax.annotate(f"AUC={auc:.3f}", (x[i], f1_means[i] + f1_stds[i] + 0.03),
                        ha="center", fontsize=7, color="#555")

    fig.tight_layout()
    path = output / "baseline_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 3. Edge type discovery heatmap ───────────────────────────────


def plot_edge_type_discovery(
    results: list[EvalResult],
    output: Path,
) -> Path:
    """Heatmap: which baselines discover which edge types.

    Rows = baselines, columns = edge types.
    Values = average number of edges of that type discovered (across codebases).
    Also shows the ground truth total for reference.
    """
    active_results = [r for r in results if r.mode == "active"]
    if not active_results:
        return _empty_figure(output / "edge_type_discovery.png", "No active results")

    edge_types = ["IMPORTS", "CALLS_API", "DATA_FLOWS_TO", "REGISTRY_WIRES"]
    # All methods except oracle
    methods = [m for m in MODEL_ORDER if m != "oracle" and m in
               {r.model_name for r in active_results}]

    by_model: dict[str, list[EvalResult]] = defaultdict(list)
    for r in active_results:
        by_model[r.model_name].append(r)

    # Compute ground truth totals from oracle results
    gt_totals: dict[str, list[int]] = defaultdict(list)
    for r in active_results:
        if r.model_name == "oracle" and r.cognitive_maps:
            final = r.cognitive_maps[-1]
            counts: Counter[str] = Counter()
            for comp in final.components.values():
                for edge in comp.edges:
                    counts[edge.type] += 1
            for et in edge_types:
                gt_totals[et].append(counts.get(et, 0))

    # Build data matrix: rows=methods, cols=edge_types
    data = np.zeros((len(methods), len(edge_types)))

    for i, bl in enumerate(methods):
        runs = by_model.get(bl, [])
        for j, et in enumerate(edge_types):
            counts_per_run = []
            for r in runs:
                if not r.cognitive_maps:
                    counts_per_run.append(0)
                    continue
                final = r.cognitive_maps[-1]
                c = sum(1 for comp in final.components.values()
                        for edge in comp.edges if edge.type == et)
                counts_per_run.append(c)
            data[i, j] = np.mean(counts_per_run) if counts_per_run else 0

    # Ground truth row
    gt_row = [np.mean(gt_totals[et]) if gt_totals[et] else 0 for et in edge_types]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Normalize to ground truth for display (fraction discovered)
    display = np.zeros_like(data)
    for j in range(len(edge_types)):
        if gt_row[j] > 0:
            display[:, j] = data[:, j] / gt_row[j]

    # Build annotation labels: show "X/Y" (discovered/total)
    annot_labels = []
    for i in range(len(methods)):
        row = []
        for j in range(len(edge_types)):
            discovered = int(round(data[i, j]))
            total = int(round(gt_row[j]))
            row.append(f"{discovered}/{total}")
        annot_labels.append(row)
    annot_array = np.array(annot_labels)

    # Clip display for colormap (over-generation still visible via annotation)
    display_clipped = np.clip(display, 0, 1)

    import seaborn as sns

    # Add horizontal separator between baselines and LLMs
    method_labels = []
    for m in methods:
        prefix = "\u2605 " if m in LLM_MODELS else ""  # star for LLMs
        method_labels.append(f"{prefix}{_label(m)}")

    sns.heatmap(
        display_clipped,
        annot=annot_array,
        fmt="s",
        xticklabels=[f"{et}\n(GT={int(gt_row[j])})" for j, et in enumerate(edge_types)],
        yticklabels=method_labels,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Fraction of GT edges discovered"},
    )

    # Draw separator line between baselines and LLMs
    baseline_count = sum(1 for m in methods if m not in LLM_MODELS)
    if 0 < baseline_count < len(methods):
        ax.axhline(y=baseline_count, color="white", linewidth=3)

    ax.set_title("Edge Type Discovery by Exploration Strategy", fontsize=13, pad=10)
    ax.set_ylabel("")

    fig.tight_layout()
    path = output / "edge_type_discovery.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Generate all figures ──────────────────────────────────────────


def generate_all(
    results: list[EvalResult],
    output: Path,
) -> list[Path]:
    """Generate all figures and return their paths."""
    output.mkdir(parents=True, exist_ok=True)
    return [
        plot_f1_vs_steps(results, output),
        plot_baseline_comparison(results, output),
        plot_edge_type_discovery(results, output),
    ]


# ── Helpers ────────────────────────────────────────────────────────


def _empty_figure(path: Path, message: str) -> Path:
    """Create a placeholder figure with a message."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ── CLI ────────────────────────────────────────────────────────────


def load_results_multi(dirs: list[Path]) -> list[EvalResult]:
    """Load results from multiple directories and deduplicate."""
    all_results: list[EvalResult] = []
    seen: set[tuple[str, str, str]] = set()
    for d in dirs:
        for r in load_results(d):
            key = (r.model_name, r.codebase_id, r.mode)
            if key not in seen:
                seen.add(key)
                all_results.append(r)
    return all_results


def filter_common_codebases(results: list[EvalResult]) -> list[EvalResult]:
    """Keep only codebases that appear in baseline results (for fair comparison).

    If baselines are present, filter LLM results to only include matching codebases.
    """
    baseline_models = {"oracle", "config-aware", "random", "bfs-import"}
    baseline_codebases: set[str] | None = None

    for r in results:
        if r.model_name in baseline_models:
            if baseline_codebases is None:
                baseline_codebases = set()
            baseline_codebases.add(r.codebase_id)

    if baseline_codebases is None:
        return results  # No baselines — keep all

    return [r for r in results
            if r.model_name in baseline_models or r.codebase_id in baseline_codebases]


def main() -> None:
    import typer
    from typing import Optional

    app = typer.Typer(help="Generate ToCS paper figures")

    @app.command()
    def run(
        results: list[Path] = typer.Option(
            [Path("./results/paper/")],
            help="Directory with EvalResult JSONs (can specify multiple)",
        ),
        output: Path = typer.Option(
            Path("./paper/figures/"), help="Output directory for figures"
        ),
    ) -> None:
        """Generate all figures from evaluation results."""
        data = load_results_multi(results)
        if not data:
            typer.echo(f"No results found in {results}")
            raise typer.Exit(1)

        # Filter to common codebases for fair comparison
        data = filter_common_codebases(data)

        models_found = sorted(set(r.model_name for r in data))
        codebases_found = sorted(set(r.codebase_id for r in data))
        typer.echo(f"Loaded {len(data)} results: {len(models_found)} methods × {len(codebases_found)} codebases")
        for m in models_found:
            count = sum(1 for r in data if r.model_name == m)
            typer.echo(f"  {m}: {count} results")

        paths = generate_all(data, output)
        for p in paths:
            typer.echo(f"Generated: {p}")
        typer.echo(f"\n{len(paths)} figures saved to {output}")

    app()


if __name__ == "__main__":
    main()
