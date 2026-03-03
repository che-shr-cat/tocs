"""Active-Passive Gap (APG) analysis.

Loads multiple EvalResult JSONs, computes APG decomposition,
and outputs summary tables in markdown and CSV formats.

APG decomposition:
  APG_total     = passive_full  - active  (how much partial observability costs)
  APG_selection = passive_oracle - active  (cost of choosing which files to open)
  APG_decision  = passive_replay - active  (cost of deciding what to do with info)
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from models import EvalResult


# ── Data structures ────────────────────────────────────────────────


@dataclass
class APGEntry:
    """APG decomposition for one model on one codebase."""

    model_name: str
    codebase_id: str
    metric: str
    active_value: float
    passive_full_value: float | None = None
    passive_oracle_value: float | None = None
    passive_replay_value: float | None = None

    @property
    def apg_total(self) -> float | None:
        if self.passive_full_value is None:
            return None
        return self.passive_full_value - self.active_value

    @property
    def apg_selection(self) -> float | None:
        if self.passive_oracle_value is None:
            return None
        return self.passive_oracle_value - self.active_value

    @property
    def apg_decision(self) -> float | None:
        if self.passive_replay_value is None:
            return None
        return self.passive_replay_value - self.active_value


@dataclass
class GapAnalysisResult:
    """Complete gap analysis across all models and codebases."""

    entries: list[APGEntry] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render as a markdown table."""
        if not self.entries:
            return "No data."

        lines = [
            "| Model | Codebase | Metric | Active | P-Full | P-Oracle | P-Replay | APG Total | APG Selection | APG Decision |",
            "|-------|----------|--------|--------|--------|----------|----------|-----------|---------------|--------------|",
        ]
        for e in self.entries:
            lines.append(
                f"| {e.model_name} | {e.codebase_id} | {e.metric} "
                f"| {e.active_value:.3f} "
                f"| {_fmt(e.passive_full_value)} "
                f"| {_fmt(e.passive_oracle_value)} "
                f"| {_fmt(e.passive_replay_value)} "
                f"| {_fmt(e.apg_total)} "
                f"| {_fmt(e.apg_selection)} "
                f"| {_fmt(e.apg_decision)} |"
            )
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Render as CSV."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "model", "codebase", "metric",
            "active", "passive_full", "passive_oracle", "passive_replay",
            "apg_total", "apg_selection", "apg_decision",
        ])
        for e in self.entries:
            writer.writerow([
                e.model_name, e.codebase_id, e.metric,
                f"{e.active_value:.4f}",
                _fmt(e.passive_full_value),
                _fmt(e.passive_oracle_value),
                _fmt(e.passive_replay_value),
                _fmt(e.apg_total),
                _fmt(e.apg_selection),
                _fmt(e.apg_decision),
            ])
        return buf.getvalue()


# ── Loading ────────────────────────────────────────────────────────


def load_results(results_dir: Path) -> list[EvalResult]:
    """Load all EvalResult JSONs from a directory."""
    results: list[EvalResult] = []
    for f in sorted(results_dir.glob("*.json")):
        if "action_log" in f.name:
            continue
        try:
            result = EvalResult.model_validate_json(f.read_text())
            results.append(result)
        except Exception:
            continue
    return results


# ── Analysis ───────────────────────────────────────────────────────


def compute_gap(results: list[EvalResult]) -> GapAnalysisResult:
    """Compute APG decomposition from a list of EvalResults.

    Groups results by (model_name, codebase_id) and computes APG
    for each metric: dependency_f1, invariant_f1, action_auc.
    """
    # Group by (model, codebase)
    groups: dict[tuple[str, str], dict[str, EvalResult]] = {}
    for r in results:
        key = (r.model_name, r.codebase_id)
        condition = _result_condition(r)
        if key not in groups:
            groups[key] = {}
        groups[key][condition] = r

    entries: list[APGEntry] = []
    metrics_to_extract = [
        ("dependency_f1", lambda r: r.map_accuracy.dependency_f1),
        ("invariant_f1", lambda r: r.map_accuracy.invariant_f1),
        ("action_auc", lambda r: r.exploration.action_auc),
    ]

    for (model, codebase), conditions in sorted(groups.items()):
        active = conditions.get("active")
        if active is None:
            continue

        for metric_name, extractor in metrics_to_extract:
            entry = APGEntry(
                model_name=model,
                codebase_id=codebase,
                metric=metric_name,
                active_value=extractor(active),
            )

            pf = conditions.get("passive-full")
            if pf:
                entry.passive_full_value = extractor(pf)

            po = conditions.get("passive-oracle")
            if po:
                entry.passive_oracle_value = extractor(po)

            pr = conditions.get("passive-replay")
            if pr:
                entry.passive_replay_value = extractor(pr)

            entries.append(entry)

    return GapAnalysisResult(entries=entries)


def compute_gap_from_dir(results_dir: Path) -> GapAnalysisResult:
    """Load results from directory and compute gap analysis."""
    results = load_results(results_dir)
    return compute_gap(results)


# ── Helpers ────────────────────────────────────────────────────────


def _result_condition(r: EvalResult) -> str:
    """Get the condition string for an EvalResult."""
    if r.passive_condition:
        return f"{r.mode}-{r.passive_condition}"
    return r.mode


def _fmt(value: float | None) -> str:
    """Format a float or None as a string."""
    if value is None:
        return "-"
    return f"{value:.3f}"
