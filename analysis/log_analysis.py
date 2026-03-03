#!/usr/bin/env python3
"""Generate a comprehensive HTML report analyzing all LLM evaluation logs.

Reads result files from ./results/, ground truth from ./data/,
and produces a self-contained HTML report at ./analysis/eval_log_analysis.html.

Only includes LLM model runs on the 3 medium codebases:
  gen-pipeline-medium-c2777a58, gen-pipeline-medium-f3784e0c, gen-pipeline-medium-fd3fe909
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = PROJECT_ROOT / "analysis" / "eval_log_analysis.html"

TARGET_CODEBASE_IDS = {
    "gen-pipeline-medium-c2777a58",
    "gen-pipeline-medium-f3784e0c",
    "gen-pipeline-medium-fd3fe909",
}

CODEBASE_ID_TO_DIR = {
    "gen-pipeline-medium-f3784e0c": "codebase_42",
    "gen-pipeline-medium-c2777a58": "codebase_123",
    "gen-pipeline-medium-fd3fe909": "codebase_999",
}

BASELINE_PREFIXES = ("bfs-import", "config-aware", "random", "oracle")

EDGE_TYPE_COLORS = {
    "IMPORTS": "#3b82f6",       # blue
    "CALLS_API": "#22c55e",     # green
    "DATA_FLOWS_TO": "#f97316", # orange
    "REGISTRY_WIRES": "#a855f7",# purple
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    """All data for a single evaluation run."""
    model_name: str
    codebase_id: str
    codebase_dir: str
    is_partial: bool
    result: dict | None          # main result JSON
    action_log: list[dict]       # action log
    conversation: list[dict]     # conversation log
    ground_truth: dict           # ground truth
    gt_edges: list[dict]         # dependency edges from GT
    gt_invariants: list[dict]    # invariants from GT
    gt_modules: list[str]        # module file paths from GT
    error_info: str = ""         # error message for PARTIAL runs

    @property
    def display_name(self) -> str:
        partial = " (PARTIAL)" if self.is_partial else ""
        return f"{self.model_name}{partial}"

    @property
    def short_codebase(self) -> str:
        return self.codebase_id.split("-")[-1][:8]


def _find_runs() -> list[RunData]:
    """Discover all LLM runs on target codebases.

    Handles two cases:
    1. Complete runs with a main result JSON (*.json without _action_log/_conversation)
    2. PARTIAL runs that only have _action_log.json and/or _conversation.json
    """
    runs = []
    seen: set[str] = set()  # keyed by base name (without suffixes)

    # First pass: collect all potential run base names
    run_bases: dict[str, dict[str, Path]] = {}  # base -> {result, action_log, conversation}

    for f in sorted(RESULTS_DIR.iterdir()):
        if not f.name.endswith(".json"):
            continue
        # Skip baselines
        if any(f.name.startswith(p) for p in BASELINE_PREFIXES):
            continue
        # Must be on a target codebase
        matched_id = None
        for cid in TARGET_CODEBASE_IDS:
            if cid in f.name:
                matched_id = cid
                break
        if not matched_id:
            continue

        name = f.name
        if name.endswith("_action_log.json"):
            base = name.replace("_action_log.json", "")
            run_bases.setdefault(base, {})["action_log"] = f
        elif name.endswith("_conversation.json"):
            base = name.replace("_conversation.json", "")
            run_bases.setdefault(base, {})["conversation"] = f
        else:
            base = name.replace(".json", "")
            run_bases.setdefault(base, {})["result"] = f

    # Second pass: build RunData for each unique base
    for base, files in sorted(run_bases.items()):
        if base in seen:
            continue
        seen.add(base)

        # Determine codebase ID
        matched_id = None
        for cid in TARGET_CODEBASE_IDS:
            if cid in base:
                matched_id = cid
                break
        if not matched_id:
            continue

        is_partial = "_PARTIAL" in base

        # Load result JSON (may not exist for PARTIAL runs)
        result = None
        if "result" in files:
            try:
                result = json.loads(files["result"].read_text(encoding="utf-8"))
            except Exception:
                pass

        # Load action log
        action_log = []
        if "action_log" in files:
            try:
                action_log = json.loads(files["action_log"].read_text(encoding="utf-8"))
            except Exception:
                pass

        # Load conversation
        conversation = []
        if "conversation" in files:
            try:
                conversation = json.loads(files["conversation"].read_text(encoding="utf-8"))
            except Exception:
                pass

        # Skip if we have absolutely nothing useful
        if result is None and not action_log and not conversation:
            continue

        # Load ground truth
        cb_dir = CODEBASE_ID_TO_DIR.get(matched_id, "")
        gt_path = DATA_DIR / cb_dir / "ground_truth.json"
        if not gt_path.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))

        # Load error info for PARTIAL runs
        error_info = ""
        if is_partial:
            error_path = RESULTS_DIR / f"{base}_error.txt"
            if error_path.exists():
                try:
                    error_info = error_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        # Extract model name from base
        model_name = base.split(f"_{matched_id}")[0]

        runs.append(RunData(
            model_name=model_name,
            codebase_id=matched_id,
            codebase_dir=cb_dir,
            is_partial=is_partial,
            result=result,
            action_log=action_log,
            conversation=conversation,
            ground_truth=gt,
            gt_edges=gt.get("dependency_edges", []),
            gt_invariants=gt.get("invariants", []),
            gt_modules=list(gt.get("modules", {}).keys()),
            error_info=error_info,
        ))

    return runs


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _gt_edge_set(gt_edges: list[dict]) -> set[tuple[str, str, str]]:
    return {(e["source"], e["target"], e["type"]) for e in gt_edges}


def _pred_edge_set_from_map(cmap: dict) -> set[tuple[str, str, str]]:
    edges = set()
    components = cmap.get("components", {})
    for filepath, comp in components.items():
        for e in comp.get("edges", []):
            etype = e.get("type", "")
            if isinstance(etype, dict):
                etype = etype.get("value", str(etype))
            edges.add((filepath, e.get("target", ""), etype))
    return edges


def _prf(pred: set, gold: set) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _edge_type_breakdown(edges: set[tuple[str, str, str]]) -> dict[str, int]:
    counts: dict[str, int] = Counter()
    for _, _, t in edges:
        counts[t] += 1
    return dict(counts)


def _opened_files_at_step(action_log: list[dict], step: int) -> set[str]:
    """Return set of files opened by (and including) the given step."""
    opened = set()
    for a in action_log:
        if a["step"] > step:
            break
        if a["action"]["type"].lower() == "open":
            opened.add(a["action"]["argument"])
    return opened


def _discoverable_edges_at_step(
    opened: set[str], gt_edges: list[dict]
) -> set[tuple[str, str, str]]:
    """Edges that are discoverable given the set of opened files.

    An edge is discoverable if both source and target have been opened.
    """
    disc = set()
    for e in gt_edges:
        if e["source"] in opened and e["target"] in opened:
            disc.add((e["source"], e["target"], e["type"]))
    return disc


def _edge_connectivity(gt_edges: list[dict]) -> dict[str, int]:
    """Count how many GT edges each file participates in (as source or target)."""
    counts: dict[str, int] = Counter()
    for e in gt_edges:
        counts[e["source"]] += 1
        counts[e["target"]] += 1
    return dict(counts)


def _strip_path(p: str) -> str:
    if not p:
        return ""
    parts = p.replace("\\", "/").split("/")
    if len(parts) > 1:
        return "/".join(parts[1:])
    return p


def _normalise_relaxed(s: dict | None) -> tuple:
    if s is None:
        return ()
    stype = s.get("type", "")
    src = _strip_path(s.get("src") or "")
    dst = _strip_path(s.get("dst") or "")
    via = _strip_path(s.get("via") or "")
    if stype == "INTERFACE_ONLY":
        if not dst and via:
            dst = via
    return (stype, src, dst)


def _relaxed_match(pred: tuple, gt: tuple) -> bool:
    if len(pred) != 3 or len(gt) != 3:
        return False
    p_type, p_src, p_dst = pred
    g_type, g_src, g_dst = gt
    if p_type != g_type:
        return False
    if g_src and g_src != p_src:
        return False
    if g_dst and g_dst != p_dst:
        return False
    return True


def _match_invariants_relaxed(
    pred_invs: list[dict], gt_invs: list[dict]
) -> list[tuple[dict, dict | None, bool]]:
    """Match predicted invariants against GT using relaxed matching.

    Returns list of (pred_inv, matched_gt_inv_or_None, is_matched).
    """
    gt_tuples = []
    for inv in gt_invs:
        gt_tuples.append(_normalise_relaxed(inv.get("structured")))

    matched_gt: set[int] = set()
    results = []
    for pred in pred_invs:
        structured = pred.get("structured")
        if structured is None:
            results.append((pred, None, False))
            continue
        pt = _normalise_relaxed(structured)
        found = False
        for gi, gt_t in enumerate(gt_tuples):
            if gi not in matched_gt and _relaxed_match(pt, gt_t):
                matched_gt.add(gi)
                results.append((pred, gt_invs[gi], True))
                found = True
                break
        if not found:
            results.append((pred, None, False))
    return results


def _classify_false_positive(
    edge: tuple[str, str, str],
    gt_edges_set: set[tuple[str, str, str]],
    gt_modules: set[str],
) -> str:
    """Classify a false positive edge."""
    src, tgt, etype = edge
    # Check edge type confusion: same src-tgt but different type
    for gs, gt_, gtype in gt_edges_set:
        if gs == src and gt_ == tgt and gtype != etype:
            return "Edge type confusion"
    # Check granularity mismatch: directory-level vs file-level
    if "/" in src and "/" in tgt:
        src_dir = "/".join(src.split("/")[:-1])
        tgt_dir = "/".join(tgt.split("/")[:-1])
        for gs, gt_, gtype in gt_edges_set:
            gs_dir = "/".join(gs.split("/")[:-1])
            gt_dir = "/".join(gt_.split("/")[:-1])
            if gs_dir == src_dir and gt_dir == tgt_dir and gtype == etype and (gs, gt_, gtype) != edge:
                return "Granularity mismatch"
    # Check if target exists in GT at all
    if tgt not in gt_modules and src not in gt_modules:
        return "True hallucination"
    # Otherwise: reasonable inference
    return "Reasonable inference"


def _classify_false_negative(
    edge: tuple[str, str, str],
    opened_files: set[str],
) -> str:
    """Classify a false negative edge."""
    src, tgt, _ = edge
    src_opened = src in opened_files
    tgt_opened = tgt in opened_files
    if src_opened and tgt_opened:
        return "Opened both but didn't report"
    elif src_opened:
        return "Opened source only"
    elif tgt_opened:
        return "Opened target only"
    else:
        return "Never opened source/target"


def _generate_commentary(run: RunData, fp_classes: dict, fn_classes: dict,
                          edge_breakdown_pred: dict, edge_breakdown_gt: dict) -> str:
    """Generate analysis commentary for a run."""
    lines = []

    # Exploration strategy
    if run.action_log:
        action_types = Counter(a["action"]["type"].lower() for a in run.action_log)
        total = sum(action_types.values())
        open_pct = action_types.get("open", 0) / total * 100 if total else 0
        list_pct = action_types.get("list", 0) / total * 100 if total else 0
        search_pct = action_types.get("search", 0) / total * 100 if total else 0
        inspect_pct = action_types.get("inspect", 0) / total * 100 if total else 0

        strategy_parts = []
        if open_pct > 60:
            strategy_parts.append("heavily file-opening focused")
        elif list_pct > 30:
            strategy_parts.append("directory-exploration heavy")
        elif search_pct > 20:
            strategy_parts.append("search-driven")

        # Check if model opened files in order of connectivity
        opened_files_list = [a["action"]["argument"] for a in run.action_log
                           if a["action"]["type"].lower() == "open"]
        connectivity = _edge_connectivity(run.gt_edges)
        if opened_files_list:
            early = opened_files_list[:len(opened_files_list)//2]
            late = opened_files_list[len(opened_files_list)//2:]
            avg_early = sum(connectivity.get(f, 0) for f in early) / max(len(early), 1)
            avg_late = sum(connectivity.get(f, 0) for f in late) / max(len(late), 1)
            if avg_early > avg_late + 1:
                strategy_parts.append("prioritizing high-connectivity files early (good)")
            elif avg_late > avg_early + 1:
                strategy_parts.append("opening peripheral files first (suboptimal)")

        lines.append(f"Exploration strategy: Action mix was {open_pct:.0f}% OPEN, "
                     f"{list_pct:.0f}% LIST, {search_pct:.0f}% SEARCH, {inspect_pct:.0f}% INSPECT. "
                     + ("; ".join(strategy_parts) + "." if strategy_parts else "Balanced approach."))

    # Belief externalization
    fn_both = fn_classes.get("Opened both but didn't report", 0)
    fn_total = sum(fn_classes.values())
    if fn_total > 0:
        extern_rate = fn_both / fn_total * 100
        if extern_rate > 30:
            lines.append(f"Belief externalization: Weak -- {fn_both}/{fn_total} "
                         f"({extern_rate:.0f}%) false negatives were edges where both files "
                         f"were opened but not reported. Model sees more than it reports.")
        elif extern_rate > 10:
            lines.append(f"Belief externalization: Moderate -- {fn_both}/{fn_total} "
                         f"({extern_rate:.0f}%) missed edges involved files that were both opened.")
        else:
            lines.append(f"Belief externalization: Strong -- only {fn_both}/{fn_total} "
                         f"missed edges had both endpoints opened.")

    # Edge type understanding
    type_accuracy = {}
    for etype in ["IMPORTS", "CALLS_API", "DATA_FLOWS_TO", "REGISTRY_WIRES"]:
        gt_count = edge_breakdown_gt.get(etype, 0)
        pred_count = edge_breakdown_pred.get(etype, 0)
        if gt_count > 0:
            ratio = pred_count / gt_count
            type_accuracy[etype] = ratio

    if type_accuracy:
        best = max(type_accuracy, key=type_accuracy.get)
        worst = min(type_accuracy, key=type_accuracy.get)
        lines.append(f"Edge type understanding: Best at {best} "
                     f"(pred/GT ratio={type_accuracy[best]:.2f}), "
                     f"worst at {worst} (ratio={type_accuracy[worst]:.2f}).")

    # Prompt improvement suggestions
    suggestions = []
    if fp_classes.get("Edge type confusion", 0) > 2:
        suggestions.append("clarify edge type definitions (especially CALLS_API vs IMPORTS)")
    if fp_classes.get("True hallucination", 0) > 2:
        suggestions.append("emphasize only reporting edges with evidence in observed code")
    if fn_both > 5:
        suggestions.append("improve belief externalization prompts to capture all observed relationships")
    type_confusion_count = fp_classes.get("Edge type confusion", 0)
    if type_confusion_count > 0:
        suggestions.append(f"reduce edge type confusion ({type_confusion_count} instances)")

    if suggestions:
        lines.append("Prompt improvement suggestions: " + "; ".join(suggestions) + ".")

    return " ".join(lines) if lines else "Insufficient data for detailed commentary."


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html.escape(str(s))


def _edge_badge(etype: str) -> str:
    color = EDGE_TYPE_COLORS.get(etype, "#6b7280")
    return f'<span class="edge-badge" style="background:{color}">{_esc(etype)}</span>'


def _progress_bar(value: float, max_val: float = 1.0, width: int = 200) -> str:
    pct = value / max_val * 100 if max_val else 0
    color = "#22c55e" if pct > 60 else "#f97316" if pct > 30 else "#ef4444"
    return (f'<div class="progress-bar" style="width:{width}px">'
            f'<div class="progress-fill" style="width:{pct:.1f}%;background:{color}"></div>'
            f'<span class="progress-text">{value:.3f}</span></div>')


def _metric_card(label: str, value: str, color: str = "#1e293b") -> str:
    return (f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{color}">{_esc(value)}</div>'
            f'<div class="metric-label">{_esc(label)}</div></div>')


CSS = """
:root {
    --bg-dark: #0f172a;
    --bg-header: #1e293b;
    --bg-card: #ffffff;
    --bg-alt: #f8fafc;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --accent: #3b82f6;
    --success: #22c55e;
    --warning: #f97316;
    --danger: #ef4444;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-alt);
    color: var(--text-primary);
    line-height: 1.6;
}

.header {
    background: var(--bg-dark);
    color: white;
    padding: 2rem 3rem;
    border-bottom: 3px solid var(--accent);
}
.header h1 { font-size: 1.8rem; font-weight: 700; }
.header p { color: #94a3b8; margin-top: 0.3rem; font-size: 0.95rem; }

.container { max-width: 1400px; margin: 0 auto; padding: 1.5rem 2rem; }

.summary-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin: 1rem 0;
    font-size: 0.9rem;
}
.summary-table th {
    background: var(--bg-header);
    color: white;
    padding: 0.7rem 1rem;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}
.summary-table td {
    padding: 0.6rem 1rem;
    border-bottom: 1px solid var(--border);
}
.summary-table tr:hover { background: #f1f5f9; }
.summary-table tr:nth-child(even) { background: var(--bg-alt); }
.summary-table tr:nth-child(even):hover { background: #f1f5f9; }

.section-card {
    background: var(--bg-card);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin: 1.5rem 0;
    overflow: hidden;
}
.section-header {
    background: var(--bg-header);
    color: white;
    padding: 1rem 1.5rem;
    font-size: 1.15rem;
    font-weight: 600;
}
.section-body { padding: 1.5rem; }

.metrics-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    min-width: 130px;
    text-align: center;
}
.metric-value { font-size: 1.3rem; font-weight: 700; }
.metric-label { font-size: 0.78rem; color: var(--text-secondary); margin-top: 0.2rem; }

.edge-badge {
    display: inline-block;
    color: white;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: monospace;
    margin: 0 0.1rem;
}

.progress-bar {
    display: inline-block;
    height: 20px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    vertical-align: middle;
}
.progress-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.progress-text {
    position: absolute;
    right: 6px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-primary);
}

details {
    border: 1px solid var(--border);
    border-radius: 6px;
    margin: 0.8rem 0;
    overflow: hidden;
}
details > summary {
    background: var(--bg-alt);
    padding: 0.7rem 1rem;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.95rem;
    user-select: none;
    border-bottom: 1px solid var(--border);
}
details > summary:hover { background: #f1f5f9; }
details[open] > summary { border-bottom: 1px solid var(--border); }
details > .detail-body { padding: 1rem; }

.timeline-step {
    display: flex;
    align-items: flex-start;
    padding: 0.5rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.88rem;
}
.timeline-step:last-child { border-bottom: none; }
.step-num {
    font-weight: 700;
    color: var(--accent);
    min-width: 40px;
    font-family: monospace;
}
.step-action {
    min-width: 100px;
    font-weight: 600;
    font-family: monospace;
}
.step-detail { flex: 1; color: var(--text-secondary); }
.step-recall { min-width: 220px; text-align: right; }
.step-flag {
    color: var(--warning);
    font-size: 0.8rem;
    font-style: italic;
    margin-left: 0.5rem;
}

.probe-card {
    background: var(--bg-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    margin: 0.8rem 0;
}
.probe-header {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
    color: var(--accent);
}

.edge-list { font-size: 0.85rem; }
.edge-list .correct { color: var(--success); }
.edge-list .error { color: var(--danger); }
.edge-list .new { font-weight: 600; }

.conv-msg {
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    border-radius: 4px;
    font-size: 0.85rem;
}
.conv-msg.user { background: #eff6ff; border-left: 3px solid var(--accent); }
.conv-msg.assistant { background: #f0fdf4; border-left: 3px solid var(--success); }
.conv-msg .role-label {
    font-weight: 700;
    font-size: 0.75rem;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.fp-table, .fn-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 0.5rem 0;
}
.fp-table th, .fn-table th {
    background: var(--bg-alt);
    padding: 0.4rem 0.8rem;
    text-align: left;
    border-bottom: 2px solid var(--border);
    font-weight: 600;
}
.fp-table td, .fn-table td {
    padding: 0.4rem 0.8rem;
    border-bottom: 1px solid var(--border);
    font-family: monospace;
    font-size: 0.8rem;
}

.commentary {
    background: #fefce8;
    border-left: 4px solid #eab308;
    padding: 1rem 1.2rem;
    border-radius: 0 6px 6px 0;
    margin: 1rem 0;
    font-size: 0.9rem;
    line-height: 1.7;
}

.tag-correct {
    background: #dcfce7;
    color: #166534;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.78rem;
    font-weight: 600;
}
.tag-error {
    background: #fef2f2;
    color: #991b1b;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.78rem;
    font-weight: 600;
}
.tag-new {
    background: #eff6ff;
    color: #1e40af;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.78rem;
    font-weight: 600;
}

code, .mono {
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.85em;
    background: #f1f5f9;
    padding: 0.1em 0.3em;
    border-radius: 3px;
}

.key-findings {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    border: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.key-findings h3 { margin-bottom: 0.7rem; color: var(--bg-header); }
.key-findings ul { padding-left: 1.5rem; }
.key-findings li { margin: 0.3rem 0; font-size: 0.92rem; }

.inv-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
    margin: 0.5rem 0;
}
.inv-table th {
    background: var(--bg-alt);
    padding: 0.4rem 0.6rem;
    text-align: left;
    border-bottom: 2px solid var(--border);
    font-weight: 600;
}
.inv-table td {
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

@media (max-width: 900px) {
    .container { padding: 1rem; }
    .metrics-row { flex-direction: column; }
    .timeline-step { flex-wrap: wrap; }
}
"""


def _build_summary_section(runs: list[RunData]) -> str:
    """Build the summary overview table and key findings."""
    parts = []
    parts.append('<div class="section-card"><div class="section-header">Summary: All LLM Runs</div>')
    parts.append('<div class="section-body">')

    # Overview table
    parts.append('<table class="summary-table"><thead><tr>')
    parts.append('<th>Model</th><th>Codebase</th><th>Status</th>'
                 '<th>Steps</th><th>Files Opened</th>'
                 '<th>Dep F1</th><th>Inv F1 (strict)</th><th>Inv F1 (relaxed)</th>'
                 '<th>Action AUC</th><th>ECE</th>')
    parts.append('</tr></thead><tbody>')

    model_scores: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        r = run.result
        ma = r.get("map_accuracy", {}) if r else {}
        ex = r.get("exploration", {}) if r else {}

        dep_f1 = ma.get("dependency_f1", 0)
        inv_f1 = ma.get("invariant_f1", 0)
        inv_f1r = ma.get("invariant_f1_relaxed", 0)
        auc = ex.get("action_auc", 0)
        ece = ma.get("confidence_ece", 0)
        steps = ex.get("steps_taken", len(run.action_log))
        files = ex.get("files_opened", 0)
        status = "PARTIAL" if run.is_partial else "Complete"
        status_color = "#f97316" if run.is_partial else "#22c55e"

        model_scores[run.model_name].append(dep_f1)

        parts.append(f'<tr>'
                     f'<td><strong>{_esc(run.model_name)}</strong></td>'
                     f'<td><code>{_esc(run.short_codebase)}</code></td>'
                     f'<td><span style="color:{status_color};font-weight:600">{status}</span></td>'
                     f'<td>{steps}</td>'
                     f'<td>{files}</td>'
                     f'<td>{_progress_bar(dep_f1)}</td>'
                     f'<td>{inv_f1:.3f}</td>'
                     f'<td>{inv_f1r:.3f}</td>'
                     f'<td>{auc:.3f}</td>'
                     f'<td>{ece:.3f}</td>'
                     f'</tr>')
    parts.append('</tbody></table>')

    # Key findings
    parts.append('<div class="key-findings"><h3>Key Findings</h3><ul>')

    # Best model by average Dep F1
    if model_scores:
        avg_scores = {m: sum(s)/len(s) for m, s in model_scores.items()}
        best = max(avg_scores, key=avg_scores.get)
        worst = min(avg_scores, key=avg_scores.get)
        parts.append(f'<li>Best average Dep F1: <strong>{_esc(best)}</strong> '
                     f'({avg_scores[best]:.3f}), worst: <strong>{_esc(worst)}</strong> '
                     f'({avg_scores[worst]:.3f})</li>')

    # Completed vs partial runs
    complete = sum(1 for r in runs if not r.is_partial)
    partial = sum(1 for r in runs if r.is_partial)
    parts.append(f'<li>{complete} complete runs, {partial} partial (error/timeout) runs</li>')

    # Edge type patterns across all models
    all_fp_types: dict[str, int] = Counter()
    all_fn_types: dict[str, int] = Counter()
    for run in runs:
        if run.result and run.result.get("cognitive_maps"):
            final_map = run.result["cognitive_maps"][-1]
            pred_edges = _pred_edge_set_from_map(final_map)
            gt_edges_set = _gt_edge_set(run.gt_edges)
            fp = pred_edges - gt_edges_set
            fn = gt_edges_set - pred_edges
            for _, _, t in fp:
                all_fp_types[t] += 1
            for _, _, t in fn:
                all_fn_types[t] += 1

    if all_fn_types:
        most_missed = max(all_fn_types, key=all_fn_types.get)
        parts.append(f'<li>Most commonly missed edge type: {_edge_badge(most_missed)} '
                     f'({all_fn_types[most_missed]} total false negatives across all runs)</li>')

    if all_fp_types:
        most_halluc = max(all_fp_types, key=all_fp_types.get)
        parts.append(f'<li>Most commonly hallucinated edge type: {_edge_badge(most_halluc)} '
                     f'({all_fp_types[most_halluc]} total false positives across all runs)</li>')

    parts.append('</ul></div>')

    # Cross-model comparison
    parts.append('<h3 style="margin-top:1.2rem">Cross-Model Comparison (Average over 3 codebases)</h3>')
    parts.append('<table class="summary-table"><thead><tr>')
    parts.append('<th>Model</th><th>Avg Dep F1</th><th>Avg Inv F1 (relaxed)</th>'
                 '<th>Avg AUC</th><th>Avg Files Opened</th><th>Runs</th></tr></thead><tbody>')

    model_data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        r = run.result or {}
        ma = r.get("map_accuracy", {})
        ex = r.get("exploration", {})
        m = run.model_name
        model_data[m]["dep_f1"].append(ma.get("dependency_f1", 0))
        model_data[m]["inv_f1r"].append(ma.get("invariant_f1_relaxed", 0))
        model_data[m]["auc"].append(ex.get("action_auc", 0))
        model_data[m]["files"].append(ex.get("files_opened", 0))

    for m, data in sorted(model_data.items()):
        n = len(data["dep_f1"])
        avg_f1 = sum(data["dep_f1"]) / n if n else 0
        avg_inv = sum(data["inv_f1r"]) / n if n else 0
        avg_auc = sum(data["auc"]) / n if n else 0
        avg_files = sum(data["files"]) / n if n else 0
        parts.append(f'<tr><td><strong>{_esc(m)}</strong></td>'
                     f'<td>{avg_f1:.3f}</td><td>{avg_inv:.3f}</td>'
                     f'<td>{avg_auc:.3f}</td><td>{avg_files:.1f}</td>'
                     f'<td>{n}</td></tr>')

    parts.append('</tbody></table>')
    parts.append('</div></div>')
    return "\n".join(parts)


def _build_run_section(run: RunData, run_idx: int) -> str:
    """Build the detailed section for a single run."""
    parts = []
    r = run.result or {}
    ma = r.get("map_accuracy", {})
    ex = r.get("exploration", {})
    cognitive_maps = r.get("cognitive_maps", [])

    dep_f1 = ma.get("dependency_f1", 0)
    dep_p = ma.get("dependency_precision", 0)
    dep_r = ma.get("dependency_recall", 0)
    inv_f1 = ma.get("invariant_f1", 0)
    inv_f1r = ma.get("invariant_f1_relaxed", 0)
    auc = ex.get("action_auc", 0)
    ece = ma.get("confidence_ece", 0)
    steps = ex.get("steps_taken", len(run.action_log))
    files = ex.get("files_opened", 0)
    unique_files = ex.get("unique_files", 0)

    status_label = "PARTIAL" if run.is_partial else "Complete"

    # Section wrapper
    parts.append(f'<details class="section-card" {"open" if run_idx == 0 else ""}>')
    parts.append(f'<summary class="section-header">Run {run_idx+1}: {_esc(run.display_name)} '
                 f'on {_esc(run.short_codebase)} '
                 f'[Dep F1={dep_f1:.3f}]</summary>')
    parts.append('<div class="section-body">')

    # ---- (a) Summary card ----
    parts.append('<h3>Summary</h3>')
    parts.append('<div class="metrics-row">')
    parts.append(_metric_card("Model", run.model_name))
    parts.append(_metric_card("Codebase", run.codebase_id))
    parts.append(_metric_card("Status", status_label,
                              "#f97316" if run.is_partial else "#22c55e"))
    parts.append(_metric_card("Dep F1", f"{dep_f1:.3f}",
                              "#22c55e" if dep_f1 > 0.6 else "#f97316" if dep_f1 > 0.3 else "#ef4444"))
    parts.append(_metric_card("Dep Precision", f"{dep_p:.3f}"))
    parts.append(_metric_card("Dep Recall", f"{dep_r:.3f}"))
    parts.append(_metric_card("Inv F1 (strict)", f"{inv_f1:.3f}"))
    parts.append(_metric_card("Inv F1 (relaxed)", f"{inv_f1r:.3f}"))
    parts.append(_metric_card("Action AUC", f"{auc:.3f}"))
    parts.append(_metric_card("ECE", f"{ece:.3f}"))
    parts.append(_metric_card("Steps Taken", str(steps)))
    parts.append(_metric_card("Files Opened", f"{files} ({unique_files} unique)"))
    parts.append('</div>')

    # Error info for PARTIAL runs
    if run.is_partial and run.error_info:
        parts.append(f'<div style="background:#fef2f2;border:1px solid #fecaca;'
                     f'border-radius:6px;padding:0.8rem 1rem;margin:0.8rem 0;'
                     f'font-size:0.85rem;color:#991b1b">'
                     f'<strong>Error:</strong> <code>{_esc(run.error_info)}</code></div>')

    # GT stats
    gt_edge_set = _gt_edge_set(run.gt_edges)
    gt_breakdown = _edge_type_breakdown(gt_edge_set)
    parts.append('<div style="margin:0.5rem 0;font-size:0.88rem;color:#64748b">')
    parts.append(f'Ground truth: {len(gt_edge_set)} edges (')
    parts.append(", ".join(f'{_edge_badge(t)} {c}' for t, c in sorted(gt_breakdown.items())))
    parts.append(f'), {len(run.gt_invariants)} invariants, {len(run.gt_modules)} modules</div>')

    # ---- (b) Action-by-action timeline ----
    if run.action_log:
        parts.append('<details><summary>Action-by-Action Timeline '
                     f'({len(run.action_log)} steps)</summary>')
        parts.append('<div class="detail-body">')

        connectivity = _edge_connectivity(run.gt_edges)
        opened_so_far: set[str] = set()
        known_dirs: set[str] = set()

        for act in run.action_log:
            step = act["step"]
            atype = act["action"]["type"].upper()
            arg = act["action"].get("argument", "")
            output = act.get("output", "")

            flags = []
            new_discoverable = set()
            running_recall = 0.0

            if atype == "OPEN":
                opened_so_far.add(arg)
                # What edges become discoverable now?
                all_disc = _discoverable_edges_at_step(opened_so_far, run.gt_edges)
                prev_opened = opened_so_far - {arg}
                prev_disc = _discoverable_edges_at_step(prev_opened, run.gt_edges)
                new_discoverable = all_disc - prev_disc
                running_recall = len(all_disc) / len(gt_edge_set) if gt_edge_set else 0

                # Flag low-connectivity files
                conn = connectivity.get(arg, 0)
                if conn <= 2 and step <= len(run.action_log) // 2:
                    flags.append(f"low connectivity ({conn} edges)")
            elif atype == "LIST":
                # Flag redundant LISTs
                if arg in known_dirs:
                    flags.append("directory already listed")
                if arg:
                    known_dirs.add(arg)
                else:
                    known_dirs.add("")
                # Update running recall based on current state
                disc = _discoverable_edges_at_step(opened_so_far, run.gt_edges)
                running_recall = len(disc) / len(gt_edge_set) if gt_edge_set else 0
            else:
                disc = _discoverable_edges_at_step(opened_so_far, run.gt_edges)
                running_recall = len(disc) / len(gt_edge_set) if gt_edge_set else 0

            parts.append('<div class="timeline-step">')
            parts.append(f'<span class="step-num">#{step}</span>')
            parts.append(f'<span class="step-action">{atype}</span>')

            detail_parts = [f'<code>{_esc(arg)}</code>']
            if atype == "OPEN" and new_discoverable:
                new_disc_str = ", ".join(
                    f'{_edge_badge(t)} <code>{_esc(s.split("/")[-1])}</code>'
                    f'&rarr;<code>{_esc(d.split("/")[-1])}</code>'
                    for s, d, t in sorted(new_discoverable)
                )
                detail_parts.append(f' &mdash; {len(new_discoverable)} new edges discoverable: {new_disc_str}')
            if flags:
                for fl in flags:
                    detail_parts.append(f'<span class="step-flag">[{_esc(fl)}]</span>')

            parts.append(f'<span class="step-detail">{"".join(detail_parts)}</span>')
            parts.append(f'<span class="step-recall">'
                         f'Discoverable recall: {_progress_bar(running_recall, 1.0, 160)}'
                         f'</span>')
            parts.append('</div>')

        parts.append('</div></details>')

    # ---- (c) Cognitive map evolution ----
    if cognitive_maps:
        parts.append(f'<details><summary>Cognitive Map Evolution '
                     f'({len(cognitive_maps)} probe points)</summary>')
        parts.append('<div class="detail-body">')

        prev_pred_edges: set[tuple[str, str, str]] = set()
        prev_correct: set[tuple[str, str, str]] = set()
        prev_errors: set[tuple[str, str, str]] = set()

        for mi, cm in enumerate(cognitive_maps):
            cm_step = cm.get("step", "?")
            comps = cm.get("components", {})
            invs = cm.get("invariants", [])
            pred_edges = _pred_edge_set_from_map(cm)
            pred_breakdown = _edge_type_breakdown(pred_edges)

            # P/R/F1 at this point
            p_, r_, f1_ = _prf(pred_edges, gt_edge_set)
            correct = pred_edges & gt_edge_set
            errors = pred_edges - gt_edge_set

            # New discoveries since last probe
            new_correct = correct - prev_correct
            new_errors = errors - prev_errors

            parts.append('<div class="probe-card">')
            parts.append(f'<div class="probe-header">Probe at step {cm_step} '
                         f'(F1={f1_:.3f}, P={p_:.3f}, R={r_:.3f})</div>')

            # Counts
            parts.append(f'<div style="font-size:0.88rem;margin:0.3rem 0">'
                         f'Components: {len(comps)} | '
                         f'Edges: {len(pred_edges)} | '
                         f'Invariants: {len(invs)}</div>')

            # Per-edge-type breakdown
            parts.append('<div style="font-size:0.85rem;margin:0.3rem 0">')
            for etype in ["IMPORTS", "CALLS_API", "DATA_FLOWS_TO", "REGISTRY_WIRES"]:
                pred_c = pred_breakdown.get(etype, 0)
                gt_c = gt_breakdown.get(etype, 0)
                color = "#22c55e" if pred_c == gt_c else "#f97316" if pred_c > 0 else "#ef4444"
                parts.append(f'{_edge_badge(etype)} {pred_c}/{gt_c} ')
            parts.append('</div>')

            # New correct discoveries
            if new_correct:
                parts.append(f'<div style="margin:0.3rem 0">'
                             f'<span class="tag-new">+{len(new_correct)} new correct</span> ')
                for s, d, t in sorted(new_correct)[:8]:
                    parts.append(f'{_edge_badge(t)} '
                                 f'<code class="mono">{_esc(s.split("/")[-1])}'
                                 f'&rarr;{_esc(d.split("/")[-1])}</code> ')
                if len(new_correct) > 8:
                    parts.append(f'... +{len(new_correct)-8} more')
                parts.append('</div>')

            # New errors
            if new_errors:
                parts.append(f'<div style="margin:0.3rem 0">'
                             f'<span class="tag-error">+{len(new_errors)} new errors</span> ')
                for s, d, t in sorted(new_errors)[:5]:
                    parts.append(f'{_edge_badge(t)} '
                                 f'<code class="mono">{_esc(s.split("/")[-1])}'
                                 f'&rarr;{_esc(d.split("/")[-1])}</code> ')
                if len(new_errors) > 5:
                    parts.append(f'... +{len(new_errors)-5} more')
                parts.append('</div>')

            parts.append('</div>')
            prev_pred_edges = pred_edges
            prev_correct = correct
            prev_errors = errors

        parts.append('</div></details>')

    # ---- (d) Conversation analysis ----
    if run.conversation:
        parts.append(f'<details><summary>Conversation Analysis '
                     f'({len(run.conversation)} messages)</summary>')
        parts.append('<div class="detail-body">')

        for ci, msg in enumerate(run.conversation):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "assistant":
                # Check if it's a probe response (JSON)
                trimmed = content.strip()
                if trimmed.startswith("{"):
                    # Parse and summarize
                    try:
                        parsed = json.loads(trimmed)
                        n_comps = len(parsed.get("components", {}))
                        n_edges = sum(len(c.get("edges", []))
                                      for c in parsed.get("components", {}).values())
                        n_invs = len(parsed.get("invariants", []))
                        summary = (f"[PROBE RESPONSE] step={parsed.get('step','?')}, "
                                   f"{n_comps} components, {n_edges} edges, {n_invs} invariants")
                        parts.append(f'<div class="conv-msg assistant">'
                                     f'<div class="role-label">Assistant (Probe)</div>'
                                     f'<code>{_esc(summary)}</code></div>')
                    except json.JSONDecodeError:
                        display = content[:300] + ("..." if len(content) > 300 else "")
                        parts.append(f'<div class="conv-msg assistant">'
                                     f'<div class="role-label">Assistant</div>'
                                     f'{_esc(display)}</div>')
                else:
                    display = content[:300] + ("..." if len(content) > 300 else "")
                    parts.append(f'<div class="conv-msg assistant">'
                                 f'<div class="role-label">Assistant (Action)</div>'
                                 f'<code>{_esc(display)}</code></div>')
            elif role == "user":
                # Truncate user prompts (they're repetitive)
                display = content[:200] + ("..." if len(content) > 200 else "")
                parts.append(f'<div class="conv-msg user">'
                             f'<div class="role-label">User</div>'
                             f'{_esc(display)}</div>')

        parts.append('</div></details>')

    # ---- (e) Error analysis ----
    if cognitive_maps:
        final_map = cognitive_maps[-1]
        pred_edges = _pred_edge_set_from_map(final_map)
        fp_edges = pred_edges - gt_edge_set
        fn_edges = gt_edge_set - pred_edges
        gt_module_set = set(run.gt_modules)

        # All opened files
        opened_files = set()
        for a in run.action_log:
            if a["action"]["type"].lower() == "open":
                opened_files.add(a["action"]["argument"])

        parts.append('<details><summary>Error Analysis '
                     f'({len(fp_edges)} FP, {len(fn_edges)} FN)</summary>')
        parts.append('<div class="detail-body">')

        # False positives
        parts.append(f'<h4>False Positives ({len(fp_edges)} edges)</h4>')
        fp_classes: dict[str, int] = Counter()
        if fp_edges:
            parts.append('<table class="fp-table"><thead><tr>'
                         '<th>Source</th><th>Target</th><th>Type</th><th>Classification</th>'
                         '</tr></thead><tbody>')
            for s, d, t in sorted(fp_edges):
                cls = _classify_false_positive((s, d, t), gt_edge_set, gt_module_set)
                fp_classes[cls] += 1
                cls_color = {
                    "Edge type confusion": "#f97316",
                    "Granularity mismatch": "#eab308",
                    "True hallucination": "#ef4444",
                    "Reasonable inference": "#64748b",
                }.get(cls, "#64748b")
                parts.append(f'<tr><td>{_esc(s)}</td><td>{_esc(d)}</td>'
                             f'<td>{_edge_badge(t)}</td>'
                             f'<td style="color:{cls_color};font-weight:600;font-family:sans-serif">'
                             f'{_esc(cls)}</td></tr>')
            parts.append('</tbody></table>')

            parts.append('<div style="margin:0.5rem 0;font-size:0.88rem">'
                         '<strong>FP breakdown:</strong> ')
            for cls, cnt in sorted(fp_classes.items(), key=lambda x: -x[1]):
                parts.append(f'{_esc(cls)}: {cnt} &nbsp; ')
            parts.append('</div>')
        else:
            parts.append('<p style="color:#22c55e;font-weight:600">No false positives.</p>')

        # False negatives
        parts.append(f'<h4 style="margin-top:1rem">False Negatives ({len(fn_edges)} edges)</h4>')
        fn_classes: dict[str, int] = Counter()
        if fn_edges:
            parts.append('<table class="fn-table"><thead><tr>'
                         '<th>Source</th><th>Target</th><th>Type</th><th>Classification</th>'
                         '</tr></thead><tbody>')
            for s, d, t in sorted(fn_edges):
                cls = _classify_false_negative((s, d, t), opened_files)
                fn_classes[cls] += 1
                cls_color = {
                    "Opened both but didn't report": "#ef4444",
                    "Opened source only": "#f97316",
                    "Opened target only": "#f97316",
                    "Never opened source/target": "#64748b",
                }.get(cls, "#64748b")
                parts.append(f'<tr><td>{_esc(s)}</td><td>{_esc(d)}</td>'
                             f'<td>{_edge_badge(t)}</td>'
                             f'<td style="color:{cls_color};font-weight:600;font-family:sans-serif">'
                             f'{_esc(cls)}</td></tr>')
            parts.append('</tbody></table>')

            parts.append('<div style="margin:0.5rem 0;font-size:0.88rem">'
                         '<strong>FN breakdown:</strong> ')
            for cls, cnt in sorted(fn_classes.items(), key=lambda x: -x[1]):
                parts.append(f'{_esc(cls)}: {cnt} &nbsp; ')
            parts.append('</div>')
        else:
            parts.append('<p style="color:#22c55e;font-weight:600">No false negatives.</p>')

        # Invariant analysis
        final_invs = final_map.get("invariants", [])
        parts.append(f'<h4 style="margin-top:1rem">Invariant Analysis '
                     f'({len(final_invs)} reported vs {len(run.gt_invariants)} GT)</h4>')

        if final_invs:
            inv_matches = _match_invariants_relaxed(final_invs, run.gt_invariants)
            parts.append('<table class="inv-table"><thead><tr>'
                         '<th>#</th><th>Type</th><th>Description</th>'
                         '<th>Structured</th><th>Match</th>'
                         '</tr></thead><tbody>')
            for ii, (pred, matched_gt, is_match) in enumerate(inv_matches):
                itype = pred.get("type", pred.get("structured", {}).get("type", "?"))
                desc = pred.get("description", "")[:120]
                struct = pred.get("structured", {})
                struct_str = f"{struct.get('type','')}: {struct.get('src','')}&rarr;{struct.get('dst','')}"
                match_label = ('<span class="tag-correct">MATCH</span>'
                               if is_match else '<span class="tag-error">NO MATCH</span>')
                parts.append(f'<tr><td>{ii+1}</td><td><code>{_esc(itype)}</code></td>'
                             f'<td>{_esc(desc)}</td>'
                             f'<td><code>{struct_str}</code></td>'
                             f'<td>{match_label}</td></tr>')
            parts.append('</tbody></table>')

            matched_count = sum(1 for _, _, m in inv_matches if m)
            parts.append(f'<div style="margin:0.5rem 0;font-size:0.88rem">'
                         f'Matched: {matched_count}/{len(final_invs)} predicted '
                         f'({matched_count}/{len(run.gt_invariants)} GT coverage)</div>')
        else:
            parts.append('<p style="color:#f97316">No invariants reported.</p>')

        parts.append('</div></details>')

        # ---- (f) Commentary ----
        edge_breakdown_pred = _edge_type_breakdown(pred_edges)
        commentary = _generate_commentary(run, fp_classes, fn_classes,
                                          edge_breakdown_pred, gt_breakdown)
        parts.append(f'<div class="commentary"><strong>Analysis:</strong> {_esc(commentary)}</div>')

    parts.append('</div></details>')
    return "\n".join(parts)


def generate_report(runs: list[RunData]) -> str:
    """Generate the full HTML report."""
    parts = []

    parts.append('<!DOCTYPE html><html lang="en"><head>')
    parts.append('<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">')
    parts.append('<title>ToCS Evaluation Log Analysis</title>')
    parts.append(f'<style>{CSS}</style>')
    parts.append('</head><body>')

    # Header
    parts.append('<div class="header">')
    parts.append('<h1>ToCS -- Evaluation Log Analysis Report</h1>')
    parts.append(f'<p>{len(runs)} LLM runs across '
                 f'{len(set(r.codebase_id for r in runs))} codebases | '
                 f'{len(set(r.model_name for r in runs))} models</p>')
    parts.append('</div>')

    parts.append('<div class="container">')

    # Summary section
    parts.append(_build_summary_section(runs))

    # Per-run sections
    for i, run in enumerate(runs):
        parts.append(_build_run_section(run, i))

    parts.append('</div>')
    parts.append('</body></html>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Discovering runs...")
    runs = _find_runs()
    print(f"Found {len(runs)} LLM runs")

    for run in runs:
        print(f"  - {run.display_name} on {run.codebase_id} "
              f"({'PARTIAL' if run.is_partial else 'OK'})")

    if not runs:
        print("No runs found! Check RESULTS_DIR and TARGET_CODEBASE_IDS.")
        sys.exit(1)

    print("\nGenerating report...")
    html_content = generate_report(runs)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html_content, encoding="utf-8")
    print(f"Report written to {OUTPUT_FILE}")
    print(f"Size: {len(html_content):,} bytes")


if __name__ == "__main__":
    main()
