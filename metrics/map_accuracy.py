"""Map accuracy metrics: score a CognitiveMap against CodebaseGroundTruth.

Provides:
1. Edge-level P/R/F1  (source, target, type) matching
2. Hallucination counts (spurious edges + phantom nodes)
3. Invariant P/R/F1   via structured-form matching
4. Contract accuracy   with forgiving type equivalence
5. Confidence ECE      (Expected Calibration Error)
6. Efficiency curves   action-AUC and observation-AUC
7. Steps-to-X-recall   first step reaching 50% / 80% edge recall
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from models import (
    ActionResult,
    ActionType,
    CodebaseGroundTruth,
    CognitiveMap,
    ExplorationMetrics,
    ExportedAPI,
    MapAccuracyMetrics,
)


# ============================================================================
# 1. Edge-level P / R / F1
# ============================================================================

EdgeTriple = tuple[str, str, str]  # (source, target, type)


def _gt_edge_set(gt: CodebaseGroundTruth) -> set[EdgeTriple]:
    """Ground-truth edges as (source, target, type) triples."""
    return {(e["source"], e["target"], e["type"]) for e in gt.dependency_edges}


def _predicted_edge_set(cmap: CognitiveMap) -> set[EdgeTriple]:
    """Predicted edges from a CognitiveMap."""
    edges: set[EdgeTriple] = set()
    for filepath, comp in cmap.components.items():
        for e in comp.edges:
            etype = e.type.value if hasattr(e.type, "value") else str(e.type)
            edges.add((filepath, e.target, etype))
    return edges


def _prf(predicted: set, gold: set) -> tuple[float, float, float]:
    """Precision, recall, F1."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def edge_prf(
    cmap: CognitiveMap, gt: CodebaseGroundTruth
) -> tuple[float, float, float]:
    """Edge-level precision, recall, F1."""
    return _prf(_predicted_edge_set(cmap), _gt_edge_set(gt))


# ============================================================================
# 2. Hallucination counts
# ============================================================================


@dataclass
class HallucinationCounts:
    hallucinated_edges: int = 0
    hallucinated_nodes: int = 0


def hallucinations(cmap: CognitiveMap, gt: CodebaseGroundTruth) -> HallucinationCounts:
    pred_edges = _predicted_edge_set(cmap)
    gt_edges = _gt_edge_set(gt)
    gt_modules = set(gt.modules.keys())

    return HallucinationCounts(
        hallucinated_edges=len(pred_edges - gt_edges),
        hallucinated_nodes=len(set(cmap.components.keys()) - gt_modules),
    )


# ============================================================================
# 3. Invariant P / R / F1  (structured-form matching)
# ============================================================================


def _normalise_structured(s: dict | None) -> tuple:
    """Normalise a structured invariant dict to a hashable tuple for matching."""
    if s is None:
        return ()
    return (
        s.get("type", ""),
        s.get("src") or "",
        s.get("dst") or "",
        s.get("via") or "",
        s.get("pattern") or "",
    )


def _gt_invariant_set(gt: CodebaseGroundTruth) -> set[tuple]:
    return {_normalise_structured(inv.structured) for inv in gt.invariants}


def _pred_invariant_set(cmap: CognitiveMap) -> set[tuple]:
    results: set[tuple] = set()
    for inv in cmap.invariants:
        if inv.structured is not None:
            d = inv.structured.model_dump() if hasattr(inv.structured, "model_dump") else {}
            results.add(_normalise_structured(d))
    return results


def invariant_prf(
    cmap: CognitiveMap, gt: CodebaseGroundTruth
) -> tuple[float, float, float]:
    """Invariant P/R/F1 via structured-form exact matching."""
    return _prf(_pred_invariant_set(cmap), _gt_invariant_set(gt))


# ── Relaxed invariant matching ──────────────────────────────────────


def _strip_path(p: str) -> str:
    """Strip leading package directories to get a canonical relative path.

    e.g. 'data_pipeline/stages/mod_a.py' → 'stages/mod_a.py'
         'data_pipeline/base.py'          → 'base.py'
         'stages/'                        → 'stages/'
    """
    if not p:
        return ""
    # Remove common single-level package prefix (e.g. data_pipeline/, text_processor/)
    parts = p.replace("\\", "/").split("/")
    if len(parts) > 1:
        return "/".join(parts[1:])
    return p


def _normalise_relaxed(s: dict | None) -> tuple:
    """Normalise for relaxed matching: (type, src, dst) with path stripping.

    Ignores `via` and `pattern`. For INTERFACE_ONLY, promotes `via` into dst
    if dst is empty (handles the field-placement confusion).
    """
    if s is None:
        return ()
    stype = s.get("type", "")
    src = _strip_path(s.get("src") or "")
    dst = _strip_path(s.get("dst") or "")
    via = _strip_path(s.get("via") or "")

    # INTERFACE_ONLY: models often put the interface file in dst instead of via.
    # Normalise by merging: if dst is empty but via is set, promote via → dst.
    # If both are set, keep dst (the model's explicit target).
    if stype == "INTERFACE_ONLY":
        if not dst and via:
            dst = via
        elif not via and dst:
            pass  # already in dst, fine

    return (stype, src, dst)


def _gt_invariant_set_relaxed(gt: CodebaseGroundTruth) -> set[tuple]:
    return {_normalise_relaxed(inv.structured) for inv in gt.invariants}


def _pred_invariant_set_relaxed(cmap: CognitiveMap) -> set[tuple]:
    results: set[tuple] = set()
    for inv in cmap.invariants:
        if inv.structured is not None:
            d = inv.structured.model_dump() if hasattr(inv.structured, "model_dump") else {}
            results.add(_normalise_relaxed(d))
    return results


def _relaxed_match(pred: tuple, gt: tuple) -> bool:
    """Check if a predicted invariant tuple matches a GT tuple under relaxed rules.

    Rules:
    - type must match exactly.
    - For src and dst: an empty field in GT acts as a wildcard (matches anything).
      If GT has a non-empty value, the predicted value must match after normalisation.
    """
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


def invariant_prf_relaxed(
    cmap: CognitiveMap, gt: CodebaseGroundTruth
) -> tuple[float, float, float]:
    """Invariant P/R/F1 via relaxed matching.

    Matches on (type, normalised_src, normalised_dst) only, ignoring
    via and pattern fields. Handles INTERFACE_ONLY field-placement confusion,
    strips leading package-directory prefixes, and treats empty GT fields
    as wildcards.

    Uses greedy 1-to-1 matching (each GT invariant matched at most once).
    """
    pred_tuples = list(_pred_invariant_set_relaxed(cmap))
    gt_tuples = list(_gt_invariant_set_relaxed(gt))

    if not pred_tuples and not gt_tuples:
        return 1.0, 1.0, 1.0

    # Greedy 1-to-1 matching: for each prediction, find first unmatched GT
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for pi, pt in enumerate(pred_tuples):
        for gi, gt_t in enumerate(gt_tuples):
            if gi not in matched_gt and _relaxed_match(pt, gt_t):
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_pred)
    precision = tp / len(pred_tuples) if pred_tuples else 0.0
    recall = tp / len(gt_tuples) if gt_tuples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# ============================================================================
# 4. Contract accuracy  (forgiving type equivalence)
# ============================================================================

# Type equivalence map: normalised form → set of equivalent strings
_TYPE_ALIASES: list[set[str]] = [
    {"dict", "Dict", "Dict[str, Any]", "dict[str, Any]"},
    {"list", "List", "List[Any]", "list[Any]"},
    {"str", "string"},
    {"int", "integer"},
    {"float", "number"},
    {"bool", "boolean"},
    {"None", "NoneType", "void", ""},
]

_OPTIONAL_RE = re.compile(r"Optional\[(.+)\]")


def normalise_type(t: str) -> str:
    """Normalise a type hint for forgiving comparison."""
    t = t.strip()
    # Optional[X] → X | None  (we just keep the inner type for matching)
    m = _OPTIONAL_RE.match(t)
    if m:
        t = m.group(1).strip()
    # X | None → X
    if " | None" in t:
        t = t.replace(" | None", "").strip()
    if "None | " in t:
        t = t.replace("None | ", "").strip()
    # Check alias groups
    for group in _TYPE_ALIASES:
        if t in group:
            return sorted(group)[0]  # canonical form (alphabetically first)
    return t.lower()


def _signature_matches(pred_sig, gt_sig) -> bool:
    """Forgiving signature comparison."""
    # Compare return types
    if normalise_type(pred_sig.return_type) != normalise_type(gt_sig.return_type):
        return False
    # Compare parameter count
    if len(pred_sig.params) != len(gt_sig.params):
        return False
    # Compare each parameter (by position)
    for pp, gp in zip(pred_sig.params, gt_sig.params):
        if pp.name != gp.name:
            return False
        if pp.type_hint and gp.type_hint:
            if normalise_type(pp.type_hint) != normalise_type(gp.type_hint):
                return False
    return True


@dataclass
class ContractScore:
    """Per-contract matching result."""

    total: int = 0
    name_matches: int = 0
    signature_matches: int = 0
    caller_matches: int = 0

    @property
    def accuracy(self) -> float:
        return self.signature_matches / self.total if self.total else 1.0

    @property
    def caller_accuracy(self) -> float:
        return self.caller_matches / self.total if self.total else 1.0


def contract_accuracy(cmap: CognitiveMap, gt: CodebaseGroundTruth) -> ContractScore:
    """Score agent's exported-API beliefs against ground-truth contracts."""
    score = ContractScore(total=len(gt.contracts))

    # Build lookup: (module, name) → BeliefExport
    pred_exports: dict[tuple[str, str], object] = {}
    for filepath, comp in cmap.components.items():
        for exp in comp.exports:
            pred_exports[(filepath, exp.name)] = exp

    for contract in gt.contracts:
        key = (contract.module, contract.name)
        pred = pred_exports.get(key)
        if pred is None:
            continue
        score.name_matches += 1

        if _signature_matches(pred.signature, contract.signature):
            score.signature_matches += 1

        # Caller set comparison
        pred_callers = set(pred.callers)
        gt_callers = set(contract.callers)
        if pred_callers == gt_callers:
            score.caller_matches += 1

    return score


# ============================================================================
# 5. Confidence ECE  (Expected Calibration Error)
# ============================================================================


def confidence_ece(cmap: CognitiveMap, gt: CodebaseGroundTruth, n_bins: int = 10) -> float:
    """Expected Calibration Error for edge confidence.

    Bins predicted edges by stated confidence, computes |avg_confidence - accuracy|
    per bin, weighted by bin size.
    """
    gt_edges = _gt_edge_set(gt)
    items: list[tuple[float, bool]] = []

    for filepath, comp in cmap.components.items():
        for e in comp.edges:
            etype = e.type.value if hasattr(e.type, "value") else str(e.type)
            correct = (filepath, e.target, etype) in gt_edges
            items.append((e.confidence, correct))

    if not items:
        return 0.0

    # Bin by confidence
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in items:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    ece = 0.0
    total = len(items)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, correct in b if correct) / len(b)
        ece += len(b) / total * abs(avg_conf - avg_acc)

    return ece


# ============================================================================
# 6. Efficiency curves + AUC
# ============================================================================


def _trapezoidal_auc(xs: list[float], ys: list[float]) -> float:
    """Trapezoidal integration. xs and ys must be same length."""
    if len(xs) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        auc += dx * (ys[i - 1] + ys[i]) / 2
    return auc


def efficiency_curves(
    maps: list[CognitiveMap],
    action_log: list[ActionResult],
    gt: CodebaseGroundTruth,
) -> tuple[list[float], list[float], float, float]:
    """Compute action-efficiency and observation-efficiency curves + AUCs.

    Returns (action_curve, observation_curve, action_auc, observation_auc).

    Convention: piecewise-constant belief between probes, trapezoidal integration.
    Each map is associated with the action step at which it was produced.
    """
    gt_edges = _gt_edge_set(gt)

    # Map step → F1
    step_f1: dict[int, float] = {}
    for m in maps:
        pred = _predicted_edge_set(m)
        _, _, f1 = _prf(pred, gt_edges)
        step_f1[m.step] = f1

    if not step_f1:
        return [], [], 0.0, 0.0

    # Action-efficiency curve: F1 at each action step (piecewise constant)
    total_actions = max(a.step for a in action_log) if action_log else max(step_f1)
    action_curve: list[float] = []
    current_f1 = 0.0
    sorted_steps = sorted(step_f1.keys())
    step_idx = 0
    for t in range(1, total_actions + 1):
        while step_idx < len(sorted_steps) and sorted_steps[step_idx] <= t:
            current_f1 = step_f1[sorted_steps[step_idx]]
            step_idx += 1
        action_curve.append(current_f1)

    # Observation-efficiency curve: F1 at each OPEN action
    open_steps = [
        a.step for a in action_log if a.action.type == ActionType.OPEN
    ]
    obs_curve: list[float] = []
    current_f1 = 0.0
    step_idx = 0
    for i, os in enumerate(open_steps):
        while step_idx < len(sorted_steps) and sorted_steps[step_idx] <= os:
            current_f1 = step_f1[sorted_steps[step_idx]]
            step_idx += 1
        obs_curve.append(current_f1)

    # AUCs (normalised by number of steps)
    action_xs = [i / max(len(action_curve), 1) for i in range(len(action_curve))]
    action_auc = _trapezoidal_auc(action_xs, action_curve) if action_curve else 0.0

    obs_xs = [i / max(len(obs_curve), 1) for i in range(len(obs_curve))]
    obs_auc = _trapezoidal_auc(obs_xs, obs_curve) if obs_curve else 0.0

    return action_curve, obs_curve, action_auc, obs_auc


# ============================================================================
# 7. Steps-to-X-recall
# ============================================================================


def steps_to_recall(
    maps: list[CognitiveMap],
    gt: CodebaseGroundTruth,
    thresholds: tuple[float, ...] = (0.5, 0.8),
) -> dict[float, int | None]:
    """For each threshold, find the first map step where edge recall >= threshold."""
    gt_edges = _gt_edge_set(gt)
    results: dict[float, int | None] = {t: None for t in thresholds}

    for m in sorted(maps, key=lambda x: x.step):
        pred = _predicted_edge_set(m)
        tp = len(pred & gt_edges)
        recall = tp / len(gt_edges) if gt_edges else 1.0
        for t in thresholds:
            if results[t] is None and recall >= t:
                results[t] = m.step

    return results


# ============================================================================
# Combined scorer
# ============================================================================


def score_map(
    cmap: CognitiveMap, gt: CodebaseGroundTruth
) -> MapAccuracyMetrics:
    """Score a single CognitiveMap against ground truth."""
    dep_p, dep_r, dep_f1 = edge_prf(cmap, gt)
    inv_p, inv_r, inv_f1 = invariant_prf(cmap, gt)
    inv_rp, inv_rr, inv_rf1 = invariant_prf_relaxed(cmap, gt)
    ece = confidence_ece(cmap, gt)

    return MapAccuracyMetrics(
        dependency_precision=dep_p,
        dependency_recall=dep_r,
        dependency_f1=dep_f1,
        invariant_precision=inv_p,
        invariant_recall=inv_r,
        invariant_f1=inv_f1,
        invariant_precision_relaxed=inv_rp,
        invariant_recall_relaxed=inv_rr,
        invariant_f1_relaxed=inv_rf1,
        confidence_ece=ece,
    )


def score_exploration(
    maps: list[CognitiveMap],
    action_log: list[ActionResult],
    gt: CodebaseGroundTruth,
) -> ExplorationMetrics:
    """Compute full exploration metrics from a sequence of maps + action log."""
    act_curve, obs_curve, act_auc, obs_auc = efficiency_curves(maps, action_log, gt)
    recall_steps = steps_to_recall(maps, gt)

    total_steps = len(action_log)
    opens = [a for a in action_log if a.action.type == ActionType.OPEN]
    unique = set(a.action.argument for a in opens if a.action.argument)

    # Final F1 from last map
    final_f1 = act_curve[-1] if act_curve else 0.0

    return ExplorationMetrics(
        information_gain_curve=[],
        action_efficiency_curve=act_curve,
        observation_efficiency_curve=obs_curve,
        action_auc=act_auc,
        observation_auc=obs_auc,
        steps_to_50_recall=recall_steps.get(0.5),
        steps_to_80_recall=recall_steps.get(0.8),
        final_efficiency=final_f1,
        steps_taken=total_steps,
        files_opened=len(opens),
        unique_files=len(unique),
    )
