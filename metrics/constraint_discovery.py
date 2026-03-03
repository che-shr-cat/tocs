"""Constraint discovery probes and compliance checking.

Provides:
1. ProbeGenerator — generates counterfactual MCQ probes from ground truth invariants
2. ComplianceChecker — AST-based checking of code against planted constraints (v1.0 stub)
3. score_constraint_discovery() — scores model responses on constraint probes
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field

from models import (
    CodebaseGroundTruth,
    ConstraintDiscoveryMetrics,
    InvariantGroundTruth,
)


# ── Probe types ────────────────────────────────────────────────────


@dataclass
class ProbeOption:
    """A single option in a constraint probe MCQ."""

    label: str
    description: str
    is_correct: bool


@dataclass
class ConstraintProbe:
    """A counterfactual MCQ probe for constraint discovery."""

    invariant_id: str
    question: str
    options: list[ProbeOption]
    correct_index: int
    explanation: str


# ── Probe generation ───────────────────────────────────────────────


class ProbeGenerator:
    """Generates counterfactual MCQ probes from ground truth invariants."""

    def __init__(
        self,
        gt: CodebaseGroundTruth,
        seed: int | None = None,
    ) -> None:
        self.gt = gt
        self.rng = random.Random(seed)

    def generate_all(self) -> list[ConstraintProbe]:
        """Generate one probe per invariant in the ground truth."""
        probes: list[ConstraintProbe] = []
        for inv in self.gt.invariants:
            probe = self._generate_probe(inv)
            if probe is not None:
                probes.append(probe)
        return probes

    def _generate_probe(self, inv: InvariantGroundTruth) -> ConstraintProbe | None:
        """Generate a probe for a single invariant."""
        s = inv.structured
        ctype = s.get("type", "")

        if ctype == "FORBIDDEN_EDGE":
            return self._probe_forbidden_edge(inv)
        elif ctype == "INTERFACE_ONLY":
            return self._probe_interface_only(inv)
        elif ctype == "VALIDATION_CHAIN":
            return self._probe_validation_chain(inv)
        else:
            # Generic fallback
            return self._probe_generic(inv)

    def _probe_forbidden_edge(self, inv: InvariantGroundTruth) -> ConstraintProbe:
        """Generate probe for FORBIDDEN_EDGE invariants."""
        s = inv.structured
        src = s.get("src")
        dst = s.get("dst")

        all_modules = list(self.gt.modules.keys())

        if src and dst:
            # Specific forbidden edge: src → dst
            violation = f"Add `from .{_mod(dst)} import ...` in {src}"
            explanation = f"{inv.description} — adding this import violates the constraint."
        elif dst:
            # No module may import dst
            # Pick a random non-dst module as the source
            candidates = [m for m in inv.involved_modules if m != dst]
            chosen_src = self.rng.choice(candidates) if candidates else "mod_a.py"
            violation = f"Add `from .{_mod(dst)} import ...` in {chosen_src}"
            explanation = f"{inv.description} — importing {dst} from any stage/helper violates this constraint."
        else:
            # Pattern-based (e.g., no stage-to-stage imports)
            stages = [m for m in inv.involved_modules if m.startswith("mod_")]
            if len(stages) >= 2:
                pair = self.rng.sample(stages, 2)
                violation = f"Add `from .{_mod(pair[1])} import ...` in {pair[0]}"
            else:
                violation = "Add a direct import between pipeline stages"
            explanation = f"{inv.description} — direct stage-to-stage imports are forbidden."

        # Generate distractors (valid changes that don't violate any constraint)
        distractors = self._generate_distractors(inv, 3)

        return self._build_probe(inv, violation, explanation, distractors)

    def _probe_interface_only(self, inv: InvariantGroundTruth) -> ConstraintProbe:
        """Generate probe for INTERFACE_ONLY invariants."""
        via = inv.structured.get("via", "base.py")
        stages = [m for m in inv.involved_modules if m != via]
        chosen = self.rng.choice(stages) if stages else "mod_a.py"

        violation = (
            f"Make {chosen} call another stage's internal method directly "
            f"instead of going through {via}"
        )
        explanation = (
            f"{inv.description} — bypassing the {via} interface breaks the uniform contract."
        )

        distractors = self._generate_distractors(inv, 3)
        return self._build_probe(inv, violation, explanation, distractors)

    def _probe_validation_chain(self, inv: InvariantGroundTruth) -> ConstraintProbe:
        """Generate probe for VALIDATION_CHAIN invariants."""
        s = inv.structured
        src = s.get("src")
        dst = s.get("dst")
        via = s.get("via")

        if src and dst and via:
            violation = (
                f"Pass data from {src} directly to {dst} without setting {via}"
            )
            explanation = (
                f"{inv.description} — skipping validation ({via}) breaks the chain."
            )
        elif src and via:
            violation = (
                f"Skip the {via} step when routing data from {src}"
            )
            explanation = (
                f"{inv.description} — the {via} step is required for data from {src}."
            )
        else:
            violation = f"Skip a required validation step in the pipeline"
            explanation = inv.description

        distractors = self._generate_distractors(inv, 3)
        return self._build_probe(inv, violation, explanation, distractors)

    def _probe_generic(self, inv: InvariantGroundTruth) -> ConstraintProbe:
        """Fallback probe for unrecognized invariant types."""
        violation = f"Modify the codebase in a way that violates: {inv.description}"
        explanation = inv.description

        distractors = self._generate_distractors(inv, 3)
        return self._build_probe(inv, violation, explanation, distractors)

    def _generate_distractors(
        self,
        inv: InvariantGroundTruth,
        count: int,
    ) -> list[str]:
        """Generate plausible distractors that DON'T violate any constraint.

        Picks changes that respect all invariants — typically using existing
        edges or valid refactorings from the ground truth.
        """
        all_modules = list(self.gt.modules.keys())
        existing_edges = {
            (e["source"], e["target"]) for e in self.gt.dependency_edges
        }

        candidates: list[str] = []

        # Distractor type 1: existing valid imports (already in the dependency graph)
        for src, tgt in existing_edges:
            candidates.append(
                f"Add `from .{_mod(tgt)} import ...` in {src}"
            )

        # Distractor type 2: adding a helper import (generally safe)
        helper_modules = [
            m for m in all_modules
            if m in ("models.py", "base.py", "helpers.py")
        ]
        for helper in helper_modules:
            for mod in all_modules:
                if mod != helper and (mod, helper) not in existing_edges:
                    candidates.append(
                        f"Add `from .{_mod(helper)} import ...` in {mod}"
                    )

        # Distractor type 3: rename a function (always safe)
        candidates.append("Rename an internal helper function in helpers.py")
        candidates.append("Add a docstring to an existing function")
        candidates.append("Extract a private helper method within a single module")

        # Filter out anything that looks like the violation
        involved = set(inv.involved_modules)
        filtered = [
            c for c in candidates
            if not _mentions_all(c, involved)
        ]

        # Include "None of the above" sometimes
        include_none = self.rng.random() < 0.3
        if include_none and count >= 2:
            pool = filtered[:] if filtered else candidates[:]
            self.rng.shuffle(pool)
            result = pool[:count - 1]
            result.append("None of the above")
            return result

        self.rng.shuffle(filtered)
        return filtered[:count] if len(filtered) >= count else (
            filtered + candidates[:count - len(filtered)]
        )[:count]

    def _build_probe(
        self,
        inv: InvariantGroundTruth,
        violation: str,
        explanation: str,
        distractors: list[str],
    ) -> ConstraintProbe:
        """Build a ConstraintProbe with shuffled options."""
        options = [
            ProbeOption(
                label=violation,
                description="This change violates a planted constraint",
                is_correct=True,
            )
        ]
        for d in distractors:
            options.append(
                ProbeOption(
                    label=d,
                    description="This change is valid",
                    is_correct=False,
                )
            )

        # Shuffle options
        self.rng.shuffle(options)
        correct_index = next(
            i for i, o in enumerate(options) if o.is_correct
        )

        return ConstraintProbe(
            invariant_id=inv.id,
            question="Which of the following changes would violate an architectural constraint?",
            options=options,
            correct_index=correct_index,
            explanation=explanation,
        )


# ── Compliance checker (v1.0 stub) ────────────────────────────────


@dataclass
class ComplianceResult:
    """Result of checking code against a constraint."""

    compliant: bool
    constraint_type: str
    details: str = ""


class ComplianceChecker:
    """AST-based checker for planted constraints.

    Checks whether a code snippet violates planted constraints.
    v1.0 stub — works on individual code strings, not integrated into pipeline.
    """

    def __init__(self, gt: CodebaseGroundTruth) -> None:
        self.gt = gt

    def check_forbidden_edge(
        self,
        code: str,
        source_module: str,
        forbidden_targets: list[str] | None = None,
    ) -> ComplianceResult:
        """Check if code imports a forbidden module.

        Args:
            code: Python source code to check.
            source_module: The module this code belongs to.
            forbidden_targets: Specific forbidden import targets.
                If None, derives from ground truth invariants.
        """
        if forbidden_targets is None:
            forbidden_targets = self._get_forbidden_targets(source_module)

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ComplianceResult(
                compliant=False,
                constraint_type="FORBIDDEN_EDGE",
                details=f"Syntax error: {e}",
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name.split(".")[-1]
                    if f"{mod_name}.py" in forbidden_targets:
                        return ComplianceResult(
                            compliant=False,
                            constraint_type="FORBIDDEN_EDGE",
                            details=f"Forbidden import: {alias.name}",
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod_name = node.module.split(".")[-1]
                    if f"{mod_name}.py" in forbidden_targets:
                        return ComplianceResult(
                            compliant=False,
                            constraint_type="FORBIDDEN_EDGE",
                            details=f"Forbidden import from: {node.module}",
                        )

        return ComplianceResult(
            compliant=True,
            constraint_type="FORBIDDEN_EDGE",
        )

    def check_interface_only(
        self,
        code: str,
        required_base: str = "base.py",
    ) -> ComplianceResult:
        """Check if code bypasses the required interface (base class).

        Looks for direct attribute access on other stage modules
        instead of going through the base interface.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ComplianceResult(
                compliant=False,
                constraint_type="INTERFACE_ONLY",
                details=f"Syntax error: {e}",
            )

        # Find all imported stage modules
        stage_imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod_name = node.module.split(".")[-1]
                if mod_name.startswith("mod_"):
                    stage_imports.add(mod_name)

        if not stage_imports:
            return ComplianceResult(
                compliant=True,
                constraint_type="INTERFACE_ONLY",
            )

        # Check for direct calls to stage internals (non-interface methods)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in stage_imports:
                        # Importing from a stage module and accessing attributes
                        return ComplianceResult(
                            compliant=False,
                            constraint_type="INTERFACE_ONLY",
                            details=(
                                f"Direct access to {node.value.id}.{node.attr} "
                                f"bypasses {required_base} interface"
                            ),
                        )

        return ComplianceResult(
            compliant=True,
            constraint_type="INTERFACE_ONLY",
        )

    def check_validation_chain(
        self,
        code: str,
        required_field: str = "is_validated",
    ) -> ComplianceResult:
        """Check if code skips a required validation step.

        Looks for data processing without checking the required field.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ComplianceResult(
                compliant=False,
                constraint_type="VALIDATION_CHAIN",
                details=f"Syntax error: {e}",
            )

        has_check = False
        for node in ast.walk(tree):
            # Look for attribute access on the required field
            if isinstance(node, ast.Attribute) and node.attr == required_field:
                has_check = True
                break
            # Also check string literals (e.g., dict access)
            if isinstance(node, ast.Constant) and node.value == required_field:
                has_check = True
                break

        if not has_check:
            return ComplianceResult(
                compliant=True,
                constraint_type="VALIDATION_CHAIN",
                details=f"Code does not interact with {required_field} (N/A)",
            )

        return ComplianceResult(
            compliant=True,
            constraint_type="VALIDATION_CHAIN",
        )

    def check_all(self, code: str, source_module: str) -> list[ComplianceResult]:
        """Run all constraint checks on a code snippet."""
        return [
            self.check_forbidden_edge(code, source_module),
            self.check_interface_only(code),
            self.check_validation_chain(code),
        ]

    def _get_forbidden_targets(self, source_module: str) -> list[str]:
        """Derive forbidden import targets for a module from ground truth."""
        forbidden: list[str] = []
        for inv in self.gt.invariants:
            s = inv.structured
            if s.get("type") != "FORBIDDEN_EDGE":
                continue
            if source_module not in inv.involved_modules:
                continue
            dst = s.get("dst")
            if dst:
                forbidden.append(dst)
            else:
                # Pattern-based: all other involved modules are forbidden
                forbidden.extend(
                    m for m in inv.involved_modules if m != source_module
                )
        return forbidden


# ── Scoring ────────────────────────────────────────────────────────


def score_constraint_discovery(
    probes: list[ConstraintProbe],
    answers: list[int],
) -> ConstraintDiscoveryMetrics:
    """Score model answers against constraint probes.

    Args:
        probes: The generated probes.
        answers: Model's selected option index for each probe (0-indexed).

    Returns:
        ConstraintDiscoveryMetrics with counterfactual_probe_accuracy.
    """
    if not probes:
        return ConstraintDiscoveryMetrics(counterfactual_probe_accuracy=0.0)

    correct = sum(
        1 for probe, answer in zip(probes, answers)
        if answer == probe.correct_index
    )
    return ConstraintDiscoveryMetrics(
        counterfactual_probe_accuracy=correct / len(probes),
    )


# ── Helpers ────────────────────────────────────────────────────────


def _mod(filename: str) -> str:
    """Strip .py extension for import-style reference."""
    return filename.replace(".py", "")


def _mentions_all(text: str, modules: set[str]) -> bool:
    """Check if text mentions ALL modules in the set."""
    return all(_mod(m) in text or m in text for m in modules)
