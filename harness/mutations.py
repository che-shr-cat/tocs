"""Mutation engine for the REVISE phase.

Provides:
1. MutationEngine — applies real mutations to codebase files on disk
2. ShamMutationEngine — generates evidence without modifying files
3. EvidenceGenerator — creates realistic-looking pytest output / tracebacks
4. score_revision() — computes belief revision metrics (BRS, inertia, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from models import (
    BeliefRevisionMetrics,
    CodebaseGroundTruth,
    CognitiveMap,
    ExportedAPI,
    Mutation,
    MutationType,
    ParameterSpec,
)


# ── Result type ─────────────────────────────────────────────────────


@dataclass
class MutationResult:
    """Result of applying (or simulating) a mutation."""

    mutation: Mutation
    post_gt: CodebaseGroundTruth
    evidence: str


# ── Evidence generation ─────────────────────────────────────────────


class EvidenceGenerator:
    """Generate realistic test failure / error output for a mutation."""

    @staticmethod
    def for_interface_break(
        module: str,
        function_name: str,
        callers: list[str],
        new_param: str = "strict",
    ) -> str:
        caller = callers[0] if callers else "runner.py"
        return (
            "========================= FAILURES =========================\n"
            "_________________ test_pipeline_runs _________________\n"
            "\n"
            "    def test_pipeline_runs():\n"
            f">       result = stage.{function_name}(records, config)\n"
            f"E       TypeError: {function_name}() missing 1 required "
            f"positional argument: '{new_param}'\n"
            "\n"
            f"{caller}:42: TypeError\n"
            "==================== 1 failed ====================\n"
        )

    @staticmethod
    def for_dependency_shift(
        src_module: str,
        dst_module: str,
        function_name: str,
        importer: str,
    ) -> str:
        src_mod = src_module.replace(".py", "")
        return (
            "Traceback (most recent call last):\n"
            f'  File "{importer}", line 3, in <module>\n'
            f"    from .{src_mod} import {function_name}\n"
            f"ImportError: cannot import name '{function_name}' "
            f"from '{src_mod}' ({src_module})\n"
            "\n"
            f"Hint: '{function_name}' has been moved to {dst_module}\n"
        )

    @staticmethod
    def for_boundary_breach(
        from_module: str,
        to_module: str,
        invariant_desc: str,
    ) -> str:
        return (
            "========================= FAILURES =========================\n"
            "_____________ test_no_direct_stage_imports _____________\n"
            "\n"
            "    def test_no_direct_stage_imports():\n"
            '        """Stages must not import each other directly."""\n'
            ">       assert not direct_imports, f'Forbidden: {direct_imports}'\n"
            f"E       AssertionError: Forbidden: {{'{from_module}' -> "
            f"'{to_module}'}}\n"
            f"E       Violation: {invariant_desc}\n"
            "\n"
            "test_boundaries.py:15: AssertionError\n"
            "==================== 1 failed ====================\n"
        )


# ── Mutation engine (real) ──────────────────────────────────────────


class MutationEngine:
    """Applies real mutations to codebase files on disk."""

    def __init__(self, codebase_dir: Path, gt: CodebaseGroundTruth) -> None:
        self.codebase_dir = Path(codebase_dir)
        self.gt = gt

    def apply(self, mutation_type: MutationType, **kwargs) -> MutationResult:
        """Apply a mutation of the given type."""
        dispatch = {
            MutationType.INTERFACE_BREAK: self.apply_interface_break,
            MutationType.DEPENDENCY_SHIFT: self.apply_dependency_shift,
            MutationType.BOUNDARY_BREACH: self.apply_boundary_breach,
        }
        fn = dispatch.get(mutation_type)
        if fn is None:
            raise ValueError(f"Unsupported mutation type: {mutation_type}")
        return fn(**kwargs)

    def apply_interface_break(
        self,
        module: str | None = None,
        function_name: str | None = None,
        new_param: str = "strict",
        new_param_type: str = "bool",
    ) -> MutationResult:
        """Add a required parameter to a function signature."""
        # Auto-select from contracts
        if module is None or function_name is None:
            for contract in self.gt.contracts:
                if contract.callers:
                    module = contract.module
                    function_name = contract.name
                    break
            else:
                raise ValueError("No contract with callers found")

        # Modify file
        filepath = self.codebase_dir / module
        content = filepath.read_text()
        pattern = rf"(def {re.escape(function_name)}\([^)]*)"
        match = re.search(pattern, content)
        if not match:
            raise ValueError(f"Cannot find def {function_name} in {module}")

        old_def = match.group(0)
        new_def = f"{old_def}, {new_param}: {new_param_type}"
        filepath.write_text(content.replace(old_def, new_def, 1))

        # Post-mutation ground truth
        post_gt = self.gt.model_copy(deep=True)
        for contract in post_gt.contracts:
            if contract.module == module and contract.name == function_name:
                contract.signature.params.append(
                    ParameterSpec(
                        name=new_param,
                        type_hint=new_param_type,
                        has_default=False,
                    )
                )

        callers = _find_callers(self.gt, module, function_name)
        affected = sorted(set([module] + callers))

        mutation = Mutation(
            id=f"ibreak_{module}_{function_name}",
            type=MutationType.INTERFACE_BREAK,
            target_module=module,
            description=(
                f"Added required '{new_param}: {new_param_type}' "
                f"parameter to {function_name} in {module}"
            ),
            is_sham=False,
            affected_modules=affected,
            affected_invariants=[],
        )
        evidence = EvidenceGenerator.for_interface_break(
            module, function_name, callers, new_param
        )
        return MutationResult(mutation=mutation, post_gt=post_gt, evidence=evidence)

    def apply_dependency_shift(
        self,
        src_module: str = "helpers.py",
        dst_module: str = "mod_d.py",
        function_name: str = "compute_checksum",
        importer: str = "mod_b.py",
    ) -> MutationResult:
        """Move a function from src_module to dst_module, update importer."""
        # 1. Extract function from source
        src_path = self.codebase_dir / src_module
        src_content = src_path.read_text()

        # Find function body (from def through next def or EOF)
        func_pattern = rf"(def {re.escape(function_name)}\(.*?\n(?:(?!def ).*\n)*)"
        func_match = re.search(func_pattern, src_content)
        func_body = func_match.group(0) if func_match else ""

        # Remove function from source
        if func_match:
            new_src = src_content.replace(func_match.group(0), "", 1)
            new_src = re.sub(r"\n{3,}", "\n\n", new_src)
            src_path.write_text(new_src)

        # 2. Add function to destination
        dst_path = self.codebase_dir / dst_module
        dst_content = dst_path.read_text()

        # Add necessary imports from the function
        extra_imports = ""
        if "hashlib" in func_body and "import hashlib" not in dst_content:
            extra_imports = "import hashlib\n"
        if "Any" in func_body and "Any" not in dst_content:
            extra_imports += "from typing import Any\n"

        if extra_imports:
            # Insert after existing imports
            lines = dst_content.split("\n")
            last_import = 0
            for i, line in enumerate(lines):
                if line.startswith("from ") or line.startswith("import "):
                    last_import = i
            lines.insert(last_import + 1, extra_imports.rstrip())
            dst_content = "\n".join(lines)

        dst_content = dst_content.rstrip() + "\n\n\n" + func_body
        dst_path.write_text(dst_content)

        # 3. Update importer
        imp_path = self.codebase_dir / importer
        imp_content = imp_path.read_text()
        src_mod = src_module.replace(".py", "")
        dst_mod = dst_module.replace(".py", "")
        imp_content = imp_content.replace(
            f"from .{src_mod} import {function_name}",
            f"from .{dst_mod} import {function_name}",
        )
        imp_path.write_text(imp_content)

        # Post-mutation ground truth
        post_gt = self.gt.model_copy(deep=True)
        # Remove old CALLS_API edge, add new IMPORTS + CALLS_API to dst
        post_gt.dependency_edges = [
            e
            for e in post_gt.dependency_edges
            if not (
                e.get("source") == importer
                and e.get("target") == src_module
                and e.get("type") == "CALLS_API"
            )
        ]
        # Replace IMPORTS edge from importer→src with importer→dst
        for e in post_gt.dependency_edges:
            if (
                e.get("source") == importer
                and e.get("target") == src_module
                and e.get("type") == "IMPORTS"
            ):
                e["target"] = dst_module
                break
        post_gt.dependency_edges.append(
            {"source": importer, "target": dst_module, "type": "CALLS_API"}
        )

        # Update module-level edges
        if importer in post_gt.modules:
            mod_edges = post_gt.modules[importer].edges
            post_gt.modules[importer].edges = [
                e
                for e in mod_edges
                if not (
                    e.get("target") == src_module
                    and e.get("type") in ("IMPORTS", "CALLS_API")
                )
            ]
            post_gt.modules[importer].edges.extend([
                {"target": dst_module, "type": "IMPORTS"},
                {"target": dst_module, "type": "CALLS_API"},
            ])

        affected = sorted(set([importer, src_module, dst_module]))
        mutation = Mutation(
            id=f"depshift_{function_name}_{src_module}_to_{dst_module}",
            type=MutationType.DEPENDENCY_SHIFT,
            target_module=src_module,
            description=f"Moved {function_name} from {src_module} to {dst_module}",
            is_sham=False,
            affected_modules=affected,
            affected_invariants=[],
        )
        evidence = EvidenceGenerator.for_dependency_shift(
            src_module, dst_module, function_name, importer
        )
        return MutationResult(mutation=mutation, post_gt=post_gt, evidence=evidence)

    def apply_boundary_breach(
        self,
        from_module: str = "mod_c.py",
        to_module: str = "mod_a.py",
        import_name: str = "IngestStage",
    ) -> MutationResult:
        """Add a forbidden direct import between modules."""
        filepath = self.codebase_dir / from_module
        content = filepath.read_text()

        to_mod = to_module.replace(".py", "")
        import_line = f"from .{to_mod} import {import_name}"

        # Insert after last import
        lines = content.split("\n")
        last_import = 0
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                last_import = i
        lines.insert(last_import + 1, import_line)
        filepath.write_text("\n".join(lines))

        # Post-mutation ground truth
        post_gt = self.gt.model_copy(deep=True)
        post_gt.dependency_edges.append(
            {"source": from_module, "target": to_module, "type": "IMPORTS"}
        )
        if from_module in post_gt.modules:
            post_gt.modules[from_module].edges.append(
                {"target": to_module, "type": "IMPORTS"}
            )

        # Find affected invariants
        affected_invariants: list[str] = []
        for inv in self.gt.invariants:
            s = inv.structured
            if s.get("type") == "FORBIDDEN_EDGE":
                if (
                    from_module in inv.involved_modules
                    and to_module in inv.involved_modules
                ) or (not s.get("src") and not s.get("dst")):
                    affected_invariants.append(inv.id)

        affected = sorted(set([from_module, to_module]))
        mutation = Mutation(
            id=f"breach_{from_module}_imports_{to_module}",
            type=MutationType.BOUNDARY_BREACH,
            target_module=from_module,
            description=(
                f"Added forbidden import: {from_module} imports "
                f"{import_name} from {to_module}"
            ),
            is_sham=False,
            affected_modules=affected,
            affected_invariants=affected_invariants,
        )
        evidence = EvidenceGenerator.for_boundary_breach(
            from_module,
            to_module,
            "stages must not import each other directly",
        )
        return MutationResult(mutation=mutation, post_gt=post_gt, evidence=evidence)


# ── Sham mutation engine ────────────────────────────────────────────


class ShamMutationEngine:
    """Generates evidence without modifying any files (no-change control)."""

    def __init__(self, codebase_dir: Path, gt: CodebaseGroundTruth) -> None:
        self.codebase_dir = Path(codebase_dir)
        self.gt = gt

    def generate_sham(self, mutation_type: MutationType) -> MutationResult:
        """Generate sham evidence: no files change, is_sham=True."""
        dispatch = {
            MutationType.INTERFACE_BREAK: self._sham_interface_break,
            MutationType.DEPENDENCY_SHIFT: self._sham_dependency_shift,
            MutationType.BOUNDARY_BREACH: self._sham_boundary_breach,
        }
        fn = dispatch.get(mutation_type)
        if fn is None:
            raise ValueError(f"Unsupported mutation type: {mutation_type}")
        return fn()

    def _sham_interface_break(self) -> MutationResult:
        module, function_name = "mod_b.py", "process"
        for contract in self.gt.contracts:
            if contract.callers:
                module = contract.module
                function_name = contract.name
                break
        callers = _find_callers(self.gt, module, function_name)

        mutation = Mutation(
            id=f"sham_ibreak_{module}_{function_name}",
            type=MutationType.INTERFACE_BREAK,
            target_module=module,
            description=f"SHAM: pretends to add parameter to {function_name}",
            is_sham=True,
            affected_modules=[],
            affected_invariants=[],
        )
        evidence = EvidenceGenerator.for_interface_break(
            module, function_name, callers
        )
        return MutationResult(
            mutation=mutation,
            post_gt=self.gt.model_copy(deep=True),
            evidence=evidence,
        )

    def _sham_dependency_shift(self) -> MutationResult:
        mutation = Mutation(
            id="sham_depshift",
            type=MutationType.DEPENDENCY_SHIFT,
            target_module="helpers.py",
            description="SHAM: pretends to move compute_checksum",
            is_sham=True,
            affected_modules=[],
            affected_invariants=[],
        )
        evidence = EvidenceGenerator.for_dependency_shift(
            "helpers.py", "mod_d.py", "compute_checksum", "mod_b.py"
        )
        return MutationResult(
            mutation=mutation,
            post_gt=self.gt.model_copy(deep=True),
            evidence=evidence,
        )

    def _sham_boundary_breach(self) -> MutationResult:
        mutation = Mutation(
            id="sham_breach",
            type=MutationType.BOUNDARY_BREACH,
            target_module="mod_c.py",
            description="SHAM: pretends to add forbidden import",
            is_sham=True,
            affected_modules=[],
            affected_invariants=[],
        )
        evidence = EvidenceGenerator.for_boundary_breach(
            "mod_c.py", "mod_a.py", "stages must not import each other"
        )
        return MutationResult(
            mutation=mutation,
            post_gt=self.gt.model_copy(deep=True),
            evidence=evidence,
        )


# ── Belief revision scoring ─────────────────────────────────────────


def score_revision(
    pre_belief: CognitiveMap,
    post_belief: CognitiveMap,
    mutation_result: MutationResult,
    pre_gt: CodebaseGroundTruth,
) -> BeliefRevisionMetrics:
    """Score belief revision after a mutation (or sham).

    Compares pre/post beliefs against pre/post ground truth to compute
    BRS, inertia_proper, impact_discovery, and gullibility_rate.
    """
    mutation = mutation_result.mutation
    post_gt = mutation_result.post_gt

    if mutation.is_sham:
        return _score_sham(pre_belief, post_belief, mutation)

    if mutation.type == MutationType.INTERFACE_BREAK:
        return _score_interface_break(
            pre_belief, post_belief, mutation, pre_gt, post_gt
        )
    return _score_edge_mutation(
        pre_belief, post_belief, mutation, pre_gt, post_gt
    )


def _score_sham(
    pre_belief: CognitiveMap,
    post_belief: CognitiveMap,
    mutation: Mutation,
) -> BeliefRevisionMetrics:
    """Score sham: nothing should change, gullibility = fraction that did."""
    pre_edges = _extract_edge_set(pre_belief)
    post_edges = _extract_edge_set(post_belief)

    changed = pre_edges.symmetric_difference(post_edges)
    total = len(pre_edges | post_edges) or 1
    gullibility = len(changed) / total

    return BeliefRevisionMetrics(
        evidence_condition="sham",
        revision_score=1.0 - gullibility,
        inertia_proper=0.0,
        impact_discovery=0.0,
        gullibility_rate=gullibility,
        revision_latency=0,
        mutation_type=mutation.type,
        pre_mutation_known_count=0,
        pre_mutation_unknown_count=0,
    )


def _score_interface_break(
    pre_belief: CognitiveMap,
    post_belief: CognitiveMap,
    mutation: Mutation,
    pre_gt: CodebaseGroundTruth,
    post_gt: CodebaseGroundTruth,
) -> BeliefRevisionMetrics:
    """Score against the contract layer for interface breaks."""
    affected_contracts: list[ExportedAPI] = []
    for contract in post_gt.contracts:
        if contract.module in mutation.affected_modules:
            affected_contracts.append(contract)

    if not affected_contracts:
        return _empty_revision(mutation)

    pre_known = pre_unknown = 0
    post_correct = pre_known_updated = pre_unknown_discovered = 0

    for post_contract in affected_contracts:
        # Find corresponding pre-gt contract
        pre_contract = _find_contract(pre_gt, post_contract.module, post_contract.name)
        pre_ok = _contract_matches_belief(pre_contract, pre_belief) if pre_contract else False
        post_ok = _contract_matches_belief(post_contract, post_belief)

        if pre_ok:
            pre_known += 1
            if post_ok:
                pre_known_updated += 1
        else:
            pre_unknown += 1
            if post_ok:
                pre_unknown_discovered += 1
        if post_ok:
            post_correct += 1

    total = len(affected_contracts)
    return BeliefRevisionMetrics(
        evidence_condition="real",
        revision_score=post_correct / total,
        inertia_proper=(1.0 - pre_known_updated / pre_known) if pre_known else 0.0,
        impact_discovery=(pre_unknown_discovered / pre_unknown) if pre_unknown else 0.0,
        gullibility_rate=0.0,
        revision_latency=0,
        mutation_type=mutation.type,
        pre_mutation_known_count=pre_known,
        pre_mutation_unknown_count=pre_unknown,
    )


def _score_edge_mutation(
    pre_belief: CognitiveMap,
    post_belief: CognitiveMap,
    mutation: Mutation,
    pre_gt: CodebaseGroundTruth,
    post_gt: CodebaseGroundTruth,
) -> BeliefRevisionMetrics:
    """Score against edge layer for dependency shifts and boundary breaches."""
    pre_gt_edges = {
        (e["source"], e["target"], e["type"]) for e in pre_gt.dependency_edges
    }
    post_gt_edges = {
        (e["source"], e["target"], e["type"]) for e in post_gt.dependency_edges
    }
    affected_edges = pre_gt_edges.symmetric_difference(post_gt_edges)

    if not affected_edges:
        return _empty_revision(mutation)

    pre_belief_edges = _extract_edge_set(pre_belief)
    post_belief_edges = _extract_edge_set(post_belief)

    pre_known = pre_unknown = 0
    post_correct = pre_known_updated = pre_unknown_discovered = 0

    for edge in affected_edges:
        should_pre = edge in pre_gt_edges
        should_post = edge in post_gt_edges
        had_pre = edge in pre_belief_edges
        has_post = edge in post_belief_edges

        pre_ok = had_pre == should_pre
        post_ok = has_post == should_post

        if pre_ok:
            pre_known += 1
            if post_ok:
                pre_known_updated += 1
        else:
            pre_unknown += 1
            if post_ok:
                pre_unknown_discovered += 1
        if post_ok:
            post_correct += 1

    total = len(affected_edges)
    return BeliefRevisionMetrics(
        evidence_condition="real",
        revision_score=post_correct / total,
        inertia_proper=(1.0 - pre_known_updated / pre_known) if pre_known else 0.0,
        impact_discovery=(pre_unknown_discovered / pre_unknown) if pre_unknown else 0.0,
        gullibility_rate=0.0,
        revision_latency=0,
        mutation_type=mutation.type,
        pre_mutation_known_count=pre_known,
        pre_mutation_unknown_count=pre_unknown,
    )


# ── Helpers ─────────────────────────────────────────────────────────


def _extract_edge_set(cmap: CognitiveMap) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    for filepath, comp in cmap.components.items():
        for e in comp.edges:
            etype = e.type.value if hasattr(e.type, "value") else str(e.type)
            edges.add((filepath, e.target, etype))
    return edges


def _find_callers(
    gt: CodebaseGroundTruth, module: str, function_name: str
) -> list[str]:
    for contract in gt.contracts:
        if contract.module == module and contract.name == function_name:
            return list(contract.callers)
    return []


def _find_contract(
    gt: CodebaseGroundTruth, module: str, name: str
) -> ExportedAPI | None:
    for c in gt.contracts:
        if c.module == module and c.name == name:
            return c
    return None


def _contract_matches_belief(
    contract: ExportedAPI, belief: CognitiveMap
) -> bool:
    comp = belief.components.get(contract.module)
    if not comp:
        return False
    for export in comp.exports:
        if export.name == contract.name:
            if len(export.signature.params) == len(contract.signature.params):
                return True
    return False


def _empty_revision(mutation: Mutation) -> BeliefRevisionMetrics:
    return BeliefRevisionMetrics(
        evidence_condition="real",
        revision_score=0.0,
        inertia_proper=0.0,
        impact_discovery=0.0,
        gullibility_rate=0.0,
        revision_latency=0,
        mutation_type=mutation.type,
        pre_mutation_known_count=0,
        pre_mutation_unknown_count=0,
    )
