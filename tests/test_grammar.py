"""Tests for generator.grammar — Pipeline pattern template."""

import pytest

from models import ArchPattern, ComplexityTier, InvariantType
from generator.grammar import (
    PipelineTemplate,
    PatternBlueprint,
    get_template,
    list_patterns,
)


@pytest.fixture
def blueprint() -> PatternBlueprint:
    """Generate a small pipeline blueprint with fixed seed."""
    return PipelineTemplate().generate(ComplexityTier.SMALL, seed=42)


@pytest.fixture
def medium_blueprint() -> PatternBlueprint:
    """Generate a medium pipeline blueprint with fixed seed."""
    return PipelineTemplate().generate(ComplexityTier.MEDIUM, seed=42)


class TestPipelineBlueprint:
    def test_metadata(self, blueprint: PatternBlueprint) -> None:
        assert blueprint.pattern == ArchPattern.PIPELINE
        assert blueprint.complexity == ComplexityTier.SMALL
        assert blueprint.package_name  # non-empty
        assert blueprint.domain in ("data_etl", "log_processing", "text_processing")

    def test_module_count(self, blueprint: PatternBlueprint) -> None:
        # Small pipeline: __init__ + models + config + utils + source
        #   + 4 stages + sink + pipeline = 11 modules
        assert 10 <= len(blueprint.modules) <= 13

    def test_has_required_roles(self, blueprint: PatternBlueprint) -> None:
        roles = {m.role for m in blueprint.modules.values()}
        assert "orchestrator" in roles
        assert "source" in roles
        assert "sink" in roles
        assert "stage" in roles
        assert "data_model" in roles
        assert "config" in roles

    def test_stage_ordering(self, blueprint: PatternBlueprint) -> None:
        stages = [
            m for m in blueprint.modules.values() if m.role == "stage"
        ]
        orders = sorted(s.order for s in stages if s.order is not None)
        assert orders == list(range(1, len(stages) + 1))

    def test_source_has_order_zero(self, blueprint: PatternBlueprint) -> None:
        sources = [m for m in blueprint.modules.values() if m.role == "source"]
        assert len(sources) == 1
        assert sources[0].order == 0

    def test_orchestrator_imports_all_stages(
        self, blueprint: PatternBlueprint
    ) -> None:
        orch = next(
            m for m in blueprint.modules.values() if m.role == "orchestrator"
        )
        stage_paths = {
            m.filepath
            for m in blueprint.modules.values()
            if m.role in ("stage", "source", "sink")
        }
        assert stage_paths.issubset(set(orch.imports_from))

    def test_stages_import_models_not_each_other(
        self, blueprint: PatternBlueprint
    ) -> None:
        models_path = next(
            m.filepath
            for m in blueprint.modules.values()
            if m.role == "data_model"
        )
        stage_paths = {
            m.filepath for m in blueprint.modules.values() if m.role == "stage"
        }
        for m in blueprint.modules.values():
            if m.role == "stage":
                assert models_path in m.imports_from
                # Stages should NOT import from other stages
                other_stages = stage_paths - {m.filepath}
                assert not other_stages.intersection(m.imports_from), (
                    f"{m.filepath} imports from another stage"
                )

    def test_dependency_edges_are_consistent(
        self, blueprint: PatternBlueprint
    ) -> None:
        for src, tgt, etype in blueprint.dependency_edges:
            assert src in blueprint.modules, f"Edge source {src} not in modules"
            assert tgt in blueprint.modules, f"Edge target {tgt} not in modules"
            if etype == "IMPORTS":
                assert tgt in blueprint.modules[src].imports_from, (
                    f"IMPORTS edge ({src} → {tgt}) not reflected in imports_from"
                )

    def test_no_self_loops(self, blueprint: PatternBlueprint) -> None:
        for src, tgt, _etype in blueprint.dependency_edges:
            assert src != tgt, f"Self-loop on {src}"

    def test_all_small_edges_are_imports(self, blueprint: PatternBlueprint) -> None:
        """Small blueprints only have IMPORTS edges."""
        for _src, _tgt, etype in blueprint.dependency_edges:
            assert etype == "IMPORTS", f"Small blueprint has non-IMPORTS edge: {etype}"


class TestInvariants:
    def test_has_invariants(self, blueprint: PatternBlueprint) -> None:
        assert len(blueprint.invariants) > 0

    def test_invariant_types_covered(self, blueprint: PatternBlueprint) -> None:
        types = {inv.type for inv in blueprint.invariants}
        assert InvariantType.DATAFLOW in types
        assert InvariantType.INTERFACE in types
        assert InvariantType.BOUNDARY in types
        assert InvariantType.INVARIANT in types
        assert InvariantType.PURPOSE in types

    def test_invariant_ids_unique(self, blueprint: PatternBlueprint) -> None:
        ids = [inv.id for inv in blueprint.invariants]
        assert len(ids) == len(set(ids))

    def test_invariant_modules_exist(self, blueprint: PatternBlueprint) -> None:
        for inv in blueprint.invariants:
            for mod in inv.involved_modules:
                assert mod in blueprint.modules, (
                    f"Invariant {inv.id} references unknown module {mod}"
                )


class TestDesignRationales:
    def test_has_rationales(self, blueprint: PatternBlueprint) -> None:
        assert len(blueprint.design_rationales) >= 3

    def test_rationale_ids_unique(self, blueprint: PatternBlueprint) -> None:
        ids = [r.id for r in blueprint.design_rationales]
        assert len(ids) == len(set(ids))

    def test_rationales_have_downstream_effects(
        self, blueprint: PatternBlueprint
    ) -> None:
        for r in blueprint.design_rationales:
            assert len(r.downstream_effects) > 0, (
                f"Rationale {r.id} has no downstream effects"
            )

    def test_rationale_modules_exist(self, blueprint: PatternBlueprint) -> None:
        for r in blueprint.design_rationales:
            for mod in r.affected_modules:
                assert mod in blueprint.modules, (
                    f"Rationale {r.id} references unknown module {mod}"
                )


class TestDeterminism:
    def test_same_seed_same_blueprint(self) -> None:
        t = PipelineTemplate()
        b1 = t.generate(ComplexityTier.SMALL, seed=123)
        b2 = t.generate(ComplexityTier.SMALL, seed=123)
        assert b1.domain == b2.domain
        assert b1.package_name == b2.package_name
        assert set(b1.modules.keys()) == set(b2.modules.keys())

    def test_different_seeds_can_differ(self) -> None:
        t = PipelineTemplate()
        domains = set()
        for seed in range(20):
            b = t.generate(ComplexityTier.SMALL, seed=seed)
            domains.add(b.domain)
        # With 3 domains and 20 seeds, we should hit at least 2
        assert len(domains) >= 2


class TestRegistry:
    def test_get_pipeline_template(self) -> None:
        t = get_template(ArchPattern.PIPELINE)
        assert isinstance(t, PipelineTemplate)

    def test_unsupported_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="No template"):
            get_template(ArchPattern.MVC)

    def test_list_patterns(self) -> None:
        patterns = list_patterns()
        assert ArchPattern.PIPELINE in patterns


# ======================================================================
# Medium blueprint tests
# ======================================================================


class TestMediumBlueprint:
    def test_metadata(self, medium_blueprint: PatternBlueprint) -> None:
        assert medium_blueprint.pattern == ArchPattern.PIPELINE
        assert medium_blueprint.complexity == ComplexityTier.MEDIUM
        assert medium_blueprint.package_name  # non-empty

    def test_module_count(self, medium_blueprint: PatternBlueprint) -> None:
        count = len(medium_blueprint.modules)
        assert 25 <= count <= 35, f"Medium blueprint has {count} modules"

    def test_edge_count(self, medium_blueprint: PatternBlueprint) -> None:
        count = len(medium_blueprint.dependency_edges)
        assert 55 <= count <= 85, f"Medium blueprint has {count} edges"

    def test_all_four_edge_types_present(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        types = {etype for _, _, etype in medium_blueprint.dependency_edges}
        assert "IMPORTS" in types, "Missing IMPORTS edges"
        assert "CALLS_API" in types, "Missing CALLS_API edges"
        assert "DATA_FLOWS_TO" in types, "Missing DATA_FLOWS_TO edges"
        assert "REGISTRY_WIRES" in types, "Missing REGISTRY_WIRES edges"

    def test_imports_are_not_everything(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        """IMPORTS should be < 70% of all edges (not everything import-discoverable)."""
        total = len(medium_blueprint.dependency_edges)
        imports_count = sum(
            1 for _, _, etype in medium_blueprint.dependency_edges
            if etype == "IMPORTS"
        )
        ratio = imports_count / total
        assert ratio < 0.70, f"IMPORTS ratio too high: {ratio:.2f}"
        # Non-IMPORTS edges should be ≥ 20 (meaningful for partial observability)
        non_imports = total - imports_count
        assert non_imports >= 20, f"Only {non_imports} non-IMPORTS edges"

    def test_has_subdirectories(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        """Medium codebases must have sub-package directories."""
        pkg = medium_blueprint.package_name
        dirs = set()
        for fp in medium_blueprint.modules:
            parts = fp.split("/")
            if len(parts) > 2:
                dirs.add(parts[1])
        assert "stages" in dirs
        assert "adapters" in dirs
        assert "middleware" in dirs
        assert "utils" in dirs
        assert "legacy" in dirs

    def test_has_required_roles(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        roles = {m.role for m in medium_blueprint.modules.values()}
        assert "orchestrator" in roles
        assert "stage" in roles
        assert "data_model" in roles
        assert "config" in roles
        assert "base" in roles
        assert "registry" in roles
        assert "adapter" in roles
        assert "middleware" in roles
        assert "utils" in roles
        assert "distractor" in roles
        assert "entry_point" in roles

    def test_stage_count(self, medium_blueprint: PatternBlueprint) -> None:
        stages = [m for m in medium_blueprint.modules.values() if m.role == "stage"]
        assert 6 <= len(stages) <= 8

    def test_stage_ordering(self, medium_blueprint: PatternBlueprint) -> None:
        stages = [m for m in medium_blueprint.modules.values() if m.role == "stage"]
        orders = sorted(s.order for s in stages if s.order is not None)
        assert orders == list(range(1, len(stages) + 1))

    def test_neutral_naming(self, medium_blueprint: PatternBlueprint) -> None:
        """Stage, adapter, and middleware files should use neutral names."""
        for m in medium_blueprint.modules.values():
            if m.role in ("stage", "adapter", "middleware"):
                filename = m.filepath.split("/")[-1]
                assert filename.startswith("mod_"), (
                    f"{m.filepath} doesn't use neutral naming"
                )

    def test_stages_import_base_not_each_other(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        base_path = next(
            m.filepath for m in medium_blueprint.modules.values()
            if m.role == "base"
        )
        stage_paths = {
            m.filepath for m in medium_blueprint.modules.values()
            if m.role == "stage"
        }
        for m in medium_blueprint.modules.values():
            if m.role == "stage":
                assert base_path in m.imports_from
                other_stages = stage_paths - {m.filepath}
                assert not other_stages.intersection(m.imports_from), (
                    f"{m.filepath} imports from another stage"
                )

    def test_adapters_wrap_stages(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        """Each adapter should have a CALLS_API edge to a stage."""
        adapter_paths = {
            m.filepath for m in medium_blueprint.modules.values()
            if m.role == "adapter"
        }
        stage_paths = {
            m.filepath for m in medium_blueprint.modules.values()
            if m.role == "stage"
        }
        for ap in adapter_paths:
            calls = [
                tgt for src, tgt, etype in medium_blueprint.dependency_edges
                if src == ap and etype == "CALLS_API"
            ]
            assert any(c in stage_paths for c in calls), (
                f"Adapter {ap} doesn't CALLS_API any stage"
            )

    def test_registry_wires_to_stages(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        """Registry should have REGISTRY_WIRES edges to all stages."""
        stage_paths = {
            m.filepath for m in medium_blueprint.modules.values()
            if m.role == "stage"
        }
        wired = {
            tgt for _, tgt, etype in medium_blueprint.dependency_edges
            if etype == "REGISTRY_WIRES"
        }
        assert stage_paths.issubset(wired), (
            f"Stages not wired: {stage_paths - wired}"
        )

    def test_distractors_have_no_incoming_edges(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        distractor_paths = {
            m.filepath for m in medium_blueprint.modules.values()
            if m.role == "distractor"
        }
        for _, tgt, _ in medium_blueprint.dependency_edges:
            assert tgt not in distractor_paths, (
                f"Distractor {tgt} has incoming edge"
            )

    def test_data_flows_to_is_linear(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        """DATA_FLOWS_TO edges should form a linear chain through stages."""
        df_edges = [
            (src, tgt) for src, tgt, etype in medium_blueprint.dependency_edges
            if etype == "DATA_FLOWS_TO"
        ]
        stage_paths = [
            m.filepath for m in medium_blueprint.modules.values()
            if m.role == "stage"
        ]
        # Should have len(stages)-1 data flow edges
        assert len(df_edges) == len(stage_paths) - 1

    def test_edges_reference_known_modules(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        for src, tgt, etype in medium_blueprint.dependency_edges:
            assert src in medium_blueprint.modules, (
                f"Edge source {src} not in modules"
            )
            assert tgt in medium_blueprint.modules, (
                f"Edge target {tgt} not in modules"
            )


class TestMediumInvariants:
    def test_invariant_count(self, medium_blueprint: PatternBlueprint) -> None:
        count = len(medium_blueprint.invariants)
        assert 8 <= count <= 16, f"Medium blueprint has {count} invariants"

    def test_invariant_types_covered(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        types = {inv.type for inv in medium_blueprint.invariants}
        assert InvariantType.DATAFLOW in types
        assert InvariantType.INTERFACE in types
        assert InvariantType.BOUNDARY in types
        assert InvariantType.INVARIANT in types
        assert InvariantType.PURPOSE in types

    def test_invariant_ids_unique(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        ids = [inv.id for inv in medium_blueprint.invariants]
        assert len(ids) == len(set(ids))

    def test_invariant_modules_exist(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        for inv in medium_blueprint.invariants:
            for mod in inv.involved_modules:
                assert mod in medium_blueprint.modules, (
                    f"Invariant {inv.id} references unknown module {mod}"
                )


class TestMediumRationales:
    def test_has_rationales(self, medium_blueprint: PatternBlueprint) -> None:
        assert len(medium_blueprint.design_rationales) >= 4

    def test_rationale_modules_exist(
        self, medium_blueprint: PatternBlueprint
    ) -> None:
        for r in medium_blueprint.design_rationales:
            for mod in r.affected_modules:
                assert mod in medium_blueprint.modules, (
                    f"Rationale {r.id} references unknown module {mod}"
                )


class TestMediumDeterminism:
    def test_same_seed_same_blueprint(self) -> None:
        t = PipelineTemplate()
        b1 = t.generate(ComplexityTier.MEDIUM, seed=42)
        b2 = t.generate(ComplexityTier.MEDIUM, seed=42)
        assert b1.domain == b2.domain
        assert set(b1.modules.keys()) == set(b2.modules.keys())
        assert len(b1.dependency_edges) == len(b2.dependency_edges)

    def test_different_seeds_produce_variation(self) -> None:
        """Different seeds should produce structurally different codebases."""
        t = PipelineTemplate()
        stage_counts = set()
        for seed in range(20):
            b = t.generate(ComplexityTier.MEDIUM, seed=seed)
            sc = sum(1 for m in b.modules.values() if m.role == "stage")
            stage_counts.add(sc)
        # With 20 seeds and range 6-8, we should see at least 2 different counts
        assert len(stage_counts) >= 2, (
            f"All 20 seeds produced the same stage count: {stage_counts}"
        )
