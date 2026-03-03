"""Architecture pattern grammar for procedural codebase generation.

Defines architectural patterns as generative templates that produce
PatternBlueprint instances — complete specifications for a codebase
including modules, dependencies, invariants, and design rationales.

v0.1: Pipeline pattern only (hardcoded). Will be made data-driven later.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from models import ArchPattern, ComplexityTier, EdgeType, InvariantType

# Neutral module names (mod_a, mod_b, ...) to prevent agents from
# inferring purpose from filenames alone.
_NEUTRAL_NAMES = [f"mod_{chr(ord('a') + i)}" for i in range(26)]


# ============================================================================
# Blueprint data structures (output of the grammar)
# ============================================================================


class ModuleSpec(BaseModel):
    """Specification for a single module in the generated codebase."""

    filepath: str
    role: str = Field(
        description="Role in the pattern (e.g., 'stage', 'orchestrator')"
    )
    purpose: str = Field(description="Why this module exists")
    exports: list[str] = Field(default_factory=list)
    imports_from: list[str] = Field(default_factory=list)
    order: int | None = Field(
        default=None, description="Stage order for pipeline patterns"
    )
    generation_hints: dict[str, str] = Field(
        default_factory=dict,
        description="Hints for codegen (e.g., input_type, output_type)",
    )


class InvariantSpec(BaseModel):
    """Specification for a planted invariant."""

    id: str
    type: InvariantType
    description: str
    involved_modules: list[str]
    rationale: str


class DesignRationaleSpec(BaseModel):
    """Specification for a planted design rationale (intentionality probing)."""

    id: str
    question: str
    answer: str
    affected_modules: list[str]
    downstream_effects: list[str]


class PatternBlueprint(BaseModel):
    """Complete blueprint for generating a codebase.

    Contains everything codegen.py and export.py need:
    - Module specs with roles, dependencies, and generation hints
    - Invariants to plant across modules
    - Design rationales for intentionality probing
    - The full dependency edge list for ground truth
    """

    pattern: ArchPattern
    complexity: ComplexityTier
    package_name: str = Field(description="Top-level package name")
    domain: str = Field(description="Thematic domain (e.g., 'data_etl')")
    domain_description: str = Field(description="What this codebase does")
    modules: dict[str, ModuleSpec]
    invariants: list[InvariantSpec]
    design_rationales: list[DesignRationaleSpec]
    dependency_edges: list[tuple[str, str, str]] = Field(
        description="(source, target, edge_type) directed typed dependency edges"
    )


# ============================================================================
# Pattern template interface
# ============================================================================


class PatternTemplate(ABC):
    """Base class for architectural pattern templates."""

    pattern: ArchPattern

    @abstractmethod
    def generate(
        self,
        complexity: ComplexityTier = ComplexityTier.SMALL,
        seed: int | None = None,
    ) -> PatternBlueprint:
        """Generate a complete blueprint for this pattern."""
        ...


# ============================================================================
# Pipeline domain definitions
# ============================================================================


@dataclass
class StageInfo:
    """Definition of a single pipeline processing stage."""

    module_name: str  # e.g., "stage_clean"
    function_name: str  # e.g., "clean_records"
    input_type: str  # e.g., "RawRecord"
    output_type: str  # e.g., "CleanedRecord"
    purpose: str  # human-readable description


@dataclass
class PipelineDomain:
    """A thematic domain for pipeline generation."""

    name: str  # e.g., "data_etl"
    package_name: str  # e.g., "data_pipeline"
    description: str
    source_description: str  # what the source reads
    sink_description: str  # what the sink writes
    raw_type: str  # type emitted by source
    stages: list[StageInfo] = field(default_factory=list)


# --- Domain catalog (hardcoded for v0.1) ---

DATA_ETL_DOMAIN = PipelineDomain(
    name="data_etl",
    package_name="data_pipeline",
    description="ETL pipeline that ingests CSV records, cleans, transforms, "
    "enriches, validates, and writes structured output.",
    source_description="Reads raw CSV rows into RawRecord objects",
    sink_description="Writes validated records as JSON lines",
    raw_type="RawRecord",
    stages=[
        StageInfo(
            module_name="stage_ingest",
            function_name="ingest_records",
            input_type="RawRecord",
            output_type="IngestedRecord",
            purpose="Reads raw input, parses format, emits typed records",
        ),
        StageInfo(
            module_name="stage_validate",
            function_name="validate_records",
            input_type="IngestedRecord",
            output_type="ValidatedRecord",
            purpose="Validates required fields, checks types, drops malformed rows",
        ),
        StageInfo(
            module_name="stage_normalize",
            function_name="normalize_records",
            input_type="ValidatedRecord",
            output_type="NormalizedRecord",
            purpose="Standardizes formats, trims whitespace, applies canonical forms",
        ),
        StageInfo(
            module_name="stage_deduplicate",
            function_name="deduplicate_records",
            input_type="NormalizedRecord",
            output_type="DeduplicatedRecord",
            purpose="Removes exact and fuzzy duplicate records by key fields",
        ),
        StageInfo(
            module_name="stage_enrich",
            function_name="enrich_records",
            input_type="DeduplicatedRecord",
            output_type="EnrichedRecord",
            purpose="Joins external lookup data (e.g., geo-IP, currency rates)",
        ),
        StageInfo(
            module_name="stage_transform",
            function_name="transform_records",
            input_type="EnrichedRecord",
            output_type="TransformedRecord",
            purpose="Derives computed fields, converts types, applies business logic",
        ),
        StageInfo(
            module_name="stage_aggregate",
            function_name="aggregate_records",
            input_type="TransformedRecord",
            output_type="AggregatedRecord",
            purpose="Groups records by key dimensions and computes summary statistics",
        ),
        StageInfo(
            module_name="stage_export",
            function_name="export_records",
            input_type="AggregatedRecord",
            output_type="ExportedRecord",
            purpose="Formats aggregated records for downstream consumption",
        ),
    ],
)

LOG_PROCESSING_DOMAIN = PipelineDomain(
    name="log_processing",
    package_name="log_processor",
    description="Log processing pipeline that ingests raw log lines, parses, "
    "filters, aggregates, and generates alert events.",
    source_description="Reads raw log lines from files or stdin",
    sink_description="Writes alert events and aggregated metrics as JSON",
    raw_type="RawLogEntry",
    stages=[
        StageInfo(
            module_name="stage_collect",
            function_name="collect_entries",
            input_type="RawLogEntry",
            output_type="CollectedEntry",
            purpose="Reads raw log lines from files or streams into typed entries",
        ),
        StageInfo(
            module_name="stage_parse",
            function_name="parse_entries",
            input_type="CollectedEntry",
            output_type="ParsedLogEntry",
            purpose="Extracts timestamp, level, message, and metadata from raw lines",
        ),
        StageInfo(
            module_name="stage_filter",
            function_name="filter_entries",
            input_type="ParsedLogEntry",
            output_type="FilteredLogEntry",
            purpose="Drops noise (debug, heartbeat) and keeps actionable entries",
        ),
        StageInfo(
            module_name="stage_correlate",
            function_name="correlate_entries",
            input_type="FilteredLogEntry",
            output_type="CorrelatedEntry",
            purpose="Groups related log entries by request ID or session",
        ),
        StageInfo(
            module_name="stage_classify",
            function_name="classify_entries",
            input_type="CorrelatedEntry",
            output_type="ClassifiedEntry",
            purpose="Assigns severity and category labels to correlated entries",
        ),
        StageInfo(
            module_name="stage_aggregate",
            function_name="aggregate_entries",
            input_type="ClassifiedEntry",
            output_type="AggregatedMetric",
            purpose="Groups by time window and category, computes counts and rates",
        ),
        StageInfo(
            module_name="stage_alert",
            function_name="check_alerts",
            input_type="AggregatedMetric",
            output_type="AlertEvent",
            purpose="Checks aggregated metrics against thresholds, emits alerts",
        ),
        StageInfo(
            module_name="stage_archive",
            function_name="archive_entries",
            input_type="AlertEvent",
            output_type="ArchivedEvent",
            purpose="Persists alert events to long-term storage with retention policy",
        ),
    ],
)

TEXT_PROCESSING_DOMAIN = PipelineDomain(
    name="text_processing",
    package_name="text_processor",
    description="NLP pipeline that reads documents, tokenizes, normalizes, "
    "analyzes sentiment/entities, and produces summaries.",
    source_description="Reads raw text documents from files",
    sink_description="Writes analysis results and summaries as JSON",
    raw_type="RawDocument",
    stages=[
        StageInfo(
            module_name="stage_tokenize",
            function_name="tokenize_documents",
            input_type="RawDocument",
            output_type="TokenizedDocument",
            purpose="Splits text into sentences and tokens with position offsets",
        ),
        StageInfo(
            module_name="stage_normalize",
            function_name="normalize_documents",
            input_type="TokenizedDocument",
            output_type="NormalizedDocument",
            purpose="Lowercases, applies Unicode normalization, strips accents",
        ),
        StageInfo(
            module_name="stage_stopwords",
            function_name="remove_stopwords",
            input_type="NormalizedDocument",
            output_type="FilteredDocument",
            purpose="Removes stopwords and high-frequency noise tokens",
        ),
        StageInfo(
            module_name="stage_lemmatize",
            function_name="lemmatize_documents",
            input_type="FilteredDocument",
            output_type="LemmatizedDocument",
            purpose="Reduces tokens to lemma forms using morphological analysis",
        ),
        StageInfo(
            module_name="stage_vectorize",
            function_name="vectorize_documents",
            input_type="LemmatizedDocument",
            output_type="VectorizedDocument",
            purpose="Converts token sequences to TF-IDF or embedding vectors",
        ),
        StageInfo(
            module_name="stage_classify",
            function_name="classify_documents",
            input_type="VectorizedDocument",
            output_type="ClassifiedDocument",
            purpose="Assigns topic and category labels using trained classifier",
        ),
        StageInfo(
            module_name="stage_summarize",
            function_name="summarize_documents",
            input_type="ClassifiedDocument",
            output_type="SummarizedDocument",
            purpose="Generates extractive summaries from classified documents",
        ),
        StageInfo(
            module_name="stage_index",
            function_name="index_documents",
            input_type="SummarizedDocument",
            output_type="IndexedDocument",
            purpose="Builds inverted index entries for search and retrieval",
        ),
    ],
)

PIPELINE_DOMAINS: list[PipelineDomain] = [
    DATA_ETL_DOMAIN,
    LOG_PROCESSING_DOMAIN,
    TEXT_PROCESSING_DOMAIN,
]


# ============================================================================
# Pipeline pattern template
# ============================================================================


class PipelineTemplate(PatternTemplate):
    """Generates Pipeline-pattern codebase blueprints.

    A pipeline is a linear chain of processing stages:
        source → stage_1 → stage_2 → ... → stage_n → sink

    Each stage receives the output of the previous stage and produces
    typed output for the next. An orchestrator module chains them together.

    Supporting modules: shared data models, configuration, utilities.
    """

    pattern = ArchPattern.PIPELINE

    def generate(
        self,
        complexity: ComplexityTier = ComplexityTier.SMALL,
        seed: int | None = None,
    ) -> PatternBlueprint:
        rng = random.Random(seed)

        # Pick a random domain
        domain = rng.choice(PIPELINE_DOMAINS)
        pkg = domain.package_name

        # Medium tier uses a different generation path
        if complexity == ComplexityTier.MEDIUM:
            return self._generate_medium(domain, pkg, rng)

        # Select stages based on complexity
        stages = self._select_stages(domain, complexity, rng)

        # Build all module specs
        modules: dict[str, ModuleSpec] = {}
        edges: list[tuple[str, str, str]] = []

        # Collect all type names for the models module
        all_types = [domain.raw_type]
        for s in stages:
            if s.output_type not in all_types:
                all_types.append(s.output_type)

        # --- __init__.py ---
        init_path = f"{pkg}/__init__.py"
        modules[init_path] = ModuleSpec(
            filepath=init_path,
            role="package_init",
            purpose=f"Package root for {domain.description.split(',')[0].lower()}",
            exports=[],
            imports_from=[],
        )

        # --- models.py (shared data types) ---
        models_path = f"{pkg}/models.py"
        modules[models_path] = ModuleSpec(
            filepath=models_path,
            role="data_model",
            purpose="Defines all data types that flow through the pipeline stages",
            exports=all_types,
            imports_from=[],
            generation_hints={
                "types": ",".join(all_types),
                "base_class": "BaseModel",
            },
        )

        # --- config.py ---
        config_path = f"{pkg}/config.py"
        modules[config_path] = ModuleSpec(
            filepath=config_path,
            role="config",
            purpose="Pipeline configuration: batch size, paths, feature flags",
            exports=["PipelineConfig"],
            imports_from=[],
            generation_hints={
                "config_fields": "batch_size,input_path,output_path,log_level",
            },
        )

        # --- utils.py ---
        utils_path = f"{pkg}/utils.py"
        modules[utils_path] = ModuleSpec(
            filepath=utils_path,
            role="utils",
            purpose="Shared utilities: logging setup, timing decorator, error helpers",
            exports=["setup_logging", "log_stage", "timed"],
            imports_from=[config_path],
            generation_hints={},
        )
        edges.append((utils_path, config_path, "IMPORTS"))

        # --- source.py ---
        source_path = f"{pkg}/source.py"
        first_stage_input = stages[0].input_type
        modules[source_path] = ModuleSpec(
            filepath=source_path,
            role="source",
            purpose=domain.source_description,
            exports=["read_source"],
            imports_from=[models_path, config_path, utils_path],
            order=0,
            generation_hints={
                "output_type": first_stage_input,
                "description": domain.source_description,
            },
        )
        edges.extend([
            (source_path, models_path, "IMPORTS"),
            (source_path, config_path, "IMPORTS"),
            (source_path, utils_path, "IMPORTS"),
        ])

        # --- Processing stages ---
        stage_paths: list[str] = []
        for i, stage in enumerate(stages):
            stage_path = f"{pkg}/{stage.module_name}.py"
            stage_paths.append(stage_path)
            modules[stage_path] = ModuleSpec(
                filepath=stage_path,
                role="stage",
                purpose=stage.purpose,
                exports=[stage.function_name],
                imports_from=[models_path, config_path, utils_path],
                order=i + 1,
                generation_hints={
                    "function_name": stage.function_name,
                    "input_type": stage.input_type,
                    "output_type": stage.output_type,
                    "description": stage.purpose,
                },
            )
            edges.extend([
                (stage_path, models_path, "IMPORTS"),
                (stage_path, config_path, "IMPORTS"),
                (stage_path, utils_path, "IMPORTS"),
            ])

        # --- sink.py ---
        sink_path = f"{pkg}/sink.py"
        last_stage_output = stages[-1].output_type
        modules[sink_path] = ModuleSpec(
            filepath=sink_path,
            role="sink",
            purpose=domain.sink_description,
            exports=["write_output"],
            imports_from=[models_path, config_path, utils_path],
            order=len(stages) + 1,
            generation_hints={
                "input_type": last_stage_output,
                "description": domain.sink_description,
            },
        )
        edges.extend([
            (sink_path, models_path, "IMPORTS"),
            (sink_path, config_path, "IMPORTS"),
            (sink_path, utils_path, "IMPORTS"),
        ])

        # --- pipeline.py (orchestrator) ---
        pipeline_path = f"{pkg}/pipeline.py"
        orchestrator_imports = [
            models_path,
            config_path,
            utils_path,
            source_path,
            *stage_paths,
            sink_path,
        ]
        modules[pipeline_path] = ModuleSpec(
            filepath=pipeline_path,
            role="orchestrator",
            purpose="Chains all stages into a single pipeline: source → stages → sink",
            exports=["run_pipeline"],
            imports_from=orchestrator_imports,
            generation_hints={
                "stage_order": ",".join(
                    [source_path] + stage_paths + [sink_path]
                ),
                "stage_functions": ",".join(
                    ["read_source"]
                    + [s.function_name for s in stages]
                    + ["write_output"]
                ),
            },
        )
        for dep in orchestrator_imports:
            edges.append((pipeline_path, dep, "IMPORTS"))

        # Build invariants and design rationales
        all_stage_paths = [source_path] + stage_paths + [sink_path]
        invariants = self._build_invariants(
            pkg, models_path, config_path, pipeline_path,
            source_path, sink_path, stage_paths, stages, all_stage_paths,
        )
        rationales = self._build_design_rationales(
            pkg, models_path, config_path, pipeline_path, utils_path,
            source_path, sink_path, stage_paths, stages, all_stage_paths,
        )

        return PatternBlueprint(
            pattern=ArchPattern.PIPELINE,
            complexity=complexity,
            package_name=pkg,
            domain=domain.name,
            domain_description=domain.description,
            modules=modules,
            invariants=invariants,
            design_rationales=rationales,
            dependency_edges=edges,
        )

    def _select_stages(
        self,
        domain: PipelineDomain,
        complexity: ComplexityTier,
        rng: random.Random,
    ) -> list[StageInfo]:
        """Select stages based on complexity tier.

        Small: first 4 domain stages (producing 10-12 files total).
        Medium: 6-8 stages from pool of 8 (producing ~30 files total).
        """
        if complexity == ComplexityTier.SMALL:
            # Use first 4 stages for backward compat with existing tests
            return list(domain.stages[:4])
        if complexity == ComplexityTier.MEDIUM:
            all_stages = list(domain.stages)
            num = rng.randint(6, min(8, len(all_stages)))
            # Always include first and last for source/sink continuity;
            # sample the rest from the middle stages.
            if num >= len(all_stages):
                return all_stages
            middle = all_stages[1:-1]
            chosen_middle = sorted(
                rng.sample(middle, num - 2),
                key=lambda s: middle.index(s),
            )
            return [all_stages[0]] + chosen_middle + [all_stages[-1]]
        return list(domain.stages[:4])

    # ------------------------------------------------------------------ #
    # MEDIUM-tier generation
    # ------------------------------------------------------------------ #

    def _generate_medium(
        self,
        domain: PipelineDomain,
        pkg: str,
        rng: random.Random,
    ) -> PatternBlueprint:
        """Generate a ~30-file pipeline blueprint with sub-packages.

        Layout:
            {pkg}/                      — package root
            {pkg}/models.py             — shared data types
            {pkg}/base.py               — StageBase ABC
            {pkg}/config.py             — PipelineConfig
            {pkg}/exceptions.py         — custom exceptions
            {pkg}/registry.py           — stage registry (importlib)
            {pkg}/stages/mod_{a..h}.py  — processing stages (neutral names)
            {pkg}/adapters/mod_{i..k}.py— adapter wrappers
            {pkg}/middleware/mod_{l..m}.py — logging / retry middleware
            {pkg}/utils/helpers.py      — shared utilities
            {pkg}/utils/formatters.py   — output formatters
            {pkg}/legacy/old_pipeline.py— dead code (distractor)
            {pkg}/legacy/compat.py      — compat shims (distractor)
            {pkg}/runner.py             — pipeline runner
            {pkg}/cli.py                — CLI entry point
        """
        stages = self._select_stages(domain, ComplexityTier.MEDIUM, rng)
        modules, edges = self._build_medium_modules(domain, pkg, stages, rng)
        invariants = self._build_medium_invariants(
            pkg, modules, edges, stages, rng,
        )
        rationales = self._build_medium_rationales(pkg, modules, stages)

        return PatternBlueprint(
            pattern=ArchPattern.PIPELINE,
            complexity=ComplexityTier.MEDIUM,
            package_name=pkg,
            domain=domain.name,
            domain_description=domain.description,
            modules=modules,
            invariants=invariants,
            design_rationales=rationales,
            dependency_edges=edges,
        )

    def _build_medium_modules(
        self,
        domain: PipelineDomain,
        pkg: str,
        stages: list[StageInfo],
        rng: random.Random,
    ) -> tuple[dict[str, ModuleSpec], list[tuple[str, str, str]]]:
        """Build modules and edges for a medium-tier pipeline blueprint."""
        modules: dict[str, ModuleSpec] = {}
        edges: list[tuple[str, str, str]] = []
        name_idx = 0  # counter for neutral module names

        # Collect all data types
        all_types = [domain.raw_type]
        for s in stages:
            if s.output_type not in all_types:
                all_types.append(s.output_type)

        # ── Package __init__.py ──────────────────────────────────────
        init_path = f"{pkg}/__init__.py"
        modules[init_path] = ModuleSpec(
            filepath=init_path,
            role="package_init",
            purpose=f"Package root for {domain.description.split(',')[0].lower()}",
            exports=[],
            imports_from=[],
        )

        # ── Infrastructure ───────────────────────────────────────────
        models_path = f"{pkg}/models.py"
        modules[models_path] = ModuleSpec(
            filepath=models_path,
            role="data_model",
            purpose="Defines all data types that flow through the pipeline stages",
            exports=all_types,
            imports_from=[],
            generation_hints={
                "types": ",".join(all_types),
                "base_class": "BaseModel",
            },
        )

        base_path = f"{pkg}/base.py"
        modules[base_path] = ModuleSpec(
            filepath=base_path,
            role="base",
            purpose="Abstract base class (StageBase) that all pipeline stages implement",
            exports=["StageBase"],
            imports_from=[models_path],
            generation_hints={"pattern": "abc"},
        )
        edges.append((base_path, models_path, "IMPORTS"))

        config_path = f"{pkg}/config.py"
        modules[config_path] = ModuleSpec(
            filepath=config_path,
            role="config",
            purpose="Pipeline configuration: batch size, paths, feature flags, stage ordering",
            exports=["PipelineConfig"],
            imports_from=[],
            generation_hints={
                "config_fields": "batch_size,input_path,output_path,log_level,stage_order",
            },
        )

        exceptions_path = f"{pkg}/exceptions.py"
        modules[exceptions_path] = ModuleSpec(
            filepath=exceptions_path,
            role="exceptions",
            purpose="Custom exception hierarchy for pipeline errors",
            exports=["PipelineError", "StageError", "ValidationError"],
            imports_from=[],
            generation_hints={},
        )

        registry_path = f"{pkg}/registry.py"
        modules[registry_path] = ModuleSpec(
            filepath=registry_path,
            role="registry",
            purpose="Stage registry: maps stage names to classes, "
                    "loaded at runtime via importlib",
            exports=["StageRegistry", "get_registry"],
            imports_from=[base_path, config_path],
            generation_hints={"pattern": "importlib"},
        )
        edges.extend([
            (registry_path, base_path, "IMPORTS"),
            (registry_path, config_path, "IMPORTS"),
        ])

        # ── Stages subpackage ────────────────────────────────────────
        stages_init = f"{pkg}/stages/__init__.py"
        modules[stages_init] = ModuleSpec(
            filepath=stages_init,
            role="package_init",
            purpose="Stages subpackage init — re-exports stage classes",
            exports=[],
            imports_from=[],
        )

        stage_paths: list[str] = []
        for i, stage in enumerate(stages):
            neutral = _NEUTRAL_NAMES[name_idx]
            name_idx += 1
            stage_path = f"{pkg}/stages/{neutral}.py"
            stage_paths.append(stage_path)
            modules[stage_path] = ModuleSpec(
                filepath=stage_path,
                role="stage",
                purpose=stage.purpose,
                exports=[stage.function_name],
                imports_from=[models_path, base_path, exceptions_path],
                order=i + 1,
                generation_hints={
                    "function_name": stage.function_name,
                    "input_type": stage.input_type,
                    "output_type": stage.output_type,
                    "description": stage.purpose,
                    "stage_name": stage.module_name,
                },
            )
            edges.extend([
                (stage_path, models_path, "IMPORTS"),
                (stage_path, base_path, "IMPORTS"),
                (stage_path, exceptions_path, "IMPORTS"),
            ])

        # DATA_FLOWS_TO between consecutive stages
        for i in range(len(stage_paths) - 1):
            edges.append((stage_paths[i], stage_paths[i + 1], "DATA_FLOWS_TO"))

        # REGISTRY_WIRES: registry dynamically loads each stage
        for sp in stage_paths:
            edges.append((registry_path, sp, "REGISTRY_WIRES"))

        # ── Adapters subpackage ──────────────────────────────────────
        num_adapters = rng.randint(2, min(4, len(stages)))
        adapted_indices = sorted(rng.sample(range(len(stages)), num_adapters))

        adapters_init = f"{pkg}/adapters/__init__.py"
        modules[adapters_init] = ModuleSpec(
            filepath=adapters_init,
            role="package_init",
            purpose="Adapters subpackage init",
            exports=[],
            imports_from=[],
        )

        adapter_paths: list[str] = []
        adapter_wraps: dict[str, str] = {}  # adapter_path → stage_path
        for idx in adapted_indices:
            neutral = _NEUTRAL_NAMES[name_idx]
            name_idx += 1
            adapter_path = f"{pkg}/adapters/{neutral}.py"
            adapter_paths.append(adapter_path)
            wrapped_stage = stages[idx]
            wrapped_path = stage_paths[idx]
            adapter_wraps[adapter_path] = wrapped_path
            modules[adapter_path] = ModuleSpec(
                filepath=adapter_path,
                role="adapter",
                purpose=(
                    f"Adapter providing external interface for "
                    f"{wrapped_stage.module_name} stage"
                ),
                exports=[f"adapt_{wrapped_stage.function_name}"],
                imports_from=[base_path, wrapped_path, models_path],
                generation_hints={
                    "wrapped_stage": wrapped_path,
                    "wrapped_function": wrapped_stage.function_name,
                    "stage_name": wrapped_stage.module_name,
                },
            )
            edges.extend([
                (adapter_path, base_path, "IMPORTS"),
                (adapter_path, wrapped_path, "IMPORTS"),
                (adapter_path, models_path, "IMPORTS"),
                (adapter_path, wrapped_path, "CALLS_API"),
            ])

        # ── Middleware subpackage ────────────────────────────────────
        middleware_kinds = ["logging", "retry", "metrics"]
        num_middleware = rng.randint(2, 3)
        chosen_mw = middleware_kinds[:num_middleware]

        mw_init = f"{pkg}/middleware/__init__.py"
        modules[mw_init] = ModuleSpec(
            filepath=mw_init,
            role="package_init",
            purpose="Middleware subpackage init",
            exports=[],
            imports_from=[],
        )

        mw_paths: list[str] = []
        for kind in chosen_mw:
            neutral = _NEUTRAL_NAMES[name_idx]
            name_idx += 1
            mw_path = f"{pkg}/middleware/{neutral}.py"
            mw_paths.append(mw_path)
            modules[mw_path] = ModuleSpec(
                filepath=mw_path,
                role="middleware",
                purpose=f"Middleware: {kind} wrapper for pipeline stages",
                exports=[f"{kind}_middleware"],
                imports_from=[base_path],
                generation_hints={"middleware_type": kind},
            )
            edges.append((mw_path, base_path, "IMPORTS"))

        # ── Utils subpackage ─────────────────────────────────────────
        utils_init = f"{pkg}/utils/__init__.py"
        modules[utils_init] = ModuleSpec(
            filepath=utils_init,
            role="package_init",
            purpose="Utils subpackage init",
            exports=[],
            imports_from=[],
        )

        helpers_path = f"{pkg}/utils/helpers.py"
        modules[helpers_path] = ModuleSpec(
            filepath=helpers_path,
            role="utils",
            purpose="Shared utilities: logging setup, timing decorator, error helpers",
            exports=["setup_logging", "log_stage", "timed"],
            imports_from=[config_path],
            generation_hints={},
        )
        edges.append((helpers_path, config_path, "IMPORTS"))

        formatters_path = f"{pkg}/utils/formatters.py"
        modules[formatters_path] = ModuleSpec(
            filepath=formatters_path,
            role="utils",
            purpose="Output formatting: JSON, CSV, table formatters",
            exports=["format_json", "format_csv", "format_table"],
            imports_from=[models_path],
            generation_hints={},
        )
        edges.append((formatters_path, models_path, "IMPORTS"))

        # Stages also import from helpers
        for sp in stage_paths:
            modules[sp].imports_from.append(helpers_path)
            edges.append((sp, helpers_path, "IMPORTS"))

        # ── Legacy / distractors ─────────────────────────────────────
        legacy_init = f"{pkg}/legacy/__init__.py"
        modules[legacy_init] = ModuleSpec(
            filepath=legacy_init,
            role="package_init",
            purpose="Legacy subpackage init (deprecated code kept for reference)",
            exports=[],
            imports_from=[],
        )

        old_pipeline_path = f"{pkg}/legacy/old_pipeline.py"
        modules[old_pipeline_path] = ModuleSpec(
            filepath=old_pipeline_path,
            role="distractor",
            purpose="Deprecated: old monolithic pipeline (kept for reference, unused)",
            exports=[],
            imports_from=[models_path],
            generation_hints={"dead_code": "true"},
        )
        edges.append((old_pipeline_path, models_path, "IMPORTS"))

        compat_path = f"{pkg}/legacy/compat.py"
        modules[compat_path] = ModuleSpec(
            filepath=compat_path,
            role="distractor",
            purpose="Compatibility shims for legacy pipeline format",
            exports=["convert_legacy_format"],
            imports_from=[models_path],
            generation_hints={},
        )
        edges.append((compat_path, models_path, "IMPORTS"))

        # ── Entry points ─────────────────────────────────────────────
        runner_path = f"{pkg}/runner.py"
        runner_imports = [config_path, registry_path, models_path, base_path]
        runner_imports.extend(mw_paths)
        modules[runner_path] = ModuleSpec(
            filepath=runner_path,
            role="orchestrator",
            purpose="Pipeline runner: loads config, resolves stages via registry, "
                    "executes pipeline with middleware",
            exports=["run_pipeline"],
            imports_from=runner_imports,
            generation_hints={
                "stage_count": str(len(stages)),
                "has_middleware": "true",
            },
        )
        edges.extend([
            (runner_path, config_path, "IMPORTS"),
            (runner_path, registry_path, "IMPORTS"),
            (runner_path, models_path, "IMPORTS"),
            (runner_path, base_path, "IMPORTS"),
            (runner_path, registry_path, "CALLS_API"),
        ])
        for mw in mw_paths:
            edges.extend([
                (runner_path, mw, "IMPORTS"),
                (runner_path, mw, "CALLS_API"),
            ])
        # Runner calls stages at runtime via registry (not via direct import)
        for sp in stage_paths:
            edges.append((runner_path, sp, "CALLS_API"))

        cli_path = f"{pkg}/cli.py"
        modules[cli_path] = ModuleSpec(
            filepath=cli_path,
            role="entry_point",
            purpose="CLI entry point: argument parsing and pipeline invocation",
            exports=["main"],
            imports_from=[runner_path, config_path],
            generation_hints={},
        )
        edges.extend([
            (cli_path, runner_path, "IMPORTS"),
            (cli_path, config_path, "IMPORTS"),
            (cli_path, runner_path, "CALLS_API"),
        ])

        return modules, edges

    def _build_medium_invariants(
        self,
        pkg: str,
        modules: dict[str, ModuleSpec],
        edges: list[tuple[str, str, str]],
        stages: list[StageInfo],
        rng: random.Random,
    ) -> list[InvariantSpec]:
        """Build 8-12 planted invariants for a medium pipeline codebase."""
        invariants: list[InvariantSpec] = []
        inv_id = 0

        # Collect paths by role
        stage_paths = [
            m.filepath for m in modules.values() if m.role == "stage"
        ]
        adapter_paths = [
            m.filepath for m in modules.values() if m.role == "adapter"
        ]
        mw_paths = [
            m.filepath for m in modules.values() if m.role == "middleware"
        ]
        base_path = next(
            m.filepath for m in modules.values() if m.role == "base"
        )
        models_path = next(
            m.filepath for m in modules.values() if m.role == "data_model"
        )
        config_path = next(
            m.filepath for m in modules.values() if m.role == "config"
        )
        registry_path = next(
            m.filepath for m in modules.values() if m.role == "registry"
        )
        runner_path = next(
            m.filepath for m in modules.values() if m.role == "orchestrator"
        )
        distractor_paths = [
            m.filepath for m in modules.values() if m.role == "distractor"
        ]
        utils_paths = [
            m.filepath for m in modules.values() if m.role == "utils"
        ]

        # ── DATAFLOW: linear stage ordering ──────────────────────────
        for i in range(len(stages) - 1):
            inv_id += 1
            invariants.append(InvariantSpec(
                id=f"inv-{inv_id:03d}",
                type=InvariantType.DATAFLOW,
                description=(
                    f"{stages[i].module_name} must produce {stages[i].output_type} "
                    f"before {stages[i+1].module_name} can consume it as "
                    f"{stages[i+1].input_type}"
                ),
                involved_modules=[stage_paths[i], stage_paths[i + 1]],
                rationale=(
                    "Pipeline stages form a linear data flow; each stage's output "
                    "type is the next stage's input type, enforcing transformation order"
                ),
            ))

        # ── INTERFACE: all stages share StageBase interface ──────────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.INTERFACE,
            description=(
                "Every processing stage inherits from StageBase and implements "
                "process(data, config) -> list[OutputType]"
            ),
            involved_modules=[base_path] + stage_paths,
            rationale=(
                "A uniform stage interface enables the registry to load stages "
                "generically and allows middleware to wrap any stage"
            ),
        ))

        # ── BOUNDARY: stages must not import each other ──────────────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.BOUNDARY,
            description=(
                "No stage module may import from any other stage module "
                "(stages communicate only via data passed by the runner)"
            ),
            involved_modules=stage_paths,
            rationale=(
                "Direct inter-stage imports would create hidden ordering "
                "dependencies and break stage isolation"
            ),
        ))

        # ── BOUNDARY: stages must not import runner (no reverse dep) ─
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.BOUNDARY,
            description=(
                "No stage, adapter, or middleware may import from runner.py "
                "(no reverse dependency on orchestrator)"
            ),
            involved_modules=[runner_path] + stage_paths + adapter_paths + mw_paths,
            rationale=(
                "The orchestrator depends on stages, not the other way around; "
                "circular dependencies would make stages untestable"
            ),
        ))

        # ── BOUNDARY: adapters must not import middleware ─────────────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.BOUNDARY,
            description=(
                "Adapter modules must not import from middleware modules "
                "(layer boundary: adapters wrap stages, middleware wraps execution)"
            ),
            involved_modules=adapter_paths + mw_paths,
            rationale=(
                "Adapters and middleware operate at different abstraction layers; "
                "cross-layer imports would blur responsibilities"
            ),
        ))

        # ── BOUNDARY: no circular imports within utils/ ──────────────
        if len(utils_paths) >= 2:
            inv_id += 1
            invariants.append(InvariantSpec(
                id=f"inv-{inv_id:03d}",
                type=InvariantType.BOUNDARY,
                description=(
                    "Utility modules must not import from each other "
                    "(no circular dependencies in utils/)"
                ),
                involved_modules=utils_paths,
                rationale=(
                    "Utility modules should be leaf dependencies; mutual imports "
                    "would create import cycles and complicate testing"
                ),
            ))

        # ── INVARIANT: config.batch_size respected by all stages ─────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.INVARIANT,
            description=(
                "PipelineConfig.batch_size must be respected by all stages "
                "and the runner for consistent batch processing"
            ),
            involved_modules=[config_path, runner_path] + stage_paths,
            rationale=(
                "Batch size is a global pipeline property; inconsistent batch "
                "handling between stages would cause data loss or duplication"
            ),
        ))

        # ── INVARIANT: registry must match config stage order ────────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.INVARIANT,
            description=(
                "StageRegistry must resolve stages in the order specified "
                "by PipelineConfig.stage_order"
            ),
            involved_modules=[registry_path, config_path, runner_path],
            rationale=(
                "Config-driven stage ordering is the single source of truth; "
                "registry resolution must respect it to maintain pipeline semantics"
            ),
        ))

        # ── PURPOSE: why stages are separated ────────────────────────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.PURPOSE,
            description=(
                "Each processing stage is a separate module to enforce "
                "single responsibility and enable independent testing"
            ),
            involved_modules=stage_paths,
            rationale=(
                "Separating stages allows each to be tested, replaced, or "
                "reordered independently; merging stages would couple concerns"
            ),
        ))

        # ── PURPOSE: why adapters exist ──────────────────────────────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.PURPOSE,
            description=(
                "Adapter modules provide a stable external interface for stages "
                "that may change their internal implementation"
            ),
            involved_modules=adapter_paths + stage_paths,
            rationale=(
                "Adapters decouple external consumers from stage internals; "
                "stage refactoring doesn't break external code"
            ),
        ))

        # ── BOUNDARY: distractors have no incoming edges ─────────────
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.BOUNDARY,
            description=(
                "Legacy modules are dead code — no active module imports from them"
            ),
            involved_modules=distractor_paths,
            rationale="Dead code is kept for reference only, not used at runtime",
        ))

        # Optionally add one more if seed allows (vary count by seed)
        if rng.random() < 0.5 and len(stages) >= 3:
            inv_id += 1
            # DATAFLOW: validation must happen before transform
            val_idx = None
            trans_idx = None
            for i, s in enumerate(stages):
                if "validat" in s.module_name:
                    val_idx = i
                if "transform" in s.module_name:
                    trans_idx = i
            if val_idx is not None and trans_idx is not None:
                invariants.append(InvariantSpec(
                    id=f"inv-{inv_id:03d}",
                    type=InvariantType.DATAFLOW,
                    description=(
                        "Data must pass through the validation stage before "
                        "reaching the transform stage (validation chain)"
                    ),
                    involved_modules=[stage_paths[val_idx], stage_paths[trans_idx]],
                    rationale=(
                        "Transforming unvalidated data risks propagating corrupt "
                        "records through the rest of the pipeline"
                    ),
                ))

        return invariants

    def _build_medium_rationales(
        self,
        pkg: str,
        modules: dict[str, ModuleSpec],
        stages: list[StageInfo],
    ) -> list[DesignRationaleSpec]:
        """Build design rationales for medium pipeline codebases."""
        rationales: list[DesignRationaleSpec] = []

        stage_paths = [
            m.filepath for m in modules.values() if m.role == "stage"
        ]
        adapter_paths = [
            m.filepath for m in modules.values() if m.role == "adapter"
        ]
        mw_paths = [
            m.filepath for m in modules.values() if m.role == "middleware"
        ]
        models_path = next(
            m.filepath for m in modules.values() if m.role == "data_model"
        )
        base_path = next(
            m.filepath for m in modules.values() if m.role == "base"
        )
        config_path = next(
            m.filepath for m in modules.values() if m.role == "config"
        )
        registry_path = next(
            m.filepath for m in modules.values() if m.role == "registry"
        )
        runner_path = next(
            m.filepath for m in modules.values() if m.role == "orchestrator"
        )

        # Why centralized models.py?
        rationales.append(DesignRationaleSpec(
            id="dr-001",
            question=(
                f"Why is there a separate {models_path} instead of "
                f"defining data types within each stage?"
            ),
            answer=(
                "Centralizing data types in models.py prevents circular imports "
                "between stages and ensures type consistency across the pipeline."
            ),
            affected_modules=[models_path] + stage_paths,
            downstream_effects=[
                "Moving types into stages would create circular imports",
                "Type changes would require editing multiple files",
                "Stages could no longer be tested independently",
            ],
        ))

        # Why StageBase ABC?
        rationales.append(DesignRationaleSpec(
            id="dr-002",
            question=(
                f"Why do all stages inherit from StageBase in {base_path}?"
            ),
            answer=(
                "A common base class enforces a uniform process() interface, "
                "enables the registry to load stages generically, and allows "
                "middleware to wrap any stage without special-casing."
            ),
            affected_modules=[base_path] + stage_paths + adapter_paths,
            downstream_effects=[
                "Removing StageBase would break registry auto-discovery",
                "Middleware would need per-stage wiring logic",
                "Adding a new stage would require updating the runner",
            ],
        ))

        # Why registry instead of direct imports?
        rationales.append(DesignRationaleSpec(
            id="dr-003",
            question=(
                f"Why does {runner_path} use {registry_path} to resolve stages "
                f"instead of importing them directly?"
            ),
            answer=(
                "Registry-based resolution allows stage ordering to be config-driven. "
                "Direct imports would hard-code the pipeline structure and require "
                "code changes to add, remove, or reorder stages."
            ),
            affected_modules=[runner_path, registry_path, config_path] + stage_paths,
            downstream_effects=[
                "Direct imports would hide architecture from config",
                "Stage ordering changes would require code edits",
                "Dynamic stage loading (e.g., plugins) would be impossible",
            ],
        ))

        # Why separate adapters?
        rationales.append(DesignRationaleSpec(
            id="dr-004",
            question=(
                "Why are there adapter modules wrapping some stages instead of "
                "exposing stages directly?"
            ),
            answer=(
                "Adapters provide a stable external interface. If a stage's "
                "internal implementation changes, only its adapter needs updating, "
                "not every external consumer."
            ),
            affected_modules=adapter_paths + stage_paths,
            downstream_effects=[
                "Direct stage access would couple external code to internals",
                "Stage API changes would break all consumers simultaneously",
                "Testing external integrations would require full stage setup",
            ],
        ))

        # Why middleware as separate modules?
        rationales.append(DesignRationaleSpec(
            id="dr-005",
            question=(
                "Why is logging/retry implemented as middleware modules "
                "rather than embedded in each stage?"
            ),
            answer=(
                "Middleware modules apply cross-cutting concerns uniformly. "
                "Embedding logging/retry in each stage would duplicate code "
                "and risk inconsistent behavior across stages."
            ),
            affected_modules=mw_paths + stage_paths + [runner_path],
            downstream_effects=[
                "Per-stage logging would diverge in format and verbosity",
                "Retry logic would be duplicated and potentially inconsistent",
                "Adding a new cross-cutting concern would touch every stage",
            ],
        ))

        return rationales

    def _build_invariants(
        self,
        pkg: str,
        models_path: str,
        config_path: str,
        pipeline_path: str,
        source_path: str,
        sink_path: str,
        stage_paths: list[str],
        stages: list[StageInfo],
        all_stage_paths: list[str],
    ) -> list[InvariantSpec]:
        """Build the set of planted invariants for a pipeline codebase."""
        invariants: list[InvariantSpec] = []
        inv_id = 0

        # DATAFLOW: linear ordering — each stage receives previous output
        for i in range(len(stages) - 1):
            inv_id += 1
            invariants.append(InvariantSpec(
                id=f"inv-{inv_id:03d}",
                type=InvariantType.DATAFLOW,
                description=(
                    f"{stages[i].module_name} must produce {stages[i].output_type} "
                    f"before {stages[i+1].module_name} can consume it as "
                    f"{stages[i+1].input_type}"
                ),
                involved_modules=[stage_paths[i], stage_paths[i + 1]],
                rationale=(
                    "Pipeline stages form a linear data flow; each stage's output "
                    "type is the next stage's input type, enforcing transformation order"
                ),
            ))

        # INTERFACE: all stages share a common function signature pattern
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.INTERFACE,
            description=(
                "Every processing stage exports a single function with signature "
                "(list[InputType], PipelineConfig) -> list[OutputType]"
            ),
            involved_modules=stage_paths,
            rationale=(
                "A uniform stage interface allows the orchestrator to chain stages "
                "generically and makes stages independently testable"
            ),
        ))

        # BOUNDARY: no direct imports between non-adjacent stages
        for i, sp_i in enumerate(stage_paths):
            for j, sp_j in enumerate(stage_paths):
                if abs(i - j) > 1:
                    inv_id += 1
                    invariants.append(InvariantSpec(
                        id=f"inv-{inv_id:03d}",
                        type=InvariantType.BOUNDARY,
                        description=(
                            f"{sp_i.split('/')[-1]} must not import from "
                            f"{sp_j.split('/')[-1]} (non-adjacent stages)"
                        ),
                        involved_modules=[sp_i, sp_j],
                        rationale=(
                            "Each stage only communicates via typed data passed by "
                            "the orchestrator; direct cross-stage imports would break "
                            "the linear pipeline contract"
                        ),
                    ))

        # BOUNDARY: stages must not import from pipeline.py (no circular deps)
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.BOUNDARY,
            description=(
                "No stage, source, or sink may import from pipeline.py "
                "(no reverse dependency on orchestrator)"
            ),
            involved_modules=[pipeline_path] + all_stage_paths,
            rationale=(
                "The orchestrator depends on stages, not the other way around; "
                "circular dependencies would make stages untestable in isolation"
            ),
        ))

        # BOUNDARY: sink must not be imported by stages
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.BOUNDARY,
            description=(
                "Processing stages must not import from sink.py "
                "(data flows forward only)"
            ),
            involved_modules=[sink_path] + stage_paths,
            rationale="Data flows strictly forward through the pipeline",
        ))

        # INVARIANT: config.BATCH_SIZE is respected by all stages
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.INVARIANT,
            description=(
                "PipelineConfig.batch_size must be respected by source, "
                "all stages, and sink for consistent batch processing"
            ),
            involved_modules=[config_path, source_path, *stage_paths, sink_path],
            rationale=(
                "Batch size is a global pipeline property; inconsistent batch "
                "handling between stages would cause data loss or duplication"
            ),
        ))

        # PURPOSE: why stages are separated
        inv_id += 1
        invariants.append(InvariantSpec(
            id=f"inv-{inv_id:03d}",
            type=InvariantType.PURPOSE,
            description=(
                "Each processing stage is a separate module to enforce "
                "single responsibility and enable independent testing"
            ),
            involved_modules=stage_paths,
            rationale=(
                "Separating stages allows each to be tested, replaced, or "
                "reordered independently; merging stages would couple concerns"
            ),
        ))

        return invariants

    def _build_design_rationales(
        self,
        pkg: str,
        models_path: str,
        config_path: str,
        pipeline_path: str,
        utils_path: str,
        source_path: str,
        sink_path: str,
        stage_paths: list[str],
        stages: list[StageInfo],
        all_stage_paths: list[str],
    ) -> list[DesignRationaleSpec]:
        """Build planted design rationales for intentionality probing."""
        rationales: list[DesignRationaleSpec] = []

        # Why centralized models.py?
        rationales.append(DesignRationaleSpec(
            id="dr-001",
            question=(
                f"Why is there a separate {models_path} instead of "
                f"defining data types within each stage?"
            ),
            answer=(
                "Centralizing data types in models.py prevents circular imports "
                "between stages and ensures type consistency across the pipeline. "
                "If each stage defined its own output type, the next stage would "
                "need to import from the previous stage, creating a chain of "
                "cross-stage dependencies."
            ),
            affected_modules=[models_path] + stage_paths,
            downstream_effects=[
                "Moving types into stages would create circular imports",
                "Type changes would require editing multiple files",
                "Stages could no longer be tested independently of each other",
            ],
        ))

        # Why don't stages import from each other?
        if len(stages) >= 2:
            s0 = stages[0]
            s1 = stages[1]
            rationales.append(DesignRationaleSpec(
                id="dr-002",
                question=(
                    f"Why doesn't {s1.module_name}.py import from "
                    f"{s0.module_name}.py directly?"
                ),
                answer=(
                    "The pipeline enforces a linear data flow through typed "
                    "interfaces. Each stage only receives data from the previous "
                    "stage via the orchestrator. Direct imports between stages would "
                    "break the pipeline contract and make stage ordering implicit "
                    "rather than explicit in pipeline.py."
                ),
                affected_modules=[stage_paths[0], stage_paths[1], pipeline_path],
                downstream_effects=[
                    "Direct imports would create hidden ordering dependencies",
                    "Stages could no longer be reordered or removed without "
                    "editing other stages",
                    "Testing a single stage would require instantiating its "
                    "predecessor",
                ],
            ))

        # Why separate config.py?
        rationales.append(DesignRationaleSpec(
            id="dr-003",
            question=(
                f"Why is {config_path} separate from {pipeline_path}?"
            ),
            answer=(
                "Configuration is injected into stages as a parameter. Separating "
                "it allows stages to be tested with different configurations "
                "without importing the orchestrator. It also enables config "
                "to be loaded from files or environment variables independently."
            ),
            affected_modules=[config_path, pipeline_path] + all_stage_paths,
            downstream_effects=[
                "Merging config into pipeline.py would force stages to import "
                "the orchestrator to access settings",
                "Testing stages in isolation would become harder",
                "Config file loading and validation logic would be coupled "
                "to pipeline execution",
            ],
        ))

        # Why does pipeline.py import all stages?
        rationales.append(DesignRationaleSpec(
            id="dr-004",
            question=(
                f"Why does {pipeline_path} import every stage explicitly "
                f"instead of discovering stages dynamically?"
            ),
            answer=(
                "Explicit imports make the pipeline's stage ordering visible in "
                "code and statically analyzable. Dynamic discovery (e.g., plugin "
                "loading) would hide the architecture from both humans and tools, "
                "making dependency analysis unreliable."
            ),
            affected_modules=[pipeline_path] + all_stage_paths,
            downstream_effects=[
                "Dynamic discovery would hide the dependency graph from "
                "static analysis",
                "Stage ordering bugs would only surface at runtime",
                "IDE navigation and refactoring tools would lose traceability",
            ],
        ))

        # Why shared utils instead of per-module helpers?
        rationales.append(DesignRationaleSpec(
            id="dr-005",
            question=(
                f"Why do all stages share {utils_path} instead of having "
                f"per-stage utility functions?"
            ),
            answer=(
                "Shared utilities (logging, timing, error formatting) enforce "
                "consistent observability across all pipeline stages. Per-stage "
                "utilities would lead to inconsistent log formats and duplicated "
                "boilerplate."
            ),
            affected_modules=[utils_path] + all_stage_paths,
            downstream_effects=[
                "Duplicated logging setup across stages",
                "Inconsistent log formats making pipeline debugging harder",
                "Per-stage utility drift over time",
            ],
        ))

        return rationales


# ============================================================================
# Template registry
# ============================================================================

_TEMPLATES: dict[ArchPattern, type[PatternTemplate]] = {
    ArchPattern.PIPELINE: PipelineTemplate,
}


def get_template(pattern: ArchPattern) -> PatternTemplate:
    """Get a pattern template by architecture pattern type."""
    cls = _TEMPLATES.get(pattern)
    if cls is None:
        supported = ", ".join(p.value for p in _TEMPLATES)
        raise ValueError(
            f"No template for pattern {pattern.value!r}. "
            f"Supported: {supported}"
        )
    return cls()


def list_patterns() -> list[ArchPattern]:
    """Return all patterns that have registered templates."""
    return list(_TEMPLATES.keys())
