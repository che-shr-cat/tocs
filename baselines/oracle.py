"""Oracle upper bound: converts ground truth directly to a CognitiveMap."""

from __future__ import annotations

from models import (
    BeliefEdge,
    BeliefExport,
    CodebaseGroundTruth,
    CognitiveMap,
    ComponentBelief,
    EdgeType,
    ModuleStatus,
)


def run(gt: CodebaseGroundTruth) -> list[CognitiveMap]:
    """Return a single CognitiveMap that perfectly matches ground truth.

    The oracle does not interact with the environment — it directly
    converts ground truth to a belief state for metric anchoring.
    """
    components: dict[str, ComponentBelief] = {}

    for filepath, mod in gt.modules.items():
        edges = [
            BeliefEdge(target=e["target"], type=EdgeType(e["type"]), confidence=1.0)
            for e in mod.edges
        ]
        # Build exports from contracts
        exports: list[BeliefExport] = []
        for contract in gt.contracts:
            if contract.module == filepath:
                exports.append(BeliefExport(
                    name=contract.name,
                    signature=contract.signature,
                    callers=list(contract.callers),
                    confidence=1.0,
                ))

        components[filepath] = ComponentBelief(
            filepath=filepath,
            status=ModuleStatus.OBSERVED,
            purpose=mod.purpose,
            edges=edges,
            exports=exports,
            confidence=1.0,
        )

    cmap = CognitiveMap(
        step=0,
        components=components,
        invariants=[],
        unexplored=[],
        uncertainty_summary="",
    )
    return [cmap]
