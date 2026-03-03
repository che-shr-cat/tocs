"""CLI entry point for codebase generation.

Usage:
  python -m generator.export --pattern pipeline --complexity medium --seed 42 --output ./data/smoke_test
  python -m generator --pattern pipeline --complexity small --output ./data/test

Both `python -m generator` and `python -m generator.export` invoke this CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from generator.export import export_fixture, export_from_blueprint
from generator.grammar import get_template
from models import ArchPattern, ComplexityTier

app = typer.Typer(help="Generate synthetic codebases for ToCS benchmarking.")


@app.command()
def generate(
    pattern: str = typer.Option("pipeline", help="Architecture pattern (pipeline)"),
    complexity: str = typer.Option("medium", help="Complexity tier: small | medium"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    output: Path = typer.Option(..., help="Output directory for generated codebase"),
) -> None:
    """Generate a codebase with ground truth."""
    arch_pattern = ArchPattern(pattern)
    tier = ComplexityTier(complexity)

    typer.echo(f"Generating {tier.value} {arch_pattern.value} codebase (seed={seed})")

    if tier == ComplexityTier.SMALL:
        # Small tier: use the hand-authored fixture for backward compat
        gt = export_fixture(output)
    else:
        # Medium+: use grammar-driven blueprint generation
        template = get_template(arch_pattern)
        blueprint = template.generate(complexity=tier, seed=seed)
        gt = export_from_blueprint(blueprint, output)

    typer.echo(f"Output:     {output}")
    typer.echo(f"Codebase:   {gt.codebase_id}")
    typer.echo(f"Modules:    {len(gt.modules)}")
    typer.echo(f"Edges:      {len(gt.dependency_edges)}")
    typer.echo(f"Invariants: {len(gt.invariants)}")
    typer.echo(f"Contracts:  {len(gt.contracts)}")

    # Summary of edge types
    edge_types: dict[str, int] = {}
    for e in gt.dependency_edges:
        etype = e.get("type", "UNKNOWN")
        edge_types[etype] = edge_types.get(etype, 0) + 1
    typer.echo(f"Edge types: {dict(sorted(edge_types.items()))}")


if __name__ == "__main__":
    app()
