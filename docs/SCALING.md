# Scaling Codebases for Frontier Model Challenge

## Current State (v0.1 Medium)
- 27-30 files, ~70-84 edges, 1 pattern (Pipeline), 5 sub-packages
- Top models (GPT-5.3-Codex, Claude Sonnet) hit F1 ~0.67 with budget=20
- Challenging but not saturating — room to push harder

## Dimension 1: LARGE Tier (50-80 files) — NEXT

The `ComplexityTier.LARGE` enum exists but has no implementation. Natural next step:

- All 8 domain stages
- More adapters (4-5), more middleware variants (3, including metrics)
- **Multiple pipeline branches**: fan-out/fan-in where stages split into parallel paths
- More distractors (test files, migration scripts, alternative implementations)
- Deeper package nesting (3 levels)
- Additional cross-cutting modules (validators, serializers, hooks)
- ~150+ edges, 25+ invariants

At 50-80 files with budget=20, agents can only open 25-40% — exploration strategy becomes critical.

## Dimension 2: Compositional Patterns (Future)

Add a second architecture pattern that **composes** with Pipeline:

- **Microservice**: multiple packages communicating via shared models/protocols — forces cross-package reasoning
- **Plugin**: host application with plugin discovery mechanism — distributed registry
- **Layered**: strict layer boundaries (presentation → service → repository → models) with dependency inversion

Start with a small codebase for the new pattern to debug, then scale up.

## Dimension 3: Harder Discovery Challenges (Future)

- **Cross-package REGISTRY_WIRES**: registry in package A loads stages from package B
- **Event-driven edges**: publish/subscribe via event bus — no direct import, no config, only matching event names
- **Dependency inversion**: interfaces in one package, implementations in another
- **More distractors**: dead imports, commented-out code, alternative implementations

## Dimension 4: Budget Pressure

- Budget=20 on large codebase (25-40% coverage) for extreme pressure
- Budget=15 for ablation studies
- Could also test budget=30 to see saturation curves

## Implementation Order

1. ✅ Medium tier (v0.1) — 27-30 files
2. **Large tier** — 50-60 files, same Pipeline pattern
3. New pattern (small) — debug the pattern grammar
4. New pattern (medium/large) — full evaluation
