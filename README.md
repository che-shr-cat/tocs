# Theory of Code Space (ToCS)

**Can AI code agents construct coherent architectural beliefs through active codebase exploration?**

ToCS is a benchmark that evaluates whether foundation models can build, maintain, and update structured understanding of software architecture under partial observability. Inspired by [Theory of Space](https://arxiv.org/abs/2602.07055) (Zhang et al., 2026), which demonstrated spatial reasoning failures in multimodal models.

## The Problem

AI coding tools ace isolated tasks (write a function, fix a bug) but struggle with complex multi-file engineering. Why? We hypothesize they cannot maintain a coherent **cognitive map** of the codebase — a structured belief about which modules exist, how they depend on each other, and what architectural constraints govern the system.

Current benchmarks test *output* (does the patch compile?). ToCS tests *understanding* (does the agent know the architecture?).

## What ToCS Measures

| Dimension | What it tests | How |
|-----------|--------------|-----|
| **Construct** | Can the agent build a dependency graph by exploring? | Graph F1 vs. exploration steps (AUC) |
| **Revise** | Can it update beliefs when interfaces change? | Belief Revision Score after mutations |
| **Exploit** | Can it use its map for downstream tasks? | Counterfactual probes (v0.1); behavioral tests (v1.0) |
| **Constraints** | Can it discover architectural rules? | Planted invariant discovery F1 |

## Key Features

- **Partial observability**: Agent opens files one at a time under a budget. SEARCH returns locations, never content.
- **Four typed edge categories**: IMPORTS (67%), CALLS_API (17%), REGISTRY_WIRES (9%), DATA_FLOWS_TO (7%) — only IMPORTS edges are discoverable by syntactic analysis.
- **Procedural codebase generator**: Deterministic, seeded, with planted invariants and verified ground truth.
- **Active-Passive Gap decomposition**: Three passive conditions isolate budget vs. selection vs. decision costs.
- **Cognitive map probing**: Structured JSON belief externalized every K actions — not just what the agent looked at, but what it *believes*.

## Quick Start

```bash
# Install
pip install -e .

# Generate a medium-complexity codebase (27-30 modules)
python -m generator --pattern pipeline --complexity medium --seed 42 \
  --output ./data/my_codebase

# Run a baseline (no API key needed)
python -m evaluation.run_eval evaluate \
  --model config-aware --codebase ./data/my_codebase --mode active \
  --budget 20 --output ./results/

# Run a frontier model (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-... python -m evaluation.run_eval evaluate \
  --model claude-sonnet-4-5-20250929 --codebase ./data/my_codebase \
  --mode active --budget 15 --probe-interval 3 --output ./results/

# Generate figures
python -m analysis.figures --results ./results/ --output ./paper/figures/

# Run tests
python -m pytest tests/ -v
```

## Baseline Results (v0.1)

On medium-complexity codebases (27-30 modules, 70-84 edges), budget=20:

| Baseline | Dep F1 | Precision | Recall | Action AUC |
|----------|--------|-----------|--------|------------|
| Oracle | 1.000 | 1.000 | 1.000 | — |
| Config-Aware | 0.577 | 0.736 | 0.475 | 0.212 |
| Random | 0.538 | 1.000 | 0.368 | 0.142 |
| BFS-Import | 0.293 | 1.000 | 0.173 | 0.079 |

**Key finding**: No rule-based baseline discovers CALLS_API or DATA_FLOWS_TO edges. These require reading and understanding function bodies — the space where LLM agents should excel.

## Project Structure

```
tocs/
├── models.py              # Pydantic schemas (CognitiveMap, GroundTruth, EvalResult)
├── generator/             # Procedural codebase generation
├── harness/               # Partial observability environment + probing
├── baselines/             # Rule-based explorers (BFS-Import, Config-Aware, Random, Oracle)
├── metrics/               # Scoring (map accuracy, gap analysis, constraint discovery)
├── evaluation/            # Model adapters (Anthropic, OpenAI) + eval pipeline
├── analysis/              # Figure generation
├── tests/                 # 310 tests
├── paper/                 # LaTeX paper source
├── data/                  # Generated codebases
└── results/               # Evaluation results
```

## Contributing

We actively welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

**Most wanted:**
- Evaluate frontier models (Claude, GPT, Gemini, open-weight)
- New architectural patterns (event-driven, microservices, plugin systems)
- New languages (TypeScript, Go)
- Scaffold-augmented evaluation (static analysis + LLM)

## Citation

```bibtex
@article{tocs2026,
  title={Theory of Code Space: Benchmarking Architectural Belief Construction
         in Code Agents Under Partial Observability},
  author={Anonymous},
  year={2026},
  note={Preprint}
}
```

## License

MIT. See [LICENSE](LICENSE).
