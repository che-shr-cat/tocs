# Contributing to ToCS

ToCS is designed as an open benchmark. Community contributions of test cases, codebase templates, and evaluation results are welcome.

## How to Contribute

### 1. Add a Codebase Template

The most valuable contribution is a new **architectural pattern template** for the procedural generator.

```
generator/templates/your_pattern/
├── README.md           # Description of the pattern and its invariants
├── skeleton.yaml       # Module structure definition
├── invariants.yaml     # Cross-module invariants to plant
├── rationales.yaml     # Design rationales for intentionality probes
└── example/            # One example generated codebase for reference
```

**Requirements:**
- Must define at least 3 cross-module invariants
- Must include at least 2 design rationales for intentionality probing
- Must be validatable via static analysis (ground truth recoverable)
- Must include at least one mutation scenario for the REVISE phase

### 2. Add a Hand-Crafted Test Case

For interesting edge cases that procedural generation can't capture:

```
data/contributed/your_case/
├── README.md           # What makes this case interesting
├── codebase/           # The actual code files
├── ground_truth.json   # Ground truth (must match models.CodebaseGroundTruth schema)
├── mutations/          # Optional: REVISE phase mutations
└── probes/             # Optional: intentionality probe questions
```

### 3. Submit Model Evaluation Results

Run the benchmark on a model we haven't tested:

```bash
python -m evaluation.run_eval --model your-model --output results/your_model/
```

Submit the results directory as a PR. Include:
- Model name and version
- API configuration (temperature, max tokens)
- Raw cognitive maps at each step
- Computed metrics

### 4. Improve Metrics or Baselines

PRs improving the scoring functions, adding new baselines, or fixing bugs are always welcome.

## Code Standards

- Python 3.11+
- Type hints on all public functions
- Pydantic models for all data structures
- pytest tests for new functionality
- Run `ruff check` before submitting

## Codebase Template Ideas (Wanted)

We'd especially welcome templates for:
- [ ] Event-driven architecture (pub/sub, message queues)
- [ ] Microservices with API gateway
- [ ] Plugin/extension system (microkernel)
- [ ] Data pipeline (ETL with validation stages)
- [ ] State machine with guards and transitions
- [ ] Hexagonal / ports-and-adapters architecture
- [ ] CQRS (Command Query Responsibility Segregation)

## License

All contributions must be under MIT license. By submitting a PR, you agree to license your contribution under MIT.
