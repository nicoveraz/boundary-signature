# Architecture

A four-layer architecture, separation enforced mechanically. This research
release contains the layers needed to reproduce the methods paper; the
clinical domain pack and production deployment are separate and not included
(see the README).

## Diagram

```
LAYER 4 — EXPERIMENT RUNNERS
   experiments/medqa_generalization/
                  │
                  ▼  consumes
LAYER 3 — DOMAIN IMPLEMENTATION
   bsig.medqa
                  │
                  ▼  satisfies contracts in
LAYER 2 — ADAPTER CONTRACTS
   bsig.adapters
                  │
                  ▼  defines interfaces over
LAYER 1 — CORE LIBRARY
   bsig.core
```

Plus reference adapter implementations (`bsig.reference`) that satisfy the
adapter contracts using common backends.

## Layer responsibilities

**`bsig.core`**: pure-algorithmic. Trajectory data model, structural-signature
computation (entropy scorers, plateau, distance), recovered-graph data
structures, deferral-curve evaluation, path classification. No I/O assumptions,
no LLM clients, no domain content.

**`bsig.adapters`**: Python Protocol classes defining the interfaces between
core and downstream consumers. Five protocols: LLMAdapter, StateCanonicalizer,
TrajectorySource, EmbeddingSource, GroundTruthExtractor.

**`bsig.medqa`**: standardized reasoning-test domain pack. Implements the
adapter contracts for MedQA-USMLE and MMLU. Includes the three experimental
conditions (CoT baseline, CoT + confidence, CoT + per-step signature
monitoring) and the CoT decomposer.

**`bsig.reference`**: reference adapter implementations. OpenAI-API client
(works with vLLM), llama.cpp and Ollama clients, an MLX adapter for Apple
Silicon, and sentence-transformer embeddings.

**`experiments/`**: thin orchestration. Pipeline and analysis scripts that load
`bsig` modules, run the relevant operation, and write Parquet/JSON outputs.

## Architectural rules (mechanically enforced)

The layer separation is enforced by import-linter rules in `.importlinter`:

- `core` does not import from any other layer.
- `adapters` does not import from domain packs or reference implementations.
- `reference` does not import from domain packs.

Run `uv run lint-imports` to check. A redundant AST-based test
(`tests/test_architecture.py`) validates the same constraints. Violations
produce a non-zero exit and clear error messages.
