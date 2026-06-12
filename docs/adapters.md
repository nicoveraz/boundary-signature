# Adapter contracts

The framework operates against five adapter contracts defined as Python
`Protocol` classes (PEP 544 structural typing) in `src/bsig/adapters/`.

## Why protocols

Adapters define the boundary between the framework's algorithmic core and
domain-specific implementations. Protocols allow:

- Multiple concrete implementations without a fixed inheritance hierarchy.
- Duck typing with type checking — `mypy --strict` validates that any
  passed-in object actually satisfies the protocol.
- Future user implementations from outside this repository to integrate
  without subclassing.

## The five protocols

### LLMAdapter

Wraps any language model. Exposes two operation modes:

- **Text generation** (`generate` / `generate_batch`): raw completion-
  style text output. Used by Conditions A, B, and Condition C's
  initial CoT step.
- **Hypothesis-distribution queries** (`get_hypothesis_distribution` /
  `get_hypothesis_distribution_batch`): probability distributions over
  a candidate hypothesis space, summing to 1.0 ± 1e-6, with key set
  matching the input hypothesis_space. Used by Condition C's per-step
  distribution monitoring.

The two pairs of methods are not interchangeable. Both batch variants
follow per-item retry semantics: each element gets up to `max_retries`
independent retries; successful items are not re-issued when other
items fail. See ADR-0005 in `docs/decisions/` for the stage-3.3a
Protocol extension that added the generate methods.

### StateCanonicalizer

Hashes raw state representations into stable identifiers. Two equivalent
raw states produce the same hash.

### TrajectorySource

Iterator over Trajectory objects. Each trajectory is a sequence of
(state, action, next_state) tuples plus an optional outcome.

### EmbeddingSource

Produces fixed-dimensional vectors from text. Used for the
distance-from-trajectory signature component.

### GroundTruthExtractor

Produces Outcome objects from raw trajectories. Multi-signal weak
supervision logic lives in implementations, not in core.

## Reference implementations

`src/bsig/reference/` contains production-ready implementations:

- `llm_openai.py` — OpenAI-API-compatible client (works with vLLM).
- `llm_local.py` — Ollama client for local smoke tests.
- `embedding_st.py` — sentence-transformer embeddings.

Each is behind an opt-in extra in `pyproject.toml` so the core library
has minimal required dependencies.

## Implementing a new adapter

For a new reasoning domain (robotics, scientific discovery, etc.), the
pattern is:

1. Create a new top-level module under `src/bsig/<domain>/`.
2. Implement each of the five protocols against your domain's data format.
3. Provide a configuration class for any tunable parameters.
4. Add tests in `tests/<domain>/` using the contract conformance test harness.
5. Update `.importlinter` to add the new module's independence constraints.

The domain pack should not import from other domain packs. If it needs
shared utilities, those go in `bsig.core` or `bsig.adapters`, not in another
domain pack.
