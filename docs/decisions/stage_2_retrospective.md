# Stage 2 retrospective

**Date:** 2026-05-03
**Scope:** Stages 2.1–2.5 — the algorithmic core of `bsig`
**Final state:** 196 tests passing, mypy --strict on 15 source files,
import-linter 4/4 contracts kept.

This is a half-page note summarizing the architectural decisions that
landed across stage 2 and any deviations from the original CLAUDE.md
§4–6 spec. Useful for future-you when the project picks up after the
gate experiment.

## Deviations from CLAUDE.md §4–6 spec

1. **Edge storage: pandas DataFrame, not NumPy structured array.** §4
   suggested structured arrays; we used pandas because pandas is
   already required, bulk groupby/filter operations are vectorized
   over the same NumPy storage, pyarrow Parquet I/O is direct, and
   debugging introspection is dramatically easier. Note recorded
   inline in CLAUDE.md §4.

2. **EdgeClass expanded from 3 to 6 values.** Original spec implied
   `consensus / underutilized / ritualized`. We added `UNCLASSIFIED`
   (= -1, sentinel for unclassified graphs), `RARE` (low VoI + low
   consensus, fourth quadrant), `MIDDLE` (non-corner cells of the
   3×3 percentile partition). No schema_version bump because column
   type/name/file structure are unchanged; old readers fail loudly
   via `EdgeClass(int)` ValueError when encountering 3 or 4.

3. **AssemblyGraph immutable + builder.** Resolved §15 Q1 toward
   immutable: `AssemblyGraph` is frozen, `AssemblyGraphBuilder` is
   single-use (`build()` consumes it; subsequent calls raise
   `BuilderConsumedError`). Eager NetworkX view at build time;
   `to_networkx()` returns the frozen view in O(1).

4. **FAISS as separate companion artifact.** Resolved §15 Q3:
   `faiss_indices/` lives in the graph artifact directory but is
   architecturally separate; `AssemblyGraph` doesn't hold the index.
   `distance_from_trajectory` takes the index map as a separate
   parameter.

5. **Function-level public API, no orchestrator.** Resolved §15 Q4
   for 0.1. Every save/load is a separate function; consumers
   compose. Tuple-return load functions explicitly rejected (see
   stages 2.2, 2.4 review threads).

6. **Sync LLMAdapter with batch method.** Resolved §15 Q5 toward
   option (c): sync `get_hypothesis_distribution` plus sync
   `get_hypothesis_distribution_batch` with per-item retry semantics
   locked in the protocol docstring. Async deferred to 0.2.

7. **`Trajectory.outcome` optional.** Resolved §15 Q2: inference-time
   trajectories are unlabeled; evaluation code asserts non-None at
   its boundary.

8. **`scikit-learn` promoted to required dep** (stage 2.5). Originally
   in `experiments` extra; promoted because every public function in
   `core/evaluation.py` needs it. CLAUDE.md §10 allowed-imports
   updated.

9. **`SignatureWeights` composite via rank-percentile normalization.**
   The composite is a weighted convex combination of
   rank-percentile-normalized components. **Signatures are
   dataset-relative** — not directly comparable across datasets
   without re-normalization. `signature_metadata.json` records
   `"normalization": "rank_percentile"` so this isn't lost.

## Invariants worth preserving across stage 3

- **Action-level VoI/consensus/classification.** All edges sharing
  `(source_id, action_id)` carry identical values; the schema stores
  them per-edge for I/O convenience. Domain packs that compute
  per-edge transition probability should add a new column, not mutate
  these.
- **Higher score = defer.** Composite and components all constructed
  this way; inverse-semantics caller-negates.
- **`target_column = 1` is defer-positive.** Inverse labels
  caller-flips before evaluation.
- **`visit_seq` is the FAISS internal ID.** Sequential from 0 within
  a single visits.parquet write; not stable across re-saves.
- **`node_id` namespace = single canonicalizer per graph.** Domain
  packs producing canonicalizers must enforce this; recovery
  validates metadata consistency per node_id and raises on mismatch.

## Open §15 questions deferred to 0.2

- **Schema migration tooling.** 0.x refuses mismatched versions
  loudly; migration helpers added when v2 ships.
- **Async LLMAdapter** as a separate protocol. Add when batch
  evaluation against vLLM motivates it.
- **Hot-reload YAML config.** Currently package-data baked at
  install; hot-reload deferred.
- **Per-edge VoI fallback tracking.** Currently coarse-grained
  counts in `graph.metadata["voi_method_summary"]`. Per-edge
  tracking needs a new edges.parquet column → schema_version 2.

## Module dependency map (final)

```
trajectory.py  -> (no internal deps)
graph.py       -> trajectory? no.  Pure pandas/networkx/numpy.
recovery.py    -> graph, trajectory
persistence.py -> graph, signature
signature.py   -> graph, trajectory
paths.py       -> graph
evaluation.py  -> (no internal deps; operates on DataFrames)
```

The architectural commitment: `evaluation.py` operates on
DataFrame-shaped inputs, not on `bsig` types. This is what lets the
same evaluation code serve clinical and MedQA domains —
demonstrating the framework's portable-architecture property.

## Operational notes

- **Test count:** 196 passing across 6 test modules + architecture
  test. ~57 of those are FAISS-gated via `pytest.importorskip`.
- **Coverage targets per CLAUDE.md §12:** core 85%, adapters 90%.
  Not formally measured during stage 2; worth running
  `pytest --cov=bsig.core --cov=bsig.adapters` before stage 3.
- **mypy override scope:** three modules have targeted disables
  (graph: `arg-type`/`operator`/`assignment` for pandas-stubs noise;
  faiss/scipy/sklearn ignore_missing_imports for missing stubs).
  Strict typing remains in force everywhere else.
- **Bootstrap CI cost:** ~2–5 minutes at 5000 iterations on
  chest-pain MIMIC scale. Dev guidance: 500 iterations.

## Next: Stage 3 (MedQA domain pack)

Per CLAUDE.md §14, smaller and simpler than clinical. Implements
`StateCanonicalizer`, `TrajectorySource`, `GroundTruthExtractor`,
`condition_a/b/c`, `decomposer.py` against MedQA-USMLE data. End-to-end
through synthetic-MCQ smoke test using a mock LLM. Roughly 1–2 weeks
of focused session time.

---

## Post-close addenda

Cross-stage observations recorded after this retrospective was written,
when implementation work in later stages surfaced something useful to
remember at stage 2's level (data model, adapter contracts, persistence).

### A1 (2026-05-03, stage 3.1): Pydantic-frozen vs dataclass-frozen error types

Pydantic models declared with `model_config = ConfigDict(frozen=True)`
raise `pydantic.ValidationError` on attempted mutation, **not**
`dataclasses.FrozenInstanceError`. Tests asserting immutability of
domain-pack raw types (which are Pydantic per stage 3.1's design
decision) need:

```python
from pydantic import ValidationError
with pytest.raises(ValidationError):
    record.field = new_value
```

Whereas tests for stage-2 core types (frozen dataclasses) use:

```python
import dataclasses
with pytest.raises(dataclasses.FrozenInstanceError):
    state.field = new_value
```

Both are valid; just different exception types. Worth knowing when
writing immutability tests for clinical raw types in stage 5 (which
will also be Pydantic per the same design rationale: validation at
construction earns its weight on heterogeneous EHR data).
