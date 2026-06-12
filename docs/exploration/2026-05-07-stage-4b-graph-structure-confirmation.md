# Stage-4b graph artifact structural confirmation

**Date**: 2026-05-07
**Purpose**: confirm that the stage-4b professional_law graph artifact
exhibits the forest-of-disjoint-paths structural pattern predicted by
the framework's documented MCQ-format limitation.

**Background**: the methods paper §5.3, §7.1, and §3.5 reference the
"single-trajectory-per-question MCQ format structurally disables the
graph-structural components" claim, anchored on the canonicalizer's
inclusion of question text verbatim in the hash. This document
confirms the structural pattern empirically from the stage-4b
professional_law N=1534 graph artifact.

## Structural metrics

From ``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law/
graph_artifact/`` (nodes.parquet + edges.parquet):

| Metric | Value | Interpretation |
|---|---|---|
| Total nodes | 8,773 | Every state across 1,534 questions × ~5.7 positions each is unique. |
| Total edges | 7,239 | Edges connect consecutive measurement positions within each question. |
| **Edge frequency = 1** | **7,239 (100.00%)** | Every edge is a singleton; no cross-question collapse. |
| Edge frequency ≥ 2 | 0 (0.00%) | The structural-collapse mass mass is zero. |
| Mean visits per node | 1.00 | Every node visited exactly once. |
| VoI mean | 0.0000 | Per-edge variance-of-information is identically zero. |
| VoI std | 0.0000 | No variability in VoI. |

## Confirmation against pre-design

This matches the pre-design expectation: the canonicalizer's
question-text-inclusion in the hash makes cross-question collapse
structurally impossible. The recovered graph is therefore a
*disjoint-paths forest* — 1,534 separate paths, one per question,
with ~5-6 states each. Per-edge VoI is undefined (every edge is
freq=1; no variance to compute).

The methods-paper claims rest on this exact property:

- **§3.5** notes ``voi_flatness`` is "structurally near-degenerate"
  on single-trajectory data — this confirms it experimentally.
- **§5.3** reports ``voi_flatness`` AUC of 0.500 ± 0.000 on
  professional_law — degenerate by construction, as expected.
- **§7.1** scopes the empirical position: "single-trajectory-per-
  question MCQ format" cannot test the graph-structural components.

## Implication for stage-6

Stage-6 chest-pain pre-registration's P1 prediction (≥30% edges with
frequency ≥ 2) is the load-bearing test of whether the framework's
composite architecture earns its complexity on multi-trajectory
data. Stage-4b's 0.00% confirms the failure mode the prediction is
testing against; stage-6 needs to show >30% on real MIMIC-IV-ED data
+ a working clinical canonicalizer (PI input on chief-complaint
clusters and discretization thresholds pending).

The stage-5 synthetic canonicalizer validation (commit 3406429)
showed 25.3% edges-freq≥2 on synthetic data with hard cluster
labels; that result was bounded by the synthetic generator's
construction and does not test P1. Real data + PI-locked
discretization is the only honest test.

## Files

- Analysis script (one-shot): inline shell above; not committed as
  a re-runnable script (single-purpose; methodology documented here).
- Source: ``~/work/eunosia/artifacts/medqa-stage-4b-mmlu-
  professional_law/graph_artifact/``
- Methods paper reference: §3.5, §5.3, §7.1.
