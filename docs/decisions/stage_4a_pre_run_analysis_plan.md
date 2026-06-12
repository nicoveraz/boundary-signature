# Stage 4a pre-run analysis plan

**Date written:** 2026-05-03 (before the run)
**Status:** pre-committed; do not revise after the run starts
**Scope:** how to interpret stage 4a's results
**Purpose:** protect against confirmation bias by committing to
interpretation criteria *before* seeing the numbers

---

## What stage 4a runs

Single command:

```bash
python experiments/medqa_generalization/scripts/03_pipeline_validation_ollama.py \
    --n-questions 1273 \
    --embedder-backend sentence-transformers \
    --embedder-model intfloat/multilingual-e5-large \
    --embedder-prefix "" \
    --checkpoint-every 50 \
    --output-dir ~/work/eunosia/artifacts/medqa-stage-4a/
```

Default `--strict=True`. Default seeds (qwen2.5 seed=42, bootstrap
seed=42). Default empty embedder prefix per OQ1 deferred decision.

**Estimated runtime:** 12-15 hours on M1 Pro (10-15 hours LLM
inference + ~10 minutes embedding pass + single-digit minutes
recovery+signature+evaluation).

**Outputs landing at the artifact path:**
- `condition_{a,b,c}_cached/` — cached trajectories per condition.
- `graph_artifact/` — recovered graph + visits + FAISS indices +
  signature scores.
- `condition_comparison.csv` — per-condition AUC + bootstrap CIs.
- `failure_mode_table.csv` — top-N high-signature trajectories.
- `repair_summary.json` — per-condition repair-rate aggregates.
- `partial_results.json` — per-question metadata.
- `checkpoint.json` — set of processed question IDs.

---

## Stats to compute (pre-committed)

These are the numbers to look at first, in order. Computed by the
script automatically; logged in the artifact directory.

### Headline metric
1. **Condition C composite AUC** with 95% bootstrap CI (n_bootstrap=5000).
2. **Condition B deferral_signal AUC** with CI.
3. **Condition A deferral_signal AUC** (constant 0.5 by construction;
   sanity check that the framework's flat-line baseline holds).

### Pairwise comparison
4. **AUC delta: C minus B** (with bootstrap CI on the delta if the
   analysis script supports it; otherwise the per-condition CIs are
   informative).

### Cross-condition agreement matrix (NEW per close-out discussion)
5. **3×3 agreement matrix:** for each pair (A,B), (A,C), (B,C),
   count how often the two conditions' `predicted_answer` matches.
   Plus marginal accuracy per condition (`predicted_answer ==
   correct_answer`).
6. **F9-style observation:** if cross-condition disagreement rate is
   above ~5%, prompt structure is materially affecting model behavior.
   This informs stage 4b's multi-prompt-variant design.

### Calibration
7. **Calibration metrics for Condition C composite** (ECE, MCE, Brier).
8. **Calibration metrics for Condition B deferral_signal** (same).

### Failure-mode inspection
9. **Top-20 highest-signature trajectories from Condition C composite.**
   For each: (predicted_answer, correct_answer, score, score_percentile,
   high_score_correct_outcome=True/False).
10. **F7-shaped case audit:** for the top-20, count how many have a
    Condition C trajectory whose per-step distribution argmax differs
    from the CoT-final-answer at any timestep. This is the explicit
    F7 detection check per O4 in retrospective notes.

### Recovered-graph diagnostics
11. **Graph density:** `(num_edges, num_nodes, mean_out_degree,
    fraction_terminal)`.
12. **Edge-frequency histogram:** how many edges have frequency=1, 2,
    3-5, 6-10, 11+? If most are frequency=1, the embedding-bin
    precision is too aggressive (or the recovery sample is too small
    to show meaningful aggregation).
13. **VoI fallback summary:** `voi_method_summary` from
    `graph.metadata`. If `posterior.global` is the largest bucket,
    the VoI estimates are noise-dominated.

### Operational sanity
14. **Repair-rate summary:** per condition, mean `repair_attempts`
    and `confidence_parsed` rate. If B's confidence_parsed rate is
    below ~95%, B's deferral_signal is partially fallback-default
    rather than model-expressed.
15. **Failure rate per condition:** how many ConditionResults have
    `success=False`?

---

## Interpretation criteria (pre-committed)

### Signal-positive

**ALL must hold:**
- Condition C composite AUC ≥ 0.55 with bootstrap CI lower bound
  ≥ 0.50.
- Condition C beats Condition B by AUC delta ≥ 0.03 (rough — sharper
  threshold deferable to stage 4b once we know the noise floor).
- Failure-mode top-20 contains at least one F7-shaped case.
- Recovered graph has at least 30% of edges at frequency ≥ 2.
- VoI fallback summary: `posterior.global` is NOT the largest bucket
  (i.e., at least some local VoI is computable).

**If all hold:** the framework demonstrates measurable signal on
the public 4-option MedQA benchmark with qwen2.5:7b. Triggers:
2. Stage 4b H100 multi-variant characterization scoping.

### Signal-negative

**EITHER:**
- Condition C composite AUC < 0.55 with bootstrap CI lower bound
  < 0.50.
- Condition C and Condition B AUCs indistinguishable (delta < 0.01
  with overlapping CIs).

**If either:** the framework does not demonstrate signal on this
configuration. Triggers:
1. Diagnostic analysis: which signature components contribute? Which
   fail? Is the recovered graph too sparse to support meaningful
   VoI?
2. Embedding-bin precision sweep per ADR-0002 (run additional local
   variants).
3. Prompt-variant analysis per F9: does a different prompt produce
   different distribution patterns? (Cheap to run locally.)
4. **Do NOT commit H100 budget** until signal-positive on at least
   one local variant.

### Signal-ambiguous

**Most likely outcome at first run.** Condition C AUC slightly above
0.5 but with bootstrap CIs that include 0.5 OR overlap Condition B's
CI substantially. Some signature components contributing, others not.

**If ambiguous:** the diagnostic analysis becomes the determining
factor. Look at:
- Per-component AUC breakdown (the `component_decomposition_table`
  function from `bsig.medqa.evaluation`). If `distance_from_trajectory`
  AUC alone is meaningful and `entropy_plateau` AUC is at chance,
  the embedding semantics are doing the work and refining them is the
  next investment.
- Recovered-graph density. If sparse, the embedding-bin sweep is the
  candidate explanation.
- Failure-mode F7-shaped case count. If zero F7-shaped cases in the
  top-20, the framework's headline claim isn't catching what it
  claims; re-examine the synthesis document.

The next move depends on which sub-finding dominates. Run additional
M1 Pro variants overnight; H100 budget waits.

---

## Decision tree (pre-committed)

```
After stage 4a completes:
  ├─ Read condition_comparison.csv → AUCs + CIs
  ├─ Read failure_mode_table.csv → top-20 inspection
  ├─ Read graph_artifact/metadata.json → recovery diagnostics
  ├─ Read repair_summary.json → operational sanity
  │
  ├─ Apply signal-positive criteria
  │   ├─ ALL hold → SIGNAL POSITIVE
  │   │   ├─ Scope stage 4b H100 multi-variant characterization
  │   │   └─ Begin methods-paper writeup foundation
  │   │
  │   └─ Some fail → check signal-negative
  │
  ├─ Apply signal-negative criteria
  │   ├─ EITHER trigger → SIGNAL NEGATIVE
  │   │   ├─ Diagnostic analysis (per components, graph, F7 cases)
  │   │   ├─ Run embedding-bin precision sweep (M1 Pro overnight)
  │   │   ├─ Run prompt-variant analysis (M1 Pro overnight)
  │   │   └─ Do NOT commit H100 budget
  │   │
  │   └─ NEITHER trigger → SIGNAL AMBIGUOUS
  │
  └─ SIGNAL AMBIGUOUS
      ├─ Per-component AUC breakdown via component_decomposition_table
      ├─ Identify dominant sub-finding
      ├─ Run targeted M1 Pro variant addressing the dominant finding
      └─ Re-evaluate after the variant completes
```

---

## What the pre-commitment protects against

Confirmation bias at three points:

1. **AUC threshold relaxation.** "0.52 isn't really that bad — the
   trend is positive." This plan commits to ≥ 0.55 with CI lower
   bound ≥ 0.50 BEFORE seeing the number. Adjusting the threshold
   downward post-hoc is moving the goalposts.

2. **Failure-mode interpretation.** "The top-20 includes some
   plausible cases — close enough to F7-shaped." This plan commits
   to an explicit F7 detection check (distribution-argmax differs
   from CoT-final-answer) BEFORE inspection. Counting hits or misses
   uses a deterministic criterion.

3. **Signal-vs-noise ambiguity.** "It's borderline but trending
   right." This plan commits to "signal-ambiguous → run more local
   variants before H100 budget" rather than letting marginal
   results rationalize compute commitment.

The plan can be revised AFTER stage 4a completes if the data
surfaces something the pre-commitment didn't anticipate. But the
revision should be explicit ("we expected X, observed Y, adjusting
Z because of specific reason") not implicit ("the threshold seems
high in retrospect").

---

## Operational checklist (before starting the run)

- [ ] M1 Pro plugged into AC power.
- [ ] `caffeinate -i` (or equivalent) running to prevent sleep.
- [ ] Free disk space at output path ≥ 2GB (artifacts are 100-500MB
      typical; buffer for FAISS indices + cached trajectories).
- [ ] Ollama service running and responsive (`ollama list` works).
- [ ] qwen2.5:7b-instruct already pulled (`ollama pull qwen2.5:7b-instruct`
      if not).
- [ ] sentence-transformers extra installed
      (`uv pip install -e '.[sentence-transformers]'`).
- [ ] First-time e5-large download will happen at the start
      (~1.1GB); run on a network connection that won't time out.
- [ ] Output directory does NOT already exist (or contains only old
      artifacts that can be discarded). For resume support, the
      script supports `--resume` if the run is interrupted.
- [ ] The MacBook is in a thermally-acceptable location (sustained
      12+ hours of high LLM load runs hot).
- [ ] After kicking off: don't touch the laptop. Don't run other
      heavy workloads (they'll compete for CPU+RAM).

---

## Post-run analysis sequence

When the run completes (morning after, ideally):

1. **Verify the run completed cleanly.** Check that
   `condition_comparison.csv`, `failure_mode_table.csv`, etc. all
   exist. Check `repair_summary.json` for per-condition failure
   rates.
2. **Compute the headline metrics** per the "Stats to compute"
   section above.
3. **Apply the interpretation criteria** without revision.
4. **Execute the decision tree** based on which outcome category the
   results fall into.

The morning-after analysis should take ~1-2 hours of focused work.
The decision (signal-positive / signal-negative / signal-ambiguous)
determines the next 1-2 weeks of project work.
