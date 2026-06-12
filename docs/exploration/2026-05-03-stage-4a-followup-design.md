# Stage 4a follow-up — N=100 re-pilot with normalize-mild, fail-severe parser

**Date:** 2026-05-03
**Status:** confirmed (PI revised the design 2026-05-03 — see Revisions)
**Companion:** `2026-05-03-stage-4a-pilot-n100.md` (the pilot writeup that motivates this)
**Architecture:** ADR-0007 (configurable parser tolerance on `OllamaLLMAdapter`)

---

## Revisions

Initial proposal accepted distributions in `[0.5, 2.5]` and normalised
them all to 1.0. PI revised on review: **normalising the 2.0 semantic-
confusion cases papers over the model's epistemic incoherence and
pollutes downstream signature signal in non-obvious ways.** Better to
fail those records loudly (data stays clean) and accept a higher
post-fix failure rate than to recover them with cosmetic
normalisation. Revised design uses tolerance `±0.05` (covers mild
arithmetic noise) and fails on anything outside.

---

## What this experiment tests

H1: relaxing the parser to accept-and-normalise distributions whose
sum is within `1.0 ± 0.05` (the "mild rounding" pattern observed at
medqa-test-5 with sum=1.10) drops Condition C's per-question failure
rate meaningfully — but does not necessarily eliminate it, since the
sum≈2.0 "semantic confusion" pattern still fails by design.

H2: the residual failures (post-fix) are concentrated in the
semantic-confusion class, which is a model-comprehension issue rather
than a parser issue. Distinguishing these classes is the point — H1
isolates the parser-tolerance bottleneck cleanly.

This is a *single, principled change* with a *single primary metric*
(post-fix Condition C failure rate). Two interesting branches:

- Failure rate drops near zero (~0-3 %): the original 18 % was
  almost entirely mild rounding. Parser tolerance was the bottleneck.
  Re-read AUCs against the original pre-committed criteria.
- Failure rate drops moderately (~5-12 %): mixed bag. The
  semantic-confusion fraction is what's left; need a different
  intervention (prompt engineering or larger model) for those.

## What changes

Three coordinated changes per ADR-0007:

**1.** `OllamaLLMAdapter.__init__` gets a new param:
```python
distribution_sum_tolerance: float = 0.05
```
Default is permissive (5 %) for typical use; tests and integration
harness can pass `0.0` (or a tiny epsilon) for strict.

**2.** `_parse_and_validate` takes the tolerance as an explicit param:
```python
def _parse_and_validate(
    raw: str,
    hypothesis_space: Sequence[str],
    sum_tolerance: float,
) -> Mapping[str, float]:
    ...
```

**3.** Validation logic — accept-and-normalise on mild rounding,
fail-loudly on severe:
```python
total = sum(distribution.values())
if abs(total - 1.0) <= sum_tolerance:
    if total != 1.0:
        # Mild arithmetic noise (e.g., 0.05 + 0.1 + 0.2 + 0.75 = 1.10).
        # Model intent is a distribution; normalise to satisfy
        # downstream consumers (FAISS index, signature components).
        distribution = {k: v / total for k, v in distribution.items()}
else:
    # Outside ±0.05: model isn't producing a coherent distribution.
    # The 2.0 "independent likelihoods" pattern lands here. Normalising
    # would preserve the rank but the values would not reflect the
    # model's epistemic state. Fail loudly to keep downstream signal
    # clean.
    raise ValueError(
        f"distribution sum {total} outside tolerance "
        f"[{1 - sum_tolerance}, {1 + sum_tolerance}]; "
        f"likely semantic incoherence rather than rounding error"
    )
```

The tolerance `±0.05` covers:
- Observed mild-rounding pattern (sum 1.05-1.15)
- Mild floating-point arithmetic noise

It excludes:
- The observed 2.0 semantic-confusion pattern (model treats entries
  as independent likelihoods rather than as a normalised distribution)
- Distributions summing to 0 (model returned all zeros — degenerate)
- Distributions far from 1.0 in either direction

## What does NOT change

- The strict `1e-6` check stays as the first branch. Well-formed
  outputs (the majority) parse without normalization side-effect.
- The retry / surgical-repair code stays exactly as-is.
- Temperature stays at 0.0 in the pilot. Stochastic-temperature
  fallback (proposed in the pilot writeup) is a separate, later
  experiment if normalize-and-accept doesn't reach <2 %.
- Prompt template stays unchanged. The prompt-design intervention
  (stronger sum-to-1 instruction) is a separate axis that would
  confound this experiment.

This is a *minimal-change* protocol: one toggle, one metric.

## How it runs

```bash
python experiments/medqa_generalization/scripts/03_pipeline_validation_ollama.py \
    --n-questions 100 \
    --embedder-backend sentence-transformers \
    --embedder-model intfloat/multilingual-e5-large \
    --embedder-prefix "" \
    --checkpoint-every 25 \
    --output-dir ~/work/eunosia/artifacts/medqa-stage-4a-pilot-n100-tol05/
```

Same fixed seeds, same prompts, same model, same hardware. Only the
parser changes (default tolerance `0.05` is picked up by the script
because the script constructs `OllamaLLMAdapter()` without overriding).

Wall-clock: ~100 minutes again (same throughput).

## Regression test fixtures (committed alongside the parser change)

Captured from real qwen2.5:7b output during the pilot diagnosis:

```python
def test_parser_normalises_mild_rounding():
    # medqa-test-5 prior, sum = 1.10
    raw = '{"A": 0.05, "B": 0.1, "C": 0.2, "D": 0.75}'
    result = _parse_and_validate(raw, ["A", "B", "C", "D"], sum_tolerance=0.05)
    assert math.isclose(sum(result.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(result["D"], 0.75 / 1.10, abs_tol=1e-9)

def test_parser_rejects_semantic_incoherence():
    # medqa-test-8 prior, sum ~= 2.0 (model emitted independent likelihoods)
    raw = '{"A": 0.9, "B": 0.85, "C": 0.05, "D": 0.05}'
    with pytest.raises(ValueError, match="outside tolerance"):
        _parse_and_validate(raw, ["A", "B", "C", "D"], sum_tolerance=0.05)
```

These pin the two failure-mode patterns directly observed in the
pilot. Future parser changes regression-test against the captured
evidence rather than synthetic hypotheticals.

## Pre-committed acceptance criteria

After the run completes:

1. **Read `repair_summary.json`. Look at C's `n_failures`.**
   - If `n_failures ≤ 3`: parser tolerance was the dominant
     bottleneck. Proceed to step 2 with full confidence in the AUCs.
   - If `3 < n_failures ≤ 10`: mixed bag — parser tolerance fixed
     part of it, semantic-confusion residual remains. Proceed to step
     2 but treat AUCs as still confounded; the residual selection
     bias matters less than at 18 % but still matters.
   - If `n_failures > 10`: the failure rate is dominated by
     non-rounding modes (or the parser change introduced new
     failures). Capture raw outputs of the new failures, re-diagnose;
     do NOT read the AUCs yet.

2. **Read `condition_comparison.csv`. Apply original
   `stage_4a_pre_run_analysis_plan.md` criteria mechanically.**
   - Signal-positive: C ≥ 0.55 AND C - B ≥ 0.03.
   - Signal-negative: C < 0.55 OR C - B ≤ 0.01 with overlapping CIs.
   - Signal-ambiguous: anywhere else.

3. **Spot-check the top-20 high-signature trajectories
   (`failure_mode_table.csv`).** Count how many have
   `high_score_correct_outcome=True`. With normalisation in place,
   if the framework is genuinely tracking failure, this fraction
   should be > 0.5; in the original pilot it was 7/20 = 0.35.

4. **Compare against the original pilot.** Did normalising change
   the AUC? By how much? This is informative regardless of direction:
   if AUC moves substantially when 18 % of cases get a normalised
   distribution, the signature is sensitive to the normalisation
   choice. If it barely moves, the original 18 dropped cases were
   not driving the score.

## What to do with each outcome

- **Failure rate ≤ 3 % AND C ≥ 0.55 AND C - B ≥ 0.03:** signal-positive
  on the corrected pipeline. Update `stage_4a_pre_run_analysis_plan.md`
  to reflect the corrected interpretation. Scope stage 4b (H100
  multi-variant).

- **Failure rate ≤ 3 % AND C < 0.55:** signal-negative on the corrected
  pipeline. The framework genuinely lacks signal at this configuration.
  Move to diagnostic cascade per the original plan: per-component AUC,
  embedding-bin sweep, prompt-variant sweep. Do NOT commit H100 budget.

- **Failure rate ≤ 3 % AND signal-ambiguous:** dominant case, per the
  original plan. Run per-component AUC breakdown to identify which
  component dominates and design the next intervention.

- **3 % < Failure rate ≤ 10 %:** parser fix partially worked.
  AUC-headlines are interpretable but with caveat. Two follow-ups:
  (a) prompt-engineering the sum-to-1 constraint (cheap, addresses
  semantic-confusion mode directly), (b) larger model (qwen2.5:14b,
  ~2-3 h pilot) to test whether the residual is model-capability.
  Run both, in either order.

- **Failure rate > 10 %:** the parser fix didn't address the dominant
  mode. Re-diagnose with raw-output capture; possible follow-ups
  include prompt engineering, larger model, structured-output mode
  if Ollama supports it for the model in question.

## Subtle interpretation caveat

The questions whose hypothesis-distribution outputs went from "fail"
to "succeed" under the relaxed parser may be systematically different
from the questions that succeeded under the original strict parser.
Specifically, the model's tendency to produce mild-rounding
distributions might correlate with question difficulty or with cases
where it has weak relative preferences (entropy higher → more
even-looking probabilities → arithmetic noise more likely to push the
sum off-1).

If those (now-recovered) cases are systematically the
high-entropy "boundary" cases the framework's signature is supposed
to flag, including them post-fix could *strengthen* the signal. If
they're noise, including them could weaken it. Worth being aware of
when reading the AUC delta vs the original pilot:

- AUC improves substantially: the recovered cases were carrying
  signal; the framework operates well when the failure rate is
  controlled.
- AUC stays flat or worsens: the recovered cases were noise; the
  framework's signal lives in the cases that always succeeded.

Both are informative. Neither falsifies the framework — the diagnostic
trail continues either way.

## Why this and not the alternatives

Considered and rejected for this experiment:

- **Prompt-engineer the sum-to-1 constraint.** Higher signal-quality
  if it works (the model produces a real distribution), but a much
  bigger intervention. Conflates parser-tolerance with prompt-design,
  making the result less interpretable. Defer to follow-up if
  normalize-and-accept doesn't suffice.

- **Switch to structured-output mode.** Ollama supports JSON-schema
  enforcement on some models. Would eliminate the parse failure
  entirely. But it's a substantial code path change and depends on
  Ollama version + model support. Higher implementation cost than
  the experiment justifies.

- **Bump repair temperature to 0.3.** Cheap (one-line change), but
  doesn't address the root cause (the model's first-pass output is
  the failure; retries with different temperature might work or
  might just produce different bad outputs). Combine with normalize
  if needed.

- **Switch to a larger model (qwen2.5:14b).** Doesn't isolate the
  fix; conflates model capability with parser tolerance. Real
  scientific experiment, but for a different question.

- **Skip the pilot and go straight to N=1273 with the fix.** Higher
  blast radius if the fix is wrong (~21 hours wasted). The pilot
  cost is ~100 minutes; there's no reason to skip it.

## Out-of-scope, but worth noting

The pilot also surfaced two non-blocking observations that deserve
follow-up after this re-pilot completes:

- **Wall-clock 1 q/min vs original 35 s/q estimate.** N=1273 is
  ~21 h not 12-15 h. M1 Pro-only stage 4a is feasible but slow.
  Worth checking against the M1 Pro thermal-throttling profile and
  whether sustained inference is hitting any cap.

- **Recovery sparsity** (429 nodes / 347 edges / 429 visits → avg
  out-degree 0.81). Most edges likely freq=1; the embedding-bin
  precision sweep (ADR-0002) is overdue. Defer to after the AUC
  story is clear; if signal-positive, sparsity matters less; if
  signal-negative, sparsity is one of the diagnostic axes.
