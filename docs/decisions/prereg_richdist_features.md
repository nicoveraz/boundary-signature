# Pre-registration — richer-distribution features (varentropy, full-vocab entropy, EPR)

**Date:** 2026-06-12 (written *before* the inference run).
**Status:** PRE-REGISTERED. First new-inference experiment in this line
(probes 14-18 were cache-only).
**Script:** `experiments/medqa_generalization/scripts/19_richdist_capture.py`
**Run:** MLX `Qwen2.5-7B-Instruct-4bit` on M1 Pro, MedQA-USMLE test split,
**N=200** (go/no-go subset; ~1.5-2.5h). Condition-C protocol EXACTLY as the
paper (same `condition_c_initial` / `condition_c_measurement` prompts, same
`Decomposer`) — reusing `ConditionC._format_initial_prompt` /
`_build_measurement_prompts` — but capturing the FULL top-K logprobs per
measurement (schema-v3 object the N=1273 v2 cache discarded). Resumable JSONL.

## What this tests

The cache-only sweep (§5.8, [[project_serial_features_exhausted]]) exhausted the
**first-moment entropy of the renormalised A/B/C/D distribution**. Literature-proven
cheap candidates that operate on a DIFFERENT measurement object remained untestable
because the v2 cache kept only the 4-way distribution. This run captures the richer
object (full next-token top-K) and tests whether features on it COMPLEMENT
`mean_entropy`:
- **Varentropy** (LogitScope/entropix): variance of surprisal over the full top-K —
  the *second moment*, not the first. Different object.
- **Full-vocab entropy** (`entropy_full`): entropy over the whole next-token
  distribution, not just A/B/C/D. Captures off-answer commitment.
- **EPR-style rate** (arXiv:2509.04492): slope/rate of full-vocab entropy across steps.

Sequence perplexity (Malinin & Gales) is NOT captured here (needs generation-token
logprobs, not exposed by `adapter.generate`); deferred.

## Honest prior

STRONG prior toward NULL: five probes already showed the answer-distribution exhausted.
These survive only if the full-distribution shape carries signal the 4-way entropy
discards. Varentropy is the best shot (genuinely 2nd-moment). Realistic: comparable or
weaker than `mean_entropy`; the live question is incremental ≥0.02.

## Pre-registered test + thresholds (committed before the run)

Per-question features = mean over steps of: `ent4` (4-way entropy, BASELINE — must
reproduce ≈0.686-shape AUC as sanity), `entropy_full`, `varentropy`, plus
`entropy_full` slope (EPR). Label = Condition-C terminal argmax vs gold.

1. **PRIMARY — incremental over mean_entropy**, per feature: 5-fold CV logistic AUC of
   {mean_entropy} vs {mean_entropy, feature}, paired bootstrap (B=2000).
   - **SIGNAL** if ΔAUC ≥ **0.02** AND 95% CI excludes 0 → real-time-deployable
     complement (same single forward pass) → pre-register held-out validation on a
     second domain/model before any claim.
   - **NULL** if ≤0 or CI includes 0.
2. **SANITY:** `mean_entropy` standalone sign-aware AUC must land in [0.62, 0.74]
   (consistent with the N=1273 0.686 at N=200 CI width); else the lean replication
   diverged from the paper protocol and the run is void.
3. **DESCRIPTIVE:** Spearman(varentropy, mean_entropy) and (entropy_full, mean_entropy)
   — redundancy diagnostic.

## Decision rule

Any feature clearing PRIMARY → fund held-out validation; it is a real-time-compatible
improvement (one forward pass). All null → the cheap single-model single-pass ceiling
(~0.69, [[project_serial_features_exhausted]]) is confirmed on the richer object too;
lock `mean_entropy` for Eunosia real-time, exploration of cheap complements CLOSED.

## RESULT (2026-06-12, N=200, MLX Qwen2.5-7B-Instruct-4bit, run as pre-registered)

Artifact `artifacts/medqa-richdist-capture/`. 200/200 captured, 0 failures.
wrong_rate=0.370. **SANITY PASSED:** mean_entropy sign-aware AUC 0.6831 ∈ [0.62,0.74],
matches paper 0.686 — lean replication valid.

- **entropy_full:** standalone 0.667, incr −0.003 [−0.012,+0.005] NULL; **Spearman 0.907
  with mean_entropy** → had signal but redundant (at measurement positions answer letters
  carry most mass, so full-vocab entropy ≈ 4-way entropy).
- **varentropy:** standalone 0.538, incr −0.000 [−0.011,+0.009] NULL; **Spearman 0.331**
  → genuinely orthogonal (as predicted, the only low-redundancy feature) but EMPTY (near
  chance). The 2nd moment carries no correctness information here.
- **entfull_slope (EPR):** standalone 0.517, incr −0.010 [−0.020,−0.000] NULL/negative.
- **All three + mean_entropy:** incr −0.016 [−0.035,+0.000] NULL.

**Decision: ALL NULL → cheap single-model single-pass ceiling (~0.69) confirmed on the
richer distribution object too. The one feature that was a genuinely different object
(varentropy) has no signal; the one with signal (entropy_full) is redundant. Cheap-complement
exploration CLOSED with real inference. Lock mean_entropy for Eunosia real-time.**
