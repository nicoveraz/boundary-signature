# Cross-quantization disagreement (E_quant_3) vs correctness on MedQA

**Date:** 2026-05-26
**Status:** RESULT.
**Script:** `experiments/medqa_generalization/scripts/12_cross_quant_comparator.py`
**Pre-registration:** `docs/decisions/prereg_cross_quant_comparator.md`
(thresholds + the stage-6 0.507 redundancy prior committed before running).

## Question

Does E_quant_3 — same-model cross-quantization disagreement (MLX-4bit vs
GGUF-Q4_K_M Qwen2.5-7B), the framework's one calibrated-novel signal (item 4
of `contribution_shape_post_literature.md`) — add information over cheap
single-run mean_entropy on a CLEAN correctness target, given the known
stage-6 redundancy (Spearman 0.507)? Same cohort/structure as the
semantic-entropy comparator.

## Setup

N=150 MedQA test (joins the semantic cohort by question_id). Both codecs
queried with `get_token_probabilities` over the answer letters at the
identical direct-answer prompt — 1 constrained forward each. Continuous
codec divergence (Jensen-Shannon, primary; L1, secondary) is the rankable
standalone signal; boolean argmax-disagreement is a secondary contingency.
`correct` = MLX-4bit argmax vs gold. Run time ~8 min (~3s/question — minutes,
not the ~1h estimated; the GGUF arm is a single forward, not a generation).

## Result (N=150, 5000-bootstrap)

| pre-registered test | threshold | result | verdict |
|---|---|---|---|
| (1) standalone JSD sign-aware AUC | ≥0.65 | **0.712** [0.628, 0.793] | clears (point est; CI low 0.628) |
| (1) standalone L1 (secondary) | — | 0.733 | — |
| (1) mean_entropy reference | — | 0.762 | reproduces semantic-comparator run |
| (2) incremental JSD over mean_entropy (CV logistic) | ≥0.02, CI excl 0 | **+0.009** [−0.010, +0.030], P(inc>0)=0.80 | **fails** |
| (3) Spearman(JSD, mean_entropy) | prior [0.3, 0.7] | **+0.733** [0.627, 0.814] | above interval (more redundant than prior) |
| (3) Spearman(JSD, semantic_entropy) | [0.0, 0.4] | **+0.253** [0.095, 0.399] | within (distinct from sampling) |
| (4) boolean argmax-disagree contingency | descriptive | 13% disagree; wrong-rate 0.70 vs 0.30 | see caveat |

## Reading — calibrated

1. **The signal is real on a clean target.** Standalone JSD 0.712 clears the
   0.65 bar (point estimate) and is far above chance. This **overturns the
   stage-6 disposition null (0.526): that null was GT-confound, not signal
   weakness.** Cross-quant disagreement predicts correctness when the target
   is aligned. mean_entropy ref 0.762 reproduces the semantic-comparator
   number (internal consistency).

2. **But it is redundant — the decisive test fails.** Incremental over
   mean_entropy is +0.009 (CI includes 0, below the 0.02 bar). Spearman
   0.733 — *above* the prior's [0.3,0.7] upper bound — explains why: JSD is
   an even more redundant re-measurement of what mean_entropy already
   captures than stage-6's 0.507 implied. **This is the base-rate outcome
   the prior predicted**; redundancy now confirmed across BOTH targets
   (disposition 0.507 → correctness 0.733), so it is a GT-independent
   property of the signal pair.

3. **The boolean flag is striking but not shown independent.** Codec
   disagreement (13% of questions) marks a 70%-wrong subset vs 30% on
   agreement. Interpretable, but given the 0.733 redundancy it almost
   certainly coincides with high-mean_entropy questions — NOT demonstrated to
   be distinct from what mean_entropy already flags. Not sold as a separate
   deployable signal without that test.

4. **Compute-constraint reading (the sharp one).** E_quant_3 requires
   standing up a *second full codec / inference stack* and yields zero
   significant incremental information over the single-run signal already in
   hand. On the framework's own compute-constraint terms it **doubles the
   inference footprint for no benefit** — actively not worth it on this task.

## Verdict for the contribution claim (item 4)

The operationalization *works* (valid standalone correctness predictor;
clean-target null overturned) but is *dominated by mean_entropy* (no
significant incremental information; Spearman 0.73). **It does not earn
first-class independent-signal status.** Item 4 narrows to: "same-model
cross-quantization disagreement is a valid but redundant correctness
predictor, dominated by single-run mean_entropy; the second-codec inference
cost is not justified by incremental information on MCQ." A bounded NEGATIVE
for E_quant_3's incremental value — the discipline worked: the prior-informed
pre-reg called the outcome, recorded without spin.

## Bounds
Correctness-prediction, NOT uncertainty-measurement (failure #5). N=150, one
base model, MCQ. **Does NOT test the graph-structural composite (P3/P4)** —
that is distributional+structural complementarity on multi-trajectory
substrate, dead on single-trajectory MCQ; a favorable mean_entropy+JSD result
would not have rescued it, and this result is in any case not favorable.
Robustness gate (full N=1273 + second base model) folds E_quant_3 in
alongside the semantic comparator.

## UPDATE — full-N=1273 + second model (Llama-3-8B), 2026-05-26

Robustness gate run at full N on Qwen2.5-7B AND Llama-3-8B (cheap arms only;
the ~10h×2 semantic replication was stopped — mean_entropy's value is already
established at 0.69/0.66/0.71, re-confirming it was the wrong use of compute).
Scripts: `12_cross_quant_comparator.py` (`--mlx-model`), diagnostics
`13_cross_model_diagnostics.py`. **mean_entropy held at 0.71 on BOTH models**
(≈ canonical 0.686) — the workhorse is robust cross-architecture.

| metric (N=1273) | Qwen2.5-7B | Llama-3-8B |
|---|---|---|
| accuracy | 59.5% | 56.6% |
| mean_entropy AUC | 0.713 | 0.712 |
| cross-quant JSD standalone | 0.695 [0.667,0.724] | 0.697 [0.667,0.724] |
| incremental over mean_entropy | +0.011 [0.003,0.019] | +0.028 [0.015,0.041] |
| vs 0.02 materiality bar | below (full CI <0.02) | point clears; CI straddles |
| Spearman(JSD, mean_entropy) | 0.729 | 0.636 |
| boolean wrong-rate disagree vs agree | 0.70 vs 0.35 (16%) | 0.76 vs 0.34 (22%) |

**Materiality threshold earned its keep:** at full N the Qwen incremental is
*significant but immaterial* (+0.011, CI excludes 0 yet entirely below 0.02).
Pre-registering materiality — not just "CI excludes 0" — is what prevents
over-reading a 0.011 bump as "E_quant_3 adds significant information."

**Model-dependence is REAL but MODEST, and partly a CONFOUND** (diagnostics
`13_…`, paired bootstrap on the shared 1273 questions):
- *Move 1:* ΔSpearman(Qwen−Llama) +0.093 [0.049,0.137] and Δincremental
  −0.0166 [−0.0316,−0.0017] both exclude 0 — the difference is real, but the
  incremental-Δ upper bound is a hair from 0 and both increments are small.
- *Move 2:* NOT a mean_entropy dynamic-range difference — mean_entropy std is
  nearly identical (0.550 Qwen / 0.559 Llama). What differs is JSD itself:
  ~2× larger and more spread on Llama (std 0.134 vs 0.081).
- *Move 3:* On AGREE & CORRECT cases (model certain+right, no
  quantization-induced answer uncertainty) Llama's codecs still diverge
  **1.81×** more (mean JSD 0.034 vs 0.019). Llama's two conversions
  (max-cache GGUF vs mlx-community 4bit) are **baseline-further-apart** than
  Qwen's (bartowski GGUF vs mlx-community 4bit).

**Reinterpretation (supersedes the earlier Qwen-only "redundant" verdict):**
E_quant_3's apparent extra value on Llama is at least partly a *conversion-
pair-distance* confound, NOT a clean architecture-level truth. The cross-quant
signal's magnitude scales with how far apart the two specific quantization
pipelines are — it partly indexes "how different are my two conversions"
rather than "how uncertain is the model." Pairing same-lineage codecs would
likely shrink Llama's increment toward Qwen's. So **neither "redundant"
(Qwen) nor "additive" (Llama) is elevatable to an architecture claim;** the
honest verdict for item 4 is: *valid standalone on both models, incremental
value over mean_entropy small and model/conversion-pair dependent, never
cleanly clearing materiality — and confounded by quantization-pipeline
distance, which weakens it as a principled UQ signal beyond the redundancy
already known.* mean_entropy remains the robust workhorse.

*Caveat on Move 3:* the agree&correct JSD gap could partly reflect Llama's
flatter distributions (higher mean_entropy 0.84 vs 0.71) inflating
distributional distance — consistent with conversion-divergence, but the
baseline-entropy alternative is not ruled out.

## Cross-references
- `docs/decisions/contribution_shape_post_literature.md` item 4 (narrowed).
- `docs/decisions/prereg_cross_quant_comparator.md`.
- `docs/exploration/2026-05-26-medqa-semantic-entropy-comparator.md` (sibling).
- stage-6 `08_eval_e_quant3.py` (the confounded-GT 0.526/0.507 prior).
