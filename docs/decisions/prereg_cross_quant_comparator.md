# Pre-registration — cross-quantization (E_quant_3) comparator on MedQA

**Date:** 2026-05-26 (written *before* the run; smoke only).
**Status:** PRE-REGISTERED.
**Script:** `experiments/medqa_generalization/scripts/12_cross_quant_comparator.py`
**Run:** N=150 MedQA-USMLE test (same cohort/order as the semantic-entropy
comparator, so question_ids join), Qwen2.5-7B-Instruct in TWO codecs —
MLX-4bit (in-process) and GGUF-Q4_K_M via llama-server :8085 — each queried
with `get_token_probabilities` over the answer letters at the **identical**
direct-answer prompt ("...The single best answer is:"). 1 constrained
forward per codec per question. Resumable JSONL cache.

## What this tests

E_quant_3 — same-model **cross-quantization disagreement** as a
deployment-cheap perturbation UQ signal (item 4 of
`contribution_shape_post_literature.md`) — on a CLEAN correctness target.
Prior stage-6 test was against disposition GT (confounded); this is the
methodologically-aligned MCQ-correctness test, same structure as the
semantic-entropy comparator.

## Honest prior (NOT an open slate)

E_quant_3 was already run at stage-6 (N=1000, disposition GT): standalone
disagreement_rate **AUC 0.526 (null)** AND **Spearman +0.507 with
mean_entropy (moderately redundant)**. The disposition nullness is plausibly
GT-confound, so re-testing standalone on clean correctness is worthwhile.
BUT the **0.507 redundancy is a signal-to-signal property, GT-independent,
and will very likely transfer.** So the prior expectation is: cross-quant is
*moderately redundant* with mean_entropy; the live question is whether it
clears incremental AUC ≥0.02 *despite* that redundancy. The pre-reg is
written expecting redundancy to be the modal outcome.

## Signal definition (MCQ-specific)

On single-step MCQ, disagreement is **boolean** per question (one
argmax-vs-argmax) → a degenerate 2-point ROC. So the rankable standalone
signal is the **continuous codec divergence**: Jensen-Shannon divergence
(primary) and L1 distance (secondary) between the two codecs' renormalised
letter distributions. The boolean argmax-disagreement is reported as a
contingency (wrong-rate | agree vs disagree), not as the AUROC signal.
`correct` = MLX-4bit argmax vs gold (the deployed prediction, consistent
with the semantic comparator).

## Pre-registered comparisons + thresholds (committed before running)

1. **Standalone:** sign-aware AUC of codec JSD vs `y_wrong`, bootstrap CI.
   - validates if AUC ≥ **0.65** (original E_quant_3 spec); falsified
     (doesn't predict correctness independently) if AUC ≤ **0.55**.
2. **Incremental over mean_entropy:** 5-fold CV logistic AUC of
   {mean_entropy} vs {mean_entropy, JSD}, paired bootstrap on the gap.
   - earns its complexity if incremental AUC ≥ **0.02** with CI excluding 0;
     dominated by mean_entropy if ≤ 0.
3. **Agreement Spearman(JSD, mean_entropy):** prior-informed prediction
   interval **[0.3, 0.7]** (centered on the stage-6 0.507). <0.3 = genuinely
   orthogonal (interesting, unexpected); >0.7 = substantially overlaps
   mean_entropy (distinctness constrained).
4. **Agreement Spearman(JSD, semantic_entropy):** predict **[0.0, 0.4]**
   (different mechanism — perturbation vs sampling). Joined from the
   semantic-entropy cache by question_id.
5. **Boolean disagreement contingency:** wrong-rate among argmax-disagree vs
   argmax-agree questions; report the rate and the lift, descriptive.

## Scope / what is NOT in this run (deferred)

- **Verbalized confidence (Condition B)** is NOT measured here → the
  three-axis composite (distributional + verbalized + perturbation) is a
  separate follow-up needing a Condition-B run, not part of this.
- This does **NOT** test the graph-structural composite (P3/P4) — that is
  dead on single-trajectory MCQ (CLAUDE.md §15). A favorable
  mean_entropy+JSD incremental AUC is *perturbation+distributional*
  complementarity, a narrower claim; it must NOT be cited as rescuing the
  P4 walk-back.
- Correctness-prediction, NOT uncertainty-measurement (failure #5 applies).

## Power / scope
N=150 (joins the semantic cohort). If standalone or incremental is
borderline, extend to full N=1273 + a second base model — folded into the
SAME robustness gate already set for the semantic-entropy result
(`2026-05-26-medqa-semantic-entropy-comparator.md`).

## Cross-references
- `contribution_shape_post_literature.md` item 4 (the claim under test).
- `project_cross_quantization_disagreement.md`, stage-6 `08_eval_e_quant3.py`.
- `prereg_semantic_entropy_comparator.md` (the sibling comparator).
