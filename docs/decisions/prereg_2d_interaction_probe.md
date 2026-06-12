# Pre-registration — 2D serial×parallel INTERACTION probe (cache-only)

**Date:** 2026-06-10 (written *before* computing any AUC; data schema
inspected, outcomes not).
**Status:** PRE-REGISTERED.
**Script:** `experiments/medqa_generalization/scripts/14_2d_interaction_probe.py`
**Data:** ZERO new inference. Joins two existing N=1273 caches by
`question_id` (= `trajectory_id` = "medqa-test-N"):
- Serial axis (MLX-4bit trajectory): `artifacts/medqa-stage-4a-n1273/condition_c_cached/states.parquet`
  → per-question `mean_entropy` (mean over reasoning steps of Shannon entropy
  in bits of the renormalised A/B/C/D `hypothesis_distribution`).
- Parallel axis (cross-quantization terminal disagreement): `artifacts/medqa-cross-quant/cache.jsonl`
  → `js_div` (Jensen-Shannon divergence between MLX-4bit and GGUF-Q4_K_M
  terminal letter distributions), plus `disagree`, `correct`.
- Label: `y_wrong = 1 - correct`, where `correct = (MLX-4bit argmax == gold)`
  — the deployed prediction, same model as the serial trajectory.

## What this tests (and what it does NOT)

The literature check (2026-06-10, [[project_2d_serial_parallel_novelty]])
found the confident-but-disagreeing interaction is DONE **cross-MODEL and as
an ADDITIVE sum** TU=AU+EU (Hamidieh et al., arXiv:2604.17112). The defensible
novelty is the INTERACTION (not the sum) on **cross-quantization** replicates
of one model. This probe is a cheap **go/no-go gate** on the interaction
before paying for a per-step dual-codec run.

The decisive question: **does the 2D interaction term beat the additive
sum?** AU-analog = `mean_entropy` (within-sample, serial). EU-analog =
`js_div` (cross-replicate disagreement, parallel).

NOT tested here (deferred, needs new inference):
- **Per-step resolution.** `js_div` is terminal-only (single direct-answer
  prompt), NOT resolved over reasoning steps. The "disagreement evolves over
  the serial trajectory" novelty (Q5) is NOT in this probe.
- Path-divergence (DTW/Fréchet), Condition-B verbalized confidence, the
  P3/P4 graph composite (dead on single-trajectory MCQ).
- A positive result is **distributional×perturbation interaction
  complementarity only** — a narrow claim.

## Honest prior (NOT an open slate)

Marginal cross-quant is already known REDUNDANT with `mean_entropy`
(stage-6: Spearman +0.507, standalone AUC 0.526 null,
[[project_mean_entropy_deployable]]). So the additive model M1 is expected to
barely beat M0. The interaction hypothesis is specifically that signal
concentrates in the CONFLICT cell (low `mean_entropy` AND high `js_div`) that
the marginal average washes out. **Modal expected outcome per discipline:
NULL** (recombination, no gain over the additive sum) — the field's additive
TU already captures most of it. The pre-reg is written expecting null; a
positive is the surprising, publishable outcome.

## Models (5-fold stratified CV, standardized features, logistic, seed=42)

- **M0** (serial baseline): `mean_entropy`
- **M1** (additive / Hamidieh-analog): `mean_entropy + js_div`
- **M2** (2D interaction): `mean_entropy + js_div + mean_entropy:js_div`

Out-of-fold predicted probabilities → `roc_auc_score(y_wrong, oof)`.

## Pre-registered comparisons + thresholds (committed before running)

1. **PRIMARY — interaction over additive:** ΔAUC = AUC(M2) − AUC(M1), paired
   bootstrap (B=2000) over questions.
   - **SIGNAL** (2D interaction is real, worth a per-step run) if ΔAUC ≥
     **0.02** AND 95% CI excludes 0.
   - **NULL** (recombination, stop) if ΔAUC ≤ 0 or CI includes 0.
2. **SECONDARY — additive over serial:** ΔAUC = AUC(M1) − AUC(M0), paired
   bootstrap. Tells whether the parallel axis adds anything at all
   additively. Prior expectation: ≈0 (redundancy).
3. **DESCRIPTIVE — conflict-quadrant contingency:** confident =
   `mean_entropy` < median; disagree = `js_div` > 75th pct. Report
   `y_wrong` rate + Wilson 95% CI in each of the 4 cells. Hypothesis:
   error elevated in {confident AND disagree} vs {confident AND agree}.
4. **REDUNDANCY CHECK:** Spearman(`js_div`, `mean_entropy`). Prior interval
   [0.3, 0.7] (transfers from stage-6 0.507).

## Decision rule

PRIMARY drives the go/no-go. SIGNAL → pre-register and fund the per-step
dual-codec run (the true 2D matrix M[step][replicate]). NULL → cross-quant
disagreement carries no deployable signal beyond `mean_entropy` even as a 2D
interaction on terminal disagreement; closes the interaction direction on
this data and documents it as a clean negative.

## RESULT (2026-06-10, N=1273, run as pre-registered)

Artifact: `artifacts/medqa-2d-interaction-probe/summary.json`. wrong_rate=0.405.

- **PRIMARY — NULL (as expected).** AUC M2(interaction)=0.7163 vs
  M1(additive)=0.7150. ΔAUC = **+0.0013**, 95% CI [−0.0030, +0.0053]. The
  2D interaction adds nothing over the additive sum. The "2D" framing
  collapses to Hamidieh's additive TU=AU+EU form. **Do NOT fund the
  per-step dual-codec run on the interaction premise.**
- **SECONDARY — positive.** M0(mean_entropy)=0.6841 → M1(additive)=0.7150.
  ΔAUC = **+0.0309**, 95% CI [+0.0173, +0.0446], excludes 0, clears the
  0.02 convention. Cross-quantization disagreement is a real, cheap,
  ADDITIVE feature (one extra quantized forward) — NOT redundant-dead. This
  is the surviving deployable result and it sits on the cross-quantization
  novelty axis ([[project_compute_constraint_orientation]]).
- **DESCRIPTIVE — strong but thin.** confident&disagree wrong=0.637 (n=80)
  vs confident&agree=0.236 (n=556): disagreement ~triples error within the
  confident stratum. The confident-but-disagreeing pattern is real but lives
  in a 6%-of-data slice (threshold-like, low-entropy tail), so it does not
  move global AUC and the linear interaction term missed it. This is a
  STAGE-4 HYPOTHESIS, not a result — a high-precision triage flag worth its
  OWN pre-registered threshold-operationalized test, NOT a post-hoc rescue
  of the null primary.
- **REDUNDANCY — confirmed.** Spearman(js_div, mean_entropy)=0.538, inside
  the predicted [0.3, 0.7].

Bottom line: 2D-interaction direction CLOSED on this data (clean negative);
cross-quant-as-additive-feature CONFIRMED (+3pp); confident-but-disagree
threshold effect logged as a separate stage-4 hypothesis.
