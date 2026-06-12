# Pre-registration — confident-but-disagreeing triage flag (cross-model held-out)

**Date:** 2026-06-10 (written *before* touching the held-out validation
cache; discovery cache already seen).
**Status:** PRE-REGISTERED.
**Script:** `experiments/medqa_generalization/scripts/15_confident_disagree_flag.py`
**Data:** ZERO new inference.
- **Discovery (already seen — contaminated):** MedQA-Qwen N=1273. The 2D
  interaction probe ([[project_2d_interaction_probe_result]]) found, as a
  pre-registered DESCRIPTIVE output, that within the confident stratum
  cross-quantization disagreement triples error: confident&disagree
  wrong=0.637 (n=80) vs confident&agree=0.236 (n=556). The linear logistic
  interaction term did NOT capture this (global ΔAUC +0.001) — the effect is
  threshold-like and lives in a 6%-of-data slice.
- **Held-out (NOT yet seen):** MedQA-Llama N=1273, `artifacts/medqa-cross-quant-llama/cache.jsonl`
  — same MedQA questions, DIFFERENT base model (Llama, MLX-4bit vs GGUF-Q4_K_M).

## Hypothesis

The confident-but-disagreeing cell is a **high-precision error-triage flag**
(not a ranking signal — it does not move AUC) that **transfers across base
models**. Operationally: a deployment can flag, label-free, the cases where
the model is internally confident yet its two quantizations disagree, and
that cell carries a much-elevated error rate.

## Operationalization (parameter-free, label-free, fixed a priori)

Per record: `confident = terminal_entropy < within-model median`;
`disagree = js_div > within-model 75th percentile`; `flag = confident AND
disagree`. Confidence uses TERMINAL entropy (`mlx_entropy` field, present in
both caches) — NOT serial mean_entropy — because no Llama per-step trajectory
exists at N=1273. This operationalization shift (serial→terminal) is itself a
robustness probe: a flag that survives it is MORE deployable (one measurement,
no trajectory). Thresholds are within-model percentiles, so the flag needs no
labels and no cross-model calibration. Applied IDENTICALLY to Qwen (reference)
and Llama (held-out test).

## Pre-registered test + thresholds (committed before seeing Llama)

Primary, on held-out MedQA-Llama:
1. **Lift:** `wrong_rate(flag) − wrong_rate(confident&agree)`.
   - **VALIDATES** if lift ≥ **0.15** AND the flag-cell wrong-rate Wilson 95%
     CI lower bound exceeds the Llama base wrong-rate.
   - **FAILS TO REPLICATE** if lift < 0.15 or the CI overlaps base rate.
   - (Discovery lift was 0.40 on Qwen-serial; 0.15 is a deliberately
     conservative transfer threshold across model + operationalization change.)
2. **Precision/coverage descriptive:** report flag-cell wrong-rate, n flagged
   (coverage), and the full 2×2 contingency with Wilson CIs, both models.

Honest prior: discovery was strong but n=80 and on a different model +
operationalization. Modal expectation per discipline: PARTIAL — the direction
transfers but attenuated; whether it clears 0.15 is the live question. This is
the [[project_mass_capture_as_signal]] pattern (decile lift medical-specific,
central null) re-examined for cross-quant disagreement.

## Scope / decision rule

VALIDATES → a deployable, model-portable triage flag; report in the methods
paper as a narrow secondary contribution. FAILS → the confident-but-disagree
effect is Qwen/medical-specific, logged as a non-replicating tail like P5b.
Either way: NOT a ranking/AUC claim, NOT the 2D-interaction claim (that is
closed, [[project_2d_interaction_probe_result]]).

## RESULT (2026-06-11, run as pre-registered)

Artifact: `artifacts/medqa-confident-disagree-flag/summary.json`.

- **PRIMARY — VALIDATES (held-out Llama).** confident&disagree wrong=0.657
  (n=67) vs confident&agree=0.234 (n=569). Lift **+0.423** (≥0.15); flag-cell
  Wilson CI [0.537, 0.759] **excludes** the Llama base rate 0.434. Both
  committed criteria met. The triage flag transfers to a different base model.
- **Caveat (reported, not buried).** Under the matched TERMINAL-entropy
  operationalization, the Qwen reference cell is WEAKER: lift +0.247, CI
  [0.346, 0.632] OVERLAPS base 0.405 (the original Qwen discovery used SERIAL
  mean_entropy, lift +0.40). So flag strength is operationalization-sensitive,
  and Qwen-terminal vs Llama-terminal are NOT statistically distinguishable
  (CIs overlap; flag cells are thin, n=43/67). Held-out passed its committed
  bar — that is the disciplined verdict — but the effect is a 5%-coverage
  high-precision triage flag, NOT a ranking signal and NOT large-n robust.
- Decision: report as a narrow, hedged secondary contribution (deployable
  triage flag, model-portable in direction, thin slice). A larger-n /
  serial-operationalization confirmation is the natural follow-up.
