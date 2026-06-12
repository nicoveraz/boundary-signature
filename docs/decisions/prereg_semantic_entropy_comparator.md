# Pre-registration — semantic-entropy comparator on MedQA

**Date:** 2026-05-26 (written *before* the powered run completed; the N=3
smoke produced no meaningful AUC, and the powered run was in flight with
no `_evaluate` output yet when this was committed).
**Status:** PRE-REGISTERED.
**Script:** `experiments/medqa_generalization/scripts/11_semantic_entropy_comparator.py`
**Run:** N=150 MedQA-USMLE test questions, 6 CoT samples/question, temp 1.0,
max-tokens 256, Qwen2.5-7B-Instruct-4bit (MLX), pinned mlx 0.31.2
(serial batch=1 generation — the rope batched-decode bug does not apply).
Resumable JSONL cache `~/work/eunosia/artifacts/medqa-semantic-entropy/cache.jsonl`.

## Why this comparator exists

The framework's compute-constraint claim ([[project_compute_constraint_orientation]],
`contribution_shape_post_literature.md` item 1) is: cheap single-run UQ is
competitive with expensive sampling-based UQ at deployment cost. Semantic
entropy (Kuhn et al.; the core metric in the Anatomy of Uncertainty paper)
is *the* established sampling-based comparator. This is the head-to-head.

## The conceptual distinction this pre-reg enforces (load-bearing)

**Correctness-prediction ≠ uncertainty-measurement.** Measuring AUROC of a
signal against correctness tests whether the signal *predicts correctness*.
It does **not** test whether the signal *measures uncertainty* as a latent
construct. These are routinely conflated (in this project's prior framing,
and in the applied-UQ field broadly — Kuhn, Anatomy, and the
selective-prediction literature all validate by correctness-correlation).
This comparator is explicit about which question each number answers.

## Pre-registered comparisons (two questions, three numbers)

### (1) Correctness-prediction parity — the primary, deployment-relevant test
- **Metric:** sign-aware AUC of `single_run_entropy` and `semantic_entropy`
  against `y_wrong = 1 - correct`, each with a 2.5/97.5 bootstrap CI
  (5000 resamples, reoriented to the point-estimate direction).
- **Headline:** the **paired** bootstrap lift `semantic_AUC − single_run_AUC`
  (same resample for both), its 95% CI, and `P(lift>0)`.
- **Pre-registered reading:**
  - lift CI **includes 0** → cheap single-run predicts correctness on par
    with expensive sampling → **supports** the compute-constraint claim
    (parity at lower cost). This is the outcome the framework's claim needs.
  - lift CI **cleanly > 0** (and material, say ≥ +0.03) → sampling carries
    extra correctness-prediction signal single-run misses → **tempers** the
    claim; report the cost/benefit honestly.
  - either signal's CI includes 0.50 → that signal is not a reliable
    correctness predictor at this N; report as null, do not spin.

### (2) Signal agreement — secondary, independent of correctness
- **Metric:** `Spearman(single_run_entropy, semantic_entropy)` over all
  questions, with a bootstrap CI.
- **Pre-registered reading:** high Spearman = the signals order questions
  similarly. **Caveat registered in advance:** on a 4-option MCQ, semantic
  equivalence collapses to "same answer letter," so semantic entropy
  degenerates to letter-agreement entropy on a tiny shared support. A high
  Spearman is therefore **partly mechanical** and is **weak** convergent-
  validity evidence — NOT proof the two measure the same latent. The strong
  convergent-validity test (open-ended generation, non-trivial semantic
  clustering) is explicitly out of scope here.

### What is NOT claimed regardless of outcome
Neither (1) nor (2) establishes that any signal **measures uncertainty** as
a construct. That requires convergent validity across distinct distributional
measures, behavior under controlled uncertainty manipulation, and
calibration-regime discrimination — none of which this run performs. The
defensible claim this run can support is, at most: *single-run predictive
entropy provides correctness-prediction comparable to sampling-based
semantic entropy on MedQA at substantially lower inference cost* — a
correctness-prediction claim under deployment cost, not an
uncertainty-measurement claim.

## Power / scope
- N=150 → ~55–60 wrong predictions (4-bit MedQA accuracy ~0.60), adequate
  for a stable sign-aware AUC + bootstrap CI but the lift CI may be wide.
- If the lift CI or either AUC CI is borderline, the resumable cache extends
  to larger N (cost ~30 s/question warm) as a cheap follow-up — recorded
  here so a later extension is not a post-hoc N-hack.

## Context for interpretation
Anatomy of Uncertainty: semantic entropy AUROC ~0.70–0.76 on TriviaQA
(factual QA), 0.33–0.60 on GSM8K (reasoning, weak/inverted). MedQA is
clinical-reasoning MCQ — closer to the GSM8K regime than TriviaQA, so
absolute AUROCs in the ~0.6 range would be unsurprising for *both* signals.
The comparison of interest is single_run-vs-semantic, not the absolute level.

## Run log / amendment (2026-05-26)

- **Run #1 INVALIDATED (max-tokens 256).** Completed N=150 but the
  semantic-entropy arm was crippled by answer-extraction failure: mean
  **3.16 of 6** samples parseable, 21% of questions with ≤1 valid sample
  (→ semantic entropy mechanically 0). Cause: max-tokens 256 truncated CoT
  before the "Answer: X" line. `single_run_entropy` AUC 0.7716 [0.6897,
  0.8449] is unaffected (clean 1-forward measurement) and stands; the
  semantic AUC 0.585, lift −0.187, Spearman −0.022 are **measurement
  artifacts, not results** — comparison inconclusive. Cache preserved:
  `artifacts/.../cache_invalid_maxtok256.jsonl`.
- **Run #2 (this pre-reg's valid run).** Amendment: max-tokens 384 + a
  **forced-extraction fallback** — when a CoT sample yields no parseable
  letter, take argmax of `get_token_probabilities` over the letters on the
  CoT continuation (a valid answer read, not a fabricated sample), removing
  the drop-unparseable downward bias. First question: 6/6 valid. The
  pre-registered (1)/(2) comparisons and readings above are unchanged; only
  the semantic-entropy measurement quality is fixed. This amendment is a
  measurement-validity fix recorded *before* run #2's results, not a
  post-hoc analytic choice.

## Cross-references
- `contribution_shape_post_literature.md` (the correctness-prediction
  precision added 2026-05-26).
- `convergence_pattern_observation.md` (calibration failure #5).
- `compute_constraint_orientation.md` (the claim under test).
