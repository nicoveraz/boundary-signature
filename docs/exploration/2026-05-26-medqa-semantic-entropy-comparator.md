# Single-run predictive entropy vs sampling-based semantic entropy on MedQA

**Date:** 2026-05-26
**Status:** RESULT (valid run #2; run #1 invalidated — see "Measurement
validity" below).
**Script:** `experiments/medqa_generalization/scripts/11_semantic_entropy_comparator.py`
**Pre-registration:** `docs/decisions/prereg_semantic_entropy_comparator.md`
(both comparisons + readings registered before results).

## Question

The framework's compute-constraint claim is that *cheap single-run UQ is
competitive with expensive sampling-based UQ at deployment cost*. Semantic
entropy (Kuhn et al.; the core metric in the Anatomy of Uncertainty paper)
is the established sampling-based comparator. This is the head-to-head,
against **correctness** — and explicitly framed as a *correctness-prediction*
comparison, not an *uncertainty-measurement* one (see
`convergence_pattern_observation.md` failure #5).

## Setup

- **Data/model:** MedQA-USMLE test, N=150; Qwen2.5-7B-Instruct-4bit (MLX),
  pinned mlx 0.31.2 (serial batch=1 — the rope batched-decode bug N/A).
- **`single_run_entropy`** (the cheap signal): Shannon entropy of
  `get_token_probabilities` over the answer letters at a direct-answer
  prompt — **1 constrained forward pass**. Predicted answer = argmax.
- **`semantic_entropy`** (the expensive comparator): Shannon entropy over
  answer letters from **6 temp=1.0 CoT generations** (max-tokens 384). On
  MCQ, semantic equivalence = same answer letter, so no NLI is needed.
- **Correctness target:** single-run argmax vs gold (the deployed prediction).
- **Stats:** sign-aware AUC vs `y_wrong = 1 − correct`, 5000-bootstrap CIs
  (reoriented to point direction); paired bootstrap on the AUC lift;
  Spearman with bootstrap CI.

## Measurement validity (why there were two runs)

**Run #1 (max-tokens 256) was invalidated.** Only **3.16 of 6** samples
were parseable on average; 21% of questions had ≤1 valid sample, forcing
semantic entropy to a mechanical 0 regardless of true variation. Cause: the
256-token cap (chosen for speed) truncated CoT before the "Answer: X" line.
`single_run_entropy` (a clean 1-forward measurement) was unaffected and
matched run #2; the semantic arm was an artifact. Cache preserved as
`cache_invalid_maxtok256.jsonl`.

**Run #2 (this result)** added a **forced-extraction fallback**: when a CoT
sample yields no parseable letter, take argmax of `get_token_probabilities`
over the letters on the CoT continuation (a valid answer read, not a
fabricated sample), removing the drop-unparseable downward bias.
Result: **6.0/6 valid samples**, **182 forced extractions (20% of all
samples)** — i.e. 1 in 5 CoT generations still did not conclude with a
parseable answer even at 384 tokens, and was rescued by the constrained
read. semantic_entropy is exactly 0 on 11% of questions now (genuine
agreement, down from the artifactual 28%); mean 0.99 bits.

## Result (N=150, 5000-bootstrap)

### (1) Correctness-prediction parity
| signal | cost | sign-aware AUC | 95% CI |
|---|---|---|---|
| `single_run_entropy` | 1 constrained forward | **0.762** | [0.682, 0.835] |
| `semantic_entropy` | 6 generations (≤384 tok) + 20% fallback fwd | **0.595** | [0.502, 0.688] |

Paired lift (semantic − single_run): **−0.167**, 95% CI **[−0.272, −0.061]**,
**P(lift>0) = 0.00**.

### (2) Signal agreement (independent of correctness)
Spearman(single_run, semantic) = **+0.309**, 95% CI **[0.156, 0.453]**.

## Reading — calibrated

**Compute-constraint claim: supported, and exceeded.** The cheap 1-forward
signal does not merely match expensive 6-generation semantic entropy on
correctness-prediction — it **significantly beats it** (lift CI excludes 0).
The direction replicates run #1 on a now-valid measurement, so it is not the
extraction artifact.

**Cost gap, stated precisely (not "1/6").** single_run = 1 constrained
forward. semantic = 6 autoregressive generations of up to 384 tokens (each
hundreds of forward passes) + a fallback forward on ~20% of samples. So the
advantage is ~6× in *model-invocation count* and roughly **two-to-three
orders of magnitude in token-generation compute**. The "1/6" shorthand
understates it.

**Signal agreement (Spearman 0.31) is a substantive finding, not just a
caveat.** Within the framework's own entropy-shape family (mean_entropy,
gap_top2, mass_capture) Phase-B re-derivation found Spearman ~0.99 — those
measure the same construct. **0.31 between single_run entropy and semantic
entropy is far weaker than within-family agreement.** So single-run
distributional signals and sampling-based semantic entropy capture
*genuinely different aspects of model behavior*: single_run measures
confidence-as-expressed-in-the-distribution (one forward); semantic entropy
measures consistency-across-sampling-variation. Both correlate with
correctness, via different paths. Consequence: the compute-constraint claim
should be stated as **"cheap and expensive measure different things; cheap
predicts correctness better here at far lower cost,"** NOT "cheap
approximates expensive." single_run is not a proxy for sampling — it is its
own signal.

## Caveats (each real, cumulatively material)

1. **Correctness-prediction, not uncertainty-measurement.** This shows
   single_run is a better *correctness predictor*; neither signal is shown
   to *measure uncertainty* as a latent construct (failure #5).
2. **MCQ-structural advantage.** On 4-option MCQ, semantic entropy collapses
   to coarse letter-agreement entropy (limited dynamic range; 11% at 0);
   single_run uses the full continuous letter distribution in one forward.
   Part of the gap is that single_run uses richer information *on this task
   shape*. **Does not generalize to open-ended generation**, where semantic
   clustering is non-trivial and the comparison could differ.
3. **20% forced extraction on the semantic arm** — a constrained answer-read
   slightly more deterministic than spontaneous sampling. Far better than
   the drop-bias it replaced, but a caveat the single_run arm does not carry.
4. **Anatomy corroboration = a limitation surfacing, not a novel defeat.**
   semantic entropy 0.595 sits in the range Anatomy reports on GSM8K
   reasoning (AUROC 0.33–0.60), vs its TriviaQA factual-QA 0.70–0.76.
   Semantic entropy is *known* to be weak on multi-step reasoning; MedQA
   sits in that regime. single_run's edge is partly that semantic entropy is
   operating outside its strongest task regime.
5. **N=150, one dataset, one model.** single_run 0.762 vs the canonical
   stage-4a MedQA `mean_entropy` 0.686 [0.657, 0.716] is consistent within
   CI (lower bound 0.682 overlaps). Do not over-read the point estimate.

## What this does NOT change

Disposition-GT stage-6 results (P5/P6 inverted, P2 null, P4 lift not
significant) stand. The aligned-GT 6.5% evaluability constraint in
MIMIC-IV-ED stands. The contribution-claim narrowings (no novel diagnostic
framework / UQ collection / clinical-awareness standalone / individual-signal
invention) do not recover. This adds one favorable MCQ data point to item 2;
it does not undo the cumulative narrowing or the clinical-transfer bind.

## Robustness gate before this anchors the methods-paper §1

This is a *supportive* data point on one cohort, not a *decisive anchor*.
Before promoting it to a §1 contribution anchor, replicate cheaply:
(a) full MedQA N=1273 (script/cache already scale); (b) a second model (the
existing Qwen-vs-Llama cross-LLM machinery); (c) ideally MMLU
professional_law. If the lift holds across these, it is decisive enough to
anchor. Until then: recorded in item 2, drafting stays deferred on the
clinical dimension.

## Cross-references
- `docs/decisions/contribution_shape_post_literature.md` item 2 (data point added).
- `docs/decisions/convergence_pattern_observation.md` (failure #5; positive-instance note).
- `docs/decisions/prereg_semantic_entropy_comparator.md` (pre-reg + run log).
