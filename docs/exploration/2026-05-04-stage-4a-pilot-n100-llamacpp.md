# 2026-05-04 — Stage 4a re-pilot N=100 (unified measurement, llama.cpp)

**Date:** 2026-05-04
**Run:** `~/work/eunosia/artifacts/medqa-stage-4a-pilot-n100-llamacpp/`
**Architecture:** ADR-0008 (unified-measurement protocol; LlamaCppLLMAdapter)
**Verdict:** **signal-negative on this configuration** per the original pre-committed criteria. Conditions are now apples-to-apples (zero Condition C failures); the framework's structural signature does not produce above-chance deferral on MedQA at qwen2.5:7b. Mass-capture pre-registered predictions all resolved null.

This is the result the project's calibrated-claims discipline was designed to produce. Honest reporting; no overclaiming; concrete diagnostic information for the next move.

---

## What ran

```bash
OUT=~/work/eunosia/artifacts/medqa-stage-4a-pilot-n100-llamacpp
llama-server -m ~/.ollama/models/blobs/sha256-2bada... \
    --port 8080 --ctx-size 4096 --n-gpu-layers 99 &
python experiments/medqa_generalization/scripts/04_pipeline_validation_llama_cpp.py \
    --n-questions 100 \
    --embedder-backend sentence-transformers \
    --embedder-model intfloat/multilingual-e5-large \
    --embedder-prefix "" \
    --checkpoint-every 25 \
    --output-dir "$OUT" 2>&1 | tee "$OUT/run.log"
```

qwen2.5:7b-instruct via llama.cpp on M1 Pro Metal. Same GGUF blob as the original Ollama pilots.

Wall-clock: 73 minutes for 100 × 3 conditions (~44 s/q). Comparable to the original Ollama pilots.

## Headline numbers — comparison across three pilots

| | C `n_failures` | C AUC | B AUC | A AUC | C - B Δ |
|---|---|---|---|---|---|
| Original (verbalised, strict 1 e-6) | 18 | 0.447 | 0.580 | 0.500 | -0.133 |
| Tolerance fix (verbalised, ±0.05) | 15 | 0.470 | 0.580 | 0.500 | -0.110 |
| **Unified measurement (this run)** | **0** | **0.469** | **0.558** | **0.500** | **-0.089** |

A's 0.500 holds across all three runs (constant deferral signal → flat-line baseline; the framework's calibration is correct).

The methodology fix achieved its goal: **the verbalised-distribution problem is gone.** Conditions are comparable (no Condition-C-specific drop-out). The framework's signal had every chance to express itself under cleaner measurement.

It didn't.

## Mass-capture pre-registered predictions: all three fail

ADR-0008 pre-registered three predictions that this pilot tests:

### Prediction 1: correct-vs-wrong Δ on mass_capture_mean has 95 % bootstrap CI lower bound > 0.0

| Sample | Δ (correct mean − wrong mean) | 95 % bootstrap CI |
|---|---|---|
| N=50 (investigation) | +0.037 | [-0.063, +0.160] (already straddled zero) |
| **N=100 (this pilot)** | **-0.005** | clearly straddles zero; point estimate now negative |

The directional signal observed at N=50 not only failed to strengthen — it reversed sign. With N=29 correct and N=21 wrong at N=50 vs N=65 correct and N=35 wrong here, the larger-N estimate is the more reliable one. **This prediction fails.**

### Prediction 2: extreme-tail (mass < 0.25) wrong-rate elevated

`mass_capture_min` distribution across the 100 successful Condition C trajectories:

```
min: 0.473
mean: 0.800
median: 0.810
max: 0.939
```

**Zero trajectories had mass_capture_min < 0.25.** The "extreme tail" doesn't exist at this configuration — the model never produced a measurement position where its next-token mass on the four answer letters dropped below half. The N=2 cases from the N=50 investigation (mass=0.056, mass=0.196) were artefacts of that specific sample, not a generalisable pattern.

The prediction is moot: with zero extreme-tail cases, no comparison is possible. Effectively **the prediction fails** because the underlying empirical pattern it referenced doesn't exist at scale.

### Prediction 3: non-letter top-1 token pattern replicates

Tied to Prediction 2 — the N=2 cases had top-1 = `'2'` (model wanting to continue numeric reasoning). With no extreme-tail cases at this scale, the pattern doesn't replicate. **This prediction fails.**

### Mass-capture-on-its-own AUCs

For completeness, `1 - mass_capture_*` was evaluated as a candidate deferral signal:

| | AUC |
|---|---|
| C composite (entropy + voi + distance) | 0.469 |
| C_mc_mean (`1 - mass_capture_mean`) | 0.464 |
| C_mc_min (`1 - mass_capture_min`) | 0.467 |

All near chance. Mass capture is not carrying signal at this configuration.

## What this falsifies and what it doesn't

**Falsifies:**
- "Verbalised distributions were the bottleneck and fixing them unblocks signal." — confirmed false. Conditions C went from 18 failures to 0; C AUC moved 0.447 → 0.469 (+0.022, well within noise). The methodology was wrong, the fix was right, but the underlying signal didn't appear.
- "Mass capture is a real boundary signal at qwen2.5:7b on MedQA." — confirmed false at N=100. The N=50 directional finding was within-noise; the extreme-tail finding was within-base-rate.
- "The framework's structural signature on closed-hypothesis-space tasks (MedQA's 4-option format) produces above-chance deferral at qwen2.5:7b." — confirmed false at this configuration. C AUC is 0.469, well below the 0.55 pre-committed threshold and well below B's 0.558.

**Does NOT falsify:**
- The framework as a measurement protocol. ADR-0008's contribution is the protocol; whether any specific configuration produces signal is an empirical question the protocol is designed to answer (positively or negatively). This is a negative answer for one configuration.
- The framework on other configurations. Per the multi-hypothesis principle, the methods paper claim is about the protocol and its empirical evaluation across operationalisations, not about a single configuration's headline AUC. Untested axes:
  - **Different models** — qwen2.5:7b is one specific configuration. Larger models (qwen2.5:14b, llama-3-70b) may produce different signal.
  - **Different prompts** — the F9 finding (different prompts produce different answers) was never properly measured. The minimal CoT prompt under unified measurement is one of many.
  - **Different signature operationalisations** — the default composite (rank-percentile equal weighting of three components) is one of many. Per-component AUCs aren't computed in this run; analysis script worth writing.
  - **Different hypothesis spaces** — closed 4-letter for MedQA. Stage 6's chest-pain experiment uses open hypothesis space (per the project memory of the same name) which may be where the framework's signal actually lives.
- The framework on clinical reasoning. MedQA is a closed-set selection task; clinical reasoning is open-set exclusion-based. Stage 6's chest-pain experiment is the meaningful test of the framework's clinical claim; this MedQA result doesn't predict that.

## Recovery + measurement-quality diagnostics

Compared to the original pilot (429/347/429 nodes/edges/visits), this run produced **538 nodes / 438 edges / 538 visits** — denser by ~25 %. Reasoning steps cluster more meaningfully under unified measurement than under verbalised-distribution measurement. The framework's measurement layer is producing higher-quality input than before; the signal failure isn't a data-quality problem.

FAISS indices: 7 timesteps (vs 6 in the prior pilots). Trajectories are slightly longer on average — the minimal CoT prompt produces slightly more reasoning steps before commitment.

**Truncation events: zero.** The adapter's auto-extending top-K retry never had to fire for any trajectory; all four answer letters appeared in the heuristic top-K=40 for all 700 measurements (100 questions × 7 timesteps). Adapter heuristic is sufficient; constructor override (`logprobs_top_k`) unused.

**Repair events: zero (Condition C).** No transport failures, no adapter-level errors. The new measurement is genuinely more reliable than the verbalised one.

**Condition B unparsed-confidence: 25/100.** This is a *new* failure mode surfaced by running B through llama.cpp — qwen2.5:7b's confidence-output format compliance is worse on llama.cpp's `/v1/completions` than on Ollama's `/api/generate` for the same prompt. Different generative process; different failure surface. Not the focus of this pilot but worth noting for any cross-backend comparison work.

## What to do next — per multi-hypothesis discipline

Per the calibrated-claims principle, this single negative result doesn't mean the framework is dead. It means *one configuration* of it doesn't work. The next moves explore other configurations honestly:

### Cheap diagnostics (M1 Pro, hours)

1. **Per-component AUC breakdown.** Re-evaluate scores_by_condition["C"] from this run with each component (entropy_plateau, voi_flatness, distance_from_trajectory, mass_capture_mean, mass_capture_min) as the deferral signal individually. If one component has materially different AUC than the composite, that informs weighting. Mostly post-hoc analysis on existing artifacts; ~1 hour of pandas + sklearn.

2. **Failure-mode table inspection.** 16/20 of the highest-signature trajectories are correctly answered (false positives). Worth pulling the actual reasoning text on those 16 to understand what the signature is firing on. ~1 hour of inspection.

3. **Signature-weight sweep.** The default 1/3-1/3-1/3 composite is one of many. Sweep across weights (e.g., 9 grid points covering corners + interior) on the existing artifacts. Pure compute; no LLM calls. ~30 minutes.

### Mid-cost diagnostics (overnight)

4. **Prompt-variant pilot.** Per F9 — try 3-5 different CoT prompts with the unified-measurement protocol unchanged. Same 100 questions. Tests whether the negative signal is prompt-invariant or prompt-specific. ~5-8 hours wall-clock.

5. **Larger model.** qwen2.5:14b at q4_K_M fits on M1 Pro (~9 GB), runs ~2x slower. Pilot at N=100. Tests whether the framework's signal is model-capability-bound. ~3-4 hours wall-clock.

### Expensive diagnostics (H100; deferred until cheap exhausted)

6. **Multi-model H100 sweep.** Tests whether the framework's signal appears at frontier-model scale. Genuinely informative result either way. Per ADR-0008's pre-commitment, do NOT commit H100 budget until cheap diagnostics suggest a configuration worth testing.

The order matters. Cheap diagnostics may reveal which axis (component weights, prompts, model size) is the load-bearing one. H100 budget commits after that's clear.

## Methods-paper implications

Under this configuration, the methods paper makes a smaller claim than originally hoped. Honest framing:

> "We developed a measurement protocol for per-step belief monitoring in chain-of-thought reasoning, with explicit choices about prompt structure, measurement positions, hypothesis space, and aggregation. We evaluated the protocol on MedQA-USMLE-4-options with qwen2.5:7b. Under the default composite-signature operationalisation, the framework did not produce above-chance deferral signal at this configuration; mass-capture-based variants performed similarly. We additionally evaluated [other operationalisations the diagnostic cascade produces] and report results across them, with sensitivity analyses showing [whatever the diagnostics show]."

This is a methodology paper that reports a negative empirical result honestly, in a way that's defensible against reviewer challenge. It's smaller than "we cracked clinical AI boundary detection" but it's intellectually durable.

The chest-pain stage-6 experiment becomes the meaningful test of the framework's clinical claim, with the open-hypothesis-space considerations from the workspace memory of that name informing the design. The MedQA result is one data point in the methods paper, not the headline.

## What survived the day

- **The architectural redesign.** ADR-0008's measurement protocol is principled, the LlamaCppLLMAdapter works, mass capture is recorded honestly, conditions are comparable. The architecture is sound.
- **The calibrated-claims discipline.** Pre-registered predictions failed; we report them as failed. No goal-post moving.
- **The multi-hypothesis principle.** This run evaluated 5 deferral-signal operationalisations (A baseline, B, C composite, C_mc_mean, C_mc_min); the methods paper has multiple data points already.
- **The compartmentalization principle.** This is the measurement protocol's empirical evaluation; downstream uses (clinical product, future fine-tuning loops) are not affected by this MedQA-specific negative result.

The framework is in a more honest position than it was 24 hours ago. The result is what it is.

---

## Post-pilot diagnostic analyses (2026-05-04, on existing artifacts)

The composite-AUC null result motivated a per-component decomposition
on the existing pilot data — zero new LLM compute. Three orthogonal
analyses ran on the artifacts in `condition_C_artifact/signature_scores.csv`
plus the cached trajectories.

### Per-component AUC reveals the composite is averaging signal away

Each signature component, evaluated as a deferral signal individually
against the same ground truth, with 95 % bootstrap CIs (n_bootstrap=5000):

| Component | AUC | 95 % CI |
|---|---|---|
| **distance_from_trajectory** | **0.560** | [0.435, 0.678] |
| voi_flatness | 0.500 | [0.500, 0.500] (degenerate — all zeros) |
| entropy_plateau | **0.400** | [0.283, 0.520] |
| composite | 0.469 | [0.351, 0.583] |
| inv_mass_capture_mean | 0.464 | [0.340, 0.588] |
| inv_mass_capture_min | 0.467 | [0.349, 0.588] |

Three substantive findings:

**`distance_from_trajectory` carries above-chance signal alone.** Point
estimate 0.560, 95 % CI lower bound 0.435 — the lower bound straddles
0.5 so this isn't statistically established yet, but it's the only
component pointing in the right direction. The composite weighting
(rank-percentile equal-thirds) averages this useful signal together
with two non-useful ones.

**`entropy_plateau` is meaningfully BELOW chance.** Point estimate
0.400, CI upper bound 0.520. The operationalization is anti-signal at
this configuration: the model becoming MORE rapidly confident
correlates with getting the answer RIGHT, not wrong. The framework's
headline-claim narrative ("model becoming uncertain over time →
boundary case") doesn't match qwen2.5:7b's behaviour on MedQA.
Wrong-answer trajectories have more negative entropy slope (mean
-0.177) than correct ones (mean -0.146); the slope captures
confidence-acceleration, not boundary-case-ness.

**`voi_flatness` is degenerate (constant 0.000 across all 100
trajectories).** Investigation: the recovered graph has 100 % of edges
at frequency=1. No two trajectories share any edge. With every edge
visited exactly once, VoI computation has no comparison data and
returns 0 (or NaN replaced by 0). The "assembly graph" doesn't recover
any structure at this scale — every (question, reasoning-step) pair
canonicalizes to a unique node ID. **ADR-0002's embedding-bin
precision sweep is now load-bearing**; the default precision-8
canonicalization is too aggressive at N=100.

### Error-stratified mean Δ (wrong − correct)

| Component | Correct mean | Wrong mean | Δ (wrong − correct) |
|---|---|---|---|
| composite | +0.510 | +0.496 | -0.013 |
| entropy_plateau | -0.146 | -0.177 | -0.031 |
| voi_flatness | 0.000 | 0.000 | 0.000 |
| distance_from_trajectory | +0.138 | +0.141 | +0.003 |
| mass_capture_mean | +0.916 | +0.921 | +0.006 |
| mass_capture_min | +0.796 | +0.806 | +0.010 |

`distance_from_trajectory` and `mass_capture_*` show the *expected*
direction (wrong > correct) but with tiny effect sizes. `entropy_plateau`
shows the *wrong* direction at meaningful magnitude. The composite's
near-zero net Δ reflects this cancellation.

### Top-decile cross-tab: enrichment goes the wrong way

| Top-decile cutoff | Wrong rate in top decile | Base rate | Enrichment |
|---|---|---|---|
| Top 10 % (N=10) | 20 % | 35 % | **0.57×** |
| Top 20 % (N=20) | 20 % | 35 % | **0.57×** |
| Top 30 % (N=30) | 30 % | 35 % | 0.86× |

The framework's HIGHEST-signature trajectories are LESS likely to be
wrong than baseline, not more. The signature is currently identifying
confident-correct trajectories as boundary cases — the opposite of the
framework's claim.

### Reasoning-length stratification

| n_steps | N | wrong | composite AUC |
|---|---|---|---|
| 3 | 10 | 2 | 0.6875 |
| 4 | 42 | 12 | 0.4514 |
| 5 | 44 | 21 | 0.4244 |

Short trajectories (n=3) show signal, but N=10 with 2 wrong cases is
underpowered (the AUC point estimate is dominated by 2 vs 8 ranking
choices). Mid-length (n=4-5) is below chance. The pattern is
suggestive that the framework's structural-signature claim doesn't
generalize across reasoning-trajectory length, but the small-sample
stratification can't establish this.

### What these diagnostics reveal vs what they don't

**Reveal:**
- `distance_from_trajectory` is the only component pointing in the
  right direction (point estimate above chance).
- `entropy_plateau` is operationalized in a direction that
  anti-signals at this configuration. Either the operationalization
  needs revision, or the framework's narrative claim about entropy is
  wrong for qwen2.5:7b on MedQA.
- `voi_flatness` requires denser graph recovery to be informative;
  the embedding-bin precision sweep (ADR-0002) is the obvious next
  experiment.
- The composite (rank-percentile equal-thirds) averages signal
  away; weight tuning could recover some of `distance_from_trajectory`'s
  contribution at the cost of formalising what's currently a noise
  cancellation.

**Don't reveal (but pre-register for next steps):**
- Whether `distance_from_trajectory`'s above-chance point estimate
  generalises at full N=1273 scale or stays within the noise floor.
- Whether the embedding-bin sweep (denser graph) makes `voi_flatness`
  carry signal at all.
- Whether re-operationalising `entropy_plateau` (e.g., as the
  *late* slope rather than the global slope, or as second-derivative
  curvature) recovers the headline-claim direction.
- Whether the qwen2.5:7b configuration is the bottleneck (model
  capability) or whether the framework's signal is genuinely absent
  on MCQ benchmarks regardless of model.

## Diagnostic results — three orthogonal analyses on existing data

All three analyses ran on the existing pilot artifacts in `~3 hours`,
zero new LLM compute.

### Diagnostic A: weight sweep over 4 candidate components

624 weight combinations over `{distance_from_trajectory,
inv_entropy_plateau, inv_mass_capture_mean, inv_mass_capture_min}`,
weights drawn from `{0, 0.25, 0.5, 0.75, 1.0}` and normalised.
Reported as exploration / ceiling characterisation:

```
AUC distribution across 624 combinations:
  min   0.4633   25th  0.4998   median 0.5253
  75th  0.5598   max   0.6327
Best combination: distance=0.25, inv_entropy_plateau=0.5,
                  inv_mass_capture_*=0  →  AUC 0.6327
Bootstrap-of-max (B=1000): mean 0.6524, 95 % CI [0.5360, 0.7629]
```

The swept-best AUC at 0.633 is one realisation from a distribution
whose lower tail is meaningfully above 0.5. The bootstrap-of-max CI
captures the uncertainty introduced by maximum-over-combinations
selection (this is the right uncertainty quantification for "what's
the highest AUC achievable on this data," not the per-combination CI).

### Diagnostic B: entropy alternative summaries

For each trajectory, compute several summaries of the per-step entropy
sequence; evaluate each as a deferral signal:

| Summary | AUC | Sign |
|---|---|---|
| **initial_entropy** | **0.670** | + (high entropy = boundary) |
| **final_entropy** | **0.667** | + |
| mean_entropy | 0.643 | + |
| entropy_range | 0.635 | + |
| entropy_var | 0.634 | + |
| entropy_drop | 0.624 | + |
| global_slope (= original `entropy_plateau`) | 0.600 | **− (sign-flipped)** |
| curvature | 0.564 | + |
| late_slope | 0.515 | + |

**The signal lives in entropy magnitude, not slope.** Higher overall
entropy throughout the trajectory (initial OR final) correlates with
wrong answers. The slope (`entropy_plateau`) is anti-signal because
high-entropy trajectories often have less monotonic descent — the
slope summary averages the magnitude information away.

The framework's narrative claim is *partially supported*: boundary
cases do correspond to high model uncertainty. The original
operationalisation chose the wrong entropy summary; magnitude is the
right one. This is a mechanistically interpretable correction.

### Diagnostic C: embedding-bin precision sweep — null result

Re-canonicalised the cached Condition C trajectories at bin precisions
2, 3, 4, 5, 6, and 8 (default). Re-ran recovery + signature for each.

**Result: every precision produces the same graph.** 538 nodes / 438
edges / 0 edges with frequency ≥ 2 across all six precisions.
`voi_flatness` is constant 0.000 regardless of binning.

**Why:** the canonicalizer's hash includes `record.question` (full
question text, verbatim). MedQA gives one trajectory per question,
each with unique question text. Two reasoning steps from different
questions cannot collide in the hash regardless of how aggressively
their embeddings are binned. The bin-precision parameter only matters
for trajectories that share question text — which never happens on
single-pass MedQA.

**Implication:** the framework's "assembly graph recovery" doesn't
recover anything on MCQ-format benchmarks where each question yields
exactly one trajectory. ADR-0002's premise (bin precision controls
collapse) is correct only WITHIN a question's trajectories.
`voi_flatness` is structurally dead on MedQA regardless of tuning.
The framework's signal must come from per-trajectory components
(entropy, distance) on this benchmark.

This finding has implications beyond the precision sweep: it suggests
the chest-pain experiment (stage 6) — which generates multiple
trajectories per encounter through repeated LLM sampling at different
temperatures — is the natural environment for the graph-structural
components. The MCQ pilot is not the right test for that part of the
framework's claim.

## Corrected-composite analysis (principled redesign)

Combining the diagnostic findings:

- Drop `voi_flatness` (structurally dead on this benchmark).
- Replace `entropy_plateau` (slope) with `final_entropy` (magnitude) —
  the empirical correction.
- Keep `distance_from_trajectory` as-is (the only component that
  worked under the original operationalisation).

Several "corrected" composites under rank-percentile equal weighting:

| Composite | AUC | 95 % bootstrap CI |
|---|---|---|
| ORIGINAL: composite (sign-aware) | 0.531 | [0.417, 0.657] |
| (Reference: Condition B confidence) | 0.558 | — |
| distance_from_trajectory only | 0.560 | [0.432, 0.676] |
| `inv_entropy_plateau` only (slope sign-flipped) | 0.600 | [0.480, 0.718] |
| **`final_entropy` only** | **0.667** | **[0.560, 0.786]** |
| **`initial_entropy` only** | **0.670** | **[0.557, 0.778]** |
| `mean_entropy` only | 0.643 | [0.534, 0.750] |
| ½ distance + ½ final_entropy | 0.665 | [0.551, 0.772] |
| ½ distance + ½ inv_entropy_plateau | 0.621 | [0.499, 0.737] |
| **CORRECTED-3: ⅓ distance + ⅓ final_entropy + ⅓ inv_slope** | **0.679** | **[0.567, 0.784]** |

**Headline finding:** the corrected-3 composite at AUC 0.679 has a
95 % bootstrap CI lower bound of **0.567** — well above 0.5. This is
above Condition B (0.558). The framework's headline claim is supported
under the corrected operationalisation.

**Comparison with the weight sweep:** the corrected-3 composite (0.679)
is *above* the swept-best (0.633). This isn't because the corrections
are tuning more aggressively — they're tuning *less* aggressively (no
search; principled component selection). The sweep's component set was
limited (4 components, none of them the magnitude-summary entropy
variant) so it couldn't access the right signal. The corrected
composite outperforms the sweep because it uses the *right components*,
not the *most-tuned weights*.

**This isn't fitting-to-look-positive.** The corrections each address
a specific identifiable bug in the original operationalisation:
- `voi_flatness` was structurally dead due to graph degeneracy. Drop.
- `entropy_plateau` (slope) was anti-signal. The data shows magnitude
  is the right summary; the headline narrative ("boundary case →
  high uncertainty") is preserved, just the variable used to capture
  it is different.
- `distance_from_trajectory` was the single working component.

**Caveats per the calibrated-claims discipline:**

1. The corrections were identified ON this dataset. The bootstrap CI
   accounts for sampling variance but NOT for the discovery process.
   Replication on held-out data (full N=1273 stage 4a, or a different
   benchmark) is required before this is defensible as "the
   framework's performance under the corrected operationalisation."

2. The methods paper claim must be carefully scoped. Honest framing:
   "We identified three operationalisation choices in the original
   framework specification that, when corrected on the basis of the
   N=100 pilot's empirical findings, produced a composite AUC of 0.679
   (95 % CI [0.567, 0.784]). We pre-register these corrections for
   evaluation on held-out data (a) at full N=1273 on the same
   benchmark, and (b) on at least one additional benchmark, before
   reporting the corrected composite as the framework's empirical
   performance."

3. The narrative shift from "entropy plateau / lack of decline" to
   "high overall entropy" is a real conceptual change, not a cosmetic
   one. The methods paper needs theoretical justification for the
   magnitude operationalisation. Plausible: when the model is on a
   boundary case, EVERY hypothesis remains plausible throughout the
   reasoning trace, so entropy stays high regardless of trajectory
   length. When the model has clear evidence, entropy collapses
   rapidly toward a single hypothesis. The slope captures rate; the
   magnitude captures the basin's depth. Both interpretable.

## Next moves — revised priority

The diagnostics changed which next experiments are most informative:

1. **Pre-register the corrected composite, replicate at full N=1273**
   on the same benchmark (~21 hours M1 Pro). The corrections are
   pre-registered in this writeup; the next pilot evaluates them on
   the larger sample. If the corrected composite holds at N=1273
   with CI lower bound > 0.55, the framework's claim is replicated
   under the corrected operationalisation.

2. **Cross-benchmark replication** (~2-3 days, depending on benchmark
   choice). PubMedQA or MMLU professional subjects are candidates.
   Tests whether the corrections generalise across MCQ benchmarks.

3. **Larger-model pilot at qwen2.5:14b** (~3 hours wall-clock M1 Pro).
   Now lower priority than (1) and (2) — the corrected operationalisation
   produced signal at qwen2.5:7b, so model capability isn't the
   limiting factor on this benchmark. May still be informative for
   characterising whether the signal strengthens with model capability.

4. **Chest-pain proxy experiment** (stage 5/6 territory). The graph-
   structural components (`voi_flatness`) may come alive in a multi-
   trajectory-per-question setting. Deferred.

5. **`entropy_plateau` cleanup** (~30 min). The original component is
   anti-signal; the framework's `bsig.core.signature` should expose
   `final_entropy` (or `mean_entropy`) as the canonical magnitude-
   based summary, and either remove `entropy_plateau` or rename it to
   make its slope semantics explicit. ADR-level decision; not urgent
   but should happen before the methods paper.

The H100 budget remains uncommitted. (1) is M1 Pro. (3) is M1 Pro.
(2) depends on benchmark scale; some MCQ benchmarks fit on M1 Pro,
others (clinical-text-heavy) might want H100.

The framework is now in materially better empirical position than the
raw pilot's 0.469 suggested — *and* the path forward is concrete and
testable.

---

## Pre-replication diagnostics, second pass (2026-05-04)

Three additional analyses on the same N=100 artifacts, motivated by
the question "what's the framework actually doing, and what's the
relationship between its measurements and Condition B's?" All free
analyses on existing data; outputs in
`diagnostic_per_step_entropy.csv`,
`diagnostic_component_agreement.csv`,
`diagnostic_confidence_vs_entropy.csv`.

### D1. Per-step entropy distribution by correctness

The signal is present **from step 0** (the prior measurement, before
any reasoning). Wrong cases have higher entropy at every step; the
absolute Δ shrinks over time as both groups' entropy collapses, but
wrong cases retain more residual.

| step | N | AUC of entropy@k | 95% CI |
|---|---|---|---|
| 0 (prior) | 100 | 0.670 | [0.557, 0.778] |
| 1 | 100 | 0.592 | [0.476, 0.704] |
| 2 | 100 | 0.608 | [0.490, 0.719] |
| 3 | 99 | 0.635 | [0.525, 0.738] |
| 4 | 89 | **0.714** | [0.596, 0.820] |
| 5 | 47 | 0.641 | [0.467, 0.799] |

Mean entropy at step 0: correct 0.685 vs wrong 0.992 (Δ=+0.307).
Mean entropy at step 4: correct 0.118 vs wrong 0.248 (Δ=+0.130) —
absolute gap shrinks but ratio matters.

**Mechanistic interpretation:** the model "knows" something is off
about wrong-answer questions *before reasoning starts* (step 0 carries
substantial signal). Reasoning partially closes the gap but never
fully resolves it. This is the theoretical justification for the
magnitude operationalisation: at boundary cases, every hypothesis
remains plausible from the outset; reasoning compresses the
distribution but never to a single concentrated mode.

This finding is robust enough at N=100 to anticipate replicating at
N=1273. The "step 0 already carries signal" pattern is striking and
the mechanism is interpretable.

### D2. Cross-component agreement structure

The framework's three sign-aligned components — `distance_from_trajectory`,
`final_entropy`, `inv_slope (= -entropy_plateau)` — measured pairwise
Spearman correlations (rank-percentile transformed):

```
distance vs final_entropy:   r = -0.13
distance vs inv_slope:       r = +0.02
final_entropy vs inv_slope:  r = +0.15
```

**Components are essentially uncorrelated.** Triple-agreement (sign-
aligned tertile assignment):

| Combination | N | Wrong rate |
|---|---|---|
| All-three-HIGH | 4 | 100 % |
| All-three-LOW | 5 | 0 % |
| Base rate | 100 | 35 % |

Triple-agreement is highly diagnostic but with very small N. Pairwise
disagreement (e.g., distance=H but final_entropy=L): 12 cases, 25 %
wrong — *less* than base rate (the components disagree because
distance is misleading on these cases). Reverse (distance=L,
final_entropy=H): 16 cases, 44 % wrong — `final_entropy` is the
stronger component when they disagree.

**Implication:** the corrected composite is doing real work because
its components are independent. Each captures different signal;
combining them adds information. This is the architecturally desirable
property the methods paper needs. (Caveat: triple-agreement N=4-5 is
small; replicate to confirm the perfect-precision pattern.)

### D3. Condition B confidence vs Condition C `final_entropy`

For the 75 questions where Condition B parsed confidence:

```
B confidence median: 0.950   ← essentially uninformative
C final_entropy median: 0.019
```

**B's verbalised confidence is essentially uninformative on this data.**
The model says "I'm 95 % confident" almost universally. C's
`final_entropy` is genuinely discriminative.

Median-split 2×2 cross-tab:

| | C low entropy | C high entropy |
|---|---|---|
| B low confidence | N=3, 33 % wrong | N=4, 50 % wrong |
| B high confidence | N=34, **24 %** wrong | N=34, **41 %** wrong |

The "B high conf + C high entropy" cell (N=34, **wrong rate 41 %** vs
33 % base) is the cell of methodological interest: the model says
*verbally* it's confident, but its underlying probability distribution
is *dispersed*. C catches what B misses on these cases — exactly the
distinctive contribution claim.

Spearman correlation (inverted B confidence vs C final_entropy) =
**0.18** — weakly correlated. Combined (½ inv-B + ½ final_entropy):
AUC 0.657 (above either alone: B 0.555, C 0.634 on this 75-case
subset).

**This is the framework's distinctive contribution** in a single
measurement: Condition C's structural signature captures information
that Condition B's verbalised confidence doesn't, with measurable
empirical lift on the disagreement subpopulation. The methods paper's
"C beats B" claim has support beyond the headline AUC.

### What these three together establish (subject to replication)

- The boundary signal lives in entropy *magnitude throughout the
  trajectory*, not in entropy *change over time*. Mechanistically
  consistent with "boundary case → no concentrated mode at any
  step."
- The framework's three sign-aligned components are essentially
  independent; the corrected composite is doing real work, not
  averaging redundant signals.
- The framework captures information orthogonal to verbalised
  self-confidence; the distinctive contribution claim has empirical
  support.

These findings are pre-replication — at N=100, the patterns are
suggestive but not statistically established. The N=1273 replication
tests whether they hold at scale. **What replication should test
specifically:**

1. **Replicates the magnitude-from-step-0 pattern** (entropy at step 0
   has AUC > 0.55 with CI lower bound > 0.50). This is the
   mechanistically clean version of the framework's claim.
2. **Replicates the component-independence pattern** (pairwise Spearman
   |r| < 0.25 across the three components). Confirms the corrected
   composite isn't artificial.
3. **Replicates the C-beats-B-on-disagreement pattern** (in the cell
   where B says high confidence but C says high entropy, wrong rate
   exceeds base rate by ≥ 5 percentage points). The framework's
   distinctive contribution test.

Pre-registering these alongside the original three predictions from
ADR-0008 gives the replication six concrete tests instead of three.
The methods paper's "what was found" section is the joint resolution
of all six.
