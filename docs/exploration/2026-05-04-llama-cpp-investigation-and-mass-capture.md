# 2026-05-04 — llama.cpp logprobs + mass-capture investigation

**Date:** 2026-05-04
**Run:** `~/work/eunosia/artifacts/mass-capture-investigation.csv`
**Purpose:** validate the measurement-protocol redesign before architectural commitment
**Outcome:** ADR-0008 accepted; mass capture recorded as measurement output; empirical claims about its detection capability deferred to stage-4

---

## Two short investigations in sequence

### Part 1: llama.cpp logprobs API check (30 min)

Verify that llama.cpp's OpenAI-compatible `/v1/completions` endpoint
exposes top-K next-token logprobs at any prompt position, with the
GGUF model bytes Ollama already has cached.

Findings:

- Install via `brew install llama.cpp` (build 9010); `llama-server`
  available as a binary.
- Ollama's GGUF blob (`~/.ollama/models/blobs/sha256-2bada…`) is a
  v3 GGUF file readable by llama.cpp directly. No conversion needed.
- `top_logprobs=25` on `/v1/completions` returns the 25 highest-
  probability tokens at the first generation position.
- Validated across 10 MedQA questions with prefix `"The best answer
  is "`: all four answer letters appear in the top-25 for 10/10
  questions; mean mass capture 0.92 (range 0.72-0.997). Reasoning
  context shifts mass downward by ~5-10 pp.
- GBNF grammar (`root ::= " A" | " B" | " C" | " D"`) constrains
  the emitted token but leaves `top_logprobs` reporting the
  pre-grammar-mask distribution. Constrained vs unconstrained-
  renormalised KL < 0.0004 nats — mathematically equivalent at the
  conditional-distribution level.
- Determinism: ~0.3 % per-letter noise across two calls at temp=0.0
  on Metal GPU. Documentable footnote.

**Decision after Part 1:** the original architectural plan was to use
constrained decoding as the primary measurement strategy, on the
grounds that it captures 100 % of mass on the answer letters by
construction. Per the *mass capture as signal* methodology insight
(workspace memory of the same name), this would have been a buried-
problem pattern — constrained decoding hides the model's reluctance
to commit. The measurement strategy switches to **unconstrained
top-K logprobs with sufficient top-K, recording mass capture as a
distinct measurement output**.

### Part 2: mass-capture-correlation investigation (30 min)

For 50 MedQA questions, measure mass capture under the chosen prefix
and cross-reference with model correctness, top-1 token, and question
length.

Findings:

- Mass capture varies from 0.056 to 0.9995. Mean 0.86, median 0.91.
- 29/50 cases correct; 21/50 wrong. Mean mass capture: 0.878 on
  correct, 0.841 on wrong. **Observed Δ = +0.037.**
- **Bootstrap 95 % CI on Δ (B=10000): [-0.063, +0.160] — straddles
  zero.** Directionally consistent (72 % of bootstrap samples have
  positive Δ) but not statistically established at N=50.
- 2/50 questions had top-1 NOT an answer letter (top-1 = `'2'`,
  the model wanting to continue numeric reasoning rather than
  commit). Both were wrong; both had mass capture below 0.20.
  **At base rate 0.42 wrong, P(2 random questions both wrong) =
  0.171** — the extreme-tail finding is consistent with chance at
  this sample size.
- Length correlation: shorter questions had higher mass capture
  (Q1: chars 266-502, mass 0.94; Q4: chars 866-1077, mass 0.88).
  Modest, monotonic-ish.

**The first interpretation drafted into ADR-0008** ("perfect-precision
boundary signal at the extreme tail," "possibly the strongest single
result the framework could report," default-composite weighting
incorporates mass capture) was an overclaim relative to the evidence.
Per the *calibrated claims* feedback memory, small-sample observations
are stage-4 hypotheses worth testing, not established results worth
baking into architectural commitments.

**Architectural commitment supported at N=50:** mass capture is
*recorded* as a measurement output. The cost is one float; the
mechanistic story is plausible; the directional evidence exists; not
recording it would discard information that might be useful at stage-
4 scale.

**Architectural commitment NOT supported at N=50:** any pre-commitment
about mass capture's *role* in the framework's signal — whether
primary, complementary, weighting factor, threshold gate, or
measurement-quality indicator only. That role is determined by
stage-4 N=1273 results under the multi-hypothesis principle.

## Pre-registered stage-4 predictions

Recorded in `feedback_calibrated_claims.md` and `project_mass_capture_as_signal.md`.
The stage-4a re-pilot (with the new measurement protocol) tests:

1. The correct-vs-wrong Δ on mass capture replicates with the lower
   bound of the 95 % bootstrap CI above 0.0.
2. Questions with mass capture below 0.25 are wrong at significantly
   higher rate than the base rate.
3. The "non-letter top-1 token in low-mass-capture cases" pattern
   from the N=50 sample replicates.

Whichever resolve positively shape the methods-paper claim; whichever
resolve null inform what mass capture is *not*. All three resolving
positively would make mass-capture-on-its-own a strong single-scalar
result. None resolving positively would still leave mass capture as
a recorded measurement-quality indicator.

## What this conversation produced beyond the architecture

A deeper-than-original framing of the project's contribution. Through
a series of corrections (verbalised-distribution circularity →
unified measurement → mass-capture-as-signal → calibrated claims),
the project's framing has shifted from "build a clinical AI tool that
detects boundary cases" to "develop and validate a measurement
protocol for chain-of-thought reasoning, with application to clinical
boundary detection." The technical work is largely the same; the
framing is sharper, the claims are more defensible.

The methodology principles articulated during this conversation are
recorded as workspace memories:

- *measurement protocol as contribution* (most foundational)
- *compartmentalization* (Phase 1 ends with chest-pain gate
  experiment; Phase 2 is the broader research program)
- *multi-hypothesis* (multiple operationalisations evaluated at
  stage 4; no single one is "the" answer)
- *open-hypothesis-space* (clinical reasoning is exclusion-based;
  flag for stage-5 design pass)
- *unified measurement* (constrained decoding over A/B/C/D at every
  reasoning-step boundary including terminal)
- *no buried problems* (workarounds that absorb unreliability into
  the framework instead of fixing the measurement)
- *fail loudly* (semantic violations stay visible)
- *mass capture as signal* (commitment-readiness recorded as
  measurement output)
- *calibrated claims* (claim strength matches evidence size; bootstrap
  CIs alongside point estimates; no architectural pre-commitment to
  unconfirmed empirical claims)

The framework's eventual contribution will be substantially stronger
under this framing than under the optimistic version of the original
plan. The price is more rigour; the gain is durability.
