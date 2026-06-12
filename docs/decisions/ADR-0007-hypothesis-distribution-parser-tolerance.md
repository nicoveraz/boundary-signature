# ADR-0007: Hypothesis-distribution parser tolerance — configurable, default permissive

**Status:** Superseded by ADR-0008 (2026-05-04)
**Date:** 2026-05-03
**Stage of origin:** stage 4a pilot follow-up

## Supersession note (2026-05-04)

This ADR was superseded one day after acceptance. It should not have been
merged. The decision to make the parser more tolerant of sum-≠-1
distributions was a workaround for a measurement-methodology problem,
not a fix for it. Recording the lesson explicitly because the project
principle (no buried problems — see workspace memory entry of the same
name) was articulated as a direct response to this mistake:

**The framework was using verbalized hypothesis distributions** —
asking the LLM to emit a JSON probability distribution as text — as
the per-step Condition C measurement. Stage-4a pilot showed
qwen2.5:7b produces sum-≠-1 distributions on 18 % of MedQA questions.
The two observed patterns were mild rounding (sum ~1.10) and semantic
confusion (sum ~2.0, model emitting independent likelihoods).

**The principled response was to switch the measurement methodology**,
not to widen the parser's tolerance. Verbalized distributions are
*the model writing about its beliefs*; what we want to measure is
*the model's actual probability distribution*. These are different
things. The parser-tolerance fix made the framework more accepting
of unreliable measurements; ADR-0008 replaces the measurement with
something that doesn't have the unreliability.

**What this ADR achieved that's still useful:** the regression-test
fixtures in ``tests/reference/test_llm_local.py`` capture two real
qwen2.5:7b output patterns. They remain useful as documentation of
*why* the verbalized-distribution approach was abandoned. The
``distribution_sum_tolerance`` constructor parameter on
``OllamaLLMAdapter`` remains in place for any caller that still uses
``get_hypothesis_distribution`` directly (e.g., third-party experiments
or comparison studies); the parameter does not affect the
unified-measurement flow.

**Cleanup commitment:** ``OllamaLLMAdapter.get_hypothesis_distribution``
and ``_parse_and_validate`` are kept in the codebase but Condition C
no longer calls them. They are candidates for removal once a sustained
period passes with no consumer; tracked in the open-questions section
of the repository's CLAUDE.md.

---

(Original ADR text below. Preserved unedited; status header above is
the current truth.)

---

## Context

The stage-3.5a `OllamaLLMAdapter._parse_and_validate` validates that the
LLM-emitted hypothesis distribution sums to 1.0 within `abs_tol=1e-6`.
This was correct under the assumption that an instruction-tuned model
asked for a probability distribution would produce arithmetically
coherent output.

The stage-4a N=100 pilot falsified that assumption empirically:
qwen2.5:7b-instruct produced sum ≠ 1.0 distributions on **18 % of
questions**. Two distinct patterns surfaced (captured live during
diagnosis 2026-05-03):

1. **Mild rounding** — model-intended distribution with arithmetic
   noise. `medqa-test-5` prior:
   `{"A":0.05, "B":0.1, "C":0.2, "D":0.75}` summing to 1.10. Relative
   ranking and rough magnitudes faithful to the model's intent;
   normalisation to 1.0 preserves both with no semantic distortion.

2. **Semantic confusion** — model emitted independent likelihoods
   rather than a normalised distribution. `medqa-test-8` prior:
   `{"A":0.9, "B":0.85, "C":0.05, "D":~0.2}` summing to ~2.0. Two
   options each rated > 80 % likely, which is a logical contradiction
   for a well-formed distribution over mutually exclusive answer
   choices. Normalisation would convert `{A:0.45, B:0.43, ...}` and
   make the distribution look reasonable, but the values would not
   reflect the model's actual epistemic state — the model never
   internalised that the probabilities should be jointly normalised.

The strict 1e-6 tolerance + `temperature=0.0` retries combined to make
both classes deterministic failures — the model emits the same broken
output on every retry, exhausts the retry budget, and the question
gets dropped via `success=False` / `failure_reason =
"hypothesis_distribution_batch_failed"`.

## Decision

**Make the sum-tolerance configurable on the adapter (constructor
parameter `distribution_sum_tolerance: float`), defaulting to 0.05.**

Behaviour:

- Sum within `[1 - tolerance, 1 + tolerance]`: accept. If not exactly
  1.0, normalise to sum=1.0 by dividing each value by the observed
  sum.
- Sum outside tolerance: raise `ValueError` (the existing pathway,
  retried by the caller and surfaced as `LLMAdapterError` if all
  retries fail).
- Numerical comparison uses `math.isclose(total, 1.0,
  abs_tol=tolerance, rel_tol=0.0)` to avoid floating-point boundary
  artefacts.

Default 0.05 is **permissive enough to absorb mild floating-point
arithmetic noise** (sums in [0.95, 1.05]) and **strict enough to
reject the semantic-confusion pattern** observed in the pilot
(sum ~2.0). Production callers can override either direction:

- Tighter (e.g., `0.0` or `1e-9`): research / regression scenarios
  where any deviation is a meaningful signal.
- Wider (e.g., `0.15`): callers who decide the borderline cases
  (1.10-style rounding) should be recovered for their use case.

## Why fail-loudly on semantic incoherence rather than normalise

Three reasons (the load-bearing argument):

1. **Downstream signal pollution.** The signature components
   (entropy, KL-divergence-shaped distance, VoI flatness) are computed
   on the distributions. Normalising a 2.0-sum distribution into
   `{A:0.474, B:0.447, ...}` looks like high-entropy borderline
   evidence, when the underlying model state is "confidently believes
   in two contradictory things." Treating it as legitimate borderline
   evidence biases the framework's evaluation toward false positives
   in a non-obvious way.

2. **Detectability.** A failed record with `failure_reason =
   "hypothesis_distribution_batch_failed"` is visible in the repair
   summary and excluded from AUC computation. A normalised-but-broken
   record disappears into the dataset and contributes ambient noise.
   Known unknowns are more honest than hidden unknowns.

3. **Interpretability of the framework's claim.** The methods paper's
   headline claim is about boundary recognition via structural
   signature on coherent reasoning trajectories. If the underlying
   distributions are silently normalised from incoherent emissions,
   the claim is at minimum ambiguous (signal on what?) and at
   maximum unfalsifiable (we never see the cases where the model
   fundamentally couldn't reason in distribution-shape).

## What this rules out (alternatives considered)

- **Wide tolerance (`±0.5` or larger) + always-normalise.** Would
  have recovered both observed failure classes, including the 2.0
  semantic-confusion pattern. Rejected because of downstream signal
  pollution (above) and because the recovery rate is achieved by
  laundering broken outputs into superficially-valid ones.

- **Hard-coded tolerance (no constructor param).** Simpler but less
  honest about the tradeoff. The strictness/permissiveness choice is
  a real tunable that callers may legitimately want different defaults
  for; making it implicit hides the lever.

- **Per-call tolerance override via `get_hypothesis_distribution`
  signature.** Would let callers vary tolerance question-by-question,
  but the use case is unclear — tolerance is an adapter-level policy,
  not a per-prompt detail. Constructor-level avoids API bloat.

- **Higher repair temperature instead of tolerance change.** Repair
  re-issues at `temperature=0.0` currently, making retries
  deterministic for format-violation patterns. Bumping temperature on
  repair would address the determinism but not the underlying issue
  (model produces sum-≠-1 distributions on first attempt; some retries
  might happen to succeed but others would fail differently).
  Decoupled from this ADR; possible follow-up if tolerance fix proves
  insufficient.

- **Prompt engineering the sum-to-1 constraint.** Higher signal-quality
  if it works (the model produces a real distribution natively), but
  conflates parser-tolerance with prompt-design. This ADR isolates
  the parser axis; prompt-design is a separate experiment.

- **Switch to structured-output mode.** Ollama supports JSON-schema
  enforcement on some models; would eliminate the parse failure at the
  serialisation layer. Substantial code path change and depends on
  Ollama version + model support. Disproportionate to the present
  problem; defer.

## Consequences

- **API surface:** `OllamaLLMAdapter.__init__` gains
  `distribution_sum_tolerance: float = 0.05`. `_parse_and_validate`
  takes the tolerance as a kwarg. No changes to the
  `LLMAdapter` Protocol — tolerance is implementation-specific to
  Ollama (other adapters with different validation semantics may
  expose different knobs).

- **Test surface:** two regression-test fixtures added in
  `tests/reference/test_llm_local.py`, capturing the medqa-test-5
  (mild rounding) and medqa-test-8 (semantic confusion) raw outputs
  observed in the pilot. Future parser changes regression-test
  against captured evidence rather than synthetic hypotheticals.

- **Default-behaviour change:** sums in `[0.95, 1.05]` that
  previously failed (e.g., sum=1.001) now pass (with normalisation).
  Existing callers' behaviour matches the prior semantics for
  sums outside that range. Backward-compatibility for the sum=0.9
  case in `tests/reference/test_llm_local.py:test_parse_and_validate_rejects_sum_mismatch`
  preserved (still rejects; updated error message).

- **Observability:** the repair-rate summary in the experiment runner
  continues to show `n_failures`. With the relaxed tolerance, this
  becomes a more honest measure of model-capability gaps (it counts
  only the cases where the model genuinely failed to produce a
  coherent distribution, not the mild-arithmetic-noise cases).

- **Stage-4a re-pilot:** an N=100 re-pilot at default tolerance
  (`~/work/eunosia/artifacts/medqa-stage-4a-pilot-n100-tol05/`) is
  the next experimental step. Results characterise the
  semantic-confusion / mild-rounding split in the original 18 %
  failure rate.

## Future revisitation triggers

This ADR is locked when:

- A second adapter (vLLM, Anthropic API, base-model adapter) ships
  and the tolerance question recurs in a different validation
  context. At that point: extract a shared `_parse_and_validate`
  utility under `bsig.adapters` with the configurable tolerance,
  and have each adapter wire its own default through.

- The default `0.05` tolerance proves wrong for production use —
  either too strict (callers consistently override to higher) or
  too permissive (silently-normalised distributions visibly degrade
  the signature signal in downstream evaluation).

- Structured-output mode (Ollama JSON-schema enforcement) becomes
  reliable across the project's target models, eliminating the
  parse-failure axis entirely and making this tolerance moot.
