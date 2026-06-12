# Corruption registry

**Started:** 2026-05-07
**Status:** active; entries appended as encountered.
**Pattern owner:** the project (no individual ADR; this is a meta-pattern).
**Promotion criterion:** if/when 5+ entries with clear category structure
accumulate by stage-6, promote the *principle* (not the registry itself)
to ``CLAUDE.md``. Until then this file lives in ``docs/decisions/``
alongside ADRs.

---

## What this is

An append-only log of corruption modes the framework has encountered,
the immediate fix applied, and the generalization status. The registry
itself is the discipline; specific framework refactors are earned by
patterns *across* registry entries (3+ similar incidents), not by any
single entry.

The pattern was decided on 2026-05-07 after a long design-pass
conversation considered ffmpeg-style "resilience to corrupted input"
as a project principle. The eager reading would have committed
speculatively to ``MeasurementOutcome`` discriminated unions,
``TrajectoryQuality`` summaries, adapter validation wrappers, and
quality-score logging — generalizing from N=1 incident (decomposer
regression). Per ``feedback_calibrated_claims.md`` and
``project_no_buried_problems.md``, that's the smoke-pattern-becoming-
confirmatory-claim antipattern.

The minimal-commitment reading is: log modes as they're encountered,
fix the immediate instance, defer generalization until the registry
shows pattern.

---

## Load-bearing principle (the qualifier matters)

**Fail loud in research paths; degrade gracefully in deployment paths.**

Without the qualifier, "degrade gracefully" hides bugs in empirical
work — exactly what the project's calibrated-claims discipline is
designed to prevent. The decomposer-regression fix worked precisely
*because* nulls were treated as wrong rather than as
gracefully-handled-failure: the loud failure mode forced diagnosis
of the underlying regex strictness.

Research code (experiment scripts, evaluation pipelines, validation
runs) should **fail loud** — silent degradation in this layer
contaminates published numbers. Deployment code (clinical
decision-support inference, real-time chat) should **degrade
gracefully** — failures here are user-visible in ways that matter
clinically; the user can't diagnose at the prompt.

The boundary is the framework's outer interface: caching outputs,
exporting results, presenting deployment signals. Fail loud at that
boundary; degrade gracefully internally only on the deployment side.

---

## Entry format

Each entry has:

- **Date** — when encountered
- **Category** — one of: ``adapter``, ``trajectory``, ``schema``,
  ``data``, ``engine``
- **Incident** — what specifically happened
- **Fix applied** — what was changed at the time
- **Generalization status** — one of: ``deferred`` (no broader
  refactor warranted yet), ``suggested`` (pattern visible but
  awaiting more entries), ``committed`` (refactor landed; see linked
  artifact)
- **Cross-references** — commits, ADRs, exploration writeups

Categories:

| Category | Meaning |
|---|---|
| ``adapter`` | LLM/embedding adapter returns malformed responses (truncated, NaN, missing logprobs, garbage tokens) |
| ``trajectory`` | Incomplete trajectories, mixed schema versions, positions where measurement failed |
| ``schema`` | Cached trajectories with field mismatches, partial fields, type mismatches |
| ``data`` | Reference-corpus corruption (missing vitals, free-text complaints, ICD codes outside scheme) |
| ``engine`` | Inference engine itself misbehaves (5xx errors mid-batch, OOM, port displaced, OOV tokens) |

---

## Entries

### 2026-05-04 — decomposer regex regression

**Category:** ``adapter``
**Incident:** Condition-C decomposer regex was too strict; LLM
outputs varied across runs (whitespace, optional bullet prefixes,
list-vs-prose phrasing). Strict pattern produced ``None`` for
non-conforming outputs; the framework treated null as
"measurement-failed" — *correct loud behavior*. Discovered when an
unexpected fraction of trajectories registered as missing reasoning
steps.
**Fix applied:** relaxed the regex to admit whitespace and bullet
variants; added test fixtures covering the encountered variants.
**Generalization status:** ``deferred``. The instance is fixed; no
broader refactor warranted from N=1. The fail-loud behavior worked.
**Cross-references:** F7 finding writeup
``docs/exploration/condition_c_end_to_end_2026-05-03.md``.

### 2026-05-07 — port-8080 contention during cross-adapter validation

**Category:** ``engine``
**Incident:** Cross-adapter agreement test crashed at q17/50 because
``llama-server`` was displaced by the local clinical-app web UI
("Pendientes Urgencia") periodically capturing port-8080 traffic.
The HTTP response returned an HTML page where JSON was expected;
the adapter raised. The 16/50 questions completed before crash gave
usable diagnostic content because the test framework streamed
per-question results — but a different test architecture would have
lost everything.
**Fix applied:** none for the contention itself (environment-
specific). Per-question streaming with ``flush=True`` was already
in place and saved the partial results.
**Generalization status:** ``committed``. Resumable-campaign
property documented at
``docs/decisions/resumable_campaign_property.md``. The pattern: any
script with expected wall-time >30 min must produce per-unit
artifacts allowing resumption from arbitrary interruption.
Stage-4b's accidentally-resumable behavior is the canonical
example. Future long-running scripts must satisfy the property
unless documented otherwise.
**Cross-references:**
``docs/exploration/2026-05-07-cross-adapter-agreement-partial.md``.

### 2026-05-07 — mlx-lm batched-decode incremental correctness limitation

**Category:** ``engine``
**Incident:** Phase C semantic-entropy implementation attempt
discovered that mlx-lm's incremental decode (passing a ``(B, 1)``
input through a model with a populated KV cache) does not preserve
correctness for ``B > 1``. Empirical probe at N=3 with identical
prefix replicated across rows showed: post-prefill cache state is
identical across rows (max-abs-diff = 0.0), step-0 logits agree
across rows, but step-1 onwards row 0 evolves consistently with
single-batch behavior while rows 1+ diverge — even with identical
inputs and identical pre-step cache state. The divergence
originates inside the model's attention or RoPE handling at
batch > 1 in incremental mode; the cache itself is not the
problem. Demonstrated against ``mlx-community/Qwen2.5-7B-Instruct-
4bit`` on mlx-lm 0.31.
**Fix applied:** Phase C batched decode reverted; ``_batched_sample_completions``
routes ``n_samples > 1`` through Phase A serial (per-sample
``_sample_completion`` calls). A hybrid approach (single batched
prefill + per-sample serial decode via cache slicing) was
prototyped and benchmarked at ~1.05x speedup vs Phase A serial on
decode-dominated workloads; did not earn its complexity for the
Phase C semantic-entropy use case and was reverted. Phase C
revisits when either mlx-lm fixes batched incremental decode or
vllm-mlx engine mode lands.
**Generalization status:** ``deferred``. The framework's
*measurement* path (``get_token_probabilities_batch`` →
``_batch_with_shared_prefix``) is unaffected — that path uses
shared-prefill + batched-prefill across N suffixes and does not
enter incremental decode at batch>1; it produces correct results
with ~2.20x speedup at 1014-token prefix (see
``2026-05-07-mlx-stage-6-shape-speedup.md``). This is a
single-instance engine limitation; the broader pattern (untrusted
upstream incremental APIs at batch>1) doesn't yet have the
2nd/3rd similar incident to earn a refactor.
**Upstream tracker:** https://github.com/ml-explore/mlx/issues/3494
(filed 2026-05-07 under user nicoveraz; bug isolated to
``mx.fast.rope`` Metal kernel at ``batch >= 2, seq_len == 1`` —
not mlx-lm). **Fix PR #3498 MERGED 2026-05-11** (commit
``76a977c`` into main; angeloskath, 3-line patch to
``mlx/backend/metal/rope.cpp`` combining ``B * N`` into kernel
y-dim grid; 22-line test mirrors the minimal repro). Maintainer
note: "Crazy that this was a bug for more than a year and nobody
encountered it."

**Release status (verified 2026-05-23):** merged to main but NOT
yet in a released wheel. Latest PyPI tag is ``mlx==0.31.2`` (cut
2026-04-22 — predates the merge). Empirical repro against the
installed 0.31.2 still reproduces the bug
(``max-abs-diff = 3.14`` for two independently-allocated but
content-identical RoPE rows at ``batch=2, seq_len=1``). Phase C
batched decode unblocks at the first released mlx-core > 0.31.2
that ships commit ``76a977c`` — **not at merge**. Re-enabling
against unreleased ``main`` would violate the ``mlx-lm>=0.31``
release pin (CLAUDE.md §13 "Pinned dependencies"); the
disciplined unblock is the next PyPI release. Re-test with the
repro above on each mlx bump.

**Fix VALIDATED ahead of release (2026-05-23).** mlx built from
source at ``main`` (commit ``2165dc0``, version
``0.32.0.dev20260523``) to de-risk Phase C before the wheel ships.
Both checks PASS: (1) rope repro ``max-abs-diff = 0.0``; (2)
batched decode N=3 on ``Qwen2.5-7B-Instruct-4bit`` at temperature=0
is **bit-identical** across rows and to the serial argmax path
(pre-fix, rows 1+ diverged). The §6.2 Amendment 1 batched design is
confirmed sound; re-enabling it on the released wheel is now
low-risk rather than speculative. Project venv restored to pinned
0.31.2 afterward (dev build was validation-only). Writeup:
``docs/exploration/2026-05-23-phase-c-rope-fix-validation.md``.
Build prerequisite worth recording: the ``metal`` shader compiler
was absent (CLT active, not full Xcode); required
``xcodebuild -downloadComponent MetalToolchain`` after license +
``-runFirstLaunch``, then build with
``DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer``.

**Cross-references:**
- ``src/bsig/reference/llm_mlx.py::_batched_sample_completions``
  (the documented routing).
- ``stage_6_mlx_adapter_pre_design_notes.md`` §6.2 Amendment 1
  (the original Phase C design; updated with the upstream-
  tracker URL).
- ``project_dev_hardware.md`` (Phase C 3-5x projection now bounded
  by the upstream limitation; current Phase A serial speedup is
  what's measured).

### 2026-05-07 — bit-identical-weights cross-adapter precondition

**Category:** ``engine``
**Incident:** Cross-adapter agreement test produced 75% argmax
agreement vs the ≥98% pre-registered threshold. Diagnosis (per
``project_diagnose_rather_than_reframe.md``): GGUF Q4_K_M and MLX
4-bit are *different quantization codecs* applied to the same base
weights. The threshold implicitly assumed bit-identical weights;
the test as run did not satisfy that precondition. The threshold
was NOT relaxed post-hoc; the precondition was documented as a
load-bearing constraint instead.
**Fix applied:** methods-paper §7.4 (threats to validity) updated
in commit ``9ccd36d`` to reflect: cross-adapter interchangeability
requires bit-identical weights; the current GGUF/MLX paths do not
provide that.
**Generalization status:** ``committed``. Provenance assertion in
``bsig.core.signature.compute_signatures`` refuses aggregation
across distinct ``adapter_name + model + quantization +
schema_version`` without explicit ``force_mix=True`` override. The
check is opportunistic: provenance keys absent from
``State.metadata`` are not enforced (no-op for that key); the
assertion becomes load-bearing as scripts incrementally populate
provenance. See ``_check_provenance_compatible`` in
``src/bsig/core/signature.py`` and tests
``tests/core/test_signature.py::test_compute_signatures_refuses_*``.
**Cross-references:**
``docs/exploration/2026-05-07-cross-adapter-agreement-partial.md``,
``project_cross_quantization_disagreement.md``.

---

## Pattern emergence (review on demand)

When 3+ entries share a category or share a generalization
status of ``suggested`` pointing at the same refactor target, that's
the trigger to lift the suggested generalization to ``committed``
and land the broader change. Until then, entries accumulate.

**Currently suggested (single instances; not yet earning refactor):**

- ``MeasurementOutcome`` discriminated union vs nullable values —
  earned by 2nd encountered ``adapter``-category mode where the
  null-as-failure pattern is the bottleneck for diagnosis. Decomposer
  regression alone is N=1; not earned.
- ``TrajectoryQuality`` summary alongside aggregates — earned by 2nd
  ``trajectory``-category mode where aggregate consumers need
  failure-rate context to decide trust. Currently zero entries.
- Schema migration utilities — earned when schema-v5 happens (the
  schema-v3→v4 migration in ADR-0009 was carefully done but bespoke;
  generalizing now would be premature).

**Explicitly deferred (do not land speculatively):**

- Adapter validation wrappers checking returned distributions for
  sanity (logprobs sum, top-K consistency). Real validation is
  best motivated by an actual incident where unchecked output
  contaminated downstream — not yet observed.
- Quality-score logging throughout the pipeline. Architectural
  change with broad surface; needs evidence the absence is the
  bottleneck.

---

## How to add an entry

When a corruption mode is encountered:

1. Apply the immediate fix to the affected code path.
2. Append a new entry to this registry following the format above.
3. Mark generalization ``deferred`` unless the new entry plus
   prior entries clearly suggest a broader refactor — then mark
   ``suggested`` and link the suggested target.
4. If a previously-``suggested`` generalization is now earned by
   this entry (i.e., 3+ similar), flip the prior entries'
   generalization status to ``committed`` and link the refactor
   PR.

The registry's value compounds with use. Each entry is small;
collectively they make the corruption-mode landscape visible and
disciplined.
