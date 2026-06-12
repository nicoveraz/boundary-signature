# Stage 3 retrospective notes (in-progress)

Running notes accumulating observations across stages 3.1–3.4. Becomes
the basis for the stage-3 retrospective document when 3.5 closes
(parallel to `stage_2_retrospective.md`'s post-stage-2.5 commit).

The structure here is "observation → why it matters → implication for
later stages." Not polished prose; just enough to write the proper
retrospective from when the time comes.

---

## Cross-stage observations

### O1: Stage-1 contracts evolved twice during stage 3, both strictly additively

**Observations:**
- Stage 3.3a added `LLMAdapter.generate` and `generate_batch` (ADR-0005).
- Stage 3.3b added `LLMAdapterError.failed_index` and `partial_results`
  fields for surgical-repair semantics.

**Why it matters:** stage 1's locked Protocol surface was right for what
stage 1 knew (Condition C was the only anticipated consumer of
distribution-style queries). Stage 3 surfaced new requirements —
Conditions A and B want raw text, not distributions; Condition C's
batch error handling wants to identify which item failed for surgical
repair. Both extensions were strictly additive: existing implementations
gained methods/fields to provide, no existing callers broke.

**The strict-addition discipline is what made the evolution
non-disruptive.** Without it, the framework would have needed Protocol
versioning, callers would have needed migration, and the architectural
rigor accumulated in stages 1-2 would have been compromised.

**Implication for stage 3.5:** be ready for a third Protocol evolution
when real Ollama behavior surfaces requirements the mocks couldn't
anticipate (timeout configuration? streaming responses? token-by-token
inspection for diagnostic purposes?). Apply the same discipline:
extensions are additive; existing callers stay correct without changes.

**Implication for stage 4:** when the H100 run with vLLM produces
unexpected behavior, the temptation will be to evolve the Protocol
again. Check first whether the issue is in the implementation rather
than the contract. The Protocol's three-stage stability suggests it's
sufficient for almost any reasonable LLM client; new requirements
should be the exception, not the default.

### O2: Empirical-vs-architectural test distinction is real and matters

**Observations:**
- Stage 3.3b's integration test validated cached-trajectories
  round-trip mechanics with synthetic data. Architectural correctness.
- Stage 3.4's pipeline-validation smoke validated end-to-end
  composition with deterministic mock LLM. Still architectural — the
  mock data was constructed not to engineer signal.
- Stage 4's H100 run will validate empirical signal at N=1273 with
  real qwen2.5:7b output. Empirical validation.

**Why it matters:** conflating the two produces tests that pass while
the framework fails empirically (or fail because of empirical noise).
Each test type has its own failure mode and its own diagnostic value.

**Smoke-2 anchor at stage 3.4:** the pipeline-validation smoke flagged
the wrong-answer high-signature case (smoke-2) as the top failure-mode
table entry. This was directionally consistent with the framework's
premise but driven by hand-constructed distributions, not by genuine
LLM behavior. Stage 4's analysis must verify the same pattern emerges
in real qwen2.5:7b output across N=1273 questions.

**Implication for stage 3.5:** the revised smoke against real Ollama is
*intermediate* — it validates that the framework runs against real
LLM behavior, but at small scale (5-10 questions, single Ollama
session) it's still primarily architectural validation, not empirical
signal validation. Be explicit about this in 3.5's design pass and the
smoke script's docstring.

**Implication for stage 4:** the H100 run is the first true empirical
validation. Plan analysis to distinguish "framework architecturally
worked" (which 3.4 already proved) from "framework empirically
detected the F7-shaped failure mode at scale" (which only stage 4 can
prove or refute).

### O3: Integration-nit pattern is consistent and predictable

**Observations:**
- Stage 2.1: graph.py needed a targeted mypy disable for pandas-stubs
  broadness on `iterrows`/`itertuples` returns.
- Stage 2.2: persistence.py round-trip tests caught NaN-vs-None handling
  for nullable string columns from Parquet read-back; the `_is_null`
  helper was added to handle both cases.
- Stage 2.4: signature.py needed `scipy.stats` to ignore_missing_imports
  in mypy config (no stubs).
- Stage 3.4 part 1: persistence.py's new `build_faiss_indices_from_visits`
  helper needed `numpy` import added (pre-existed elsewhere in the
  module's call chain, but not at module top); plus a `# type: ignore[arg-type]`
  on a pandas-stubs-broad `groupby` key.

**Why it matters:** each integration nit was small individually but they
arrived predictably whenever a module was first integrated with another.
The pattern is "stage X writes module M, tests pass; stage X+1 integrates
M with N, integration surfaces 1-3 small nits that need fixing." The
fix is always small (typically <5 LOC), but ignoring the pattern means
underestimating implementation time.

**The stage 3.4 design pass anticipated this** ("be ready for 2-3 small
issues that need fixing during stage 3.4 implementation"). The
prediction held: 2 nits surfaced and were fixed inline. Stage 3.5 will
likely have similar — Ollama's HTTP behavior has edge cases that mock
LLMs don't capture (timeouts, partial responses, encoding quirks per
ADR-0005's documented concerns).

**Implication for stage 3.5:** allocate small buffer in the
implementation session for integration nits. Don't treat the first
fix as "stage went badly"; treat it as the expected pattern.

### O4: F7 finding's framing implications still need stage-4 validation

**Observations:**
- Stage 3.4's pipeline-validation smoke surfaced the F7-shaped pattern
  (wrong answer + persistent argmax-on-A distributions) at N=3 with
  hand-constructed fixtures. The failure-mode table flagged smoke-2.
- Whether the framing holds at N=1273 with real qwen2.5:7b output is
  empirical and only testable at stage 4.

**Why it matters:** the synthesis document at `e3e09ce` commits to
"internal trajectory disagreement is the headline claim" based on a
single qwen2.5 observation (the original F7) and the pipeline-validation
smoke's directional consistency. Both are necessary but not sufficient
evidence.

**Implication for stage 3.5:** if the revised smoke against real Ollama
on 5-10 questions produces patterns INCONSISTENT with F7 (e.g.,
hypothesis-distribution argmax tracks final-answer argmax in most
cases, contradicting F7's "they disagreed"), surface this immediately
as a serious finding before stage 4. Either F7 was an outlier or the
framework's empirical premise needs revisiting.

**Implication for stage 4:** the analysis script must explicitly check
the F7 pattern. Build a "frequency of CoT-final-answer-vs-distribution-
argmax disagreement" stat into the summary report. Without this stat,
stage 4 can't tell whether the framework is detecting what its claim
says it detects.

### O5: Two-commit-per-stage pattern when integration risk is real

**Observation:** stage 3.4 explicitly committed in two parts (3.4 part
1: evaluation extensions + FAISS helper; 3.4 part 2: pipeline
validation script). The split was anticipated in the design pass
("stage 3.4's commit doesn't have to be atomic; small integration-fix
commits during implementation are appropriate").

**Why it matters:** atomic commits are good for code-review
attributability but bad when integration nits surface mid-stage —
they create the choice between rolling back or pushing through with
mixed-quality commits. The two-commit pattern preserved both clean
parts AND made integration nits visible (the "added numpy import"
was its own line in part 1's commit message).

**Implication for stage 3.5:** plan for 2-3 commits if Ollama
integration surfaces real friction. The reference adapter
(`bsig.reference.llm_local`) lands in one commit; the revised smoke
test that exercises it lands in another; integration-fix commits
between them are fine.

---

## Stage-3.5-specific items surfaced by writing this file

### S5_1: bsig.reference.llm_local needs to satisfy ALL FIVE LLMAdapter methods

Per ADR-0005 + the LLMAdapterError extension, the Protocol surface is
now: `generate`, `generate_batch`, `get_hypothesis_distribution`,
`get_hypothesis_distribution_batch`, `get_metadata`. The Ollama client
must implement all five.

**Implementation order matters** (the four LLM-call methods aren't
symmetric in difficulty):
- `generate` — foundational. Maps directly to Ollama's `/api/generate`
  endpoint. ~30 lines including HTTP handling, retry logic, response
  parsing.
- `get_hypothesis_distribution` — the trickier one. Requires the full
  prompt-construction-plus-distribution-parsing pipeline that conditions
  variously assume. Format hypothesis prompt template, call generate,
  parse JSON, validate sum-to-1, handle parse failures via repair
  re-issue.
- `generate_batch` and `get_hypothesis_distribution_batch` — loops over
  the single-shot variants since Ollama doesn't batch natively. Per-item
  retry semantics fall out of the loop structure for free.
- `get_metadata` — trivial. Last.

Total ~200-300 lines for the full adapter.

### S5_2: Surgical-repair compliance is optional but worth implementing

Per the LLMAdapterError extension docstring, adopters can populate
`failed_index` and `partial_results` for the surgical-repair path, OR
leave them None for atomic-fallback. For Ollama specifically, since
batch is implemented as a loop, partial-state tracking is essentially
free: maintain a results list, accumulate successful calls, on retry
exhaustion return `LLMAdapterError(failed_index=k,
partial_results=results[:k])`.

**Typing detail to verify in the design pass.** Stage 3.3b's
`LLMAdapterError.partial_results` is typed as `object` — permissive
enough to accept both `Sequence[Mapping[str, float]]` (for
distribution-batch failures) and `Sequence[str]` (for generate-batch
failures). This works at runtime but loses type safety at use sites
(`exc.partial_results[0]` returns `object`, not the specific element
type). For 0.1 the permissive typing is fine; the design pass should
verify this is the case the docstring documents both shapes
explicitly. If a tighter type discipline becomes needed, two paths:
generic LLMAdapterError parameterized over the partial-result type, or
two separate exception classes (one per batch operation type). Both
deferable until typing pain shows.

### S5_3: Include `bsig.reference.embedding_st` in stage 3.5

**Strong recommendation (revised from "lean stage 3.5 does just
Ollama").** The case for embedding_st in 3.5 is significantly stronger
than the case against:

**Case for inclusion.** Real embeddings on real reasoning text are
necessary for meaningful canonicalization. The mock embedder produces
deterministic-but-semantically-meaningless vectors; two reasoning
steps about completely different topics get embeddings that are
essentially random with respect to their content similarity. This
means the action_id aggregation that recovery does is operating on
content-similarity-blind hashes — recovery's edge frequency
distribution will look very different with real embeddings vs mock
embeddings.

If stage 4a (M1 Pro preliminary) runs against real Ollama with mock
embeddings, the recovered graph's edge structure isn't representative
of what stage 4b (H100) would produce. The signature scores computed
against that graph aren't predictive of stage 4b's signature scores.
The "preliminary signal validation" loses much of its value because
the signal mechanism (recovery + signature) is operating on
essentially-random embedding semantics.

**Case against inclusion.** Two new modules instead of one. Real
embeddings introduce performance considerations the framework hasn't
yet exercised (multilingual-e5-large model loading: 5-10s, ~2GB RAM).
Stage 3.5 becomes the first time these matter operationally.

**Verdict: include both.** The against-case is real but bounded.
embedding_st is a thin sentence-transformers wrapper — maybe 80 lines
including model loading, batched embed calls, L2-normalization, and
the EmbeddingSource Protocol implementation. Less than the LLM
adapter. Not architecturally novel.

**Sub-stage split: 3.5a + 3.5b.**
- **3.5a:** `bsig.reference.llm_local` (Ollama). Revised smoke against
  real LLM with mock embedder. Validates the LLM adapter end-to-end.
  ~600 lines including tests; one focused session.
- **3.5b:** `bsig.reference.embedding_st` (sentence-transformers +
  multilingual-e5-large). Revised smoke against both real components.
  Validates the full real-component pipeline. ~400 lines including the
  second-smoke configuration; one focused session.

Same architectural logic as the 3.3a/3.3b split: keep the simpler
component standalone, add the more complex one in a second commit.
Each gauntlet checkpoint validates one new component against the
existing pipeline.

### S5_4: Stage 4a collapses into stage 3.5b at scale

If 3.5b ships with the revised smoke against both real Ollama and
real embedder, then **the smoke at N=1273 questions IS effectively
stage 4a.** Architecturally the same code as 3.5b's small-scale
smoke; operationally just a `--n-questions 1273` parameter change
(the pipeline-validation script already accepts `--output-dir`;
extending with `--n-questions` is trivial).

This compresses the stage-4 framing meaningfully:

- **Stage 3.5a:** Ollama adapter, smoke at small scale (3-10
  questions) against real LLM with mock embedder.
- **Stage 3.5b:** embedding adapter, smoke at small scale against
  both real components.
- **Stage 4a:** same revised smoke from 3.5b at full N=1273 scale.
  One real-data preliminary result for the headline experiment.
  ~1-2 days of M1 Pro compute. Output: a real AUC number that
  informs whether stage 4b is worth the H100 budget.
- **Stage 4b:** H100 multi-variant characterization. Conditional on
  4a's signal.

The "stage 4 H100 run" becomes "stage 4b H100 multi-variant
characterization, contingent on stage 4a's local preliminary result."
The H100 commitment is now a contingent decision based on local
results, not a speculative one.

This framing also tightens the gate-experiment-budget argument: if
4a's preliminary AUC is well below 0.5 (no signal at all), 4b's H100
budget shifts toward investigating WHY (different prompt, different
model, different recovery config) rather than just running the
multi-variant grid as planned.

---

## Cross-stage observations (continued)

### O6: The retrospective-notes pattern is a forecasting tool, not just documentation

**Observation:** Writing the stage-3 retrospective notes file (this
file) surfaced three stage-3.5-relevant items (S5_1, S5_2, S5_3) that
weren't yet flagged anywhere. Subsequent engagement (stage-3.4
close-out review) elaborated S5_1 with implementation order, surfaced
the S5_2 typing detail, strengthened the S5_3 recommendation
substantially, and produced S5_4 (stage 4a collapses into 3.5b).

**Why it matters:** the notes are doing real architectural work, not
just recording past decisions. The act of articulating "what's been
learned" forces specificity that surfaces "what comes next has these
implications." Without the notes file, S5_4's stage-4a/3.5b boundary
collapse insight would have emerged during stage 3.5's design pass —
fine, but later, with less time to absorb implications.

**Implication for stage 3.5 (and beyond):** maintain the running
notes file across stages. Re-read before each stage's design pass.
Half an hour of reading at the start of a design pass beats two hours
of mid-implementation re-discovery.

**Implication for the proper stage-3 retrospective:** when 3.5
closes, write the proper retrospective by curating these notes. The
substantive observations (O1-O8) become section headings; the
stage-3.5-specific items (S5_*) become "resolved during 3.5" or
"deferred to stage 4" entries.

### O7: Realistic mocks reduce integration-nit accumulation

**Observation:** Stage 3.5a's tests used `httpx.MockTransport`, which
simulates the actual httpx client behavior (request routing, response
construction, error semantics) rather than mocking the abstraction
layer above it. Result: **zero integration nits at first real-component
run** (Ollama test ran clean without code changes). Contrast with
stage 3.4's two integration nits (numpy import, pandas-stubs
broadness on groupby) which surfaced when production code paths
weren't fully exercised by abstraction-heavy mocks.

**Why it matters:** the integration-nit pattern (O3) fires when a
stage composes new code with existing code in ways tests don't
exercise. Realistic mocks — transport-level for HTTP, fixture-driven
for LLM output — exercise the same code paths as production, just
with deterministic responses. The boundary at which integration nits
accumulate is much narrower than with abstraction-heavy mocks.

**Three abstraction levels for mocking, with trade-offs:**
- High abstraction ("just return this dict"): fastest tests, least
  realistic, most integration nits at first real run.
- Mid abstraction (mock the underlying client / forward call):
  some realism, moderate test fidelity.
- Realistic (transport-level / tiny test model that actually runs):
  highest fidelity, slowest tests, fewest integration nits.

**Implication for stage 3.5b** (sentence-transformers embedder): the
`SentenceTransformerEmbedder` class is thin enough that
high-abstraction mocking is probably right (mock the `embed()` method
to return a fixed numpy array). The class has limited surface area
where mocks can lie about reality. But worth being aware of the
trade-off — if integration nits surface during 3.5b's first real
run, the mock level is the place to look.

**Implication for stages 5-6** (clinical pack + chest-pain
experiment runner): EHR data loaders, multi-signal weak supervision,
and the runner's full-pipeline composition all have rich behavior
that abstraction-heavy mocks fail to capture. Budget time for
realistic fixture construction. The testing investment compounds —
realistic mocks catch integration issues during unit testing rather
than at composition.

### O8.5: Condition C's predicted_answer is a framework-level invariant

**Observation:** Stage 3.5b's checkpoint+resume implementation
initially reconstructed `predicted_answer` from
`argmax(trajectory.states[-1].hypothesis_distribution)`. This is
correct for Conditions A (one-hot) and B (confidence-weighted) but
**structurally incorrect for Condition C** because the framework's
headline claim (F7 / internal trajectory disagreement) explicitly
relies on these two being potentially different. The reconstruction
silently changed predicted_answer for the F7-shaped trajectories the
framework was designed to detect. Caught only by the live integration
verification — the unit tests exercised cached-trajectory shape
correctness but not predicted_answer semantics.

**Framework-level invariant (record this for stage 5/6):**

> Condition C's predicted_answer MUST be sourced from the original
> CoT generation (the model's "Final answer: X" line), NOT from
> `trajectory.states[-1].hypothesis_distribution`'s argmax. The
> framework's headline claim relies on these being potentially
> different. Code that conflates them is structurally incorrect,
> not just buggy.

The trajectory carries both pieces of information separately. The
hypothesis_distribution is the per-step LLM-queried belief; the
final_answer is the CoT integration step. F7 is precisely the case
where they disagree. Any code that derives final_answer from the
distribution erases the framework's signal.

**Implication for stages 5-6** (clinical pack + chest-pain
experiment runner): "predicted disposition" or whatever the
clinical-domain final answer is must follow the same pattern —
sourced from the CoT generation, persisted alongside the trajectory
(not derived from the trajectory's distributions). The bug pattern
in 3.5b's reconstruction will recur if the analogous clinical
reconstruction logic isn't aware of this invariant. Worse: in the
clinical domain, the "F7-shaped cases" don't have the dress-rehearsal
anchor we have for MedQA, so a buggy reconstruction would be harder
to catch by inspection.

**Implication for any future cached-trajectories format change:**
the source-of-truth for predicted_answer must be preserved through
serialization round-trips. Stage 3.5b ships this via
`partial_results.json` (per-condition metadata sidecar). If a future
schema change loses this sidecar, the surgical-repair-equivalent of
this bug will surface elsewhere.

### O8: Low-N runs legitimately produce class-imbalance failures

**Observation:** The pipeline-validation script at N=2 against real
Ollama produced both questions answered correctly → wrong-answer
target had 0 positives → `condition_comparison` would have raised
`EvaluationError("no positive cases")`. The script's
early-warning-and-exit pattern handled this gracefully, but only
because it was designed in.

**Why it matters:** this isn't a bug in the framework or the
runner; it's a legitimate small-N edge case. The framework's
EvaluationError is the right behavior at the framework level
(metrics undefined when one class is missing). The runner's
graceful handling is the right behavior at the runner level
(low-N runs should produce informative output, not stack traces).

**The pattern is generalizable:** stage 5/6's clinical experiment
will have low-prevalence target labels (e.g., specific
clinical-pathway outcomes); future cross-domain experiments
(MMLU professional subjects) could have similar issues; small
diagnostic runs at any stage may hit this.

**Implication for stage 5/6 runners and any future experiment
script:** detect and handle the no-class-balance case gracefully.
The pattern from stage 3.5a's script:

```python
n_pos = int((ground_truth[target_column] == 1).sum())
n_neg = int((ground_truth[target_column] == 0).sum())
if n_pos == 0 or n_neg == 0:
    print(
        "WARNING: ground truth has only one class — "
        "condition_comparison cannot run. Common at small N."
    )
    return 0
```

**Documentation addition:** noted in `bsig.core.evaluation`'s module
docstring that low-N runs may legitimately have class-imbalance
issues; runner scripts should handle this case rather than letting
the framework's EvaluationError surface as a stack trace.

---

## Open at end of stage 3.4

- ADR-0006 (gate-metric revision per F7 reframing): deferred to stage 4
  setup. Recorded as "stub" in stage 3.3a's synthesis addendum.
- Peer-review-defenses section in synthesis document: "next docs-only
  session" per stage-3.3a-close discussion. Half a page. Still pending.
- Workspace data/README.md edit (made on disk during stage-3.2
  exploration; not committed because it lives outside the
  boundary-signature repo). Address at workspace-housekeeping pass.
- Workspace-level git repo (recommended in stage 3.2 close-out): would
  cover workspace docs without crossing repo boundaries. Not now.

---

## Update protocol

Append observations as they emerge across stages 3.5 and any later
stage-3 follow-up work. When 3.5 closes, write the proper
`stage_3_retrospective.md` (parallel to `stage_2_retrospective.md`'s
final form) using these notes as the input. The proper retrospective
gets a "Post-close addenda" section like stage 2's, where stage-4
observations land if they're stage-3-relevant.
