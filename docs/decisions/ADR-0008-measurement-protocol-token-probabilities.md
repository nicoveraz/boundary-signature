# ADR-0008: Measurement protocol as contribution; LLMAdapter extended with token-probability querying; mass capture as recorded measurement

**Status:** Accepted
**Date:** 2026-05-04
**Stage of origin:** stage 4a follow-up; supersedes ADR-0007
**Companion:** workspace memory entries — *measurement protocol as contribution*,
*compartmentalization*, *multi-hypothesis*, *unified measurement*, *no buried problems*,
*mass capture as signal*, *calibrated claims*

## Drafting history (2026-05-04)

This ADR went through three drafts within a single session, each
revision tightening the framing per the project's stated principles:

1. *Initial draft:* constrained-decoding-as-primary-measurement
   language. Locked the GBNF grammar approach as the measurement
   strategy.
2. *Revision after methodology correction:* mass capture identified
   as a candidate signal that constrained decoding hides; the
   measurement strategy switches to unconstrained logprobs with
   mass-capture recording. Status set to Draft pending an empirical
   investigation.
3. *Revision after calibrated-claims correction (this version):* the
   N=50 mass-capture investigation results were initially over-
   interpreted ("perfect-precision boundary signal at extreme tail",
   "primary signal possibly dominating other components"). The
   evidence at N=50 is directionally consistent but does not reject
   the null (95 % bootstrap CI on the correct-vs-wrong mean
   difference Δ = +0.037 spans [-0.063, +0.160]). The N=2
   extreme-tail finding is consistent with chance (P = 0.171 given
   base rate 0.42). Architecturally, mass-capture *recording* is
   well-supported regardless of these outcomes; mass-capture-based
   *signature* claims become stage-4 hypotheses, NOT pre-committed
   architectural design.

The arc reflects the project's stated principles: address
limitations through redesign (not workarounds); compartmentalize
empirical claims from architectural commitments; report claim
strength calibrated to evidence size.

## Context

Stage 4a's N=100 pilot revealed that the framework's per-step Condition C
measurement (verbalized hypothesis distribution emitted by the LLM as JSON
text) was unreliable: 18 % of qwen2.5:7b runs produced sum-≠-1 distributions
that the parser rejected. ADR-0007 attempted a workaround (configurable
parser tolerance, normalize-mild / fail-severe). The post-fix re-pilot
recovered only 3 of 18 failures — the workaround addressed a small
fraction of the symptom and none of the underlying problem.

The underlying problem is structural: **verbalized distributions are
not measurements of the model's beliefs.** They are *the model writing
about its beliefs* — a generative process that's subject to the same
prompt-sensitivity, format-conformance, and arithmetic-coherence issues
as any other text generation. Asking the model to emit a JSON
probability distribution is asking it to perform two tasks at once
(reason about the question; serialize a coherent JSON object); failures
in the second task contaminate the measurement of the first.

A 30-minute investigation 2026-05-04 confirmed that llama.cpp's
OpenAI-compatible `/v1/completions` endpoint exposes top-K next-token
logprobs at any prompt position. Validated on 10 MedQA questions: with
`top_k=25`, all four answer letters appear in the top-K for 10/10
questions; mass capture over A/B/C/D ranges 0.72-0.997 (mean 0.92).

A second 30-minute investigation 2026-05-04 (N=50, mass-capture
correlation) revealed that **the fraction of next-token mass that lands
on the answer letters before renormalisation is information about the
model's commitment-readiness state** — not a defect to be hidden via
constrained decoding or absorbed silently into renormalisation. The
findings are *suggestive* at this sample size and warrant testing at
stage-4 scale:

- Mass capture varies from 0.056 to 0.9995 across the 50 questions.
- Mean mass capture: 0.878 on correct (n=29), 0.841 on wrong (n=21).
  Observed Δ = +0.037. **Bootstrap 95 % CI on Δ: [-0.063, +0.160] —
  straddles zero.** Directionally consistent but not statistically
  established at this N.
- Two of 50 questions had top-1 NOT an answer letter (top-1 = `'2'`,
  the model wanting to continue numeric reasoning rather than commit).
  Both cases were wrong; both had mass capture below 0.20. **At base
  rate 0.42 wrong, P(2 random questions both wrong) = 0.171** — the
  extreme-tail finding is consistent with chance at this sample size.
- Length correlation: shorter questions had higher mass capture (Q1
  = 0.94, Q4 = 0.88). Modest, monotonic-ish.

The mechanistic story is plausible — when the model would rather
continue reasoning than commit, mass capture drops, and the renormalised
distribution becomes a smaller-and-smaller share of the model's actual
next-token belief. **Whether this pattern reflects a real boundary
signal or an artefact of small sample is a stage-4 empirical question
under the multi-hypothesis principle.** Pre-registered predictions
(in the workspace memory *mass capture as signal*) for stage-4 testing:
the Δ correlation strengthens at N=1273 with CI lower bound > 0, the
extreme-tail (mass < 0.25) wrong-rate exceeds the base rate
significantly, and the non-letter-top-1 pattern replicates.

Architecturally, the cost of recording mass capture is one float per
measurement; the architectural commitment is well-supported regardless
of how the empirical predictions resolve. This ADR records architectural
commitments around *what gets recorded*; what role mass capture plays in
the framework's signal is determined by stage-4 evaluation, not pre-
committed here.

## Decisions

### 1. The framework is reframed as a *measurement protocol*

The framework's contribution is the protocol — explicit choices about
prompt structure, measurement positions, hypothesis space, top-K
parameter, terminal-answer determination. Not a discovery about LLM
cognition.

**Implication for the methods paper:** the title and structure shift
to a methodology paper. Title direction:
"A Measurement Protocol for Per-Step Belief Monitoring in
Chain-of-Thought Reasoning, with Application to Clinical Boundary
Detection." The methodology section is a primary contribution; every
protocol choice is named, justified, alternatives acknowledged,
sensitivity tested where practical.

**Implication for the architecture:** the framework commits to *one*
concrete protocol via this ADR (the unified-measurement protocol with
mass-capture recording specified below). The architecture supports
alternative protocols without locking in this one. A future
`MeasurementProtocol` abstraction may centralise the prompt-structure
/ decomposer / measurement-prefix / hypothesis-space / mass-capture-
threshold decisions when a second concrete protocol motivates the
refactor.

### 2. Unified measurement strategy

Condition C's per-step monitoring and predicted-answer extraction
are *the same procedure*. At each reasoning-step boundary k
(0 ≤ k ≤ N, where N is the number of decomposed reasoning steps),
the model's belief distribution over the answer choices is read via
unconstrained top-K next-token logprobs, renormalised over the answer
letters. The predicted answer is `argmax(distributions[N])` — the
argmax of the terminal measurement.

There is no separate predicted-answer extraction from CoT text. The
CoT prompt becomes minimal: "Reason step by step about this
question. Show your reasoning." No "End with: Final answer: X"
instruction. The reasoning content is whatever the model produces;
the answer is read by measurement.

This eliminates:
- The verbalised-distribution measurement (formerly used for per-step
  monitoring).
- The cross-method comparison between CoT-extracted final answer and
  per-step verbalised distributions (formerly the framework's
  internal-disagreement signal). Disagreement now is between
  measurement_at_step_k and measurement_at_step_N — within-method,
  commensurable comparisons.
- Prompt-induced variation in final-answer extraction (F9 finding).
  The reasoning text varies; the measurement protocol is invariant.

### 3. Mass capture is a recorded measurement output

For each measurement, the framework records BOTH:

- The renormalised conditional distribution
  P(letter | letter ∈ {A,B,C,D}) over the answer letters.
- The mass-capture fraction Σ P(letter) — what fraction of the
  model's next-token mass landed on answer letters before
  renormalisation.

Mass capture < 1.0 is information about the model's commitment-
readiness state at the measurement position. The framework records
it; whether it functions as a primary signal, a complementary
component, or a measurement-quality indicator only is a stage-4
empirical question, NOT pre-committed by this ADR.

Candidate uses for mass capture in signature scoring (all to be
evaluated empirically at stage 4 under the multi-hypothesis principle):

- **As a separate component** in the composite signature.
- **As a weighting factor** so low-mass-capture states get less
  influence in entropy/distance computations.
- **As a threshold gate** where measurements with mass capture below
  a configurable threshold trigger an explicit boundary flag separately
  from the composite signature.
- **As a measurement-quality indicator only**, recorded alongside the
  conditional distribution but not directly contributing to scoring.

The default composite weighting is determined by stage 4 results,
not pre-committed here. The multi-hypothesis principle says results
are reported across multiple operationalisations regardless; no single
weighting is "the" answer.

### 4. LLMAdapter Protocol gains `get_token_probabilities` (returns mass capture)

A new method, strictly additive (per the stage-1 evolution pattern;
matches the pattern from ADR-0005 which added `generate`):

```python
@dataclass(frozen=True)
class TokenProbabilityResult:
    """Output of get_token_probabilities. ``distribution`` is the
    renormalised conditional over the requested token_set;
    ``mass_capture`` is the fraction of next-token mass that landed on
    token_set before renormalisation. Both are required for the
    framework's downstream signature computation."""
    distribution: Mapping[str, float]
    mass_capture: float

class LLMAdapter(Protocol):
    ...existing methods...

    def get_token_probabilities(
        self,
        prompt: str,
        token_set: Sequence[str],
        top_k: int = 40,
        max_retries: int = 2,
    ) -> TokenProbabilityResult:
        """Read the model's next-token distribution at the position
        immediately following the prompt. Returns the renormalised
        conditional distribution over ``token_set`` and the mass-capture
        fraction.

        Implementations request enough top_logprobs to ensure all
        token_set members are reliably present (default 40).
        Token-aliasing (leading-space variants like " A" vs "A") is
        handled by the implementation; the returned distribution's
        keys are exactly ``set(token_set)``.

        Adapters that lack logprobs API access (e.g. Anthropic API)
        cannot implement this method; downstream code (Condition C)
        requires it.
        """
        ...

    def get_token_probabilities_batch(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        top_k: int = 40,
        max_retries: int = 2,
    ) -> Sequence[TokenProbabilityResult]: ...
```

Per-item retry semantics on the batch variant (matching the existing
`_batch` patterns). Adapters that can populate `failed_index` and
`partial_results` on the raised `LLMAdapterError` enable surgical-
repair downstream.

The existing `get_hypothesis_distribution` and
`get_hypothesis_distribution_batch` methods stay in the Protocol.
No consumer in the project's own code calls them after this ADR
lands; they remain available for third-party adapters or future
comparison studies. Cleanup (removing them) is a candidate for a
future major-version break if no consumer materializes.

### 5. `LlamaCppLLMAdapter` is the reference implementation

A new reference adapter (`src/bsig/reference/llm_llama_cpp.py`),
behind a new opt-in extra (`llama_cpp = ["httpx>=0.27"]`).
Communicates with a llama.cpp server via OpenAI-compatible HTTP API.

Key implementation points (locked here for architectural clarity;
subject to refinement during the implementation pass):

- `top_k = 40` default (validated as safe margin; investigation showed
  K=25 is empirically sufficient for MedQA's 4-letter set).
- Token aliasing: the adapter normalises leading-space and leading-
  newline variants (` A`, `A`, `\nA`) when matching against `token_set`,
  summing probability across variants for each canonical token.
- Mass-capture computation: sum of (post-aliasing) probability over
  every member of `token_set` before renormalisation. Returned as a
  field on `TokenProbabilityResult`.
- Constrained-decoding via GBNF grammar: NOT used by default.
  Constraints would force the model to emit a letter, hiding the
  mass-capture signal that this ADR establishes as load-bearing.
  Available as an OPT-IN behaviour for callers that want predicted-
  answer output safety (e.g., when generating a final answer for
  display, separate from the framework's measurement).
- Determinism caveat: Metal-GPU FP-order produces ~0.3 % per-letter
  noise across calls at temp=0.0. Documented in `get_metadata()` as
  the `determinism_class` field. Methods-paper reproducibility footnote
  will acknowledge this.

### 6. Condition C is rewritten around the unified measurement

`bsig.medqa.conditions.condition_c.ConditionC.run` becomes:

1. Generate CoT (`generate(initial_prompt_minimal)`). Minimal prompt:
   "Reason step by step about this question. Show your reasoning."
2. Decompose CoT into N reasoning steps (`Decomposer.decompose`).
   The decomposer no longer extracts a final answer; that role is
   removed.
3. For each step boundary k in `0..N`:
   - Build a measurement prompt: question + choices + reasoning-up-to-step-k
     + measurement prefix.
   - Call `llm.get_token_probabilities(prompt, ["A","B","C","D"])` and
     get back `TokenProbabilityResult{distribution, mass_capture}`.
4. Build trajectory with N+1 states. Each state's
   `hypothesis_distribution` is the result's `distribution`; each
   state's `metadata` carries `mass_capture` as a field.
5. `predicted_answer = argmax(distributions[N])`.

The verbalised-distribution flow (current `condition_c_hypothesis.txt`
prompt + JSON parsing + `_parse_and_validate`) is removed from
Condition C. The prompt template is replaced with a measurement-prefix
template (`condition_c_measurement.txt`, content: "The best answer is ").

### 7. Cached-trajectories format absorbs the change

The `Trajectory.states[k].hypothesis_distribution` field already
holds a `Mapping[str, float]`; the new measurement produces values
of the same type. Mass capture is added as a metadata field on each
State (the existing `metadata: dict[str, Any]` accommodates it
without schema change). Parquet round-trip handling already supports
arbitrary metadata fields.

The framework's data model has been built around the abstract concept
of "per-step distribution" rather than the specific generative source.
Condition C's measurement-source change is invisible to recovery,
signature, and evaluation — except that mass capture becomes
available as input to signature components.

## Why this and not the alternatives

- **Wider parser tolerance.** Tried via ADR-0007. Recovered 3 of 18
  pilot failures. Doesn't address the underlying methodology problem.
  Superseded by this ADR.

- **Constrained decoding via GBNF grammar.** Initial recommendation
  during the architectural design pass. Empirically equivalent to
  unconstrained-renormalised at the conditional-distribution level
  (KL < 0.0004 nats on the validation sample), but **hides mass
  capture** — the model's reluctance to commit to a letter is
  forced into commitment by the grammar mask. The mass-capture-
  correlation investigation produced suggestive (though not
  statistically significant at N=50) evidence that mass capture
  carries information about model commitment-readiness; hiding it
  before stage-4 evaluation can determine its role would be a
  buried-problem pattern.

- **Prompt engineering of the verbalised-distribution prompt.** Higher
  signal-quality for some questions; doesn't solve the fundamental
  issue (the model is performing two tasks: reasoning + JSON
  serialisation). Conflates measurement quality with prompt design.

- **Switch to structured-output mode (Ollama JSON-schema).** Would
  fix the parse failures but not the underlying methodology problem.
  Verbalised distributions remain a generative process; their values
  remain prompt-sensitive in non-obvious ways.

- **Keep verbalised distributions; add token-probabilities as a
  cross-check.** Two measurements where one suffices. Adds complexity
  without resolving the fundamental issue.

- **Switch to a larger model (qwen2.5:14b).** Doesn't fix the
  methodology — larger models also produce verbalised distributions
  with the same circularity, and the token-probability + mass-capture
  measurement protocol is independent of model size.

The clean answer is to switch the measurement and record mass capture
as a first-class output.

## Consequences

### Code changes (separate implementation step from this ADR)

- New: `src/bsig/reference/llm_llama_cpp.py` — `LlamaCppLLMAdapter`
- New: `tests/reference/test_llm_llama_cpp.py` — adapter tests with
  `httpx.MockTransport`, regression fixtures from this investigation
- New: `src/bsig/medqa/prompts/condition_c_measurement.txt` — the
  measurement-prefix template ("The best answer is ")
- New: `TokenProbabilityResult` dataclass in `bsig.adapters.base` (or
  `bsig.adapters.llm` — to be decided in the implementation pass)
- Modified: `src/bsig/adapters/llm.py` — Protocol gains
  `get_token_probabilities` and `get_token_probabilities_batch`
- Modified: `src/bsig/medqa/conditions/condition_c.py` — rewritten
  around `get_token_probabilities`; minimal initial prompt; no
  answer-letter extraction from CoT; mass capture stored in State
  metadata
- Modified: `src/bsig/medqa/prompts/condition_c_initial.txt` —
  minimal CoT instruction (no "Final answer: X")
- Modified: `src/bsig/medqa/conditions/decomposer.py` —
  `answer_letter` extraction made optional/deprecated; not removed
  yet to preserve back-compat for any external consumer
- Modified: `src/bsig/core/signature.py` — components gain optional
  mass-capture awareness (default: ignore; opt-in: use as
  weighting factor or as a separate composite component)
- Modified: `pyproject.toml` — new `llama_cpp` extra
- Modified: `experiments/medqa_generalization/scripts/03_pipeline_validation_ollama.py`
  — likely renamed to `…_llama_cpp.py` and reworked, OR a parallel
  `04_pipeline_validation_llama_cpp.py` is added so the Ollama
  variant remains runnable for comparison studies. To be decided
  during the implementation pass.

### Doc changes

- ADR-0007: Status changed from Accepted to Superseded by ADR-0008.
  Done.
- This ADR (0008): Accepted (after mass-capture-correlation
  investigation findings incorporated). Done.
- `docs/decisions/stage_4a_pre_run_analysis_plan.md`: updated to
  reflect that the original strict-tolerance plan was mid-course-
  corrected via ADR-0008. Original plan preserved unedited;
  amendment recorded in a new "Revisions" section.
- `CLAUDE.md` §5 (LLMAdapter Protocol): updated to describe
  `get_token_probabilities` alongside the existing methods.
- `CLAUDE.md` §15 (open questions): the verbalised-distribution
  question is closed (resolved by this ADR). New entries: cleanup
  of `get_hypothesis_distribution` once no consumer remains; the
  `MeasurementProtocol` abstraction; specific weighting of
  mass-capture in the default signature composite.
- `docs/framework/boundary_aware_reasoning_general_framework_v0.3.md`:
  major revision incorporating the locked methodology principles.
  Large; deferred to a dedicated session.

### Cost / wall-clock

The new measurement is structurally equivalent in cost to the old
one: 1 generation call (CoT) + N+1 single-token forward-pass +
softmax calls (per-step measurements). The per-step measurements
are if anything cheaper than the old verbalised-distribution calls
because they generate one token, not a JSON object.

### Re-pilot expectations

Stage 4a re-pilot under the new protocol should show:
- C `n_failures` near zero (no parser failure mode in the new
  measurement; only transport-layer failures remain).
- C composite AUC, B AUC, A AUC at face value comparable across
  conditions (no Condition-C-specific drop-out).
- **Mass-capture-on-its-own** evaluated as a candidate deferral signal
  (`1 - mass_capture`), reported alongside the composite. Stage-4
  data is what determines whether this is a strong, weak, or null
  signal at scale.
- The N=50 investigation's pre-registered predictions (correct-vs-wrong
  Δ CI lower bound > 0; extreme-tail wrong-rate elevated; non-letter
  top-1 pattern replicates) are tested in this run. Whichever resolve
  positively shape the methods-paper claim; whichever resolve null
  inform what mass capture is *not*.
- Wall-clock comparable to the original pilot (~100 minutes for N=100).

If C's composite AUC remains low after the methodology fix, the
framework's broader signal genuinely needs scrutiny. The diagnostic
cascade per the original `stage_4a_pre_run_analysis_plan.md` applies
in that case — multi-component analysis, embedding-bin sweep,
prompt-variant sweep — interpreted under the multi-hypothesis
principle (no single operationalisation is "the" answer).

## Implementation-pass design questions — resolved 2026-05-04

The four implementation-pass design questions identified during the
ADR's drafting were resolved in a focused design pass before
implementation began. The resolutions:

1. **`TokenProbabilityResult` lives in `bsig.adapters.llm`**, co-located
   with the Protocol method that returns it. The `LLMAdapterError`-in-
   base precedent doesn't transfer (exceptions are caller-cross-cutting,
   caught generically; return types are method-specific). Keeping
   `base` focused on cross-adapter abstractions and exceptions
   prevents it from becoming a dumping ground.

2. **`top_k` is NOT exposed at the Protocol level.** The adapter
   handles internally based on `len(token_set)` (default heuristic:
   `effective_top_k = max(40, 10 × len(token_set))`). The analogy with
   `max_retries` (caller-tolerance, model-orthogonal) doesn't transfer
   to `top_k` (model-tokenization property, adapter-driven). Callers
   don't have the information to set `top_k` correctly; adapters do.
   `LlamaCppLLMAdapter` exposes `logprobs_top_k` as a constructor
   parameter for callers who need to override the heuristic.

3. (Subsumed by #2.) Q3 was the same axis as Q2 at the Protocol-
   design level. Resolution applies.

4. **Permissive batch contract** (per-item retry semantics matching
   existing `get_hypothesis_distribution_batch`). `failed_index` and
   `partial_results` on raised `LLMAdapterError` are encouraged but
   not required; atomic-repair fallback in the caller handles
   adapters that can't preserve partial state. `LlamaCppLLMAdapter`
   implements surgical-repair (loop pattern transfers from
   `OllamaLLMAdapter`).

**Two refinements identified during the design pass that strengthen
this ADR's commitments:**

5. **`TokenProbabilityResult` gains a third field, `truncated_members:
   tuple[str, ...] = ()`.** When a `token_set` member is below the
   adapter's effective top-K even after one auto-extending retry, the
   adapter populates `truncated_members` with that letter and returns
   `P=0` for it. Empty tuple in the common case (full coverage).
   Surfaces measurement-quality information rather than silently
   biasing the renormalised conditional. Methods paper can report
   the rate of truncation across the test set as a measurement-
   quality indicator. Consistent with the *no buried problems* and
   *mass capture as signal* principles.

6. **`mass_capture` is a structured field on `State`, not a metadata
   dict entry.** Mass capture is a load-bearing universal measurement
   (any token-probability protocol over a finite hypothesis space
   produces it); the type-checked structured field reflects its
   first-class status. Future protocol-specific per-position fields
   (decision-point indices, sample counts, exclusion targets) live in
   `metadata` for extensibility. The `State` data model gains
   `mass_capture: float | None = None` as an Optional field for
   backward-compatibility with existing trajectories.

The cached-trajectories Parquet schema bumps from v1 to v2 to mark
the addition of the `mass_capture` column. Readers handle both
versions; v1 trajectories deserialise with `mass_capture=None`.

These six resolutions (4 questions + 2 refinements) are the
architectural ground truth for the implementation pass.

## Future revisitation triggers

This ADR is locked when:

- A second adapter (vLLM, Anthropic API, base-model adapter) ships
  and reveals a contract gap in `get_token_probabilities` or
  `TokenProbabilityResult`.
- A second measurement protocol is designed (e.g., the
  exclusion-trajectory protocol for stage-6 chest-pain). At that
  point the `MeasurementProtocol` abstraction becomes worth
  introducing, and Condition C's current orchestration becomes one
  implementation among several.
- The framework's claim is empirically validated at full N=1273
  scale and the methods paper begins; "what gets presented as the
  contribution" becomes load-bearing and this ADR's language about
  protocol-as-contribution gets re-examined in context.
- The mass-capture signal turns out to be empirically dominant or
  dominated relative to the existing signature components.
  Default-composite weighting gets revised; the multi-hypothesis
  principle says we report across operationalisations regardless,
  but the default may shift.
