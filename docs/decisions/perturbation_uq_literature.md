# Perturbation-based uncertainty quantification — literature positioning

**Date:** 2026-05-23
**Status:** ACCEPTED (positioning record; not a code change)
**Placement:** `docs/decisions/`, NOT CLAUDE.md. Same standard as the
corruption registry — design-record-worthy, but has not earned
CLAUDE.md promotion through extensive practice.
**Predecessors / related:**
- `stage_6_chest_pain_pre_design_notes.md` (E_quant_3 cross-quantization
  disagreement; E_perturb_1 added alongside it by this record).
- `project_cross_quantization_disagreement.md` (project memory anchoring
  the E_quant_3 research direction).
- `feedback_calibrated_claims.md` (the discipline this record enforces:
  claim strength matched to evidence; no novelty claim on established
  methodology).
- `project_diagnose_rather_than_reframe.md` (applied below — the SRE
  caveat surfaces a real tension between the two perturbation axes,
  recorded rather than papered over).

## Why this record exists

Literature engagement on perturbation-based uncertainty quantification
(UQ) surfaced established prior work that repositions the framework's
contribution claim. The cross-quantization disagreement signal
(E_quant_3, already pre-registered) and any temperature-perturbation
extension (E_perturb_1, added by this record) both sit inside the
**perturbation-based UQ family**. The honest contribution is *not* the
perturbation intuition itself — that is established — but the specific
adaptation, axis combination, validation domain, and discipline pattern.

This record fixes the positioning before stage-6 so the methods paper
and the pre-registration cite the family correctly and do not
overclaim novelty.

---

## 1. Established literature

- **SPUQ** — Gao, Zhang, Mouatadid & Das, *Spectrum of Perturbation for
  Uncertainty Quantification in LLMs*, EACL 2024 Long Papers,
  pp. 2336–2346, doi:10.18653/v1/2024.eacl-long.143. Code:
  `github.com/intuit-ai-research/SPUQ`. Perturbs the **prompt** and the
  **temperature**, then aggregates output variation into an uncertainty
  estimate. Reports ~50% average ECE reduction. Foundational reference
  for the family.

- **Monte Carlo Temperature (MCT)** — Cecere, Bacciu, Fernández Tobías &
  Mantrach, 2025 (Amazon). Treats **temperature as a varied axis** rather
  than a fixed parameter, marginalising over it; achieves statistical
  parity with oracle temperatures without per-task hyperparameter
  optimisation.

- **Inv-Entropy** — Nov 2025. Introduces **Temperature Sensitivity of
  Uncertainty (TSU)**, a metric specifically for how the output
  distribution's *shape changes under temperature*. Establishes
  distribution-shape-change-under-temperature as a named, measured
  quantity.

Read together: perturbing the prompt, perturbing/marginalising
temperature, and measuring distribution-shape change under temperature
are all **established** methodology. Any framework axis that perturbs
prompt or temperature is an adaptation of this family, not a new idea.

---

## 2. Critical caveat — low-temperature collapse (SRE)

The Sampling-Reproducibility-Equivalence (SRE) caveat: **low-temperature
samples collapse to consistent results.** A perturbation method that
samples at very low temperature does not carry signal, because the
output barely varies. The signal-bearing variants of temperature
perturbation are the **higher-temperature** ones. Any
temperature-perturbation extension must operate in a temperature regime
where output actually varies, or it measures nothing.

**This caveat is axis-specific, and that matters here (see §3).** It
applies to *temperature*-sourced perturbation (E_perturb_1). It does
**not** apply to *quantization*-sourced perturbation (E_quant_3): two
quantization codecs disagree on argmax at temperature=0 precisely
because the perturbation source is the weight codec, not sampling noise.
The empirical anchor (2026-05-07 cross-adapter test, commit `4dac3d4`)
observed ~25% argmax disagreement between GGUF-Q4_K_M and MLX-4bit at
the terminal argmax position — i.e. in exactly the low-temperature
regime where temperature perturbation would have collapsed. The two
axes therefore probe complementary structure.

---

## 3. The framework's position

The framework's contribution within the perturbation-UQ family is:

1. **Adaptation to the clinical-reasoning domain.** The family has been
   validated mostly on general QA / calibration benchmarks. Stage-6
   chest-pain is a clinical-reasoning validation domain with
   weak-supervision ground truth (`need_for_consultation`), not a
   calibration benchmark.

2. **Combination of perturbation axes.** Quantization (E_quant_3) +
   temperature (E_perturb_1) + prompt (SPUQ-style, available as a
   configurable axis) probe different structure. The quantization axis
   is the least-explored in the cited literature and survives the SRE
   low-temperature-collapse caveat that constrains temperature
   perturbation — making the *combination*, not any single axis, the
   defensible contribution.

3. **The calibrated-claims discipline pattern**, applied throughout:
   pre-registration committed before data, Bonferroni-corrected
   confirmatory family with exploratory signals held outside it,
   bootstrap CIs at all sample sizes, honest narrowing of rejected
   mechanisms (mass-capture). See `feedback_calibrated_claims.md`.

4. **The schema-v3 architecture** (raw top-K logprobs cached at full
   fidelity; computations are post-hoc derivations) enabling new scorers
   — including perturbation-based ones — to be developed *without*
   re-running inference. See `project_measurement_vs_computation.md`.

The empirical findings remain first-class contributions independent of
the family positioning: cross-domain `mean_entropy`, B-vs-C
complementarity, the mass-capture honest narrowing, the Phase-B
re-derivation, and stage-6 results when they land.

---

## 4. Implications for Phase C semantic entropy

Phase C (multi-sample semantic entropy) is itself a temperature-driven
sampling method and therefore squarely inside the perturbation family.
Concretely:

- **`intuit-ai-research/SPUQ` is the reference code** for prompt +
  temperature perturbation. When implementing the perturbation axis,
  follow its aggregation approach rather than inventing one.

- **Temperature perturbation becomes a configurable option** on the
  Phase C generation path, not a fixed behaviour. The sampling
  temperature for semantic-entropy completions is already a parameter;
  expose a perturbation schedule (e.g. a small set of temperatures
  marginalised à la MCT) as configuration, defaulting off until
  E_perturb_1 earns it.

- **Respect the SRE caveat in defaults.** Any temperature-perturbation
  schedule must live in a regime where outputs vary; do not default to
  near-zero temperatures, which would collapse the signal (§2).

- **The Phase C engine path is unblocked** as of 2026-05-23: the
  `mx.fast.rope` batched-decode bug (PR #3498) is validated fixed (see
  `docs/exploration/2026-05-23-phase-c-rope-fix-validation.md`), pending
  only a released mlx wheel. Perturbation-axis work and the batched
  decode re-enable are independent.

---

## 5. What NOT to claim

- **Do NOT claim novelty on perturbation-based UQ.** SPUQ (2024) is the
  foundational reference; the intuition is established.
- **Do NOT claim temperature variation as an uncertainty signal is
  novel.** MCT (2025) and Inv-Entropy/TSU (2025) establish it.
- **Do NOT present E_perturb_1 as a new method.** It is an adaptation
  of SPUQ-style temperature perturbation to the clinical domain,
  reported as exploratory additive evidence.

**What IS claimable:** the specific empirical findings (cross-domain
`mean_entropy`, B-vs-C complementarity, mass-capture honest narrowing,
Phase-B re-derivation, stage-6 results when they land); the *combination*
of perturbation axes (quantization + temperature + prompt); the clinical
validation context; the calibrated-claims discipline pattern; and the
schema-v3 architecture enabling post-hoc scorer development.

---

## 6. Per-token uncertainty literature

A third literature family — per-token uncertainty from the output
probability distribution — surfaced after cross-model disagreement
(E_quant_3) and perturbation (E_perturb_1). It bears directly on the
framework because the schema-v3 infrastructure already caches per-step
top-K logprobs at full fidelity, so per-token scorers are post-hoc
derivations requiring no new inference (see
`project_measurement_vs_computation.md`). See also
`convergence_pattern_observation.md` for the meta-pattern across all
three families.

### Established literature

- **LogitScope** — IBM, 2026 (`github.com/IBM/logitscope`). Open-source
  framework quantifying token-level uncertainty via **entropy and
  varentropy** (variance of entropy across positions) from probability
  distributions; single forward pass, no labels, no extra models.
  Introduces the low/high-entropy × low/high-varentropy **quadrant
  framing** for token classification. Reference implementation for
  per-token measurement.

- **EPR (Entropy Production Rate)** — arXiv:2509.04492, Springer 2026,
  doi:10.1007/978-3-032-21289-4_8. Designed for **black-box API access
  where only top-K logprobs per token are available** — the same setting
  as the framework's adapter abstraction. Single-sequence, non-greedy
  decoding; improves token-level hallucination detection over prior
  methods. EPR is the **slope/rate** signal across tokens.

- **HaluNet** — arXiv:2512.24562. Multi-granular uncertainty modeling
  combining token-level probability uncertainty with semantic
  embeddings; multi-branch, lightweight one-pass. Relevant to the
  framework's combination-of-signals approach.

- **Entropy-Based Inference Scaling** — Meskarian, 2025 (article).
  Finding: **elevated entropy across consecutive tokens when the model
  fabricates**, with spikes at the transition where factual knowledge
  ends and confabulation begins. This is the
  boundary-detection-at-fabrication-onset signal that maps to stage-6
  **P8**.

- **Semantic Energy** — arXiv:2508.14496. **Counterpoint:** argues
  entropy is insufficient and operates on **penultimate-layer logits**
  instead. Cited as an alternative perspective; not adopted (the
  framework's black-box / top-K-logprob setting precludes
  penultimate-layer access, consistent with the EPR setting).

### Framework position

The framework **adapts established per-token methods**; it does not
introduce them. Per-token entropy, varentropy, and the entropy-rate /
fabrication-onset signal are established. The framework's contribution
is the specific **aggregation choices** (which per-token signals are
combined, and how they roll up to a trajectory-level deferral signal),
**validated empirically in the clinical-reasoning domain**, under the
**calibrated-claims discipline**, on the **schema-v3 architecture** that
makes per-token scorers derivable post-hoc.

### Implications for Phase B / Phase C

- **Varentropy as a new scorer** (LogitScope). Add a trajectory-level
  varentropy aggregate alongside `mean_entropy`; tests dynamics
  (variance of per-position entropy) as complementary to magnitude. Maps
  to E_token_2.
- **EPR slope/rate as a trajectory-level aggregate.** The entropy-rate
  framing across tokens is the natural slope-form signal; it relates to
  the (deprecation-pending) `entropy_plateau` slope operationalisation,
  which may regain meaning on per-token trajectories. Maps to the
  dynamics half of E_token_1/E_token_2.
- **Fabrication-onset boundary signal** (Meskarian) maps to stage-6 P8
  (entropy spike at the transition into confabulation). Per-token
  measurement across the full generation, not just hypothesis positions,
  is what exposes it.
- **LogitScope is the reference implementation** for any future
  per-token engineering work.

### What NOT to claim

- **Do NOT claim novelty on per-token entropy, varentropy, EPR, or
  token-level hallucination detection.** All established (LogitScope, EPR,
  HaluNet, Meskarian 2025).
- **Do NOT present E_token_1 / E_token_2 as new methods.** They are
  adaptations reported as exploratory additive evidence.

**What IS claimable:** the specific aggregation choices (which signals
combine how), the clinical-reasoning validation domain, the
calibrated-claims discipline, and the schema-v3 architecture enabling
post-hoc per-token scorer derivation.

---

## 7. Toolkit collections (method libraries, not diagnostic frameworks)

A class of prior art is the **method-collection library**: a unified
interface over a battery of UE/UQ methods. This is architecturally
distinct from a *diagnostic framework* (which interprets patterns across
signals to produce a decision). A toolkit gives you the signals; it does
not tell you what their joint pattern means.

- **LM-Polygraph** — Fadeeva et al. 2023 (arXiv:2311.07383); benchmarking
  paper arXiv:2406.15627, TACL 2025; `github.com/IINemo/lm-polygraph`. A
  Python framework implementing a large battery of uncertainty-estimation
  methods under one interface, with a benchmark for consistent
  cross-language / cross-task evaluation. **Widely adopted** (hundreds of
  researchers and companies). Categorically a **toolkit**, not a
  diagnostic framework.
- **UncertaintyZoo** — arXiv:2512.06406, Dec 2025;
  `github.com/Paddingbuta/UncertaintyZoo`. 29 UQ methods across five
  categories, plugin-oriented. Same category as LM-Polygraph: toolkit,
  not diagnostic framework.

**Compute profile.** Both assume **substantial inference capacity** for
the methods they implement — semantic entropy at N=10, multi-sample and
ensemble approaches. They are libraries of methods, agnostic to the
deployment budget; the heavy methods in the battery presuppose
infrastructure boundary-signature's regime does not have
(`compute_constraint_orientation.md`).

**Framework position.** boundary-signature is not a toolkit and should
not claim to be a "unified UQ measurement collection" — LM-Polygraph
occupies that space and is widely adopted. The architectural distinction
is real: boundary-signature is a *measurement protocol with a diagnostic
decomposition*, not a library of interchangeable methods. The packaging
decision is **standalone, not a plugin** to such a toolkit
(`standalone_framework_decision.md`): implementing inside an
infrastructure-assuming toolkit interface would obscure the
single-run/constrained-deployment differentiator.

**What NOT to claim:** novelty on collecting UQ methods under a unified
interface (LM-Polygraph, UncertaintyZoo).

## 8. Diagnostic decomposition frameworks

The closest cousins: frameworks that **decompose** uncertainty and map
the decomposition to interpretation or intervention.

- **Anatomy of Uncertainty in LLMs** — Taparia et al., arXiv:2603.24967,
  Mar 2026; `github.com/adityataparia/LLM-Uncertainty`. Decomposes
  uncertainty into three **causal sources** — input ambiguity, knowledge
  gaps, decoding randomness — each measured by separate sampling (K=5
  paraphrases, M=5 LoRA adapters, N=5 stochastic samples) and mapped to
  interventions (clarification / RAG / sampling adjustment). Validated on
  TriviaQA and GSM8K. **Empirical results partial:** the knowledge
  component (U_knowledge) is near chance (AUROC ~0.5) on TriviaQA; all
  three components are weak on GSM8K reasoning (AUROC 0.33–0.60); the
  interaction analysis surfaces unresolved underconfidence in the
  high-confidence regime (ECE 0.635 when both signals low). The authors
  explicitly note operational-proxy limitations and non-orthogonality of
  the components. **Compute profile:** ~15× single-inference cost per
  measurement (K=5 + M=5 + N=5) **plus** ensemble training of the M LoRA
  adapters; reported on NVIDIA H100 80GB. AUROC 0.5–0.76 across sources.
- **Grammars of Formal Uncertainty** — arXiv:2505.20047. A 25-metric
  taxonomy for neurosymbolic systems with signal fusion for selective
  verification. **Compute profile:** multiple metric extractions per
  query.
- **Cognometry v0** — (Awesome-LLM-Uncertainty list). A 9-signal pooled
  logistic-regression hallucination detector, cross-validated on 8
  benchmarks, with pre-declared failure modes. **Compute profile:**
  multi-signal collection including an NLI-contradiction step.
- **Multi-Layered Mitigation** — MDPI 2025. A three-layer architecture
  (input governance / evidence-grounded generation / post-response
  verification) with a supervisory agent for escalation. **Compute
  profile:** assumes RAG infrastructure plus fine-tuning capacity.

**Comparator design (from the full paper, fetched 2026-05-26).** Anatomy
evaluates against **answer correctness** (Rouge-L ≥ 0.3 / bidirectional
NLI) on TriviaQA + GSM8K — the aligned output-reliability target. It uses
**semantic entropy (NLI-clustered) as its core metric**, NOT a baseline,
and **does not benchmark against other UQ methods** (no
semantic-entropy-vs-P(True)/max-prob/etc. table) — so the
value-add-over-baseline comparator gap is field-wide, not specific to this
project. AUROCs: TriviaQA 0.71–0.76 (input/decoding), knowledge ~0.50
(null); **GSM8K reasoning 0.33–0.60, with Gemma 3 inverted at 0.334** —
i.e. their reasoning-domain UQ is weak/inverted too, mirroring stage-6's
entropy inversion. Useful implications: (1) the proven-method comparator
in this space *is* semantic entropy; (2) this project's MCQ `mean_entropy`
(0.686 MedQA / 0.664 law vs correctness) is in-range with their
TriviaQA semantic-entropy and beats their GSM8K reasoning — at single-run
vs N=5-sampling cost (suggestive, pending a head-to-head on shared data).

**Framework position — the load-bearing distinction.** Anatomy of
Uncertainty is the closest prior work, and the distinction is the
**decomposition axis**:

- Anatomy decomposes by **causal source** (where uncertainty *originates*:
  input / knowledge / decoding).
- boundary-signature decomposes by **measurement signal** (what is
  *measurable from the model's behaviour*: entropy / gap / mass-capture /
  cross-quantization disagreement).

These are **different decomposition levels, not competing frameworks**,
and they live in **different compute regimes**. Anatomy requires K+M+N
sampling plus an ensemble of LoRA adapters (~15× single-inference cost
plus training, on H100 80GB); boundary-signature operates from
**single-run inference** on 4-bit-quantized weights on Apple Silicon
(`compute_constraint_orientation.md`). And Anatomy's reasoning-domain
results came back weak (GSM8K AUROC 0.33–0.60), which is precisely where
boundary-signature's clinical-reasoning validation is genuinely additive
rather than redundant. The authors' acknowledged operational-proxy
limitations support positioning boundary-signature as **complementary**.
(On performance: boundary-signature's AUROC is *in the same range* as
Anatomy's TriviaQA sources at a fraction of the per-measurement cost —
but this is a **cross-dataset, indicative** comparison, not a head-to-head;
see the hedge in `contribution_shape_post_literature.md` item 2.)

**What NOT to claim:** novelty on "a diagnostic framework for LLM UQ" —
Anatomy of Uncertainty occupies that space (at the causal-source level).
Claim the *measurement-signal decomposition level* and the single-run
operationalization, not the idea of decomposition itself.

## 9. Model-layer uncertainty awareness

- **ConfiDx** — Dec 2025. A **fine-tuned**, uncertainty-aware clinical
  LLM that verbalizes uncertainty in its outputs; requires fine-tuning on
  annotated clinical data. Categorically a **model-layer** solution.
  **Compute profile:** training-time substantial (fine-tuning +
  annotated clinical datasets at scale); deployment-time comparable to
  the base model. The cost — and the data-annotation requirement — is
  front-loaded into training, which is exactly the capacity
  boundary-signature's regime lacks (no fine-tuning, single-person team).

**Framework position.** ConfiDx operates at a **different layer of the
deployment stack** than boundary-signature: it bakes uncertainty
awareness into the model weights via fine-tuning, whereas
boundary-signature is a **measurement-layer** post-hoc extraction from a
deployed (unmodified) model. These are **complementary, not competing** —
a ConfiDx-style model could itself be measured by a boundary-signature
protocol. "Clinical uncertainty awareness" as a standalone contribution
is not claimable (ConfiDx achieves it through fine-tuning); the
differentiated claim is the *measurement-layer architecture* that needs
no fine-tuning and no annotated uncertainty labels.

**What NOT to claim:** novelty on clinical uncertainty awareness as such
(ConfiDx exists). The complementary architectural pattern
(measurement-layer vs model-layer) remains differentiated.

---

## §2 methods-paper content — STATUS

The methods paper draft exists (`docs/paper/draft.md` §2 Related work).
A "Perturbation-based uncertainty quantification" paragraph was added to
§2 citing Gao et al. 2024, Cecere et al. 2025, and Inv-Entropy 2025, and
a "Per-token uncertainty" paragraph citing LogitScope (2026), EPR
(2026), HaluNet, Meskarian 2025, and Semantic Energy (2026), closing
with the synthesis sentence that the per-position/per-token and
perturbation choices *adapt* established methods to the clinical context
under calibrated-claims discipline — the contribution being the
aggregation choices and their validation, not the signals themselves.
If §2 is later restructured, preserve that framing. No formal
bibliography section exists in the draft yet; citations are inline
author-year / arXiv-id and should be reconciled into a bibliography when
one is created.

**Architectural-distinction paragraphs (added 2026-05-23).** §2 now also
carries three positioning paragraphs distinguishing boundary-signature
from (a) toolkit collections (LM-Polygraph, UncertaintyZoo —
architecturally different: protocol+decomposition, not a method library),
(b) diagnostic-decomposition frameworks (Anatomy of Uncertainty as
closest cousin — measurement-signal vs causal-source decomposition;
single-run vs K+M+N sampling; complementary clinical-reasoning validation
where their GSM8K results were weak), and (c) model-layer awareness
(ConfiDx — measurement-layer vs model-layer fine-tuning, complementary).
The revised contribution claim is consolidated in
`contribution_shape_post_literature.md`. §7.4 (threats to validity) notes
that literature engagement narrowed the contribution and that the
calibrated-claims discipline itself required iteration
(`convergence_pattern_observation.md`, now three calibration-failure
modes).

**Compute-constraint orientation (added 2026-05-23, load-bearing).** The
distinctive differentiator missed across the earlier narrowing is that
every cited diagnostic / model-layer / toolkit solution assumes
infrastructure (multi-sample, ensembles, fine-tuning, RAG, H100) absent
from boundary-signature's deployment regime (Apple Silicon, 4-bit,
single-run, on-prem, single-person team). Each literature section above
now carries a **COMPUTE PROFILE** note, and §1 of the draft leads the
contribution with the constraint orientation (framed as a *demonstrated*
capability, with the cross-dataset performance comparison explicitly
hedged). Canonical reference: `compute_constraint_orientation.md`. The
packaging question is **resolved → standalone**
(`standalone_framework_decision.md`), superseding the open-question file.
**Venue implication flagged for PI:** a constraint-oriented
clinical-deployment framing may point at clinical-informatics venues
(JAMIA, JAMA Network Open) rather than pure-methodology UQ venues
(NeurIPS/ICLR/ACL) — a PI decision, not resolved here.
