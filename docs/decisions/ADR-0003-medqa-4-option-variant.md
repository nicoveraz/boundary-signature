# ADR-0003: MedQA-USMLE 4-option variant via GBaker, 5-option deferred

**Status:** Accepted
**Date:** 2026-05-03
**Stage of origin:** 3.1 / pre-3.2 grounded exploration

The MedQA experiment loads `GBaker/MedQA-USMLE-4-options` (1273 test,
10178 train, options always `{A, B, C, D}`). The originally-targeted
`bigbio/med_qa` mirror is broken in modern `datasets` versions
(deprecated loading-script format); GBaker is the working Parquet-format
mirror with equivalent content.

The framework's MedQA experiment is therefore "evaluation against the
public 4-option benchmark" — the standard format used by Hager et al.,
Singhal et al., and the medical-LLM benchmarking literature generally.
The original 5-option USMLE-USMLE format is not publicly accessible and
would require a separate data-acquisition project; deferred to a
hypothetical 0.2 or external-collaboration phase.

**Operational implications:** the LLM hypothesis space passed to
`LLMAdapter.get_hypothesis_distribution` is always 4 letters for this
source. `MCQRawState.record.choices.keys()` is the canonical
hypothesis space — already the convention from stage 3.1, no code
change. The 4-option constraint is a property of the data source, not
the framework; switching to 5-option later is a loader change, not a
framework change.

**Methods-paper precision:** describe results as "evaluation against
the public 4-option MedQA-USMLE benchmark," not "evaluation against
USMLE-format clinical reasoning." The 4-option variant is a benchmark
adaptation, not the underlying clinical task. This distinction matters
when comparing to clinician-baseline numbers, which are typically
reported against the 5-option original format.
