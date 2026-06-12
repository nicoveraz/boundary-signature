# medqa_generalization

The framework's generality experiment. Tests whether boundary-aware reasoning
(Condition C) outperforms confidence-based deferral (Condition B) on the
MedQA-USMLE benchmark, with cross-LLM and cross-domain (MMLU) replication.

See CLAUDE.md §9 and the project's general-framework document Part III for
full specification.

## Status

Pre-implementation. Pipeline scripts will live in `scripts/` and consume
`bsig.medqa` adapter implementations.

## Compute budget

- M1 Pro plumbing: 2–3 days
- Single H100 session: 6–8 hours, < $30
- M1 Pro analysis: 2–3 weeks
