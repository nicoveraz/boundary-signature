# ADR-0002: MedQA embedding-bin precision deferred to stage 4

**Status:** Deferred
**Date:** 2026-05-03
**Stage of origin:** 3.1

`MCQCanonicalizationConfig.embedding_bin_precision` defaults to 8
decimal places; this is a placeholder pending real-data sensitivity
analysis, not a tuned value. Synthetic embeddings
(`DeterministicMockEmbedder` from `tests/medqa/conftest.py`) cannot
tell us how aggressively bin precision should collapse
semantically-similar reasoning steps — that's a property of the real
`intfloat/multilingual-e5-large` embedder applied to actual MedQA
reasoning content, neither of which is exercised at stage 3.1.

**Resolution:** stage 4 (H100 run for MedQA) sweeps precision in
{4, 6, 8, 10, 12}, measures (a) recovered-graph node count, (b) edge
count, (c) density (mean out-degree), (d) per-bucket entropy of node
visits, per setting. Pick the elbow on the node-count curve where
collapse stops being aggressive (more precision adds nodes only
marginally). Update the default in `MCQCanonicalizationConfig` and
record the sweep in stage 4's analysis notes.

**Why this is recorded as an ADR:** the placeholder default works for
all stage 3 work and would be easy to forget. Tracking it explicitly
prevents the "we shipped the placeholder and forgot" failure mode that
would surface as anomalous results during stage 4.
