# boundary-signature

A token-probability **measurement protocol** for per-step belief monitoring in
chain-of-thought (CoT) reasoning, with a deferral/selective-prediction
evaluation harness. This is the research code and data behind the paper
*"A Measurement Protocol for Per-Step Belief Monitoring in Chain-of-Thought
Reasoning"* ([`docs/paper/draft.md`](docs/paper/draft.md)).

The protocol reads the model's next-token distribution at each reasoning-step
boundary, renormalises over the answer space (e.g. `{A,B,C,D}` for MCQ),
records the renormalised conditional plus the *mass-capture* fraction and the
full top-K logprobs, and derives per-trajectory scorers — chiefly
`mean_entropy` — evaluated against answer-correctness ground truth.

## What this repository is (and is not)

- **It is** the reproducibility artifact for the methods paper: the framework
  (`bsig`), the MedQA/MMLU experiment pipeline, the pre-registration trail, and
  the cached measurements needed to recompute every published number.
- **It is not** a production uncertainty service. Eunosia's deployed
  uncertainty/calibration stack (`eunosia-uncertainty`) and clinical domain
  pack are separate and closed. Nothing here depends on them; nothing here
  exposes them.

## Headline result

On MedQA-USMLE (N=1273) `mean_entropy` reaches sign-aware AUC **0.686
[0.657, 0.716]** against the wrong-answer indicator, replicating cross-domain
on MMLU professional_law (N=1534) at **0.664 [0.636, 0.690]**. A systematic
sweep (paper §5.7–5.8) shows that — under single-model, single-run, 4-bit,
black-box constraints — no cheap trajectory-dynamics, perturbation, verbalised-
confidence, or richer-distribution feature complements `mean_entropy`; the only
additive gain comes from cross-quantization disagreement, which is itself
label-sensitive. The negative-results map is part of the contribution.

## Reproduce without an LLM (measurement vs computation)

The `measurements/` directory ships the cached per-step measurements (embeddings
stripped; ~11 MB). Because scorers are cheap derivations of cached measurements,
you can recompute the paper's numbers without re-running inference:

```bash
uv pip install -e '.[experiments]'
python experiments/medqa_generalization/scripts/16_belief_dynamics_probe.py
python experiments/medqa_generalization/scripts/14_2d_interaction_probe.py
```

To re-run the **measurement** itself (the LLM forward passes), install a backend
extra (`mlx` on Apple Silicon, or `llama_cpp`) and run the capture scripts
(`03c`/`04`/`19`); see each script's header.

## Architecture

Four layers, separation enforced mechanically by `import-linter`
(`uv run lint-imports`) and an AST test (`tests/test_architecture.py`):

```
bsig.core        pure algorithms (trajectory model, signature scorers, evaluation)
bsig.adapters    Protocol contracts (LLM, embedding, canonicalizer, ...)
bsig.medqa       MedQA/MMLU domain pack (loaders, conditions A/B/C, decomposer)
bsig.reference   reference adapters (llama.cpp, MLX, sentence-transformers)
```

## Install

```bash
uv pip install -e '.[all,dev]'      # everything, for development
uv pip install -e '.[experiments]'  # minimal: recompute from cached measurements
```

## Pre-registration / discipline

Every empirical claim is pre-registered with quantitative thresholds before the
data that tests it (`docs/decisions/prereg_*.md`, `docs/decisions/stage_4*`).
The git history is the audit trail.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citation

Citation metadata will be added on publication; for now cite the repository.
