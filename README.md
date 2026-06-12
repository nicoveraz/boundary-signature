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
- **It is not** a production uncertainty service.

## Headline result

On MedQA-USMLE (N=1273) `mean_entropy` reaches sign-aware AUC **0.686
[0.657, 0.716]** against the wrong-answer indicator, replicating cross-domain
on MMLU professional_law (N=1534) at **0.664 [0.636, 0.690]**.

The signal is present before the model writes a single reasoning token:
eventually-wrong answers start with much higher entropy over the answer
options, and the gap narrows as reasoning resolves.

![Median answer entropy per reasoning step, correct vs wrong (MedQA N=1273)](docs/figures/entropy_trajectory.png)

*(regenerate with `python experiments/medqa_generalization/scripts/make_figure_entropy_trajectory.py`)*

### What we tested, and what actually helps

A systematic sweep — every signal is cheap (single model, single forward pass,
black-box) and tested for incremental AUC over `mean_entropy`:

| Signal | Standalone AUC | Adds over `mean_entropy`? |
|---|---|---|
| **`mean_entropy`** (mean per-step answer entropy) | **0.686** | — *this is the signal* |
| Cross-quantization disagreement | additive | **+0.031** — but needs a 2nd model, and label-sensitive |
| Trajectory dynamics (volatility · argmax flips · monotonicity · margin) | ≤ 0.68 | no (all CIs include 0) |
| Verbalised confidence (ask the model) | 0.541 | no |
| Logit-noise perturbation (one model) | — | no (it re-encodes entropy) |
| Varentropy · full-vocab entropy · entropy-production rate | ≤ 0.67 | no |

Under these constraints the cheap single-pass ceiling is ≈0.69 AUC, reached by
`mean_entropy` alone. The only additive gain (cross-quantization disagreement)
requires a second quantized model and its gain attenuates on the deployed
prediction. **The negative-results map is part of the contribution** — every
row is pre-registered or run as an explicit exploratory probe (paper §5.7–5.8).

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
