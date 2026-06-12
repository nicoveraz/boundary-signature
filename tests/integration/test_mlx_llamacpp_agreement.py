"""Cross-adapter agreement test: LlamaCppLLMAdapter vs MLXLLMAdapter.

This is the methodological gate (§7.1 of stage_6_mlx_adapter pre-design)
that certifies both adapters as interchangeable for the framework's
measurement protocol. Until it passes, results computed under
``MLXLLMAdapter`` cannot be reported as framework findings (only as
MLX-specific exploratory results).

**Hardware-gated**: skipped unless both adapters are available
- ``LlamaCppLLMAdapter``: requires a running llama.cpp server at
  ``LLAMACPP_HOST`` (default ``http://localhost:8080``) loaded with
  the same model weights as the MLX path
- ``MLXLLMAdapter``: requires Apple Silicon + ``mlx-lm`` installed +
  the same model weights at ``MLX_MODEL_PATH``

**Test scope**: 50 questions from MedQA-USMLE test split (deterministic
slice). For each question, both adapters run Condition C's per-step
measurement protocol. We compare:

1. **Predicted answer agreement** (token-probability argmax of the
   terminal measurement): ≥ 98 % across ~250 measurement positions
   (50 questions × ~5 positions).
2. **Mass-capture absolute difference**: ≤ 0.01 in 95 % of comparisons.
3. **Per-position entropy absolute difference** (in nats): ≤ 0.05 in
   95 % of comparisons.

Disagreement above the documented Metal-vs-CUDA fp noise floor
(~0.3 % per-letter probability per spec §11) indicates a real
adapter-protocol divergence and triggers investigation, NOT silent
default to one adapter.

**Run**:

    LLAMACPP_HOST=http://localhost:8080 \\
    MLX_MODEL_PATH=mlx-community/Qwen2.5-7B-Instruct-4bit \\
    uv run pytest tests/integration/test_mlx_llamacpp_agreement.py -v

Without env vars set, the test is skipped (returns CI green via
pytest.skip rather than failing on missing hardware).

**On a substantive disagreement**: investigate which adapter is
correct (compare against a reference small model where both adapters
should produce bit-identical outputs at deterministic seed); document
divergence; update the framework's adapter-selection guidance. Both
adapters remain available; the divergence is reported as a known
limitation rather than silently resolved.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


# ============================================================
# Skip conditions
# ============================================================

LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST")
MLX_MODEL_PATH = os.environ.get("MLX_MODEL_PATH")

# Acceptance criteria from stage_6_mlx_adapter_pre_design_notes §7.1
N_QUESTIONS = 50
ANSWER_AGREEMENT_THRESHOLD = 0.98
MASS_CAPTURE_TOLERANCE = 0.01
MASS_CAPTURE_QUANTILE = 0.95
ENTROPY_TOLERANCE_NATS = 0.05
ENTROPY_QUANTILE = 0.95


def _both_available() -> bool:
    if not LLAMACPP_HOST or not MLX_MODEL_PATH:
        return False
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    try:
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


# ============================================================
# Test data
# ============================================================


def _load_medqa_50() -> list[dict[str, Any]]:
    """Load the deterministic 50-question MedQA-USMLE test slice.

    Uses GBaker/MedQA-USMLE-4-options via HuggingFace datasets
    (the loader the framework standardised on per ADR-0003). The
    50-question slice is the first 50 of the test split, preserving
    order across runs.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        pytest.skip(f"datasets package not available: {exc}")
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options", split=f"test[:{N_QUESTIONS}]"
    )
    return [
        {
            "question_id": f"medqa-test-{i}",
            "question": row["question"],
            "options": dict(row["options"]),
            "correct_letter": row["answer_idx"],
        }
        for i, row in enumerate(ds)
    ]


# ============================================================
# Per-adapter measurement (the framework's Condition C protocol)
# ============================================================


@dataclass
class MeasurementSnapshot:
    """Per-question measurement output: snapshot of what each
    adapter measured, used for cross-adapter comparison."""

    question_id: str
    predicted_letter: str  # argmax of terminal token-probability measurement
    per_position_entropy: list[float]  # bits, one per measurement position
    per_position_mass_capture: list[float]  # ∈ [0, 1], one per position
    n_positions: int


def _measure_question(
    adapter: Any,
    question: dict[str, Any],
    measurement_positions: Sequence[str],
) -> MeasurementSnapshot:
    """Run the framework's measurement protocol on a single question
    against pre-built measurement positions.

    Cross-adapter agreement requires that BOTH adapters measure at
    the **same** prompt-and-CoT positions; otherwise the comparison
    measures CoT-decoding divergence rather than adapter-protocol
    divergence. The orchestration generates CoT once (on llama.cpp,
    by convention) and passes the resulting positions to both
    adapters' ``_measure_question`` calls.
    """
    per_position_entropy = []
    per_position_mc = []
    final_distribution = None
    token_set = list(question["options"].keys())  # ['A', 'B', 'C', 'D']
    for pos_text in measurement_positions:
        result = adapter.get_token_probabilities(pos_text, token_set)
        # Entropy of renormalised distribution (in bits)
        h = sum(
            -p * math.log2(p)
            for p in result.distribution.values()
            if p > 0
        )
        per_position_entropy.append(h)
        per_position_mc.append(result.mass_capture)
        final_distribution = result.distribution

    # Predicted answer = argmax of terminal measurement
    if final_distribution is None:
        raise RuntimeError(
            f"No measurement positions for {question['question_id']}"
        )
    predicted = max(final_distribution.items(), key=lambda kv: kv[1])[0]

    return MeasurementSnapshot(
        question_id=question["question_id"],
        predicted_letter=predicted,
        per_position_entropy=per_position_entropy,
        per_position_mass_capture=per_position_mc,
        n_positions=len(measurement_positions),
    )


def _build_cot_prompt(question: dict[str, Any]) -> str:
    """Minimal CoT prompt — same template both adapters use."""
    options_text = "\n".join(
        f"{letter}: {text}"
        for letter, text in question["options"].items()
    )
    return (
        f"{question['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Reason step by step.\n"
    )


_MEASUREMENT_SUFFIX = "\nThe best answer is"


def _split_into_measurement_positions(prompt: str, cot: str) -> list[str]:
    """Split the prompt+CoT into measurement positions.

    Each position ends with a letter-biasing suffix ("The best answer
    is") so the model's next-token distribution concentrates on
    answer-letter tokens. This mirrors the framework's actual
    ``condition_c_measurement.txt`` template; without it, mass
    capture is zero at positions where the model would otherwise
    emit reasoning tokens rather than letters.

    Three positions:
    - **prior**: question + measurement suffix, no CoT
    - **mid**: question + first half of CoT + measurement suffix
    - **terminal**: question + full CoT + measurement suffix

    Both adapters measure at the same three positions; the comparison
    isolates adapter-protocol divergence at the per-position level.
    """
    half = len(cot) // 2 if cot else 0
    return [
        prompt + _MEASUREMENT_SUFFIX,
        prompt + cot[:half] + _MEASUREMENT_SUFFIX,
        prompt + cot + _MEASUREMENT_SUFFIX,
    ]


# ============================================================
# Comparison + agreement metrics
# ============================================================


@dataclass
class AgreementReport:
    n_questions: int
    n_position_comparisons: int
    answer_agreement_rate: float
    mc_diff_quantile: float  # 95th percentile of |Δ|
    mc_within_tolerance_rate: float
    h_diff_quantile: float  # 95th percentile of |Δ| in nats
    h_within_tolerance_rate: float
    passes: bool


def _compute_agreement(
    llamacpp_snapshots: list[MeasurementSnapshot],
    mlx_snapshots: list[MeasurementSnapshot],
) -> AgreementReport:
    assert len(llamacpp_snapshots) == len(mlx_snapshots)
    n_questions = len(llamacpp_snapshots)

    # Predicted-answer agreement
    n_answer_matches = sum(
        1
        for a, b in zip(llamacpp_snapshots, mlx_snapshots, strict=True)
        if a.predicted_letter == b.predicted_letter
    )
    answer_agreement = n_answer_matches / n_questions

    # Per-position comparisons (mass capture, entropy)
    mc_diffs: list[float] = []
    h_diffs: list[float] = []
    for a, b in zip(llamacpp_snapshots, mlx_snapshots, strict=True):
        n = min(len(a.per_position_mass_capture), len(b.per_position_mass_capture))
        for i in range(n):
            mc_diffs.append(
                abs(a.per_position_mass_capture[i] - b.per_position_mass_capture[i])
            )
            # Entropy from bits → nats: multiply by ln(2)
            h_a = a.per_position_entropy[i] * math.log(2)
            h_b = b.per_position_entropy[i] * math.log(2)
            h_diffs.append(abs(h_a - h_b))

    n_position_comparisons = len(mc_diffs)
    mc_diffs_sorted = sorted(mc_diffs)
    h_diffs_sorted = sorted(h_diffs)

    def _quantile(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = int(round(q * (len(sorted_vals) - 1)))
        return sorted_vals[idx]

    mc_q = _quantile(mc_diffs_sorted, MASS_CAPTURE_QUANTILE)
    h_q = _quantile(h_diffs_sorted, ENTROPY_QUANTILE)

    mc_within = sum(1 for d in mc_diffs if d <= MASS_CAPTURE_TOLERANCE) / max(
        1, len(mc_diffs)
    )
    h_within = sum(1 for d in h_diffs if d <= ENTROPY_TOLERANCE_NATS) / max(
        1, len(h_diffs)
    )

    passes = (
        answer_agreement >= ANSWER_AGREEMENT_THRESHOLD
        and mc_q <= MASS_CAPTURE_TOLERANCE
        and h_q <= ENTROPY_TOLERANCE_NATS
    )

    return AgreementReport(
        n_questions=n_questions,
        n_position_comparisons=n_position_comparisons,
        answer_agreement_rate=answer_agreement,
        mc_diff_quantile=mc_q,
        mc_within_tolerance_rate=mc_within,
        h_diff_quantile=h_q,
        h_within_tolerance_rate=h_within,
        passes=passes,
    )


# ============================================================
# Test
# ============================================================


@pytest.mark.skipif(
    not _both_available(),
    reason="Both adapters required: set LLAMACPP_HOST + MLX_MODEL_PATH "
    "and install mlx-lm + httpx",
)
def test_mlx_llamacpp_cross_adapter_agreement(tmp_path: Path) -> None:
    """Run MedQA-50 through both adapters; compare outputs against
    pre-registered acceptance criteria.

    Per stage_6_mlx_adapter pre-design §8.1:
    - predicted answer agreement ≥ 98%
    - 95th percentile mass capture |Δ| ≤ 0.01
    - 95th percentile per-position entropy |Δ| (nats) ≤ 0.05

    On failure: report diagnostics; do NOT auto-default to one adapter.
    Both adapters remain available; the divergence is documented and
    investigated.
    """
    from bsig.reference.llm_llama_cpp import LlamaCppLLMAdapter
    from bsig.reference.llm_mlx import MLXLLMAdapter

    questions = _load_medqa_50()

    llamacpp = LlamaCppLLMAdapter(host=LLAMACPP_HOST)
    mlx = MLXLLMAdapter(model=MLX_MODEL_PATH)

    import sys
    import time
    print(f"\nRunning MedQA-{N_QUESTIONS} through both adapters...", flush=True)
    llamacpp_snapshots = []
    mlx_snapshots = []
    overall_start = time.time()
    for i, q in enumerate(questions):
        q_start = time.time()
        # Generate CoT once on llama.cpp; both adapters measure at
        # the same positions so the comparison isolates adapter-
        # protocol divergence from CoT-decoding divergence.
        cot_prompt = _build_cot_prompt(q)
        # Reduce max_tokens to bound CoT cost (most MedQA CoTs are
        # well under 200 tokens).
        cot_text = llamacpp.generate(
            cot_prompt, max_tokens=256, temperature=0.0,
        )
        gen_t = time.time() - q_start
        positions = _split_into_measurement_positions(cot_prompt, cot_text)

        m_start = time.time()
        llamacpp_snapshots.append(_measure_question(llamacpp, q, positions))
        ll_t = time.time() - m_start

        m_start = time.time()
        mlx_snapshots.append(_measure_question(mlx, q, positions))
        mlx_t = time.time() - m_start

        elapsed = time.time() - overall_start
        eta_s = (elapsed / (i + 1)) * (N_QUESTIONS - i - 1)
        print(
            f"  q={i+1:>2d}/{N_QUESTIONS}  qid={q['question_id']}  "
            f"gen={gen_t:.1f}s  ll={ll_t:.1f}s  mlx={mlx_t:.1f}s  "
            f"ll_pred={llamacpp_snapshots[-1].predicted_letter}  "
            f"mlx_pred={mlx_snapshots[-1].predicted_letter}  "
            f"agree={'Y' if llamacpp_snapshots[-1].predicted_letter == mlx_snapshots[-1].predicted_letter else 'N'}  "
            f"ETA={eta_s/60:.1f}m",
            flush=True,
        )
        sys.stdout.flush()

    report = _compute_agreement(llamacpp_snapshots, mlx_snapshots)

    # Save full report for downstream analysis
    report_path = tmp_path / "agreement_report.json"
    report_path.write_text(json.dumps(
        {
            "n_questions": report.n_questions,
            "n_position_comparisons": report.n_position_comparisons,
            "answer_agreement_rate": report.answer_agreement_rate,
            "mc_diff_quantile_p95": report.mc_diff_quantile,
            "mc_within_tolerance_rate": report.mc_within_tolerance_rate,
            "h_diff_quantile_p95": report.h_diff_quantile,
            "h_within_tolerance_rate": report.h_within_tolerance_rate,
            "passes": report.passes,
            "thresholds": {
                "answer_agreement_threshold": ANSWER_AGREEMENT_THRESHOLD,
                "mc_tolerance": MASS_CAPTURE_TOLERANCE,
                "mc_quantile": MASS_CAPTURE_QUANTILE,
                "h_tolerance_nats": ENTROPY_TOLERANCE_NATS,
                "h_quantile": ENTROPY_QUANTILE,
            },
        },
        indent=2,
    ))
    print(f"\nReport saved to {report_path}")

    print(f"\n=== Cross-adapter agreement report ===")
    print(f"Questions: {report.n_questions}")
    print(f"Position comparisons: {report.n_position_comparisons}")
    print(
        f"Answer agreement: {report.answer_agreement_rate * 100:.1f}% "
        f"(threshold: {ANSWER_AGREEMENT_THRESHOLD * 100:.0f}%)"
    )
    print(
        f"Mass capture P95 |Δ|: {report.mc_diff_quantile:.4f} "
        f"(tolerance: {MASS_CAPTURE_TOLERANCE})"
    )
    print(
        f"Mass capture within tolerance: "
        f"{report.mc_within_tolerance_rate * 100:.1f}%"
    )
    print(
        f"Entropy P95 |Δ| (nats): {report.h_diff_quantile:.4f} "
        f"(tolerance: {ENTROPY_TOLERANCE_NATS})"
    )
    print(
        f"Entropy within tolerance: "
        f"{report.h_within_tolerance_rate * 100:.1f}%"
    )
    print(f"PASSES: {report.passes}")

    assert report.answer_agreement_rate >= ANSWER_AGREEMENT_THRESHOLD, (
        f"Predicted-answer agreement {report.answer_agreement_rate * 100:.1f}% "
        f"below threshold {ANSWER_AGREEMENT_THRESHOLD * 100}%; "
        f"investigate before treating adapters as interchangeable."
    )
    assert report.mc_diff_quantile <= MASS_CAPTURE_TOLERANCE, (
        f"Mass capture P95 |Δ| = {report.mc_diff_quantile:.4f} "
        f"exceeds tolerance {MASS_CAPTURE_TOLERANCE}; "
        f"adapters disagree on measurement-quality field."
    )
    assert report.h_diff_quantile <= ENTROPY_TOLERANCE_NATS, (
        f"Entropy P95 |Δ| (nats) = {report.h_diff_quantile:.4f} "
        f"exceeds tolerance {ENTROPY_TOLERANCE_NATS}; "
        f"adapters disagree on the framework's primary signal."
    )
