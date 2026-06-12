#!/usr/bin/env python
"""Pipeline-validation smoke test for the MedQA generalization experiment.

This is **NOT just a "does the pipeline run" check**. It's the first
end-to-end exercise of the framework's full stack against deterministic
mock data, and its primary value is **debugging surface for stage 4's
H100 run**.

When stage 4 produces unexpected results (AUC=0.51, weird trajectory
patterns, etc.), this script's clean prior runs prove the framework
end-to-end works correctly with controlled inputs. Deviation in stage
4 must be in the LLM behavior or the empirical signal, not in the
framework's plumbing.

Pipeline exercised:
  load 3 fixtured records
  -> run Conditions A, B, C against ScriptedMockLLM
  -> attach outcomes via AnswerKeyGroundTruthExtractor
  -> persist Condition C cached trajectories (round-trip via
     MedQAPrerecoveredTrajectorySource)
  -> run recovery on Condition C trajectories
  -> build FAISS indices from visits via build_faiss_indices_from_visits
  -> compute_signatures for all conditions
  -> condition_comparison + failure_mode_table
  -> persist evaluation outputs to artifacts/medqa-smoke/

Determinism: seeds locked; mock LLM returns scripted responses keyed
on question_id; mock embedder is hash-seeded; bootstrap CIs use
fixed random_seed. Two runs of this script produce identical
artifacts.

Usage:
    python 02_pipeline_validation_local.py
    python 02_pipeline_validation_local.py --output-dir /tmp/smoke
    python 02_pipeline_validation_local.py --no-strict   # skip assertions

For performance validation at larger scale, use stage 3.5's revised
smoke test against real Ollama. This script is structural validation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bsig.adapters.llm import TokenProbabilityResult
from bsig.core.persistence import (
    build_faiss_indices_from_visits,
    save_embedding_indices,
    save_graph,
    save_signature_scores,
)
from bsig.core.recovery import RecoveryConfig, recover_assembly_graph
from bsig.core.signature import SignatureWeights, compute_signatures
from bsig.medqa import (
    AnswerKeyGroundTruthExtractor,
    ConditionA,
    ConditionB,
    ConditionC,
    Decomposer,
    MCQActionCanonicalizer,
    MCQStateCanonicalizer,
    MedQAPrerecoveredTrajectorySource,
    MedQARawRecord,
    condition_comparison,
    failure_mode_table,
    load_all_versions,
    save_cached_trajectories,
)
from bsig.medqa.conditions._helpers import with_outcome


DEFAULT_OUTPUT_DIR = Path.home() / "work" / "eunosia" / "artifacts" / "medqa-smoke"


# ============================================================
# Inline fixtures (3 questions). The smoke is structural, so the
# fixtures are self-contained — not loaded from tests/medqa/fixtures
# to avoid src/test boundary crossings.
# ============================================================

_FIXTURES: list[dict[str, Any]] = [
    {
        "question_id": "smoke-1",
        "question": "Patient A: clinical vignette one. What's the diagnosis?",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "correct_answer": "B",
        "usmle_step": "step1",
        "initial_cot": (
            "Reasoning step 1: Consider the patient's symptoms.\n"
            "Reasoning step 2: Apply diagnostic criteria.\n"
            "Reasoning step 3: Eliminate alternatives.\n"
            "\n"
            "Final answer: B\n"
            "Confidence: 0.75\n"
        ),
        "distributions": [
            {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            {"A": 0.2, "B": 0.5, "C": 0.15, "D": 0.15},
            {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1},
            {"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.05},
        ],
    },
    {
        "question_id": "smoke-2",
        "question": "Patient B: clinical vignette two. What's the diagnosis?",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "correct_answer": "B",
        "usmle_step": "step2&3",
        "initial_cot": (
            "Reasoning step 1: Patient presentation suggests a particular pattern.\n"
            "Reasoning step 2: Lab values point toward A.\n"
            "Reasoning step 3: A is most consistent with the overall picture.\n"
            "\n"
            "Final answer: A\n"
            "Confidence: 0.6\n"
        ),
        "distributions": [
            {"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2},
            {"A": 0.5, "B": 0.2, "C": 0.15, "D": 0.15},
            {"A": 0.65, "B": 0.15, "C": 0.1, "D": 0.1},
            {"A": 0.6, "B": 0.2, "C": 0.1, "D": 0.1},
        ],
    },
    {
        "question_id": "smoke-3",
        "question": "Patient C: clinical vignette three. What's the diagnosis?",
        "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "correct_answer": "B",
        "usmle_step": "step1",
        "initial_cot": (
            "Reasoning step 1: Symptoms suggest cardiac etiology.\n"
            "Reasoning step 2: ECG findings consistent with anterior MI.\n"
            "Reasoning step 3: Cardiac biomarkers should be elevated.\n"
            "Reasoning step 4: Treatment is reperfusion therapy.\n"
            "\n"
            "Final answer: C\n"
            "Confidence: 0.85\n"
        ),
        "distributions": [
            {"A": 0.2, "B": 0.3, "C": 0.4, "D": 0.1},
            {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1},
            {"A": 0.05, "B": 0.1, "C": 0.8, "D": 0.05},
            {"A": 0.05, "B": 0.05, "C": 0.85, "D": 0.05},
            {"A": 0.05, "B": 0.05, "C": 0.85, "D": 0.05},
        ],
    },
]


# ============================================================
# Mock LLM and embedder (inline to avoid test/script boundary)
# ============================================================


class _SmokeScriptedLLM:
    """Returns fixture-driven responses keyed on question_id extracted
    from the prompt. Implements the full LLMAdapter Protocol."""

    def __init__(self, fixtures: list[dict[str, Any]]) -> None:
        self._fixtures_by_id = {f["question_id"]: f for f in fixtures}
        self._qid_re = re.compile(r"Patient ([A-Z]):")

    def _qid_from_prompt(self, prompt: str) -> str | None:
        m = self._qid_re.search(prompt)
        if not m:
            return None
        # Map "Patient A" -> "smoke-1", etc.
        letter = m.group(1)
        return {"A": "smoke-1", "B": "smoke-2", "C": "smoke-3"}.get(letter)

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> str:
        qid = self._qid_from_prompt(prompt)
        if qid is None or qid not in self._fixtures_by_id:
            return ""
        return self._fixtures_by_id[qid]["initial_cot"]

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_retries: int = 2,
    ) -> Sequence[str]:
        return [self.generate(p) for p in prompts]

    def get_token_probabilities(
        self,
        prompt: str,
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> TokenProbabilityResult:
        n = len(token_set)
        return TokenProbabilityResult(
            distribution={t: 1.0 / n for t in token_set},
            mass_capture=1.0,
            truncated_members=(),
        )

    def get_token_probabilities_batch(
        self,
        prompts: Sequence[str],
        token_set: Sequence[str],
        max_retries: int = 2,
    ) -> Sequence[TokenProbabilityResult]:
        if not prompts:
            return []
        qid = self._qid_from_prompt(prompts[0])
        captured = (
            self._fixtures_by_id[qid]["distributions"]
            if qid is not None and qid in self._fixtures_by_id
            else None
        )
        n = len(token_set)
        results: list[TokenProbabilityResult] = []
        for i in range(len(prompts)):
            dist = (
                captured[i]
                if captured is not None and i < len(captured)
                else {t: 1.0 / n for t in token_set}
            )
            # Re-normalise just in case fixture distributions don't
            # exactly cover token_set (defensive).
            keyed = {t: float(dist.get(t, 0.0)) for t in token_set}
            total = sum(keyed.values())
            if total > 0:
                keyed = {t: v / total for t, v in keyed.items()}
            else:
                keyed = {t: 1.0 / n for t in token_set}
            results.append(
                TokenProbabilityResult(
                    distribution=keyed, mass_capture=1.0, truncated_members=()
                )
            )
        return results

    def get_metadata(self) -> Mapping[str, str]:
        return {"model": "_SmokeScriptedLLM", "model_version": "1"}


class _SmokeMockEmbedder:
    """Hash-seeded L2-normalized embeddings. Same as
    DeterministicMockEmbedder in tests/medqa/conftest.py."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        import hashlib
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self.embed(t) for t in texts])

    def get_metadata(self) -> Mapping[str, str]:
        return {
            "model": "_SmokeMockEmbedder",
            "model_version": "1",
            "dim": str(self._dim),
        }

    @property
    def dimension(self) -> int:
        return self._dim


# ============================================================
# Pipeline
# ============================================================


def main(output_dir: Path, *, strict: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline validation smoke (strict={strict})")
    print(f"Output: {output_dir}")
    print()

    # ---- 1. Build records from fixtures ----
    records = [
        MedQARawRecord(
            question_id=f["question_id"],
            question=f["question"],
            choices=f["choices"],
            answer_letter=f["correct_answer"],
            usmle_step=f["usmle_step"],
        )
        for f in _FIXTURES
    ]
    print(f"Records: {len(records)}")

    # ---- 2. Build conditions ----
    embedder = _SmokeMockEmbedder(dim=8)
    llm = _SmokeScriptedLLM(_FIXTURES)
    extractor = AnswerKeyGroundTruthExtractor()
    decomposer = Decomposer()

    cond_a = ConditionA(llm, decomposer=decomposer)
    cond_b = ConditionB(llm, decomposer=decomposer)
    cond_c = ConditionC(
        llm,
        state_canonicalizer=MCQStateCanonicalizer(embedder),
        action_canonicalizer=MCQActionCanonicalizer(embedder),
        embedder=embedder,
        decomposer=decomposer,
    )

    # ---- 3. Run conditions ----
    results_by_condition: dict[str, list] = {"A": [], "B": [], "C": []}
    for record in records:
        for cond_id, cond in [("A", cond_a), ("B", cond_b), ("C", cond_c)]:
            result = cond.run(record)
            outcome = extractor.extract(record)
            traj = with_outcome(result.trajectory, outcome)
            results_by_condition[cond_id].append((result, traj))

    # ---- 4. Persist Condition C cached trajectories + round-trip ----
    c_trajectories = [traj for _, traj in results_by_condition["C"]]
    cached_path = output_dir / "condition_c_cached"
    save_cached_trajectories(
        c_trajectories,
        cached_path,
        overwrite=True,
        source_dataset="smoke-fixtures",
        condition_id="condition_c",
    )
    reloaded = MedQAPrerecoveredTrajectorySource(cached_path).load_all()
    if strict:
        assert len(reloaded) == len(c_trajectories), (
            "round-trip: trajectory count mismatch"
        )
    print(f"Cached trajectories round-trip: {len(reloaded)} OK")

    # ---- 5. Recovery ----
    recovery_result = recover_assembly_graph(
        c_trajectories, RecoveryConfig(voi_local_prior_min_count=1)
    )
    if strict:
        assert recovery_result.graph.num_nodes > 0, "recovery: empty graph nodes"
        assert recovery_result.graph.num_edges > 0, "recovery: empty graph edges"
        assert len(recovery_result.visits) > 0, "recovery: empty visits"
    print(
        f"Recovery: {recovery_result.graph.num_nodes} nodes, "
        f"{recovery_result.graph.num_edges} edges, "
        f"{len(recovery_result.visits)} visits"
    )

    # Persist graph
    save_graph(recovery_result.graph, output_dir / "graph_artifact", overwrite=True)

    # ---- 6. FAISS indices ----
    embedding_indices, manifest = build_faiss_indices_from_visits(
        recovery_result.visits,
        dimension=embedder.dimension,
        embedding_model="_SmokeMockEmbedder",
    )
    if embedding_indices:
        save_embedding_indices(
            embedding_indices,
            output_dir / "graph_artifact",
            manifest,
            overwrite=True,
        )
    print(f"FAISS indices: {len(embedding_indices)} timesteps")

    # ---- 7. Compute signatures for each condition's trajectories ----
    weights = SignatureWeights()
    scores_by_condition: dict[str, pd.DataFrame] = {}
    for cond_id, results in results_by_condition.items():
        trajectories = [traj for _, traj in results]
        # For A and B (single-state trajectories with no per-step
        # distributions over reasoning), compute_signatures still runs
        # — entropy_plateau returns 0 (one distribution),
        # voi_flatness returns 0 (no actions),
        # distance_from_trajectory returns 0 (no embeddings).
        scores = compute_signatures(
            trajectories,
            recovery_result.graph,
            recovery_result.visits,
            embedding_indices,
            weights,
        )
        scores_by_condition[cond_id] = scores
        if strict:
            assert not scores["composite"].isna().all(), (
                f"signatures for {cond_id}: all composite scores NaN"
            )
    print(f"Signatures computed for {sorted(scores_by_condition)}")

    # Persist signature scores per condition
    for cond_id, scores in scores_by_condition.items():
        cond_dir = output_dir / f"condition_{cond_id}_artifact"
        cond_dir.mkdir(parents=True, exist_ok=True)
        # Need a graph artifact for save_signature_scores to write into.
        # Reuse the main graph_artifact for all three.
        save_signature_scores(
            scores,
            weights,
            output_dir / "graph_artifact",
            overwrite=True,
            graph_artifact_path=output_dir / "graph_artifact",
            prompt_versions=load_all_versions(),
        )
        # Also save per-condition copies for diagnostic inspection
        scores.to_csv(cond_dir / "signature_scores.csv", index=False)

    # ---- 8. Build A and B per-condition DataFrames for condition_comparison ----
    a_compare = pd.DataFrame(
        {
            "trajectory_id": [r.question_id for r, _ in results_by_condition["A"]],
            "deferral_signal": [r.deferral_signal for r, _ in results_by_condition["A"]],
        }
    )
    b_compare = pd.DataFrame(
        {
            "trajectory_id": [r.question_id for r, _ in results_by_condition["B"]],
            "deferral_signal": [r.deferral_signal for r, _ in results_by_condition["B"]],
        }
    )
    c_compare = scores_by_condition["C"]

    # Ground truth: target=1 means "model got it wrong, should defer."
    ground_truth = pd.DataFrame(
        {
            "trajectory_id": [
                r.question_id for r, _ in results_by_condition["C"]
            ],
            "needs_consultation": [
                int(r.predicted_answer != record.answer_letter)
                for record, (r, _) in zip(records, results_by_condition["C"], strict=True)
            ],
        }
    )

    # Need both classes present for evaluation. With 3 fixtures and
    # answers (B, A, C) vs correct (B, B, B): correct/wrong split is
    # 1 correct (smoke-1) and 2 wrong (smoke-2, smoke-3) → both classes.
    n_pos = int((ground_truth["needs_consultation"] == 1).sum())
    n_neg = int((ground_truth["needs_consultation"] == 0).sum())
    if strict:
        assert n_pos > 0 and n_neg > 0, (
            f"smoke fixtures don't produce both classes: pos={n_pos}, neg={n_neg}"
        )
    print(f"Ground truth: {n_pos} pos, {n_neg} neg")

    cmp = condition_comparison(
        {"A": a_compare, "B": b_compare, "C": c_compare},
        ground_truth,
        target_column="needs_consultation",
        compute_ci=False,
    )
    print("\nCondition comparison:")
    print(cmp.to_string(index=False))

    if strict:
        assert set(cmp["condition_id"]) == {"A", "B", "C"}, (
            "condition_comparison: missing conditions"
        )
        assert not cmp["roc_auc"].isna().any(), (
            "condition_comparison: NaN AUCs"
        )
        a_auc = float(cmp[cmp["condition_id"] == "A"].iloc[0]["roc_auc"])
        assert math.isclose(a_auc, 0.5, abs_tol=1e-6), (
            f"condition A's constant-signal should yield AUC=0.5, got {a_auc}"
        )

    cmp.to_csv(output_dir / "condition_comparison.csv", index=False)

    # ---- 9. Failure-mode table on Condition C ----
    fmt = failure_mode_table(
        c_compare, ground_truth, "needs_consultation", top_n=10
    )
    print("\nFailure-mode table (top 10):")
    print(fmt.to_string(index=False))
    fmt.to_csv(output_dir / "failure_mode_table.csv", index=False)

    # ---- 10. Repair-rate instrumentation (per the project-arc note) ----
    print("\nRepair-rate summary:")
    repair_summary: dict[str, dict[str, float | int]] = {}
    for cond_id, results in results_by_condition.items():
        n_results = len(results)
        n_failures = sum(1 for r, _ in results if not r.success)
        repair_attempts = [
            int(r.metadata.get("repair_attempts", 0)) for r, _ in results
        ]
        confidence_parsed = [
            r.metadata.get("confidence_parsed") for r, _ in results
        ]
        repair_summary[cond_id] = {
            "n_results": n_results,
            "n_failures": n_failures,
            "mean_repair_attempts": (
                float(np.mean(repair_attempts)) if repair_attempts else 0.0
            ),
            "n_confidence_parsed": sum(
                1 for c in confidence_parsed if c is True
            ),
            "n_confidence_unparsed": sum(
                1 for c in confidence_parsed if c is False
            ),
        }
    print(json.dumps(repair_summary, indent=2))

    with (output_dir / "repair_summary.json").open("w") as fp:
        json.dump(repair_summary, fp, indent=2)

    print(f"\nPipeline validation complete. Artifacts in {output_dir}")
    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enable structural assertions (default; use --no-strict to disable)",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Skip structural assertions; log warnings instead",
    )
    args = parser.parse_args()
    sys.exit(main(args.output_dir, strict=args.strict))


if __name__ == "__main__":
    cli()
