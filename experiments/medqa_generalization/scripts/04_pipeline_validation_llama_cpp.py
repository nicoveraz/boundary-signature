#!/usr/bin/env python
"""Pipeline validation against llama.cpp + unified-measurement protocol (ADR-0008).

Parallel to ``03_pipeline_validation_ollama.py``. Same pipeline shape
(Conditions A/B/C, recovery, signature, condition_comparison) but with
the post-ADR-0008 architecture:

- ``LlamaCppLLMAdapter`` instead of ``OllamaLLMAdapter``. Reads
  next-token logprobs at the measurement position; returns
  TokenProbabilityResult (renormalised conditional + mass capture +
  truncated members).
- Condition C uses the unified-measurement protocol: predicted answer
  is argmax of the terminal measurement, NOT extracted from the CoT
  text. State.mass_capture is populated from the measurement.
- Stage-4 mass-capture analysis: in addition to the composite
  signature, the runner reports ``mass_capture_mean`` and
  ``mass_capture_min`` as candidate deferral signals (per the
  multi-hypothesis principle and the calibrated-claims pre-registered
  predictions in ADR-0008). The methods paper will report results
  across these operationalisations.

Usage:
    # Smoke (5 questions, mock embedder)
    python 04_pipeline_validation_llama_cpp.py

    # ADR-0008 re-pilot at N=100 (sentence-transformers required).
    # Tee the run log INSIDE the output directory, not as a sibling
    # in artifacts/, so all run artifacts are co-located.
    OUT=~/work/eunosia/artifacts/medqa-stage-4a-pilot-n100-llamacpp
    mkdir -p "$OUT"
    python 04_pipeline_validation_llama_cpp.py \\
        --n-questions 100 \\
        --embedder-backend sentence-transformers \\
        --embedder-model intfloat/multilingual-e5-large \\
        --embedder-prefix "" \\
        --checkpoint-every 25 \\
        --output-dir "$OUT" \\
        2>&1 | tee "$OUT/run.log"

    # Stage 4b cross-benchmark: one subject per invocation.
    OUT=~/work/eunosia/artifacts/medqa-stage-4b-mmlu-professional_law
    mkdir -p "$OUT"
    python 04_pipeline_validation_llama_cpp.py \\
        --benchmark mmlu --mmlu-subject professional_law \\
        --n-questions 1534 \\
        --embedder-backend sentence-transformers \\
        --embedder-model intfloat/multilingual-e5-large \\
        --embedder-prefix "" \\
        --checkpoint-every 50 \\
        --output-dir "$OUT" \\
        2>&1 | tee "$OUT/run.log"

    # Resume an interrupted run
    python 04_pipeline_validation_llama_cpp.py --resume

The llama.cpp server must be running before this script is invoked:
    llama-server -m <gguf-path> --port 8080 --ctx-size 4096 --n-gpu-layers 99
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    MedQAQuestionLoader,
    MedQARawRecord,
    MMLULoader,
    condition_comparison,
    failure_mode_table,
    load_all_versions,
    save_cached_trajectories,
)
from bsig.medqa.conditions._helpers import with_outcome
from bsig.reference.llm_llama_cpp import LlamaCppLLMAdapter

DEFAULT_OUTPUT_DIR = (
    Path.home()
    / "work"
    / "eunosia"
    / "artifacts"
    / "medqa-stage-4a-pilot-n100-llamacpp"
)


# ============================================================
# Mock embedder (mirrors 03 + 02 scripts)
# ============================================================


class _MockEmbedder:
    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
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
            "model": "_MockEmbedder",
            "model_version": "1",
            "dim": str(self._dim),
        }

    @property
    def dimension(self) -> int:
        return self._dim


def _build_embedder(backend: str, model: str | None, prefix: str = "") -> Any:
    if backend == "mock":
        return _MockEmbedder(dim=8)
    if backend == "sentence-transformers":
        from bsig.reference.embedding_st import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=model or "intfloat/multilingual-e5-large",
            prefix=prefix,
        )
    raise SystemExit(f"unknown embedder backend: {backend!r}")


# ============================================================
# Checkpoint + resume (identical pattern to 03 script)
# ============================================================


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def _condition_cache_path(output_dir: Path, cond_id: str) -> Path:
    return output_dir / f"condition_{cond_id.lower()}_cached"


def _load_checkpoint(output_dir: Path) -> set[str]:
    cp = _checkpoint_path(output_dir)
    if not cp.exists():
        return set()
    data = json.loads(cp.read_text())
    return set(data.get("processed_question_ids", []))


def _save_checkpoint(output_dir: Path, processed_ids: set[str]) -> None:
    cp = _checkpoint_path(output_dir)
    cp.write_text(
        json.dumps({"processed_question_ids": sorted(processed_ids)}, indent=2)
    )


def _save_partial_results(
    output_dir: Path,
    results_by_condition: dict[str, list],
) -> None:
    partial_path = output_dir / "partial_results.json"
    serialized = {}
    for cond_id, results in results_by_condition.items():
        serialized[cond_id] = [
            {
                "question_id": r.question_id,
                "predicted_answer": r.predicted_answer,
                "deferral_signal": (
                    r.deferral_signal
                    if not (
                        isinstance(r.deferral_signal, float)
                        and math.isnan(r.deferral_signal)
                    )
                    else None
                ),
                "success": r.success,
                "failure_reason": r.failure_reason,
                "metadata": dict(r.metadata),
            }
            for r, _ in results
        ]
    partial_path.write_text(json.dumps(serialized, indent=2))


def _checkpoint_trajectories(
    output_dir: Path,
    results_by_condition: dict[str, list],
    source_dataset: str,
) -> None:
    """Save trajectories for each condition. Filter success=False per
    the ConditionResult contract (same as 03)."""
    for cond_id, results in results_by_condition.items():
        trajectories = [traj for r, traj in results if r.success]
        if not trajectories:
            continue
        save_cached_trajectories(
            trajectories,
            _condition_cache_path(output_dir, cond_id),
            overwrite=True,
            source_dataset=source_dataset,
            condition_id=f"condition_{cond_id.lower()}",
        )


# ============================================================
# Pipeline
# ============================================================


def main(args: argparse.Namespace) -> int:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pipeline validation against llama.cpp (strict={args.strict})")
    print(f"Output: {output_dir}")
    print(f"LLM host: {args.llm_host}")
    print(f"Model (logged): {args.llm_model}")
    print(f"Embedder backend: {args.embedder_backend}")
    print(f"N questions: {args.n_questions}, split: {args.split}")
    print()

    # ---- Resume / checkpoint state ----
    processed_ids: set[str] = (
        _load_checkpoint(output_dir) if args.resume else set()
    )
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} questions already processed")

    # ---- 1. Load records ----
    if args.benchmark == "mmlu":
        if not args.mmlu_subject:
            print(
                "ERROR: --benchmark mmlu requires --mmlu-subject "
                "(e.g., professional_law)",
                file=sys.stderr,
            )
            return 2
        mmlu_loader = MMLULoader(
            subject=args.mmlu_subject, split=args.split
        )
        all_records = list(
            islice(mmlu_loader.iter_records(), args.n_questions)
        )
        print(
            f"Benchmark: MMLU subject={args.mmlu_subject!r}, "
            f"split={args.split}"
        )
    else:
        medqa_loader = MedQAQuestionLoader(split=args.split)
        all_records = list(
            islice(medqa_loader.iter_records(), args.n_questions)
        )
        print(f"Benchmark: MedQA-USMLE-4-options, split={args.split}")
    pending_records = [r for r in all_records if r.question_id not in processed_ids]
    print(f"Records: {len(all_records)} total, {len(pending_records)} pending")
    if not pending_records and not args.resume:
        print("No pending records and not resuming; nothing to do.")
        return 0

    # ---- 2. Build conditions ----
    embedder = _build_embedder(
        args.embedder_backend, args.embedder_model, args.embedder_prefix
    )
    llm = LlamaCppLLMAdapter(
        model=args.llm_model,
        host=args.llm_host,
        timeout=args.timeout,
    )
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

    # ---- 3. Run conditions with checkpointing ----
    results_by_condition: dict[str, list] = {"A": [], "B": [], "C": []}

    if args.resume and processed_ids:
        from bsig.medqa.conditions.result import ConditionResult

        partial_path = output_dir / "partial_results.json"
        if not partial_path.exists():
            raise SystemExit(
                f"--resume requested but partial_results.json missing at "
                f"{partial_path}. Cannot reconstruct ConditionResult fields."
            )
        partial_data = json.loads(partial_path.read_text())

        for cond_id in ("A", "B", "C"):
            cache_path = _condition_cache_path(output_dir, cond_id)
            if not cache_path.exists():
                continue
            loaded = MedQAPrerecoveredTrajectorySource(cache_path).load_all()
            partial_by_qid = {
                p["question_id"]: p
                for p in partial_data.get(cond_id, [])
            }
            for traj in loaded:
                meta_record = partial_by_qid.get(traj.trajectory_id)
                if meta_record is None:
                    raise SystemExit(
                        f"--resume: cached trajectory {traj.trajectory_id!r} "
                        f"in condition {cond_id} has no entry in "
                        f"partial_results.json. Cannot reconstruct."
                    )
                deferral = meta_record["deferral_signal"]
                if deferral is None:
                    deferral = math.nan
                synthetic = ConditionResult(
                    question_id=traj.trajectory_id,
                    predicted_answer=meta_record["predicted_answer"],
                    deferral_signal=float(deferral),
                    trajectory=traj,
                    raw_llm_output=None,
                    metadata={
                        **meta_record.get("metadata", {}),
                        "resumed": True,
                    },
                    success=meta_record.get("success", True),
                    failure_reason=meta_record.get("failure_reason"),
                )
                results_by_condition[cond_id].append((synthetic, traj))
        print(
            f"Resume: loaded cached trajectories — "
            f"{len(results_by_condition['A'])} A, "
            f"{len(results_by_condition['B'])} B, "
            f"{len(results_by_condition['C'])} C"
        )

    try:
        from tqdm import tqdm
        progress = tqdm(pending_records, desc="Conditions A/B/C")
    except ImportError:
        progress = pending_records

    source_dataset = "GBaker/MedQA-USMLE-4-options"
    for i, record in enumerate(progress):
        for cond_id, cond in [("A", cond_a), ("B", cond_b), ("C", cond_c)]:
            result = cond.run(record)
            outcome = extractor.extract(record)
            traj = with_outcome(result.trajectory, outcome)
            results_by_condition[cond_id].append((result, traj))
        processed_ids.add(record.question_id)

        if (i + 1) % args.checkpoint_every == 0:
            _save_checkpoint(output_dir, processed_ids)
            _save_partial_results(output_dir, results_by_condition)
            _checkpoint_trajectories(
                output_dir, results_by_condition, source_dataset
            )

    _save_checkpoint(output_dir, processed_ids)
    _save_partial_results(output_dir, results_by_condition)
    _checkpoint_trajectories(output_dir, results_by_condition, source_dataset)
    print(f"All {len(pending_records)} records processed.")

    # ---- 4. Round-trip verification ----
    c_trajectories = [traj for r, traj in results_by_condition["C"] if r.success]
    cached_path = _condition_cache_path(output_dir, "C")
    reloaded = MedQAPrerecoveredTrajectorySource(cached_path).load_all()
    if args.strict:
        assert len(reloaded) == len(c_trajectories), "round-trip count mismatch"
    print(f"Cached trajectories round-trip: {len(reloaded)} OK")

    # ---- 5. Recovery ----
    recovery_result = recover_assembly_graph(
        c_trajectories, RecoveryConfig(voi_local_prior_min_count=1)
    )
    if args.strict:
        assert recovery_result.graph.num_nodes > 0, "recovery: empty graph"
        assert len(recovery_result.visits) > 0, "recovery: empty visits"
    print(
        f"Recovery: {recovery_result.graph.num_nodes} nodes, "
        f"{recovery_result.graph.num_edges} edges, "
        f"{len(recovery_result.visits)} visits"
    )
    save_graph(recovery_result.graph, output_dir / "graph_artifact", overwrite=True)

    # ---- 6. FAISS indices ----
    embedding_indices, manifest = build_faiss_indices_from_visits(
        recovery_result.visits,
        dimension=embedder.dimension,
        embedding_model=embedder.get_metadata().get("model", "unknown"),
    )
    if embedding_indices:
        save_embedding_indices(
            embedding_indices,
            output_dir / "graph_artifact",
            manifest,
            overwrite=True,
        )
    print(f"FAISS indices: {len(embedding_indices)} timesteps")

    # ---- 7. Compute signatures (now includes mass_capture_mean / _min) ----
    weights = SignatureWeights()
    scores_by_condition: dict[str, pd.DataFrame] = {}
    for cond_id, results in results_by_condition.items():
        trajectories = [traj for _, traj in results]
        scores = compute_signatures(
            trajectories,
            recovery_result.graph,
            recovery_result.visits,
            embedding_indices,
            weights,
        )
        scores_by_condition[cond_id] = scores
        if args.strict:
            assert not scores["composite"].isna().all(), (
                f"signatures for {cond_id}: all composite NaN"
            )
    print(f"Signatures computed for {sorted(scores_by_condition)}")

    save_signature_scores(
        scores_by_condition["C"],
        weights,
        output_dir / "graph_artifact",
        overwrite=True,
        graph_artifact_path=output_dir / "graph_artifact",
        prompt_versions=load_all_versions(),
    )
    for cond_id, scores in scores_by_condition.items():
        cond_dir = output_dir / f"condition_{cond_id}_artifact"
        cond_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(cond_dir / "signature_scores.csv", index=False)

    # ---- 8. Condition comparison: composite + mass_capture-on-its-own ----
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

    # Mass-capture-on-its-own as deferral signal (1 - mass_capture so
    # higher = more boundary-flagged). Per ADR-0008's pre-registered
    # predictions, this is the candidate signature for the methods
    # paper if the patterns observed at N=50 generalise.
    c_mc_mean_compare = pd.DataFrame(
        {
            "trajectory_id": c_compare["trajectory_id"],
            "deferral_signal": 1.0 - c_compare["mass_capture_mean"],
        }
    )
    c_mc_min_compare = pd.DataFrame(
        {
            "trajectory_id": c_compare["trajectory_id"],
            "deferral_signal": 1.0 - c_compare["mass_capture_min"],
        }
    )

    ground_truth = pd.DataFrame(
        {
            "trajectory_id": [r.question_id for r, _ in results_by_condition["C"]],
            "needs_consultation": [
                int(r.predicted_answer != record.answer_letter)
                for record, (r, _) in zip(
                    all_records, results_by_condition["C"], strict=True
                )
            ],
        }
    )

    n_pos = int((ground_truth["needs_consultation"] == 1).sum())
    n_neg = int((ground_truth["needs_consultation"] == 0).sum())
    print(f"Ground truth: {n_pos} pos, {n_neg} neg")

    if n_pos == 0 or n_neg == 0:
        print(
            "WARNING: ground truth has only one class — condition_comparison "
            "cannot run. Common at small N if all questions go same way."
        )
        return 0

    cmp = condition_comparison(
        {
            "A": a_compare,
            "B": b_compare,
            "C": c_compare,
            "C_mc_mean": c_mc_mean_compare,
            "C_mc_min": c_mc_min_compare,
        },
        ground_truth,
        target_column="needs_consultation",
        score_columns_per_condition={
            "A": "deferral_signal",
            "B": "deferral_signal",
            "C": "composite",
            "C_mc_mean": "deferral_signal",
            "C_mc_min": "deferral_signal",
        },
        compute_ci=False,
    )
    print("\nCondition comparison (composite + mass-capture variants):")
    print(cmp.to_string(index=False))

    if args.strict:
        a_auc = float(cmp[cmp["condition_id"] == "A"].iloc[0]["roc_auc"])
        assert math.isclose(a_auc, 0.5, abs_tol=1e-6), (
            f"Condition A's constant signal should yield AUC=0.5, got {a_auc}"
        )

    cmp.to_csv(output_dir / "condition_comparison.csv", index=False)

    # ---- 9. Failure-mode table ----
    fmt = failure_mode_table(
        c_compare, ground_truth, "needs_consultation", top_n=20
    )
    print(f"\nFailure-mode table (top {min(20, len(fmt))}):")
    print(fmt.to_string(index=False))
    fmt.to_csv(output_dir / "failure_mode_table.csv", index=False)

    # ---- 10. Repair-rate + mass-capture summary ----
    print("\nRepair-rate summary:")
    repair_summary: dict[str, dict[str, Any]] = {}
    for cond_id, results in results_by_condition.items():
        n_results = len(results)
        n_failures = sum(1 for r, _ in results if not r.success)
        repair_attempts = [
            int(r.metadata.get("repair_attempts", 0)) for r, _ in results
        ]
        confidence_parsed = [
            r.metadata.get("confidence_parsed") for r, _ in results
        ]
        truncated_events = [
            int(r.metadata.get("n_truncated_member_events", 0)) for r, _ in results
        ]
        repair_summary[cond_id] = {
            "n_results": n_results,
            "n_failures": n_failures,
            "mean_repair_attempts": (
                float(np.mean(repair_attempts)) if repair_attempts else 0.0
            ),
            "n_confidence_parsed": sum(1 for c in confidence_parsed if c is True),
            "n_confidence_unparsed": sum(1 for c in confidence_parsed if c is False),
            "mean_truncated_member_events": (
                float(np.mean(truncated_events)) if truncated_events else 0.0
            ),
            "n_trajectories_with_truncation": sum(
                1 for t in truncated_events if t > 0
            ),
        }
    print(json.dumps(repair_summary, indent=2))

    with (output_dir / "repair_summary.json").open("w") as fp:
        json.dump(repair_summary, fp, indent=2)

    # ---- 11. Mass-capture distribution analysis (per ADR-0008 predictions) ----
    print("\nMass-capture distribution (Condition C):")
    mc_summary = _mass_capture_summary(
        results_by_condition["C"], all_records
    )
    print(json.dumps(mc_summary, indent=2))
    with (output_dir / "mass_capture_summary.json").open("w") as fp:
        json.dump(mc_summary, fp, indent=2)

    print(f"\nPipeline validation complete. Artifacts in {output_dir}")
    return 0


def _mass_capture_summary(
    c_results: list,
    all_records: list,
) -> dict[str, Any]:
    """Per-trajectory mass-capture metrics aggregated for ADR-0008's
    pre-registered prediction tests. Surfaces:

    - Distribution of mass_capture_mean and mass_capture_min across
      successful Condition C trajectories.
    - Δ between correct and wrong groups (point estimate; bootstrap CI
      computation deferred to a downstream analysis script).
    - Count of trajectories with any state below 0.25 mass_capture
      (the extreme-tail threshold from the N=50 investigation).
    - Wrong-rate among those extreme-tail trajectories vs base rate.
    """
    record_by_qid = {r.question_id: r for r in all_records}
    rows: list[dict[str, Any]] = []
    for r, traj in c_results:
        if not r.success:
            continue
        record = record_by_qid.get(r.question_id)
        if record is None:
            continue
        captures = [
            s.mass_capture for s in traj.states if s.mass_capture is not None
        ]
        if not captures:
            continue
        mc_mean = float(np.mean(captures))
        mc_min = float(min(captures))
        correct = (r.predicted_answer == record.answer_letter)
        rows.append(
            {
                "question_id": r.question_id,
                "mass_capture_mean": mc_mean,
                "mass_capture_min": mc_min,
                "correct": correct,
            }
        )

    if not rows:
        return {"n": 0, "note": "no Condition C measurements with mass_capture"}

    means_correct = [r["mass_capture_mean"] for r in rows if r["correct"]]
    means_wrong = [r["mass_capture_mean"] for r in rows if not r["correct"]]
    extreme_rows = [r for r in rows if r["mass_capture_min"] < 0.25]
    extreme_wrong = sum(1 for r in extreme_rows if not r["correct"])
    base_wrong = sum(1 for r in rows if not r["correct"])

    return {
        "n_total": len(rows),
        "n_correct": len(means_correct),
        "n_wrong": len(means_wrong),
        "mass_capture_mean": {
            "min": float(min(r["mass_capture_mean"] for r in rows)),
            "mean": float(np.mean([r["mass_capture_mean"] for r in rows])),
            "median": float(np.median([r["mass_capture_mean"] for r in rows])),
            "max": float(max(r["mass_capture_mean"] for r in rows)),
        },
        "mass_capture_min": {
            "min": float(min(r["mass_capture_min"] for r in rows)),
            "mean": float(np.mean([r["mass_capture_min"] for r in rows])),
            "median": float(np.median([r["mass_capture_min"] for r in rows])),
            "max": float(max(r["mass_capture_min"] for r in rows)),
        },
        "correct_vs_wrong_mass_capture_mean_delta": (
            float(np.mean(means_correct) - np.mean(means_wrong))
            if means_correct and means_wrong
            else None
        ),
        "extreme_tail_mass_lt_0_25": {
            "n_trajectories": len(extreme_rows),
            "n_wrong": extreme_wrong,
            "wrong_rate": (
                extreme_wrong / len(extreme_rows) if extreme_rows else None
            ),
            "base_wrong_rate": base_wrong / len(rows) if rows else None,
        },
    }


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--benchmark",
        choices=["medqa", "mmlu"],
        default="medqa",
        help="Source benchmark. 'medqa' uses GBaker/MedQA-USMLE-4-options "
        "(stage 4a). 'mmlu' uses cais/mmlu and requires --mmlu-subject "
        "(stage 4b cross-benchmark replication).",
    )
    parser.add_argument(
        "--mmlu-subject",
        default=None,
        help="Required when --benchmark=mmlu. The cais/mmlu subject "
        "config (e.g., professional_law, professional_accounting, "
        "professional_medicine, formal_logic, elementary_mathematics).",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen2.5:7b-instruct",
        help="Model identifier for reproducibility logging only. The "
        "actual model loaded is whatever GGUF the llama.cpp server was "
        "started with.",
    )
    parser.add_argument("--llm-host", default="http://localhost:8080")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--embedder-backend",
        choices=["mock", "sentence-transformers"],
        default="mock",
    )
    parser.add_argument("--embedder-model", default=None)
    parser.add_argument("--embedder-prefix", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true", default=False)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", action="store_true", default=True)
    strict_group.add_argument(
        "--no-strict", dest="strict", action="store_false"
    )
    args = parser.parse_args()
    sys.exit(main(args))


if __name__ == "__main__":
    cli()
