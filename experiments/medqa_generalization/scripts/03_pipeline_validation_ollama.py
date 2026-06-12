#!/usr/bin/env python
"""Pipeline validation against real Ollama (and optionally real embedder).

The de-risking step for stage 4a. Stage 3.5a ships this with mock
embedder; stage 3.5b extends with --embedder-backend
sentence-transformers. After 3.5b lands,
``--n-questions 1273 --embedder-backend sentence-transformers`` IS
stage 4a (per S5_4 in
``docs/decisions/stage_3_retrospective_notes.md``).

Pipeline shape identical to
``02_pipeline_validation_local.py`` — same Conditions A/B/C, same
recovery, same signature scoring, same condition_comparison.
Difference: real LLM (Ollama) instead of scripted mock; real records
from MedQAQuestionLoader instead of inline fixtures.

**Operational notes for stage 4a (--n-questions 1273):**
- Estimated runtime: 20-25 hours of M1 Pro compute.
- Use ``--checkpoint-every 50`` (default) to write incremental
  progress to disk; restart with ``--resume`` skips already-
  processed questions.
- Plug in the laptop. Use ``caffeinate -i`` or similar to prevent
  sleep. Don't run on battery.

Usage:
    # Stage 3.5a smoke (5 questions, mock embedder)
    python 03_pipeline_validation_ollama.py

    # Same with explicit args
    python 03_pipeline_validation_ollama.py \\
        --n-questions 5 --llm-model qwen2.5:7b-instruct

    # Stage 4a (after 3.5b lands; sentence-transformers required)
    python 03_pipeline_validation_ollama.py \\
        --n-questions 1273 \\
        --embedder-backend sentence-transformers \\
        --checkpoint-every 50

    # Resume an interrupted run
    python 03_pipeline_validation_ollama.py --resume
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
    condition_comparison,
    failure_mode_table,
    load_all_versions,
    save_cached_trajectories,
)
from bsig.medqa.conditions._helpers import with_outcome
from bsig.reference.llm_local import OllamaLLMAdapter

DEFAULT_OUTPUT_DIR = (
    Path.home() / "work" / "eunosia" / "artifacts" / "medqa-ollama-smoke"
)


# ============================================================
# Mock embedder (3.5a default; 3.5b adds sentence-transformers path)
# ============================================================


class _MockEmbedder:
    """Hash-seeded L2-normalized embeddings. Same shape as the
    DeterministicMockEmbedder in tests/medqa/conftest.py and the
    one in 02_pipeline_validation_local.py."""

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


def _build_embedder(
    backend: str, model: str | None, prefix: str = ""
) -> Any:
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
# Checkpoint + resume (per stage-3.5b recommendation: use
# cached-trajectories Parquet format so partial runs produce
# partial-but-usable cached data, and resume skips already-
# processed question_ids).
# ============================================================


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def _condition_cache_path(output_dir: Path, cond_id: str) -> Path:
    return output_dir / f"condition_{cond_id.lower()}_cached"


def _load_checkpoint(output_dir: Path) -> set[str]:
    """Return the set of question_ids already processed."""
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
    """Write per-condition partial-results JSON (just the
    ConditionResult metadata; trajectories are persisted separately
    via save_cached_trajectories at each checkpoint)."""
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
    """Save current trajectories for each condition as cached-
    trajectories Parquet (overwrite). Re-write at each checkpoint;
    the cumulative cache grows monotonically."""
    for cond_id, results in results_by_condition.items():
        # Filter success=False per ConditionResult contract (result.py): the
        # runner drops failed results before persistence so the cached
        # parquet is uniformly well-formed (Condition C failures have
        # embedding=None which would later violate recovery's uniform-
        # embedding invariant).
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

    print(f"Pipeline validation against Ollama (strict={args.strict})")
    print(f"Output: {output_dir}")
    print(f"Model: {args.llm_model}")
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
    loader = MedQAQuestionLoader(split=args.split)
    all_records = list(islice(loader.iter_records(), args.n_questions))
    pending_records = [r for r in all_records if r.question_id not in processed_ids]
    print(f"Records: {len(all_records)} total, {len(pending_records)} pending")
    if not pending_records and not args.resume:
        print("No pending records and not resuming; nothing to do.")
        return 0

    # ---- 2. Build conditions ----
    embedder = _build_embedder(
        args.embedder_backend, args.embedder_model, args.embedder_prefix
    )
    llm = OllamaLLMAdapter(
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

    # On resume: load existing cached trajectories from each condition's
    # cache directory; reconstruct in-memory results_by_condition by
    # reading partial_results.json for the per-question metadata
    # (predicted_answer, deferral_signal, etc.). Reconstructing
    # predicted_answer from the trajectory's distribution-argmax would
    # be wrong for Condition C — the CoT-final-answer and the distribution-
    # argmax can differ (per F7), and the original ConditionResult
    # captured the CoT-final-answer. partial_results.json is the source
    # of truth for these fields.
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
                # NaN was serialized as None — restore.
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
        progress = pending_records  # fallback if tqdm not installed

    source_dataset = "GBaker/MedQA-USMLE-4-options"
    for i, record in enumerate(progress):
        for cond_id, cond in [("A", cond_a), ("B", cond_b), ("C", cond_c)]:
            result = cond.run(record)
            outcome = extractor.extract(record)
            traj = with_outcome(result.trajectory, outcome)
            results_by_condition[cond_id].append((result, traj))
        processed_ids.add(record.question_id)

        # Checkpoint every N questions
        if (i + 1) % args.checkpoint_every == 0:
            _save_checkpoint(output_dir, processed_ids)
            _save_partial_results(output_dir, results_by_condition)
            _checkpoint_trajectories(
                output_dir, results_by_condition, source_dataset
            )

    # Final checkpoint
    _save_checkpoint(output_dir, processed_ids)
    _save_partial_results(output_dir, results_by_condition)
    _checkpoint_trajectories(output_dir, results_by_condition, source_dataset)
    print(f"All {len(pending_records)} records processed.")

    # ---- 4. Verify cached-trajectories round-trip ----
    # Final checkpoint already wrote condition_c_cached; just verify
    # we can read back what we wrote. The cache is filtered to
    # success=True per the ConditionResult contract, so compare against
    # the same filtered set.
    c_trajectories = [traj for r, traj in results_by_condition["C"] if r.success]
    cached_path = _condition_cache_path(output_dir, "C")
    reloaded = MedQAPrerecoveredTrajectorySource(cached_path).load_all()
    if args.strict:
        assert len(reloaded) == len(c_trajectories), "round-trip count mismatch"
    print(f"Cached trajectories round-trip: {len(reloaded)} OK")

    # ---- 5. Recovery ----
    # success=False trajectories carry embedding=None (failure-result
    # construction in condition_c.py); recovery's uniform-embedding
    # invariant requires they're filtered out.
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

    # ---- 7. Compute signatures ----
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

    # Persist signature scores
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

    # ---- 8. condition_comparison ----
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
        {"A": a_compare, "B": b_compare, "C": c_compare},
        ground_truth,
        target_column="needs_consultation",
        compute_ci=False,
    )
    print("\nCondition comparison:")
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

    # ---- 10. Repair-rate summary ----
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
        repair_summary[cond_id] = {
            "n_results": n_results,
            "n_failures": n_failures,
            "mean_repair_attempts": (
                float(np.mean(repair_attempts)) if repair_attempts else 0.0
            ),
            "n_confidence_parsed": sum(1 for c in confidence_parsed if c is True),
            "n_confidence_unparsed": sum(1 for c in confidence_parsed if c is False),
        }
    print(json.dumps(repair_summary, indent=2))

    with (output_dir / "repair_summary.json").open("w") as fp:
        json.dump(repair_summary, fp, indent=2)

    print(f"\nPipeline validation complete. Artifacts in {output_dir}")
    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-questions", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--embedder-backend",
        choices=["mock", "sentence-transformers"],
        default="mock",
    )
    parser.add_argument(
        "--embedder-model",
        default=None,
        help="(only used when --embedder-backend=sentence-transformers; "
        "defaults to intfloat/multilingual-e5-large)",
    )
    parser.add_argument(
        "--embedder-prefix",
        default="",
        help="Prefix prepended to every text before embedding "
        "(e.g., 'passage: ' or 'query: ' for e5 models). Default empty "
        "per OQ1 stage-3.5b decision; the prefix question is deferred "
        "to stage 4a as an experiment-design choice.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=50)
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
