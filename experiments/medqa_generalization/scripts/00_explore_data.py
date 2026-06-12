#!/usr/bin/env python
"""Exploration: examine the MedQA-USMLE-4-options dataset.

Numbered 00 to mark as pre-pipeline exploration; never imported by
other scripts. Use as a template for similar dataset exploration on
new sources (MMLU professional subjects, future dataset versions,
etc.).

Surfaces: schema, sizes, class balance, question length stats, sanity
check that ``answer`` text matches ``options[answer_idx]``.

Promoted from inline ``uv run python <<EOF`` exploration done during
stage 3.2 pre-design (see
``docs/decisions/stage_3_2_pre_design_notes.md`` for the original
findings).

Usage:
    python 00_explore_data.py
    python 00_explore_data.py --split train
    python 00_explore_data.py --dataset-name OtherOrg/SomeMCQDataset
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter

from datasets import load_dataset


def main(dataset_name: str, split: str) -> int:
    print(f"Loading {dataset_name} (split={split})...")
    ds = load_dataset(dataset_name, split=split)
    print()

    print("=== Splits ===")
    full = load_dataset(dataset_name)
    for split_name in full:
        print(f"  {split_name}: {len(full[split_name])}")
    print()

    print("=== Schema (test split) ===")
    print(full["test"].features if "test" in full else ds.features)
    print()

    print("=== First example (truncated) ===")
    ex = ds[0]
    for k, v in ex.items():
        s = repr(v)
        print(f"  {k!r}: {type(v).__name__} = {s[:200]}{'...' if len(s) > 200 else ''}")
    print()

    print(f"=== Choice-count histogram ({split}) ===")
    if "options" in ex:
        counts = Counter(len(row["options"]) for row in ds)
        print(f"  {dict(counts)}")
    print()

    if "meta_info" in ex:
        print(f"=== meta_info distribution ({split}) ===")
        meta_counts = Counter(row["meta_info"] for row in ds)
        print(f"  {dict(meta_counts)}")
        print()

    if "answer_idx" in ex:
        print(f"=== answer_idx distribution ({split}) ===")
        ans_counts = Counter(row["answer_idx"] for row in ds)
        print(f"  {dict(ans_counts)}")
        print()

    if "question" in ex:
        print(f"=== Question length stats (chars, {split}) ===")
        lens = [len(row["question"]) for row in ds]
        print(
            f"  min={min(lens)}, max={max(lens)}, "
            f"mean={statistics.mean(lens):.0f}, "
            f"median={statistics.median(lens):.0f}"
        )
        print()

    if "answer" in ex and "options" in ex and "answer_idx" in ex:
        print(f"=== answer-vs-options[answer_idx] consistency ({split}) ===")
        mismatches = sum(
            1 for r in ds if r["answer"] != r["options"][r["answer_idx"]]
        )
        print(f"  mismatches: {mismatches} / {len(ds)}")
        print()

    return 0


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset-name",
        default="GBaker/MedQA-USMLE-4-options",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to inspect (e.g. train, test)",
    )
    args = parser.parse_args()
    sys.exit(main(args.dataset_name, args.split))


if __name__ == "__main__":
    cli()
