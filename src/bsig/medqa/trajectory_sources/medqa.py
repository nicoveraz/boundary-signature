"""MedQA-USMLE 4-option dataset loader.

Source: ``GBaker/MedQA-USMLE-4-options`` on HuggingFace Hub. Per
ADR-0003, the originally-named ``bigbio/med_qa`` mirror is broken in
modern ``datasets`` versions; GBaker is the working Parquet-format
mirror.

``MedQAQuestionLoader`` does NOT implement ``TrajectorySource``. It
yields raw ``MedQARawRecord`` instances; trajectory construction
happens in stage 3.3 Conditions A/B/C, which wrap raw records with
LLM-driven reasoning loops.

The HuggingFace ``datasets`` package is lazy-imported per the FAISS
pattern from ``bsig.core.persistence``: import inside the loader
function with a clear ``ImportError`` hint if missing. Users not
running MedQA experiments don't pull in the ~200MB+ ``datasets``
package.

Discarded HF-row fields:
- ``metamap_phrases``: list of UMLS-extracted clinical phrases. The
  framework reasons from question text via the LLM and embedder
  directly, not from pre-extracted concepts.
- ``answer``: full text of the correct option. Redundant with
  ``answer_idx`` lookup against ``options``; the framework uses the
  letter, not the text.

Question ID synthesis: ``f"medqa-{split}-{idx}"`` where ``idx`` is
the row's position in the HuggingFace ``DatasetDict[split]``. Stable
across loads of the same dataset version. If the upstream dataset is
re-versioned and reorders, IDs would shift — that's a known risk the
loader cannot mitigate. Pin ``revision=...`` at H100-run time per
ADR-0003 / Q2 if reproducibility across HF-side re-versions matters.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

from bsig.medqa.canonicalization.state import MedQARawRecord


class MedQAQuestionLoader:
    """Yields ``MedQARawRecord`` instances from
    ``GBaker/MedQA-USMLE-4-options``. Single-shot iterator (re-
    instantiate to re-iterate).
    """

    def __init__(
        self,
        split: Literal["train", "test"] = "test",
        dataset_name: str = "GBaker/MedQA-USMLE-4-options",
    ) -> None:
        self._split = split
        self._dataset_name = dataset_name

    def iter_records(self) -> Iterator[MedQARawRecord]:
        ds = self._load()
        for idx, row in enumerate(ds):
            yield MedQARawRecord(
                question_id=f"medqa-{self._split}-{idx}",
                question=row["question"],
                choices=dict(row["options"]),
                answer_letter=row["answer_idx"],
                usmle_step=row.get("meta_info"),
            )

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            "loader_name": "MedQAQuestionLoader",
            "loader_version": "1",
            "dataset_name": self._dataset_name,
            "split": self._split,
        }

    def _load(self) -> Any:
        try:
            from datasets import load_dataset  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "MedQAQuestionLoader requires the 'datasets' package. "
                "Install with: uv pip install -e '.[medqa]'"
            ) from exc
        return load_dataset(self._dataset_name, split=self._split)
