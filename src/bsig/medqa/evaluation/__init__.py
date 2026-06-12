"""MedQA-specific evaluation extensions building on bsig.core.evaluation.

Five public functions:
- ``stratified_deferral_auc``: per-stratum AUC (e.g., usmle_step).
- ``cross_llm_comparison``: per-LLM AUC on the same dataset.
- ``cross_domain_comparison``: per-domain AUC with separate ground truths.
- ``condition_comparison``: per-condition AUC; THE methods-paper-headline
  analysis. Supports subset_mask for the eventual ADR-0006 gate-metric
  revision.
- ``failure_mode_table``: per-question diagnostic for inspecting the
  highest-signature trajectories.
"""
from __future__ import annotations

from bsig.medqa.evaluation.cross_comparison import (
    condition_comparison,
    cross_domain_comparison,
    cross_llm_comparison,
)
from bsig.medqa.evaluation.diagnostic import failure_mode_table
from bsig.medqa.evaluation.stratified import stratified_deferral_auc

__all__ = [
    "condition_comparison",
    "cross_domain_comparison",
    "cross_llm_comparison",
    "failure_mode_table",
    "stratified_deferral_auc",
]
