"""bsig.medqa — MedQA-USMLE / MMLU domain pack.

Public surface (re-exported here parallel to ``bsig.core``):
- ``MedQARawRecord`` / ``MCQRawState`` — Pydantic raw types.
- ``MCQStateCanonicalizer`` / ``MCQCanonicalizationConfig``.
- ``AnswerKeyGroundTruthExtractor``.
- ``load_prompt`` / ``load_prompt_version`` / ``load_all_versions``.

Stages 3.2-3.5 add: trajectory sources (HuggingFace + cached),
decomposer, conditions A/B/C, MedQA-specific evaluation extensions,
and the smoke-test pipeline.
"""
from __future__ import annotations

from bsig.medqa._prompts import (
    PROMPT_NAMES,
    load_all_versions,
    load_prompt,
    load_prompt_version,
)
from bsig.medqa.canonicalization import (
    MCQActionCanonicalizationConfig,
    MCQActionCanonicalizer,
    MCQCanonicalizationConfig,
    MCQRawState,
    MCQStateCanonicalizer,
    MedQARawRecord,
    ReasoningStepRawAction,
)
from bsig.medqa.conditions import (
    NEUTRAL_DEFERRAL_SIGNAL,
    ConditionA,
    ConditionB,
    ConditionC,
    ConditionResult,
    Decomposer,
    DecomposerConfig,
    DecomposerError,
    DecomposerResult,
)
from bsig.medqa.evaluation import (
    condition_comparison,
    cross_domain_comparison,
    cross_llm_comparison,
    failure_mode_table,
    stratified_deferral_auc,
)
from bsig.medqa.ground_truth import AnswerKeyGroundTruthExtractor
from bsig.medqa.trajectory_sources import (
    CACHED_TRAJECTORIES_SCHEMA_VERSION,
    MedQAPrerecoveredTrajectorySource,
    MedQAQuestionLoader,
    MMLULoader,
    iter_cached_trajectories,
    load_cached_trajectories,
    save_cached_trajectories,
)

__all__ = [
    "AnswerKeyGroundTruthExtractor",
    "CACHED_TRAJECTORIES_SCHEMA_VERSION",
    "ConditionA",
    "ConditionB",
    "ConditionC",
    "ConditionResult",
    "Decomposer",
    "DecomposerConfig",
    "DecomposerError",
    "DecomposerResult",
    "MCQActionCanonicalizationConfig",
    "MCQActionCanonicalizer",
    "MCQCanonicalizationConfig",
    "MCQRawState",
    "MCQStateCanonicalizer",
    "MMLULoader",
    "MedQAPrerecoveredTrajectorySource",
    "MedQAQuestionLoader",
    "MedQARawRecord",
    "NEUTRAL_DEFERRAL_SIGNAL",
    "PROMPT_NAMES",
    "ReasoningStepRawAction",
    "condition_comparison",
    "cross_domain_comparison",
    "cross_llm_comparison",
    "failure_mode_table",
    "iter_cached_trajectories",
    "load_all_versions",
    "load_cached_trajectories",
    "load_prompt",
    "load_prompt_version",
    "save_cached_trajectories",
    "stratified_deferral_auc",
]
