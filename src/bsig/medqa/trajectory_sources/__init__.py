"""MedQA / MMLU trajectory sources.

- ``MedQAQuestionLoader``: yields raw ``MedQARawRecord`` instances
  from HuggingFace ``GBaker/MedQA-USMLE-4-options``. Live-mode
  source. Stage 3.3 conditions consume these.
- ``MMLULoader``: yields ``MedQARawRecord`` instances from
  ``cais/mmlu`` filtered to a single subject. Used by stage-4b
  cross-benchmark replication. Reuses ``MedQARawRecord`` as the
  4-option MCQ output shape.
- ``MedQAPrerecoveredTrajectorySource``: satisfies ``TrajectorySource``
  against a cached-trajectories artifact directory. Reader for
  Condition C's cached output.
- ``serialization`` module: format-locked save/load functions for the
  cached-trajectories Parquet schema.
"""
from __future__ import annotations

from bsig.medqa.trajectory_sources.medqa import MedQAQuestionLoader
from bsig.medqa.trajectory_sources.mmlu import MMLULoader
from bsig.medqa.trajectory_sources.prerecovered import (
    MedQAPrerecoveredTrajectorySource,
)
from bsig.medqa.trajectory_sources.serialization import (
    CACHED_TRAJECTORIES_SCHEMA_VERSION,
    iter_cached_trajectories,
    load_cached_trajectories,
    save_cached_trajectories,
)

__all__ = [
    "CACHED_TRAJECTORIES_SCHEMA_VERSION",
    "MMLULoader",
    "MedQAPrerecoveredTrajectorySource",
    "MedQAQuestionLoader",
    "iter_cached_trajectories",
    "load_cached_trajectories",
    "save_cached_trajectories",
]
