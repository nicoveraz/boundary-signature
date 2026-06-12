"""bsig — boundary-aware reasoning framework.

Public API surface. Re-exports from layers ``core`` and ``adapters``.
The MedQA domain pack (``bsig.medqa``) and reference implementations
(``bsig.reference``) are imported directly from their submodules to keep
the top-level namespace dependency-light.
"""
from __future__ import annotations

from bsig.adapters.base import (
    AdapterError,
    AdapterMetadata,
    CanonicalizationError,
    LLMAdapterError,
)
from bsig.adapters.canonicalizer import StateCanonicalizer
from bsig.adapters.embedding import EmbeddingSource
from bsig.adapters.ground_truth import GroundTruthExtractor
from bsig.adapters.llm import LLMAdapter
from bsig.adapters.trajectory_source import TrajectorySource
from bsig.core.graph import (
    AssemblyGraph,
    AssemblyGraphBuilder,
    BuilderConsumedError,
    EdgeClass,
)
from bsig.core.calibration import (
    CalibrationResult,
    apply_threshold,
    roc_threshold_table,
    threshold_at_fpr,
    threshold_at_sensitivity,
)
from bsig.core.evaluation import (
    EvaluationError,
    calibration_metrics,
    component_decomposition_table,
    deferral_auc,
    deferral_curve,
    operating_points,
)
from bsig.core.paths import (
    PathsConfig,
    compute_assembly_indices,
)
from bsig.core.recovery import (
    RecoveryConfig,
    RecoveryResult,
    recover_assembly_graph,
)
from bsig.core.signature import (
    SignatureWeights,
    compute_signatures,
    mean_entropy,
)
from bsig.core.trajectory import Action, Outcome, State, Trajectory

__version__ = "0.1.0"

__all__ = [
    "Action",
    "AdapterError",
    "AdapterMetadata",
    "AssemblyGraph",
    "AssemblyGraphBuilder",
    "BuilderConsumedError",
    "CalibrationResult",
    "CanonicalizationError",
    "EdgeClass",
    "EmbeddingSource",
    "EvaluationError",
    "GroundTruthExtractor",
    "LLMAdapter",
    "LLMAdapterError",
    "Outcome",
    "PathsConfig",
    "RecoveryConfig",
    "RecoveryResult",
    "SignatureWeights",
    "State",
    "StateCanonicalizer",
    "Trajectory",
    "TrajectorySource",
    "apply_threshold",
    "calibration_metrics",
    "component_decomposition_table",
    "compute_assembly_indices",
    "compute_signatures",
    "deferral_auc",
    "deferral_curve",
    "mean_entropy",
    "operating_points",
    "recover_assembly_graph",
    "roc_threshold_table",
    "threshold_at_fpr",
    "threshold_at_sensitivity",
]
