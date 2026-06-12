"""bsig.core — pure-algorithmic core.

Domain-independent. Required dependencies only (numpy, scipy, networkx,
pandas, pyarrow, pydantic). Must not import from any other ``bsig``
layer; enforced by ``.importlinter`` and ``tests/test_architecture.py``.
"""
from __future__ import annotations

from bsig.core.graph import (
    SCHEMA_VERSION,
    AssemblyGraph,
    AssemblyGraphBuilder,
    BuilderConsumedError,
    DensityStats,
    Edge,
    EdgeClass,
    NodeAttributes,
)
from bsig.core.evaluation import (
    EvaluationError,
    EvaluationWarning,
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
from bsig.core.persistence import (
    MANIFEST_SCHEMA_VERSION,
    SIGNATURE_METADATA_SCHEMA_VERSION,
    ArtifactExistsError,
    ArtifactNotFoundError,
    FaissNotInstalledError,
    PersistenceError,
    SchemaVersionError,
    build_faiss_indices_from_visits,
    load_assembly_indices,
    load_embedding_indices,
    load_graph,
    load_signature_scores,
    load_signature_weights,
    load_visits,
    save_assembly_indices,
    save_embedding_indices,
    save_graph,
    save_signature_scores,
    save_visits,
)
from bsig.core.recovery import (
    RecoveryConfig,
    RecoveryResult,
    recover_assembly_graph,
)
from bsig.core.signature import (
    SignatureWeights,
    compute_signatures,
    distance_from_trajectory,
    entropy_plateau,
    voi_flatness,
)
from bsig.core.trajectory import Action, Outcome, State, Trajectory

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SIGNATURE_METADATA_SCHEMA_VERSION",
    "Action",
    "ArtifactExistsError",
    "ArtifactNotFoundError",
    "AssemblyGraph",
    "AssemblyGraphBuilder",
    "BuilderConsumedError",
    "DensityStats",
    "Edge",
    "EdgeClass",
    "EvaluationError",
    "EvaluationWarning",
    "FaissNotInstalledError",
    "NodeAttributes",
    "Outcome",
    "PathsConfig",
    "PersistenceError",
    "RecoveryConfig",
    "RecoveryResult",
    "SchemaVersionError",
    "SignatureWeights",
    "State",
    "Trajectory",
    "build_faiss_indices_from_visits",
    "calibration_metrics",
    "component_decomposition_table",
    "compute_assembly_indices",
    "compute_signatures",
    "deferral_auc",
    "deferral_curve",
    "distance_from_trajectory",
    "entropy_plateau",
    "load_assembly_indices",
    "load_embedding_indices",
    "load_graph",
    "load_signature_scores",
    "load_signature_weights",
    "load_visits",
    "operating_points",
    "recover_assembly_graph",
    "save_assembly_indices",
    "save_embedding_indices",
    "save_graph",
    "save_signature_scores",
    "save_visits",
    "voi_flatness",
]
