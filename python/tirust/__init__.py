"""TIRust: batched tensor reduction with one shared Symbolica native kernel.

Import Expression, Float, Model and TensorExpression from this package when using
the standalone wheel. Objects from a separately compiled Symbolica Community
wheel have a different native identity and must not be mixed with this host.
"""

from pathlib import Path as _Path

from ._tirust import (
    BatchResult,
    ComplexFloat,
    ContractionResult,
    Dual,
    E,
    Expression,
    FeynmanDiagram,
    Float,
    KinematicsBatch,
    LaneError,
    Model,
    PreparedGraph,
    PreparedNumerator,
    ReductionError,
    ReductionPlan,
    S,
    TensorExpression,
    Workspace,
    __version__,
    configure_data_directory,
    initialize,
)

# Only this standalone facade selects package assets. A Community host links the
# same Rust provider without this facade and retains its own data configuration.
configure_data_directory(_Path(__file__).with_name("data"), validate=False)
del _Path

__all__ = [
    "BatchResult", "ComplexFloat", "ContractionResult", "Dual", "E",
    "Expression", "FeynmanDiagram", "Float", "KinematicsBatch", "LaneError",
    "Model", "PreparedGraph", "PreparedNumerator", "ReductionError",
    "ReductionPlan", "S", "TensorExpression", "Workspace", "__version__",
    "initialize", "configure_data_directory",
]
