"""TIRust native API; upstream names are aliases of their owner stub types."""
from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from ._types.symbolica.core import ComplexFloat as ComplexFloat, E as E, Expression as Expression, Float as Float, S as S
from ._types.symbolica.community.hepkit import FeynmanDiagram as FeynmanDiagram, Model as Model
from ._types.symbolica.community.tensor import TensorExpression as TensorExpression

__version__: str
Number: TypeAlias = int | float | complex | str | Expression | Float | ComplexFloat | tuple["Number", "Number"]
NumericInput: TypeAlias = Number | "Dual"
DType: TypeAlias = Literal["f64", "double-double", "float"]
Algorithm: TypeAlias = Literal["hybrid", "pv", "ibp"]
Engine: TypeAlias = Literal["direct", "mass_taylor", "legacy_noop", "generic", "six_point", "six_point_hybrid", "certified_tensor", "affine_power", "seven_point"]
Convention: TypeAlias = Literal["iregi", "oneloop", "madloop"]
Shape: TypeAlias = Sequence[Sequence[int]]
Monomial: TypeAlias = tuple[int, int, int, int]
Status: TypeAlias = Literal["success", "invalid_input", "master_failure", "unstable"]
OutputNumber: TypeAlias = complex | ComplexFloat | "Dual"
Laurent: TypeAlias = tuple[OutputNumber, OutputNumber, OutputNumber, OutputNumber]

GenericEngineExecution = TypedDict("GenericEngineExecution", {
    "engine": Literal["generic"],
    "execution": dict[str, int],
    "certificate_execution": dict[str, int],
})

PackedEngineExecution = TypedDict("PackedEngineExecution", {
    "engine": Literal["six_point", "seven_point"],
    "parent_input_rows": int,
    "child_batch_calls": int,
    "packed_child_rows": int,
    "parent_batch_calls": int,
    "packed_parent_rows": int,
    "child_execution": dict[str, int],
    "parent_execution": dict[str, int],
    "child_certificate_execution": dict[str, int],
    "parent_certificate_execution": dict[str, int],
})

HybridEngineExecution = TypedDict("HybridEngineExecution", {
    "engine": Literal["six_point_hybrid"],
    "primary": PackedEngineExecution,
    "fallback": dict[str, int],
    "fallback_certificate_execution": dict[str, int],
    "fallback_rows": int,
    "fallback_successes": int,
    "fallback_batches": int,
    "fallback_preparation_attempts": int,
    "fallback_preparation_error": str | None,
})

CertifiedEngineExecution = TypedDict("CertifiedEngineExecution", {
    "engine": Literal["certified_tensor"],
    "execution": dict[str, int],
    "certificate_execution": dict[Literal["cartesian", "gram"], dict[str, int]],
})

AffineChildMethod = TypedDict("AffineChildMethod", {
    "propagators": int, "powers": list[int], "requested": Algorithm,
    "actual": Algorithm | None, "rows": int, "notes": list[str],
})

AffinePowerEngineExecution = TypedDict("AffinePowerEngineExecution", {
    "engine": Literal["affine_power"], "input_rows": int,
    "certificate_execution": dict[str, int], "profiles": int,
    "dag_nodes": int, "dag_edges": int, "packed_child_rows": int,
    "child_batch_calls": int, "parent_batch_calls": int, "parent_rows": int,
    "quadrature_work_rows": int, "child_execution": dict[str, int],
    "parent_execution": dict[str, int], "child_methods": list[AffineChildMethod],
})

MassTaylorEngineExecution = TypedDict("MassTaylorEngineExecution", {
    "engine": Literal["mass_taylor"], "input_rows": int,
    "certificate_execution": dict[str, int],
    "user_components": int, "internal_components": int,
    "unit_execution": dict[str, int], "projection_execution": dict[str, int],
    "unit_algorithm_used": Algorithm, "requested_algorithm": Algorithm,
})

LegacyNoOpEngineExecution = TypedDict("LegacyNoOpEngineExecution", {
    "engine": Literal["legacy_noop"], "input_rows": int,
    "accepted_rows": int, "rejected_rows": int, "query_count": int,
    "certificate_execution": dict[str, int], "child": "EngineExecution | None",
    "child_preparation_attempts": int, "child_preparation_error": str | None,
})

DirectRescueTier = TypedDict("DirectRescueTier", {"bits": int, "rows": int, "successes": int})

DirectEngineExecution = TypedDict("DirectEngineExecution", {
    "engine": Literal["direct"], "input_rows": int, "direct_successes": int,
    "fallback_rows": int, "fallback_successes": int,
    "precision_rescue": list[DirectRescueTier], "symbolic_rows": int,
    "rotation_rows": int, "rotation_failures": int, "checked_rows": int, "checked_successes": int,
    "fallback_preparation_error": str | None, "fallback": NotRequired["EngineExecution"],
})
EngineExecution: TypeAlias = DirectEngineExecution | MassTaylorEngineExecution | LegacyNoOpEngineExecution | GenericEngineExecution | PackedEngineExecution | HybridEngineExecution | CertifiedEngineExecution | AffinePowerEngineExecution

class ReductionError(RuntimeError): ...
class LaneError(ReductionError): ...

def initialize() -> None:
    """Initialize this provider. Repeated calls are safe."""
    ...

def configure_data_directory(directory: str | PathLike[str], *, validate: bool = True) -> None:
    """Select data for this standalone host; optionally validate every owner hash."""
    ...

class Dual:
    """Transport-only Symbolica Taylor coefficients, not an arithmetic engine."""
    def __new__(cls, shape: Shape, values: Sequence[Number]) -> Dual: ...
    @property
    def shape(self) -> list[list[int]]: ...
    @property
    def values(self) -> tuple[Any, ...]: ...
    @property
    def primal(self) -> Any: ...
    def __len__(self) -> int: ...

class ReductionPlan:
    """Immutable reduction structure; dimension_shift=h means D=4+2h-2ε."""
    def __new__(cls, propagators: int, rank: int, *, powers: Sequence[int] | None = None,
        dimension_shift: int = 0, algorithm: Algorithm = "hybrid",
        convention: Convention = "iregi", invariants: Literal["shifts", "scalar_products"] = "shifts",
        input_policy: Literal["preserve", "legacy_on_shell"] = "preserve",
        on_shell_threshold: str | None = None, tensor_weights: Literal["components", "multinomial"] = "components",
        check_scalar_products: bool = False) -> ReductionPlan: ...
    @property
    def propagators(self) -> int: ...
    @property
    def rank(self) -> int: ...
    @property
    def powers(self) -> list[int]: ...
    @property
    def dimension_shift(self) -> int: ...
    @property
    def parameter_count(self) -> int: ...
    @property
    def basis(self) -> list[Monomial]: ...
    @property
    def preparation_notes(self) -> list[str]: ...
    @property
    def input_layout(self) -> dict[str, Any]: ...
    def workspace(self, *, dtype: DType = "f64", precision: int | None = None,
        accuracy_bits: int | None = None, dual_shape: Shape | None = None,
        profiling: bool = False) -> Workspace: ...
    def pack(self, *, shifts: Sequence[Sequence[Sequence[NumericInput]]],
        masses_squared: Sequence[Sequence[NumericInput]], mu: Sequence[NumericInput],
        scalar_products: Sequence[Sequence[NumericInput]] | None = None) -> KinematicsBatch:
        """Pack time-first shifts, squared masses, and mu (not mu-squared)."""
        ...

class KinematicsBatch:
    """Immutable owned input references, created by ReductionPlan.pack."""
    @property
    def rows(self) -> int: ...
    def __len__(self) -> int: ...

class Workspace:
    """Owned scratch, created by plan.workspace; independent workspaces may run concurrently."""
    @property
    def dtype(self) -> DType: ...
    @property
    def precision(self) -> int: ...
    @property
    def dual_shape(self) -> list[list[int]] | None: ...
    @property
    def engine(self) -> Engine:
        """Actual owned workspace path, including capability fallbacks."""
        ...
    @property
    def preparation_notes(self) -> list[str]: ...
    def set_regulators(
        self,
        *,
        mu_uv2: float | None = None,
        mu_ir2: float | None = None,
        delta_uv: float = 0.0,
        delta_ir1: float = 0.0,
        delta_ir2: float = 0.0,
        collier_normalization: bool = False,
    ) -> None: ...
    def set_precision_rescue(self, enabled: bool) -> None:
        """Allow (default) or forbid rerunning refused float64 lanes at higher
        precision; with enabled=False every lane is computed in binary64 only and
        refused lanes are returned Unstable."""
    def precision_rescue(self) -> bool: ...
    def reduce_batch(self, batch: KinematicsBatch | Sequence[Sequence[NumericInput]]) -> BatchResult: ...
    def contract(self, result: BatchResult, *, monomials: Sequence[Monomial],
        coefficients: Sequence[Sequence[NumericInput]]) -> ContractionResult: ...

class BatchResult:
    @property
    def rows(self) -> int: ...
    @property
    def basis(self) -> list[Monomial]: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def precision(self) -> int: ...
    @property
    def dual_shape(self) -> list[list[int]] | None: ...
    @property
    def laurent_names(self) -> tuple[str, str, str, str]: ...
    @property
    def statuses(self) -> tuple[Status, ...]: ...
    @property
    def success(self) -> tuple[bool, ...]: ...
    @property
    def diagnostics(self) -> tuple[tuple[str, ...], ...]: ...
    @property
    def fallback_used(self) -> list[bool]: ...
    @property
    def analytic_used(self) -> list[bool]: ...
    @property
    def execution_stats(self) -> dict[str, int]: ...
    @property
    def stats(self) -> dict[str, int]: ...
    @property
    def engine_execution(self) -> EngineExecution:
        """Captured Rust counters; packed and certificate work stays nested."""
        ...
    @property
    def profile(self) -> dict[str, float] | None: ...
    def lane(self, row: int) -> tuple[Laurent, ...]:
        """Return complete moments; raise LaneError for failed lanes."""
        ...
    def lanes(self) -> tuple[tuple[Laurent, ...] | None, ...]: ...
    def __len__(self) -> int: ...

class ContractionResult:
    @property
    def rows(self) -> int: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def precision(self) -> int: ...
    @property
    def dual_shape(self) -> list[list[int]] | None: ...
    @property
    def laurent_names(self) -> tuple[str, str, str, str]: ...
    @property
    def statuses(self) -> tuple[Status, ...]: ...
    @property
    def success(self) -> tuple[bool, ...]: ...
    @property
    def diagnostics(self) -> tuple[tuple[str, ...], ...]: ...
    @property
    def fallback_used(self) -> list[bool]: ...
    @property
    def analytic_used(self) -> list[bool]: ...
    def lane(self, row: int) -> Laurent:
        """Return native Laurent slots; raise LaneError for failed lanes."""
        ...
    def lanes(self) -> tuple[Laurent | None, ...]: ...
    def __len__(self) -> int: ...

class PreparedNumerator:
    @staticmethod
    def from_expression(expression: Expression, variables: Sequence[Expression]) -> PreparedNumerator: ...
    @staticmethod
    def from_tensor(numerator: TensorExpression, variables: Sequence[Expression], *, component: Sequence[int] | None = None) -> PreparedNumerator: ...
    @property
    def rank(self) -> int: ...
    @property
    def expression(self) -> Expression: ...
    @property
    def variables(self) -> list[Expression]: ...
    @property
    def monomials(self) -> list[Monomial]: ...
    @property
    def terms(self) -> list[tuple[Monomial, Expression]]: ...
    def evaluate_batch(self, parameters: Sequence[Expression], rows: Sequence[Sequence[NumericInput]], *,
        dtype: DType = "f64", precision: int | None = None, dual_shape: Shape | None = None) -> tuple[tuple[OutputNumber, ...], ...]: ...

class PreparedGraph:
    @staticmethod
    def from_diagram(diagram: FeynmanDiagram, *, source: Literal["supplied", "feynman_rules"] = "supplied",
        external_wavefunctions: bool = False, component: Sequence[int] | None = None) -> PreparedGraph: ...
    @staticmethod
    def from_dot(model: Model, dot: str, *, source: Literal["supplied", "feynman_rules"] = "supplied",
        external_wavefunctions: bool = False, component: Sequence[int] | None = None) -> PreparedGraph: ...
    @property
    def name(self) -> str: ...
    @property
    def propagator_count(self) -> int: ...
    @property
    def diagram(self) -> FeynmanDiagram: ...
    @property
    def numerator(self) -> PreparedNumerator: ...
    @property
    def external_edges(self) -> list[int]: ...
    @property
    def propagators(self) -> list[dict[str, Any]]: ...
