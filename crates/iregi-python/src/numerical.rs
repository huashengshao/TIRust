//! Owned batched workspaces, inputs and failure-aware results.

use std::{collections::HashMap, sync::Arc};

use iregi_rust::{
    Algorithm, BatchResult, CertificateExecutionStats, ContractionResult, ContractionWorkspace,
    Convention, ExecutionProfile, ExecutionStats, InputPolicy, InvariantInput, KinematicsBatch,
    LaneStatus, NumeratorBatch, ReductionSettings, TensorEngineExecution, TensorEngineKind,
    TensorReductionPlan, TensorWeights, TensorWorkspace, TensorWorkspaceOptions, basis::Monomial,
};
use pyo3::{
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use symbolica::{
    domains::float::{Complex, DoubleFloat, Float},
    evaluate::EvaluationDomain,
};

use crate::{
    reduction_error,
    scalar::{PythonNumber, checked_shape, components, objects, output_components, rows},
};

const LAURENT: [&str; 4] = ["uv_single", "ir_single", "ir_double", "finite"];

#[pyclass(
    frozen,
    name = "ReductionPlan",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyReductionPlan {
    pub(crate) inner: Arc<TensorReductionPlan>,
}

#[pymethods]
impl PyReductionPlan {
    /// Prepare immutable recurrence structure. dimension_shift is h in D=4+2h-2ε.
    #[new]
    #[pyo3(signature = (propagators, rank, *, powers=None, dimension_shift=0, algorithm="hybrid", convention="iregi", invariants="shifts", input_policy="preserve", on_shell_threshold=None, tensor_weights="components", check_scalar_products=false))]
    fn new(
        propagators: usize,
        rank: u16,
        powers: Option<Vec<u16>>,
        dimension_shift: i16,
        algorithm: &str,
        convention: &str,
        invariants: &str,
        input_policy: &str,
        on_shell_threshold: Option<String>,
        tensor_weights: &str,
        check_scalar_products: bool,
    ) -> PyResult<Self> {
        let mut settings = ReductionSettings::new(propagators, rank);
        if let Some(powers) = powers {
            settings.powers = powers;
        }
        settings.dimension_shift = dimension_shift;
        settings.algorithm = match algorithm {
            "hybrid" => Algorithm::Hybrid,
            "ibp" => Algorithm::Ibp,
            "pv" => Algorithm::PassarinoVeltman,
            _ => {
                return Err(PyValueError::new_err(
                    "algorithm must be 'hybrid', 'pv', or 'ibp'",
                ));
            }
        };
        settings.convention = match convention {
            "iregi" => Convention::IregiNative,
            "oneloop" => Convention::OneLoop,
            "madloop" => Convention::MadLoop,
            _ => {
                return Err(PyValueError::new_err(
                    "convention must be 'iregi', 'oneloop', or 'madloop'",
                ));
            }
        };
        settings.invariants = match invariants {
            "shifts" => InvariantInput::FromShifts,
            "scalar_products" => InvariantInput::SuppliedScalarProducts,
            _ => {
                return Err(PyValueError::new_err(
                    "invariants must be 'shifts' or 'scalar_products'",
                ));
            }
        };
        settings.input_policy = match input_policy {
            "preserve" if on_shell_threshold.is_none() => InputPolicy::Preserve,
            "legacy_on_shell" => on_shell_threshold
                .map_or_else(InputPolicy::legacy_onshell, |threshold| {
                    InputPolicy::LegacyOnShell { threshold }
                }),
            "preserve" => {
                return Err(PyValueError::new_err(
                    "on_shell_threshold requires input_policy='legacy_on_shell'",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "input_policy must be 'preserve' or 'legacy_on_shell'",
                ));
            }
        };
        settings.tensor_weights = match tensor_weights {
            "components" => TensorWeights::Components,
            "multinomial" => TensorWeights::Multinomial,
            _ => {
                return Err(PyValueError::new_err(
                    "tensor_weights must be 'components' or 'multinomial'",
                ));
            }
        };
        let plan = if check_scalar_products {
            if settings.invariants != InvariantInput::SuppliedScalarProducts {
                return Err(PyValueError::new_err(
                    "check_scalar_products requires invariants='scalar_products'",
                ));
            }
            TensorReductionPlan::new_checked_cartesian(settings)
        } else {
            TensorReductionPlan::new(settings)
        }
        .map_err(reduction_error)?;
        Ok(Self {
            inner: Arc::new(plan),
        })
    }

    #[getter]
    fn propagators(&self) -> usize {
        self.inner.settings().propagators
    }
    #[getter]
    fn rank(&self) -> u16 {
        self.inner.settings().rank
    }
    #[getter]
    fn powers(&self) -> Vec<u16> {
        self.inner.settings().powers.clone()
    }
    #[getter]
    fn dimension_shift(&self) -> i16 {
        self.inner.settings().dimension_shift
    }
    #[getter]
    fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }
    #[getter]
    fn basis(&self) -> Vec<(u16, u16, u16, u16)> {
        basis(self.inner.basis())
    }
    #[getter]
    fn preparation_notes(&self) -> Vec<String> {
        self.inner.preparation_notes().to_vec()
    }

    #[getter]
    fn input_layout(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let layout = self.inner.input_layout();
        let output = PyDict::new(py);
        output.set_item("shifts", (layout.shifts.start, layout.shifts.end))?;
        output.set_item(
            "scalar_products",
            layout
                .scalar_products
                .as_ref()
                .map(|range| (range.start, range.end)),
        )?;
        output.set_item("masses_squared", (layout.masses.start, layout.masses.end))?;
        output.set_item("mu", layout.scale)?;
        output.set_item("metric", "+---")?;
        output.set_item("scale_is_squared", false)?;
        Ok(output.unbind())
    }

    /// Construct owned, reusable numeric scratch. Precision and sample axes are distinct.
    #[pyo3(signature = (*, dtype="f64", precision=None, accuracy_bits=None, dual_shape=None, profiling=false))]
    fn workspace(
        &self,
        dtype: &str,
        precision: Option<u32>,
        accuracy_bits: Option<u32>,
        dual_shape: Option<Vec<Vec<usize>>>,
        profiling: bool,
    ) -> PyResult<PyWorkspace> {
        PyWorkspace::create(
            Arc::clone(&self.inner),
            dtype,
            precision,
            accuracy_bits,
            dual_shape,
            profiling,
        )
    }

    /// Pack shifts[batch][propagator][4], squared masses[batch][propagator], mu[batch].
    #[pyo3(signature = (*, shifts, masses_squared, mu, scalar_products=None))]
    fn pack(
        &self,
        py: Python<'_>,
        shifts: &Bound<'_, PyAny>,
        masses_squared: &Bound<'_, PyAny>,
        mu: &Bound<'_, PyAny>,
        scalar_products: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyKinematicsBatch> {
        let shift_rows = objects(shifts)?;
        let mass_rows = rows(masses_squared)?;
        let scales = objects(mu)?;
        let product_rows = scalar_products.map(rows).transpose()?;
        let count = shift_rows.len();
        if mass_rows.len() != count
            || scales.len() != count
            || product_rows
                .as_ref()
                .is_some_and(|rows| rows.len() != count)
        {
            return Err(PyValueError::new_err(
                "shifts, masses_squared, mu and scalar_products must have equal batch length",
            ));
        }
        let n = self.inner.settings().propagators;
        let layout = self.inner.input_layout();
        if layout.scalar_products.is_some() != product_rows.is_some() {
            return Err(PyValueError::new_err(
                "scalar_products must be supplied exactly when the plan uses invariants='scalar_products'",
            ));
        }
        let mut output = Vec::with_capacity(count);
        for lane in 0..count {
            let propagators = rows(shift_rows[lane].bind(py))?;
            if propagators.len() != n
                || propagators.iter().any(|shift| shift.len() != 4)
                || mass_rows[lane].len() != n
            {
                return Err(PyValueError::new_err(format!(
                    "lane {lane}: expected {n} four-component shifts and {n} squared masses"
                )));
            }
            let mut packed: Vec<_> = propagators.into_iter().flatten().collect();
            if let Some(products) = &product_rows {
                let expected = layout.scalar_products.as_ref().unwrap().len();
                if products[lane].len() != expected {
                    return Err(PyValueError::new_err(format!(
                        "lane {lane}: expected {expected} packed scalar products"
                    )));
                }
                packed.extend(products[lane].iter().map(|value| value.clone_ref(py)));
            }
            packed.extend(mass_rows[lane].iter().map(|value| value.clone_ref(py)));
            packed.push(scales[lane].clone_ref(py));
            output.push(packed);
        }
        Ok(PyKinematicsBatch {
            plan: Arc::clone(&self.inner),
            rows: output,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ReductionPlan(propagators={}, rank={}, powers={:?}, dimension_shift={})",
            self.propagators(),
            self.rank(),
            self.powers(),
            self.dimension_shift()
        )
    }
}

#[pyclass(
    frozen,
    name = "KinematicsBatch",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyKinematicsBatch {
    plan: Arc<TensorReductionPlan>,
    rows: Vec<Vec<Py<PyAny>>>,
}

#[pymethods]
impl PyKinematicsBatch {
    #[getter]
    fn rows(&self) -> usize {
        self.rows.len()
    }
    fn __len__(&self) -> usize {
        self.rows.len()
    }
    fn __repr__(&self) -> String {
        format!(
            "KinematicsBatch(rows={}, parameters={}, scale=mu)",
            self.rows.len(),
            self.plan.parameter_count()
        )
    }
}

struct Engine<T: PythonNumber> {
    prototype: T,
    workspace: TensorWorkspace<T>,
    contractions: HashMap<Vec<[u16; 4]>, ContractionWorkspace<T>>,
}

impl<T: PythonNumber> Engine<T>
where
    Complex<T>: EvaluationDomain,
{
    fn new(
        plan: &TensorReductionPlan,
        prototype: T,
        shape: Option<&[Vec<usize>]>,
        accuracy: Option<u32>,
        profiling: bool,
    ) -> PyResult<Self> {
        let options = TensorWorkspaceOptions {
            profiling,
            ..Default::default()
        };
        let mut workspace = if let Some(shape) = shape {
            TensorWorkspace::new_dual_with_options(
                plan,
                prototype.clone(),
                checked_shape(shape)?,
                options,
            )
            .map_err(reduction_error)?
        } else {
            TensorWorkspace::new_with_options(plan, prototype.clone(), options)
                .map_err(reduction_error)?
        };
        if let Some(bits) = accuracy {
            workspace
                .set_accuracy_precision(bits)
                .map_err(reduction_error)?;
        }
        Ok(Self {
            prototype,
            workspace,
            contractions: HashMap::new(),
        })
    }

    fn parse(
        &self,
        py: Python<'_>,
        input: &[Vec<Py<PyAny>>],
        parameters: usize,
        shape: Option<&[Vec<usize>]>,
    ) -> PyResult<KinematicsBatch<T>> {
        let components_per_argument = shape.map_or(1, <[Vec<usize>]>::len);
        let capacity = input
            .len()
            .checked_mul(parameters)
            .and_then(|size| size.checked_mul(components_per_argument))
            .ok_or_else(|| PyValueError::new_err("input batch dimensions overflow"))?;
        let mut values = Vec::with_capacity(capacity);
        for (lane, row) in input.iter().enumerate() {
            if row.len() != parameters {
                return Err(PyValueError::new_err(format!(
                    "lane {lane}: expected {parameters} arguments, received {}",
                    row.len()
                )));
            }
            for (argument, value) in row.iter().enumerate() {
                values.extend(components(value.bind(py), &self.prototype, shape).map_err(
                    |error| {
                        PyValueError::new_err(format!("lane {lane}, argument {argument}: {error}"))
                    },
                )?);
            }
        }
        Ok(KinematicsBatch {
            rows: input.len(),
            components: components_per_argument,
            values,
        })
    }
}

enum Workspaces {
    F64(Engine<f64>),
    Dd(Engine<DoubleFloat>),
    Float(Engine<Float>),
}
enum Results {
    F64(BatchResult<f64>),
    Dd(BatchResult<DoubleFloat>),
    Float(BatchResult<Float>),
}
enum Contracts {
    F64(ContractionResult<f64>),
    Dd(ContractionResult<DoubleFloat>),
    Float(ContractionResult<Float>),
}

#[pyclass(
    name = "Workspace",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyWorkspace {
    plan: Arc<TensorReductionPlan>,
    dtype: String,
    precision: u32,
    shape: Option<Vec<Vec<usize>>>,
    inner: Workspaces,
}

impl PyWorkspace {
    fn create(
        plan: Arc<TensorReductionPlan>,
        dtype: &str,
        precision: Option<u32>,
        accuracy: Option<u32>,
        shape: Option<Vec<Vec<usize>>>,
        profiling: bool,
    ) -> PyResult<Self> {
        if let Some(shape) = &shape {
            checked_shape(shape)?;
        }
        let (inner, bits) = match dtype {
            "f64" => {
                if precision.is_some_and(|bits| bits != 53) {
                    return Err(PyValueError::new_err(
                        "f64 arithmetic has 53 bits; omit precision or set 53",
                    ));
                }
                (
                    Workspaces::F64(Engine::new(
                        &plan,
                        0.,
                        shape.as_deref(),
                        accuracy,
                        profiling,
                    )?),
                    53,
                )
            }
            "double-double" => {
                if precision.is_some_and(|bits| bits != 106) {
                    return Err(PyValueError::new_err(
                        "double-double arithmetic has 106 bits; omit precision or set 106",
                    ));
                }
                (
                    Workspaces::Dd(Engine::new(
                        &plan,
                        DoubleFloat::from(0.),
                        shape.as_deref(),
                        accuracy,
                        profiling,
                    )?),
                    106,
                )
            }
            "float" => {
                let bits = precision.unwrap_or(128);
                // Fallible upstream parser validates the backend's precision domain.
                let prototype = Float::parse("0", Some(bits)).map_err(PyValueError::new_err)?;
                (
                    Workspaces::Float(Engine::new(
                        &plan,
                        prototype,
                        shape.as_deref(),
                        accuracy,
                        profiling,
                    )?),
                    bits,
                )
            }
            _ => {
                return Err(PyValueError::new_err(
                    "dtype must be 'f64', 'double-double', or 'float'",
                ));
            }
        };
        Ok(Self {
            plan,
            dtype: dtype.into(),
            precision: bits,
            shape,
            inner,
        })
    }
}

#[pymethods]
impl PyWorkspace {
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }
    #[getter]
    fn precision(&self) -> u32 {
        self.precision
    }
    #[getter]
    fn dual_shape(&self) -> Option<Vec<Vec<usize>>> {
        self.shape.clone()
    }

    /// Actual owned Rust workspace path, including generic capability fallbacks.
    #[getter]
    fn engine(&self) -> &'static str {
        match &self.inner {
            Workspaces::F64(engine) => engine_name(engine.workspace.engine_kind()),
            Workspaces::Dd(engine) => engine_name(engine.workspace.engine_kind()),
            Workspaces::Float(engine) => engine_name(engine.workspace.engine_kind()),
        }
    }

    #[getter]
    fn preparation_notes(&self) -> Vec<String> {
        match &self.inner {
            Workspaces::F64(engine) => engine.workspace.preparation_notes().to_vec(),
            Workspaces::Dd(engine) => engine.workspace.preparation_notes().to_vec(),
            Workspaces::Float(engine) => engine.workspace.preparation_notes().to_vec(),
        }
    }

    /// Evaluate one genuine Rust batch; failed lanes keep independent diagnostics.
    fn reduce_batch(
        &mut self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
    ) -> PyResult<PyBatchResult> {
        let owned;
        let input = if let Ok(batch) = batch.extract::<PyRef<'_, PyKinematicsBatch>>() {
            if !Arc::ptr_eq(&batch.plan, &self.plan) {
                return Err(PyValueError::new_err(
                    "packed batch belongs to a different ReductionPlan",
                ));
            }
            owned = batch
                .rows
                .iter()
                .map(|row| row.iter().map(|value| value.clone_ref(py)).collect())
                .collect::<Vec<Vec<_>>>();
            &owned
        } else {
            owned = rows(batch)?;
            &owned
        };
        let (result, execution, diagnostics, profile) = match &mut self.inner {
            Workspaces::F64(engine) => run(engine, input, &self.plan, self.shape.as_deref(), py)
                .map(|(result, execution, diagnostics, profile)| {
                    (Results::F64(result), execution, diagnostics, profile)
                })?,
            Workspaces::Dd(engine) => run(engine, input, &self.plan, self.shape.as_deref(), py)
                .map(|(result, execution, diagnostics, profile)| {
                    (Results::Dd(result), execution, diagnostics, profile)
                })?,
            Workspaces::Float(engine) => run(engine, input, &self.plan, self.shape.as_deref(), py)
                .map(|(result, execution, diagnostics, profile)| {
                    (Results::Float(result), execution, diagnostics, profile)
                })?,
        };
        let stats = execution.summary().map_err(reduction_error)?;
        py.check_signals()?;
        Ok(PyBatchResult {
            plan: Arc::clone(&self.plan),
            dtype: self.dtype.clone(),
            precision: self.precision,
            shape: self.shape.clone(),
            result,
            stats,
            execution,
            diagnostics,
            profile,
        })
    }

    /// Contract supplied numerical coefficients through Symbolica's batched core.
    #[pyo3(signature = (result, *, monomials, coefficients))]
    fn contract(
        &mut self,
        py: Python<'_>,
        result: PyRef<'_, PyBatchResult>,
        monomials: Vec<[u16; 4]>,
        coefficients: &Bound<'_, PyAny>,
    ) -> PyResult<PyContractionResult> {
        if !Arc::ptr_eq(&result.plan, &self.plan) {
            return Err(PyValueError::new_err(
                "result belongs to a different ReductionPlan",
            ));
        }
        if result.precision != self.precision || result.shape != self.shape {
            return Err(PyValueError::new_err(
                "result and workspace must use the same working precision and exact Dual layout",
            ));
        }
        let input = rows(coefficients)?;
        let contracted = match (&mut self.inner, &result.result) {
            (Workspaces::F64(engine), Results::F64(moments)) => Contracts::F64(contract(
                engine,
                moments,
                &self.plan,
                &monomials,
                &input,
                self.shape.as_deref(),
                py,
            )?),
            (Workspaces::Dd(engine), Results::Dd(moments)) => Contracts::Dd(contract(
                engine,
                moments,
                &self.plan,
                &monomials,
                &input,
                self.shape.as_deref(),
                py,
            )?),
            (Workspaces::Float(engine), Results::Float(moments)) => Contracts::Float(contract(
                engine,
                moments,
                &self.plan,
                &monomials,
                &input,
                self.shape.as_deref(),
                py,
            )?),
            _ => {
                return Err(PyTypeError::new_err(
                    "result and workspace must use the same numerical type",
                ));
            }
        };
        Ok(PyContractionResult {
            result: contracted,
            dtype: self.dtype.clone(),
            precision: self.precision,
            shape: self.shape.clone(),
            diagnostics: result.diagnostics.clone(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Workspace(dtype={:?}, precision={}, components={})",
            self.dtype,
            self.precision,
            self.shape.as_ref().map_or(1, Vec::len)
        )
    }
}

fn run<T: PythonNumber>(
    engine: &mut Engine<T>,
    input: &[Vec<Py<PyAny>>],
    plan: &TensorReductionPlan,
    shape: Option<&[Vec<usize>]>,
    py: Python<'_>,
) -> PyResult<(
    BatchResult<T>,
    TensorEngineExecution,
    Vec<Vec<String>>,
    Option<ExecutionProfile>,
)>
where
    Complex<T>: EvaluationDomain,
{
    let packed = engine.parse(py, input, plan.parameter_count(), shape)?;
    // Captures only Rust-owned, Send numeric data and the actual Rust workspace.
    // PyO3's Ungil bound verifies this; no unsafe Send implementations exist.
    let result = py
        .detach(|| engine.workspace.reduce_batch(&packed))
        .map_err(reduction_error)?;
    Ok((
        result,
        engine.workspace.engine_execution(),
        engine.workspace.last_diagnostics().to_vec(),
        engine.workspace.last_profile(),
    ))
}

fn contract<T: PythonNumber>(
    engine: &mut Engine<T>,
    moments: &BatchResult<T>,
    plan: &TensorReductionPlan,
    monomials: &[[u16; 4]],
    input: &[Vec<Py<PyAny>>],
    shape: Option<&[Vec<usize>]>,
    py: Python<'_>,
) -> PyResult<ContractionResult<T>>
where
    Complex<T>: EvaluationDomain,
{
    let packed = engine.parse(py, input, monomials.len(), shape)?;
    if packed.rows != moments.rows() {
        return Err(PyValueError::new_err(
            "coefficient batch and tensor result have different lane counts",
        ));
    }
    let support = monomials.to_vec();
    if !engine.contractions.contains_key(&support) {
        let contraction = plan
            .contraction_plan(&monomials.iter().copied().map(Monomial).collect::<Vec<_>>())
            .map_err(reduction_error)?;
        let workspace = if let Some(shape) = shape {
            ContractionWorkspace::new_dual(
                &contraction,
                engine.prototype.clone(),
                checked_shape(shape)?,
            )
            .map_err(reduction_error)?
        } else {
            ContractionWorkspace::new(&contraction, engine.prototype.clone())
                .map_err(reduction_error)?
        };
        engine.contractions.insert(support.clone(), workspace);
    }
    let workspace = engine.contractions.get_mut(&support).unwrap();
    let numerator = NumeratorBatch {
        rows: packed.rows,
        components: packed.components,
        values: packed.values,
    };
    py.detach(|| workspace.contract_batch(moments, &numerator))
        .map_err(reduction_error)
}

#[pyclass(
    frozen,
    name = "BatchResult",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyBatchResult {
    plan: Arc<TensorReductionPlan>,
    dtype: String,
    precision: u32,
    shape: Option<Vec<Vec<usize>>>,
    result: Results,
    stats: ExecutionStats,
    execution: TensorEngineExecution,
    diagnostics: Vec<Vec<String>>,
    profile: Option<ExecutionProfile>,
}

macro_rules! result_call {
    ($self:ident, $name:ident) => {
        match &$self.result {
            Results::F64(value) => value.$name(),
            Results::Dd(value) => value.$name(),
            Results::Float(value) => value.$name(),
        }
    };
}

impl PyBatchResult {
    fn statuses_ref(&self) -> &[LaneStatus] {
        match &self.result {
            Results::F64(value) => &value.statuses,
            Results::Dd(value) => &value.statuses,
            Results::Float(value) => &value.statuses,
        }
    }
}

#[pymethods]
impl PyBatchResult {
    #[getter]
    fn rows(&self) -> usize {
        result_call!(self, rows)
    }
    #[getter]
    fn basis(&self) -> Vec<(u16, u16, u16, u16)> {
        basis(self.plan.basis())
    }
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }
    #[getter]
    fn precision(&self) -> u32 {
        self.precision
    }
    #[getter]
    fn dual_shape(&self) -> Option<Vec<Vec<usize>>> {
        self.shape.clone()
    }
    #[getter]
    fn laurent_names(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        (LAURENT[0], LAURENT[1], LAURENT[2], LAURENT[3])
    }
    #[getter]
    fn statuses(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.statuses_ref().iter().map(status_name))?.unbind())
    }
    #[getter]
    fn success(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.statuses_ref()
                .iter()
                .map(|status| *status == LaneStatus::Success),
        )?
        .unbind())
    }
    #[getter]
    fn diagnostics(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        diagnostic_tuples(self.statuses_ref(), &self.diagnostics, py)
    }
    #[getter]
    fn fallback_used(&self) -> Vec<bool> {
        match &self.result {
            Results::F64(value) => value.fallback_used.clone(),
            Results::Dd(value) => value.fallback_used.clone(),
            Results::Float(value) => value.fallback_used.clone(),
        }
    }
    #[getter]
    fn analytic_used(&self) -> Vec<bool> {
        match &self.result {
            Results::F64(value) => value.analytic_used.clone(),
            Results::Dd(value) => value.analytic_used.clone(),
            Results::Float(value) => value.analytic_used.clone(),
        }
    }
    #[getter]
    fn execution_stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        stats(self.stats, py)
    }
    #[getter]
    fn stats(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        stats(self.stats, py)
    }
    /// Actual engine and nested work counters, captured for this batch.
    #[getter]
    fn engine_execution(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        engine_execution(&self.execution, py)
    }
    #[getter]
    fn profile(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(profile) = self.profile else {
            return Ok(None);
        };
        let output = PyDict::new(py);
        output.set_item("arguments_seconds", profile.arguments.as_secs_f64())?;
        output.set_item(
            "scalar_masters_seconds",
            profile.scalar_masters.as_secs_f64(),
        )?;
        output.set_item("reduction_seconds", profile.reduction.as_secs_f64())?;
        output.set_item("total_seconds", profile.total.as_secs_f64())?;
        Ok(Some(output.unbind()))
    }
    fn lane(&self, py: Python<'_>, row: usize) -> PyResult<Py<PyAny>> {
        check_lane(self.statuses_ref(), row)?;
        match &self.result {
            Results::F64(value) => tensor_lane(value, row, self.shape.as_deref(), py),
            Results::Dd(value) => tensor_lane(value, row, self.shape.as_deref(), py),
            Results::Float(value) => tensor_lane(value, row, self.shape.as_deref(), py),
        }
    }
    fn lanes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = (0..self.rows())
            .map(|row| {
                if self.statuses_ref()[row] == LaneStatus::Success {
                    self.lane(py, row)
                } else {
                    Ok(py.None())
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, values)?.into_any().unbind())
    }
    fn __len__(&self) -> usize {
        self.rows()
    }
    fn __repr__(&self) -> String {
        format!(
            "BatchResult(rows={}, success={}, moments={}, dtype={:?})",
            self.rows(),
            self.statuses_ref()
                .iter()
                .filter(|status| **status == LaneStatus::Success)
                .count(),
            self.plan.basis().len(),
            self.dtype
        )
    }
}

#[pyclass(
    frozen,
    name = "ContractionResult",
    module = "symbolica.community.hepkit.tensor_integrals"
)]
pub struct PyContractionResult {
    result: Contracts,
    dtype: String,
    precision: u32,
    shape: Option<Vec<Vec<usize>>>,
    diagnostics: Vec<Vec<String>>,
}

impl PyContractionResult {
    fn statuses_ref(&self) -> &[LaneStatus] {
        match &self.result {
            Contracts::F64(value) => &value.statuses,
            Contracts::Dd(value) => &value.statuses,
            Contracts::Float(value) => &value.statuses,
        }
    }
}

#[pymethods]
impl PyContractionResult {
    #[getter]
    fn rows(&self) -> usize {
        self.statuses_ref().len()
    }
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }
    #[getter]
    fn precision(&self) -> u32 {
        self.precision
    }
    #[getter]
    fn dual_shape(&self) -> Option<Vec<Vec<usize>>> {
        self.shape.clone()
    }
    #[getter]
    fn laurent_names(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        (LAURENT[0], LAURENT[1], LAURENT[2], LAURENT[3])
    }
    #[getter]
    fn statuses(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, self.statuses_ref().iter().map(status_name))?.unbind())
    }
    #[getter]
    fn success(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(
            py,
            self.statuses_ref()
                .iter()
                .map(|status| *status == LaneStatus::Success),
        )?
        .unbind())
    }
    #[getter]
    fn diagnostics(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        diagnostic_tuples(self.statuses_ref(), &self.diagnostics, py)
    }
    #[getter]
    fn fallback_used(&self) -> Vec<bool> {
        match &self.result {
            Contracts::F64(value) => value.fallback_used.clone(),
            Contracts::Dd(value) => value.fallback_used.clone(),
            Contracts::Float(value) => value.fallback_used.clone(),
        }
    }
    #[getter]
    fn analytic_used(&self) -> Vec<bool> {
        match &self.result {
            Contracts::F64(value) => value.analytic_used.clone(),
            Contracts::Dd(value) => value.analytic_used.clone(),
            Contracts::Float(value) => value.analytic_used.clone(),
        }
    }
    fn lane(&self, py: Python<'_>, row: usize) -> PyResult<Py<PyAny>> {
        check_lane(self.statuses_ref(), row)?;
        match &self.result {
            Contracts::F64(value) => contracted_lane(value, row, self.shape.as_deref(), py),
            Contracts::Dd(value) => contracted_lane(value, row, self.shape.as_deref(), py),
            Contracts::Float(value) => contracted_lane(value, row, self.shape.as_deref(), py),
        }
    }
    fn lanes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = (0..self.rows())
            .map(|row| {
                if self.statuses_ref()[row] == LaneStatus::Success {
                    self.lane(py, row)
                } else {
                    Ok(py.None())
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, values)?.into_any().unbind())
    }
    fn __len__(&self) -> usize {
        self.rows()
    }
}

fn tensor_lane<T: PythonNumber>(
    result: &BatchResult<T>,
    row: usize,
    shape: Option<&[Vec<usize>]>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let values = result
        .lane(row)
        .map_err(|status| crate::LaneError::new_err(format!("lane {row}: {status:?}")))?;
    let moments = values
        .chunks_exact(4 * result.components)
        .map(|moment| {
            let poles = moment
                .chunks_exact(result.components)
                .map(|component| output_components(component, shape, py))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, poles)?.into_any().unbind())
        })
        .collect::<PyResult<Vec<Py<PyAny>>>>()?;
    Ok(PyTuple::new(py, moments)?.into_any().unbind())
}

fn contracted_lane<T: PythonNumber>(
    result: &ContractionResult<T>,
    row: usize,
    shape: Option<&[Vec<usize>]>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let values = result
        .lane(row)
        .map_err(|status| crate::LaneError::new_err(format!("lane {row}: {status:?}")))?;
    let poles = values
        .chunks_exact(result.components)
        .map(|component| output_components(component, shape, py))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, poles)?.into_any().unbind())
}

fn check_lane(statuses: &[LaneStatus], row: usize) -> PyResult<()> {
    let status = statuses.get(row).ok_or_else(|| {
        PyIndexError::new_err(format!(
            "lane {row} is outside batch of length {}",
            statuses.len()
        ))
    })?;
    if *status != LaneStatus::Success {
        return Err(crate::LaneError::new_err(format!("lane {row}: {status:?}")));
    }
    Ok(())
}

fn status_name(status: &LaneStatus) -> &'static str {
    match status {
        LaneStatus::Success => "success",
        LaneStatus::InvalidInput(_) => "invalid_input",
        LaneStatus::MasterFailure(_) => "master_failure",
        LaneStatus::Unstable(_) => "unstable",
    }
}

fn diagnostics(statuses: &[LaneStatus], notes: &[Vec<String>]) -> Vec<Vec<String>> {
    statuses
        .iter()
        .enumerate()
        .map(|(row, status)| {
            let mut output = notes.get(row).cloned().unwrap_or_default();
            match status {
                LaneStatus::Success => {}
                LaneStatus::InvalidInput(reason)
                | LaneStatus::MasterFailure(reason)
                | LaneStatus::Unstable(reason) => output.insert(0, reason.clone()),
            }
            output
        })
        .collect()
}

fn diagnostic_tuples(
    statuses: &[LaneStatus],
    notes: &[Vec<String>],
    py: Python<'_>,
) -> PyResult<Py<PyTuple>> {
    let notes = diagnostics(statuses, notes)
        .into_iter()
        .map(|row| PyTuple::new(py, row).map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, notes)?.unbind())
}

fn basis(values: &[Monomial]) -> Vec<(u16, u16, u16, u16)> {
    values
        .iter()
        .map(|Monomial([a, b, c, d])| (*a, *b, *c, *d))
        .collect()
}

fn stats(stats: ExecutionStats, py: Python<'_>) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    for (name, value) in [
        ("input_rows", stats.input_rows),
        ("argument_rows", stats.argument_rows),
        ("master_family_batches", stats.master_family_batches),
        ("master_rows", stats.master_rows),
        ("pentagon_box_rows", stats.pentagon_box_rows),
        ("reduction_rows", stats.reduction_rows),
        ("fallback_rows", stats.fallback_rows),
        ("fallback_successes", stats.fallback_successes),
        ("analytic_rows", stats.analytic_rows),
    ] {
        output.set_item(name, value)?;
    }
    Ok(output.unbind())
}

fn engine_name(kind: TensorEngineKind) -> &'static str {
    match kind {
        TensorEngineKind::Direct => "direct",
        TensorEngineKind::Generic => "generic",
        TensorEngineKind::LegacyNoOp => "legacy_noop",
        TensorEngineKind::SixPoint => "six_point",
        TensorEngineKind::SixPointHybrid => "six_point_hybrid",
        TensorEngineKind::CertifiedTensor => "certified_tensor",
        TensorEngineKind::MassTaylor => "mass_taylor",
        TensorEngineKind::AffinePower => "affine_power",
        TensorEngineKind::SevenPoint => "seven_point",
    }
}

fn algorithm_name(algorithm: Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Ibp => "ibp",
        Algorithm::PassarinoVeltman => "pv",
        Algorithm::Hybrid => "hybrid",
    }
}

fn certificate_stats(stats: CertificateExecutionStats, py: Python<'_>) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    output.set_item("input_rows", stats.input_rows)?;
    output.set_item("evaluated_rows", stats.evaluated_rows)?;
    output.set_item("passes", stats.passes)?;
    output.set_item("reduction_batches", stats.reduction_batches)?;
    Ok(output.unbind())
}

fn engine_execution(execution: &TensorEngineExecution, py: Python<'_>) -> PyResult<Py<PyDict>> {
    let output = PyDict::new(py);
    // These are the actual public Rust counters, without flattening packed rows
    // or certificate work into the facade's logical-input summary.
    macro_rules! packed {
        ($kind:expr, $value:ident) => {{
            output.set_item("engine", engine_name($kind))?;
            output.set_item("parent_input_rows", $value.parent_input_rows)?;
            output.set_item("child_batch_calls", $value.child_batch_calls)?;
            output.set_item("packed_child_rows", $value.packed_child_rows)?;
            output.set_item("parent_batch_calls", $value.parent_batch_calls)?;
            output.set_item("packed_parent_rows", $value.packed_parent_rows)?;
            output.set_item("child_execution", stats($value.child_execution, py)?)?;
            output.set_item("parent_execution", stats($value.parent_execution, py)?)?;
            output.set_item(
                "child_certificate_execution",
                certificate_stats($value.child_certificate_execution, py)?,
            )?;
            output.set_item(
                "parent_certificate_execution",
                certificate_stats($value.parent_certificate_execution, py)?,
            )?;
        }};
    }
    match execution {
        TensorEngineExecution::Direct(direct) => {
            output.set_item("engine", engine_name(TensorEngineKind::Direct))?;
            output.set_item("input_rows", direct.input_rows)?;
            output.set_item("direct_successes", direct.direct_successes)?;
            output.set_item("fallback_rows", direct.fallback_rows)?;
            output.set_item("fallback_successes", direct.fallback_successes)?;
            let tiers = PyList::empty(py);
            for tier in &direct.precision_rescue {
                let entry = PyDict::new(py);
                entry.set_item("bits", tier.bits)?;
                entry.set_item("rows", tier.rows)?;
                entry.set_item("successes", tier.successes)?;
                tiers.append(entry)?;
            }
            output.set_item("precision_rescue", tiers)?;
            output.set_item("symbolic_rows", direct.symbolic_rows)?;
            output.set_item("fallback_preparation_error", direct.fallback_preparation_error.clone())?;
            if let Some(child) = &direct.fallback {
                output.set_item("fallback", engine_execution(child, py)?)?;
            }
        }
        TensorEngineExecution::Generic {
            execution,
            certificate_execution,
        } => {
            output.set_item("engine", engine_name(TensorEngineKind::Generic))?;
            output.set_item("execution", stats(*execution, py)?)?;
            output.set_item(
                "certificate_execution",
                certificate_stats(*certificate_execution, py)?,
            )?;
        }
        TensorEngineExecution::LegacyNoOp {
            input_rows, accepted_rows, rejected_rows, query_count,
            certificate_execution, child, child_preparation_attempts, child_preparation_error,
        } => {
            output.set_item("engine", engine_name(TensorEngineKind::LegacyNoOp))?;
            output.set_item("input_rows", input_rows)?;
            output.set_item("accepted_rows", accepted_rows)?;
            output.set_item("rejected_rows", rejected_rows)?;
            output.set_item("query_count", query_count)?;
            output.set_item("certificate_execution", certificate_stats(*certificate_execution, py)?)?;
            if let Some(child) = child {
                output.set_item("child", engine_execution(child, py)?)?;
            } else {
                output.set_item("child", py.None())?;
            }
            output.set_item("child_preparation_attempts", child_preparation_attempts)?;
            output.set_item("child_preparation_error", child_preparation_error)?;
        }
        TensorEngineExecution::SixPoint(value) => packed!(TensorEngineKind::SixPoint, value),
        TensorEngineExecution::MassTaylor(value) => {
            output.set_item("engine", engine_name(TensorEngineKind::MassTaylor))?;
            output.set_item("certificate_execution", certificate_stats(value.certificate_execution, py)?)?;
            output.set_item("input_rows", value.input_rows)?;
            output.set_item("user_components", value.user_components)?;
            output.set_item("internal_components", value.internal_components)?;
            output.set_item("unit_execution", stats(value.unit_execution, py)?)?;
            output.set_item("projection_execution", stats(value.projection_execution, py)?)?;
            output.set_item("unit_algorithm_used", algorithm_name(value.unit_algorithm_used))?;
            output.set_item("requested_algorithm", algorithm_name(value.requested_algorithm))?;
        }
        TensorEngineExecution::AffinePower(value) => {
            output.set_item("engine", engine_name(TensorEngineKind::AffinePower))?;
            output.set_item("input_rows", value.input_rows)?;
            output.set_item(
                "certificate_execution",
                certificate_stats(value.certificate, py)?,
            )?;
            output.set_item("profiles", value.profiles)?;
            output.set_item("dag_nodes", value.dag_nodes)?;
            output.set_item("dag_edges", value.dag_edges)?;
            output.set_item("packed_child_rows", value.packed_child_rows)?;
            output.set_item("child_batch_calls", value.child_batch_calls)?;
            output.set_item("parent_batch_calls", value.parent_batch_calls)?;
            output.set_item("parent_rows", value.parent_rows)?;
            output.set_item("quadrature_work_rows", value.quadrature_work_rows)?;
            output.set_item("child_execution", stats(value.child_execution, py)?)?;
            output.set_item("parent_execution", stats(value.parent_execution, py)?)?;
            let methods = value
                .child_methods
                .iter()
                .map(|method| {
                    let item = PyDict::new(py);
                    item.set_item("propagators", method.propagators)?;
                    item.set_item("powers", &method.powers)?;
                    item.set_item("requested", algorithm_name(method.requested))?;
                    item.set_item("actual", method.actual.map(algorithm_name))?;
                    item.set_item("rows", method.rows)?;
                    item.set_item("notes", &method.notes)?;
                    Ok(item.unbind())
                })
                .collect::<PyResult<Vec<_>>>()?;
            output.set_item("child_methods", methods)?;
        }
        TensorEngineExecution::CertifiedTensor {
            execution,
            certificate_execution,
        } => {
            output.set_item("engine", engine_name(TensorEngineKind::CertifiedTensor))?;
            output.set_item("execution", stats(*execution, py)?)?;
            let certificates = PyDict::new(py);
            certificates.set_item(
                "cartesian",
                certificate_stats(certificate_execution.cartesian, py)?,
            )?;
            certificates.set_item("gram", certificate_stats(certificate_execution.gram, py)?)?;
            output.set_item("certificate_execution", certificates)?;
        }
        TensorEngineExecution::SixPointHybrid {
            primary,
            fallback,
            fallback_certificate_execution,
            fallback_rows,
            fallback_successes,
            fallback_batches,
            fallback_preparation_attempts,
            fallback_preparation_error,
        } => {
            output.set_item("engine", engine_name(TensorEngineKind::SixPointHybrid))?;
            output.set_item(
                "primary",
                engine_execution(&TensorEngineExecution::SixPoint(primary.clone()), py)?,
            )?;
            output.set_item("fallback", stats(*fallback, py)?)?;
            output.set_item(
                "fallback_certificate_execution",
                certificate_stats(*fallback_certificate_execution, py)?,
            )?;
            output.set_item("fallback_rows", fallback_rows)?;
            output.set_item("fallback_successes", fallback_successes)?;
            output.set_item("fallback_batches", fallback_batches)?;
            output.set_item(
                "fallback_preparation_attempts",
                fallback_preparation_attempts,
            )?;
            output.set_item("fallback_preparation_error", fallback_preparation_error)?;
        }
        TensorEngineExecution::SevenPoint(value) => packed!(TensorEngineKind::SevenPoint, value),
    }
    Ok(output.unbind())
}
