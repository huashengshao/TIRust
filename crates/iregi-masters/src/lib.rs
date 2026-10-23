//! Scalar masters evaluated with Symbolica's instruction-major batch engine.
//!
//! Every input row contains the OneLOop arguments (squared invariants, squared
//! masses, then the squared scale). Dual components are argument-major and
//! component-minor, using Symbolica's Taylor-coefficient convention. The batch
//! dimension is independent of the Dual shape. No numerical algebra or
//! differentiation is implemented here.
//!
//! The UV/IR split is native metadata: A0 has UV residue m², B0 has UV residue
//! one, and C0/D0/dB0 are UV finite. Thus a scaleless B0 retains UV=1 and IR=-1
//! even though its combined Laurent series vanishes.
#![forbid(unsafe_code)]

use std::{error::Error, fmt};

mod bubble_neighborhood;
mod derivative_domain;
mod dual_validation;
mod exact_plan;
mod normalization;
mod pentagon6;
mod scale_transport;
pub use dual_validation::validate_dual_structure;
pub use exact_plan::ExactMasterPlan;
pub use pentagon6::{
    Pentagon6Batch, Pentagon6Diagnostic, Pentagon6Evaluator, Pentagon6LaneError, Pentagon6Options,
    Pentagon6Plan,
};
#[cfg(feature = "native-batch")]
mod native_batch;

pub use oneloop::{NativeFloat, ScalarIntegral};
pub use symbolica::domains::{
    dual::{DualNumberStructure, HyperDual},
    float::{Complex, DoubleFloat, Float, FloatLike},
};
use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{float::SingleFloat, rational::Rational},
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
};

// Five physical outputs and one private derivative-domain discriminant.
const OUTPUTS: usize = 6;

/// Construction or whole-buffer errors. Domain failures are reported per lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MasterError {
    InvalidPrecision,
    InvalidShape(String),
    SizeOverflow,
    InputLength { expected: usize, actual: usize },
    Preparation(String),
    Evaluation(String),
}

impl fmt::Display for MasterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPrecision => f.write_str("working precision must be positive"),
            Self::InvalidShape(reason) => write!(f, "invalid Dual shape: {reason}"),
            Self::SizeOverflow => f.write_str("master batch dimensions overflow"),
            Self::InputLength { expected, actual } => {
                write!(f, "expected {expected} master inputs, received {actual}")
            }
            Self::Preparation(reason) => write!(f, "master preparation failed: {reason}"),
            Self::Evaluation(reason) => write!(f, "master evaluation failed: {reason}"),
        }
    }
}
impl Error for MasterError {}

/// A failed sample. Other lanes in the batch remain usable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneError {
    NonFiniteInput { argument: usize, component: usize },
    ComplexInvariant { argument: usize },
    UpperHalfPlaneMass { argument: usize },
    InvalidScale,
    UndefinedScalelessDerivative,
    UnsupportedDerivativeDomain,
    NonFiniteOutput,
}

impl fmt::Display for LaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput { argument, component } => write!(
                f,
                "nonfinite master input at argument {argument}, component {component}"
            ),
            Self::ComplexInvariant { argument } => {
                write!(f, "external squared invariant {argument} must be real")
            }
            Self::UpperHalfPlaneMass { argument } => {
                write!(f, "squared mass {argument} must have nonpositive imaginary part")
            }
            Self::InvalidScale => f.write_str("mu_squared must be positive and real"),
            Self::UndefinedScalelessDerivative => {
                f.write_str("dB0 is undefined at the fully scaleless point")
            }
            Self::UnsupportedDerivativeDomain => f.write_str(
                "these Dual seeds cross a scalar-limit branch without a supported analytic neighborhood",
            ),
            Self::NonFiniteOutput => {
                f.write_str("master is singular or unsupported at this kinematic point")
            }
        }
    }
}
impl Error for LaneError {}

/// Borrowed Laurent coefficients of one successful lane.
///
/// Each slice contains one value for ordinary inputs, or every component of
/// the prepared Dual shape. `total_single = uv + ir`.
#[derive(Clone, Copy, Debug)]
pub struct MasterLane<'a, T: FloatLike> {
    pub finite: &'a [Complex<T>],
    pub uv: &'a [Complex<T>],
    pub ir: &'a [Complex<T>],
    pub ir2: &'a [Complex<T>],
    pub total_single: &'a [Complex<T>],
}

/// Owned results and per-lane failures, retaining native UV/IR separation.
#[derive(Clone, Debug)]
pub struct MasterBatch<T: FloatLike> {
    components: usize,
    values: Vec<Complex<T>>,
    errors: Vec<Option<LaneError>>,
}

impl<T: FloatLike> MasterBatch<T> {
    pub fn row_count(&self) -> usize {
        self.errors.len()
    }

    pub fn component_count(&self) -> usize {
        self.components
    }

    pub fn errors(&self) -> &[Option<LaneError>] {
        &self.errors
    }

    /// Borrow one lane. An out-of-range lane is a caller indexing error.
    pub fn lane(&self, lane: usize) -> Result<MasterLane<'_, T>, &LaneError> {
        if let Some(error) = &self.errors[lane] {
            return Err(error);
        }
        let width = self.components;
        let row = &self.values[lane * OUTPUTS * width..(lane + 1) * OUTPUTS * width];
        Ok(MasterLane {
            finite: &row[..width],
            total_single: &row[width..2 * width],
            ir2: &row[2 * width..3 * width],
            uv: &row[3 * width..4 * width],
            ir: &row[4 * width..5 * width],
        })
    }
}

/// Instruction-major execution choices for ordinary scalar masters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MasterBackend {
    SymbolicaBatch,
    RustBatch,
}

/// Default scalar backend selected by this crate's compiled feature set.
/// Dual plans always use Symbolica's batch interpreter.
pub const fn default_master_backend() -> MasterBackend {
    if cfg!(feature = "native-batch") {
        MasterBackend::RustBatch
    } else {
        MasterBackend::SymbolicaBatch
    }
}

#[derive(Clone)]
enum PreparedMaster<T: NativeFloat> {
    Symbolica(BatchExpressionEvaluator<Complex<T>>),
    #[cfg(feature = "native-batch")]
    Rust(native_batch::NativeBatchEvaluator<T>),
}

/// Prepared scalar-master instructions and reusable batch workspaces.
///
/// Constants are converted directly from exact rationals at the prototype's
/// precision. Inputs retain their supplied precision; no f64 conversion or
/// upward precision padding is performed. Construct a fresh evaluator when
/// increasing working precision, and supply equally precise input values.
#[derive(Clone)]
pub struct MasterEvaluator<T: NativeFloat> {
    family: ScalarIntegral,
    prototype: T,
    shape: Option<Vec<Vec<usize>>>,
    components: usize,
    evaluator: PreparedMaster<T>,
    input_workspace: Vec<Complex<T>>,
    output_workspace: Vec<Complex<T>>,
    valid_lanes: Vec<usize>,
    normalization_scales: Vec<Option<normalization::BinaryScale>>,
    constant_evaluator: Option<Box<MasterEvaluator<T>>>,
    constant_input_workspace: Vec<Complex<T>>,
    constant_lanes: Vec<usize>,
    scale_transport: Option<scale_transport::ScaleTransport<T>>,
}

impl<T: NativeFloat> MasterEvaluator<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(family: ScalarIntegral, prototype: T) -> Result<Self, MasterError> {
        Self::new_with_backend(family, prototype, default_master_backend())
    }

    /// Select the native generated kernels or the exact-expression interpreter.
    /// Both execute instructions across batches; neither loops over scalar calls.
    pub fn new_with_backend(
        family: ScalarIntegral,
        prototype: T,
        backend: MasterBackend,
    ) -> Result<Self, MasterError> {
        match backend {
            MasterBackend::SymbolicaBatch => ExactMasterPlan::new(family)?.instantiate(prototype),
            MasterBackend::RustBatch => {
                #[cfg(feature = "native-batch")]
                {
                    if prototype.get_precision() == 0 {
                        return Err(MasterError::InvalidPrecision);
                    }
                    let evaluator = native_batch::NativeBatchEvaluator::new(family, &prototype)
                        .map_err(MasterError::Preparation)?;
                    Ok(Self {
                        family,
                        prototype,
                        shape: None,
                        components: 1,
                        evaluator: PreparedMaster::Rust(evaluator),
                        input_workspace: Vec::new(),
                        output_workspace: Vec::new(),
                        valid_lanes: Vec::new(),
                        normalization_scales: Vec::new(),
                        constant_evaluator: None,
                        constant_input_workspace: Vec::new(),
                        constant_lanes: Vec::new(),
                        scale_transport: None,
                    })
                }
                #[cfg(not(feature = "native-batch"))]
                {
                    Err(MasterError::Preparation(
                        "RustBatch requires the native-batch feature".into(),
                    ))
                }
            }
        }
    }

    pub fn backend(&self) -> MasterBackend {
        match &self.evaluator {
            PreparedMaster::Symbolica(_) => MasterBackend::SymbolicaBatch,
            #[cfg(feature = "native-batch")]
            PreparedMaster::Rust(_) => MasterBackend::RustBatch,
        }
    }

    /// Bound native temporary storage without changing the vectorized kernels.
    /// Returns false for the Symbolica backend, whose workspace is managed by Symbolica.
    pub fn set_native_batch_policy(
        &mut self,
        policy: oneloop::rust_codegen::runtime::BatchExecutionPolicy,
    ) -> bool {
        #[cfg(feature = "native-batch")]
        if let PreparedMaster::Rust(evaluator) = &mut self.evaluator {
            evaluator.set_policy(policy);
            return true;
        }
        let _ = policy;
        false
    }

    /// Prepare static or dynamic Symbolica Dual shapes. Shape members retain
    /// Symbolica's order and factorial normalization without reinterpretation.
    ///
    /// Seeded triangle/box inputs support generic massive physical and Euclidean
    /// points, including strictly lower-half-plane complex masses. Their real
    /// mass parts must be positive and invariant jets real; zero-width masses
    /// require real mass jets. Selected Gram, threshold and Cayley boundaries,
    /// and unsupported massless/scaleless boundaries return
    /// [`LaneError::UnsupportedDerivativeDomain`]. Constant Dual kinematics
    /// retain the ordinary scalar domain, including the soft-box relabelling.
    /// B0/dB0 use exact order-dependent Taylor neighborhoods at zero momentum;
    /// a massless propagator there must have every mass-jet component zero.
    pub fn new_dual<D: DualNumberStructure>(
        family: ScalarIntegral,
        prototype: T,
        dual: D,
    ) -> Result<Self, MasterError> {
        ExactMasterPlan::new_dual(family, dual)?.instantiate(prototype)
    }

    fn from_exact(
        family: ScalarIntegral,
        prototype: T,
        shape: Option<Vec<Vec<usize>>>,
        exact: ExpressionEvaluator<Complex<Rational>>,
    ) -> Result<Self, MasterError> {
        let bits = prototype.get_precision();
        if bits == 0 {
            return Err(MasterError::InvalidPrecision);
        }
        let components = shape.as_ref().map_or(1, Vec::len);
        let evaluator = exact
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        prototype.from_rational(&value.re),
                        prototype.from_rational(&value.im),
                    )
                },
                bits,
            )
            .into_batch_evaluator();
        Ok(Self {
            family,
            prototype,
            shape,
            components,
            evaluator: PreparedMaster::Symbolica(evaluator),
            input_workspace: Vec::new(),
            output_workspace: Vec::new(),
            valid_lanes: Vec::new(),
            normalization_scales: Vec::new(),
            constant_evaluator: None,
            constant_input_workspace: Vec::new(),
            constant_lanes: Vec::new(),
            scale_transport: None,
        })
    }

    pub fn family(&self) -> ScalarIntegral {
        self.family
    }

    pub fn component_count(&self) -> usize {
        self.components
    }

    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.shape.as_deref()
    }

    pub fn binary_precision(&self) -> u32 {
        self.prototype.get_precision()
    }

    pub fn evaluate_batch(
        &mut self,
        input: &[Complex<T>],
        rows: usize,
    ) -> Result<MasterBatch<T>, MasterError> {
        let row_width = self
            .family
            .arity()
            .checked_mul(self.components)
            .ok_or(MasterError::SizeOverflow)?;
        let expected = rows
            .checked_mul(row_width)
            .ok_or(MasterError::SizeOverflow)?;
        if input.len() != expected {
            return Err(MasterError::InputLength {
                expected,
                actual: input.len(),
            });
        }
        let result_width = OUTPUTS
            .checked_mul(self.components)
            .ok_or(MasterError::SizeOverflow)?;
        let result_len = rows
            .checked_mul(result_width)
            .ok_or(MasterError::SizeOverflow)?;
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = MasterBatch {
            components: self.components,
            values: vec![zero.clone(); result_len],
            errors: vec![None; rows],
        };
        self.input_workspace.clear();
        self.valid_lanes.clear();
        self.normalization_scales.clear();
        self.constant_input_workspace.clear();
        self.constant_lanes.clear();
        for (lane, point) in input.chunks_exact(row_width).enumerate() {
            if let Err(error) = self.validate_point(point) {
                result.errors[lane] = Some(error);
                continue;
            }
            // Constant kinematics retain the scalar boundary prescription.
            // Scale-only jets are transported afterwards by a separate exact
            // Laurent program, avoiding unused kinematic derivative 0/0.
            if self.components > 1 && !self.kinematics_seeded(point) {
                self.constant_lanes.push(lane);
                self.constant_input_workspace.extend(
                    point
                        .chunks_exact(self.components)
                        .map(|argument| argument[0].clone()),
                );
                continue;
            }
            self.valid_lanes.push(lane);
            let start = self.input_workspace.len();
            self.input_workspace.extend_from_slice(point);
            let normalization = if self.components > 1
                && matches!(self.family, ScalarIntegral::B0 | ScalarIntegral::DB0)
            {
                normalization::BinaryScale::for_positive_real(&point[self.components])
                    .or_else(|| {
                        normalization::BinaryScale::for_positive_real(&point[2 * self.components])
                    })
                    .and_then(|scale| scale.bounded_by(&point[..3 * self.components]))
            } else {
                None
            };
            if let Some(scale) = normalization {
                for value in &mut self.input_workspace[start..start + 3 * self.components] {
                    scale.divide(value);
                }
                // Keep the normalized kinematics and scale separately moderate.
                // B0's exact scale dependence is restored below; DB0 has no
                // scale dependence. This avoids differentiating logarithms at
                // an extreme scale just because a kinematic seed is nonzero.
                for component in 0..self.components {
                    self.input_workspace[start + 3 * self.components + component] = Complex::new(
                        if component == 0 {
                            self.prototype.one()
                        } else {
                            self.prototype.zero()
                        },
                        self.prototype.zero(),
                    );
                }
            }
            self.normalization_scales.push(normalization);
            // Reuse OneLOop's exact soft-box relabelling. Nonconstant jets at
            // massless boundaries are rejected by validate_point below.
            if self.family == ScalarIntegral::D0 {
                if self.components == 1 {
                    oneloop::regular_d0_labelling(&mut self.input_workspace[start..]);
                } else {
                    let mut primals: Vec<_> = point
                        .chunks_exact(self.components)
                        .map(|argument| argument[0].clone())
                        .collect();
                    if oneloop::regular_d0_labelling(&mut primals) {
                        // Every kinematic jet is zero in this permitted soft
                        // domain; only the primal labels need permutation.
                        for (argument, primal) in primals.into_iter().enumerate() {
                            self.input_workspace[start + argument * self.components] = primal;
                        }
                    }
                }
            }
        }
        let valid = self.valid_lanes.len();
        self.output_workspace.resize(valid * result_width, zero);
        match &mut self.evaluator {
            PreparedMaster::Symbolica(evaluator) => evaluator
                .evaluate_batch(&self.input_workspace, &mut self.output_workspace, valid)
                .map_err(|error| MasterError::Evaluation(error.to_string()))?,
            #[cfg(feature = "native-batch")]
            PreparedMaster::Rust(evaluator) => evaluator
                .evaluate_batch(&self.input_workspace, &mut self.output_workspace, valid)
                .map_err(MasterError::Evaluation)?,
        }
        self.restore_bubble_scale(input, row_width, result_width)?;
        for (prepared, &lane) in self.valid_lanes.iter().enumerate() {
            let point = &input[lane * row_width..(lane + 1) * row_width];
            let check_discriminant = self.kinematics_seeded(point)
                && (matches!(self.family, ScalarIntegral::C0 | ScalarIntegral::D0)
                    || (matches!(self.family, ScalarIntegral::B0 | ScalarIntegral::DB0)
                        && !point[0].is_zero()));
            let regular_pseudothreshold = self.regular_bubble_pseudothreshold(
                &self.input_workspace[prepared * row_width..(prepared + 1) * row_width],
            );
            let output =
                &mut self.output_workspace[prepared * result_width..(prepared + 1) * result_width];
            if self.family == ScalarIntegral::DB0 {
                if let Some(scale) = self.normalization_scales[prepared] {
                    for value in &mut output[..5 * self.components] {
                        scale.divide(value);
                    }
                }
            }
            let domain = &output[5 * self.components];
            if check_discriminant
                && (!domain.re.is_finite()
                    || !domain.im.is_finite()
                    || (domain.is_zero() && !regular_pseudothreshold))
            {
                result.errors[lane] = Some(LaneError::UnsupportedDerivativeDomain);
            } else if output[..5 * self.components]
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                result.errors[lane] = Some(LaneError::NonFiniteOutput);
            } else {
                result.values[lane * result_width..(lane + 1) * result_width]
                    .clone_from_slice(output);
            }
        }
        if !self.constant_lanes.is_empty() {
            if self.constant_evaluator.is_none() {
                self.constant_evaluator =
                    Some(Box::new(Self::new(self.family, self.prototype.clone())?));
            }
            let constants = self
                .constant_evaluator
                .as_mut()
                .unwrap()
                .evaluate_batch(&self.constant_input_workspace, self.constant_lanes.len())?;
            for (prepared, &lane) in self.constant_lanes.iter().enumerate() {
                if let Some(error) = &constants.errors[prepared] {
                    result.errors[lane] = Some(error.clone());
                } else {
                    for slot in 0..OUTPUTS {
                        result.values[lane * result_width + slot * self.components] =
                            constants.values[prepared * OUTPUTS + slot].clone();
                    }
                }
            }
            let scale_offset = (self.family.arity() - 1) * self.components;
            let scale_seeded = self.constant_lanes.iter().any(|&lane| {
                input[lane * row_width + scale_offset + 1..(lane + 1) * row_width]
                    .iter()
                    .any(|value| !value.is_zero())
            });
            if scale_seeded {
                if self.scale_transport.is_none() {
                    self.scale_transport = Some(scale_transport::ScaleTransport::new(
                        self.shape.as_ref().unwrap(),
                        &self.prototype,
                    )?);
                }
                self.input_workspace.clear();
                self.valid_lanes.clear();
                for &lane in &self.constant_lanes {
                    if result.errors[lane].is_some() {
                        continue;
                    }
                    self.valid_lanes.push(lane);
                    let start = self.input_workspace.len();
                    self.input_workspace.resize(
                        start + 7 * self.components,
                        Complex::new(self.prototype.zero(), self.prototype.zero()),
                    );
                    for slot in 0..4 {
                        self.input_workspace[start + slot * self.components] =
                            result.values[lane * result_width + slot * self.components].clone();
                    }
                    let scale = &input[lane * row_width + scale_offset..(lane + 1) * row_width];
                    let offset = start + 4 * self.components;
                    self.input_workspace[offset..offset + self.components].clone_from_slice(scale);
                    if let Some(normalization) =
                        normalization::BinaryScale::for_positive_real(&scale[0])
                            .and_then(|normalization| normalization.bounded_by(scale))
                    {
                        for value in &mut self.input_workspace[offset..offset + self.components] {
                            normalization.divide(value);
                        }
                    }
                    self.input_workspace[start + 5 * self.components] =
                        self.input_workspace[offset].clone();
                }
                let valid = self.valid_lanes.len();
                self.output_workspace.resize(
                    valid * result_width,
                    Complex::new(self.prototype.zero(), self.prototype.zero()),
                );
                self.scale_transport.as_mut().unwrap().evaluate_batch(
                    &self.input_workspace,
                    &mut self.output_workspace,
                    valid,
                )?;
                for (prepared, &lane) in self.valid_lanes.iter().enumerate() {
                    let output = &self.output_workspace
                        [prepared * result_width..(prepared + 1) * result_width];
                    if output
                        .iter()
                        .any(|value| !value.re.is_finite() || !value.im.is_finite())
                    {
                        result.errors[lane] = Some(LaneError::NonFiniteOutput);
                    } else {
                        result.values[lane * result_width..(lane + 1) * result_width]
                            .clone_from_slice(output);
                    }
                }
            }
        }
        Ok(result)
    }

    /// Adapter for caller-defined static Dual types. Pass `|dual| &dual.values`.
    /// All values must have the shape used when preparing this evaluator.
    pub fn evaluate_dual_batch<D: DualNumberStructure>(
        &mut self,
        input: &[D],
        rows: usize,
        components: impl for<'a> Fn(&'a D) -> &'a [Complex<T>],
    ) -> Result<MasterBatch<T>, MasterError> {
        let expected = rows
            .checked_mul(self.family.arity())
            .ok_or(MasterError::SizeOverflow)?;
        if input.len() != expected {
            return Err(MasterError::InputLength {
                expected,
                actual: input.len(),
            });
        }
        let shape = self.shape.as_ref().ok_or_else(|| {
            MasterError::InvalidShape("use new_dual before passing Dual values".into())
        })?;
        let capacity = input
            .len()
            .checked_mul(self.components)
            .ok_or(MasterError::SizeOverflow)?;
        let mut flattened = Vec::with_capacity(capacity);
        for value in input {
            if value.get_shape() != shape.iter().map(Vec::as_slice).collect::<Vec<_>>()
                || components(value).len() != self.components
            {
                return Err(MasterError::InvalidShape(
                    "input shape differs from prepared shape".into(),
                ));
            }
            flattened.extend_from_slice(components(value));
        }
        self.evaluate_batch(&flattened, rows)
    }

    pub fn evaluate_hyperdual_batch(
        &mut self,
        input: &[HyperDual<Complex<T>>],
        rows: usize,
    ) -> Result<MasterBatch<T>, MasterError> {
        self.evaluate_dual_batch(input, rows, |value| &value.values)
    }

    /// Adapter for real-valued static or dynamic Dual inputs, embedding each
    /// component into the complex master domain without changing precision.
    pub fn evaluate_real_dual_batch<D: DualNumberStructure>(
        &mut self,
        input: &[D],
        rows: usize,
        components: impl for<'a> Fn(&'a D) -> &'a [T],
    ) -> Result<MasterBatch<T>, MasterError> {
        let expected = rows
            .checked_mul(self.family.arity())
            .ok_or(MasterError::SizeOverflow)?;
        if input.len() != expected {
            return Err(MasterError::InputLength {
                expected,
                actual: input.len(),
            });
        }
        let shape = self.shape.as_ref().ok_or_else(|| {
            MasterError::InvalidShape("use new_dual before passing Dual values".into())
        })?;
        let capacity = input
            .len()
            .checked_mul(self.components)
            .ok_or(MasterError::SizeOverflow)?;
        let mut flattened = Vec::with_capacity(capacity);
        for value in input {
            if value.get_shape() != shape.iter().map(Vec::as_slice).collect::<Vec<_>>()
                || components(value).len() != self.components
            {
                return Err(MasterError::InvalidShape(
                    "input shape differs from prepared shape".into(),
                ));
            }
            flattened.extend(
                components(value)
                    .iter()
                    .map(|value| Complex::new(value.clone(), value.zero())),
            );
        }
        self.evaluate_batch(&flattened, rows)
    }

    pub fn evaluate_real_hyperdual_batch(
        &mut self,
        input: &[HyperDual<T>],
        rows: usize,
    ) -> Result<MasterBatch<T>, MasterError> {
        self.evaluate_real_dual_batch(input, rows, |value| &value.values)
    }

    fn restore_bubble_scale(
        &mut self,
        input: &[Complex<T>],
        row_width: usize,
        result_width: usize,
    ) -> Result<(), MasterError> {
        if self.family != ScalarIntegral::B0
            || !self.normalization_scales.iter().any(Option::is_some)
        {
            return Ok(());
        }
        if self.scale_transport.is_none() {
            self.scale_transport = Some(scale_transport::ScaleTransport::new(
                self.shape.as_ref().unwrap(),
                &self.prototype,
            )?);
        }
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut transport_input = Vec::new();
        let mut prepared_indices = Vec::new();
        for (prepared, (&lane, normalization)) in self
            .valid_lanes
            .iter()
            .zip(&self.normalization_scales)
            .enumerate()
        {
            let Some(normalization) = normalization else {
                continue;
            };
            prepared_indices.push(prepared);
            let start = transport_input.len();
            transport_input.resize(start + 7 * self.components, zero.clone());
            transport_input[start..start + 4 * self.components].clone_from_slice(
                &self.output_workspace
                    [prepared * result_width..prepared * result_width + 4 * self.components],
            );
            let scale = &input[lane * row_width + 3 * self.components..(lane + 1) * row_width];
            let offset = start + 4 * self.components;
            transport_input[offset..offset + self.components].clone_from_slice(scale);
            let scale_normalization = normalization::BinaryScale::for_positive_real(&scale[0])
                .and_then(|normalization| normalization.bounded_by(scale));
            if let Some(scale_normalization) = scale_normalization {
                for value in &mut transport_input[offset..offset + self.components] {
                    scale_normalization.divide(value);
                }
            }
            transport_input[start + 5 * self.components] =
                Complex::new(self.prototype.one(), self.prototype.zero());
            let exponent = scale_normalization
                .map_or(0, normalization::BinaryScale::exponent)
                .checked_sub(normalization.exponent())
                .ok_or(MasterError::SizeOverflow)?;
            transport_input[start + 6 * self.components] =
                Complex::new(self.prototype.from_i64(exponent), self.prototype.zero());
        }
        let mut output = vec![zero; prepared_indices.len() * result_width];
        self.scale_transport.as_mut().unwrap().evaluate_batch(
            &transport_input,
            &mut output,
            prepared_indices.len(),
        )?;
        for (row, prepared) in prepared_indices.into_iter().enumerate() {
            // The original normalized kinematic guard remains authoritative.
            self.output_workspace
                [prepared * result_width..prepared * result_width + 5 * self.components]
                .clone_from_slice(
                    &output[row * result_width..row * result_width + 5 * self.components],
                );
        }
        Ok(())
    }

    fn validate_point(&self, point: &[Complex<T>]) -> Result<(), LaneError> {
        for (slot, value) in point.iter().enumerate() {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(LaneError::NonFiniteInput {
                    argument: slot / self.components,
                    component: slot % self.components,
                });
            }
        }
        let momenta = match self.family {
            ScalarIntegral::A0 => 0,
            ScalarIntegral::B0 | ScalarIntegral::DB0 => 1,
            ScalarIntegral::C0 => 3,
            ScalarIntegral::D0 => 6,
        };
        let arguments = self.family.arity();
        let primal = |argument: usize| &point[argument * self.components];
        for argument in 0..momenta {
            if !primal(argument).im.is_zero() {
                return Err(LaneError::ComplexInvariant { argument });
            }
        }
        for argument in momenta..arguments - 1 {
            let mass = primal(argument);
            if mass.im > mass.im.zero() {
                return Err(LaneError::UpperHalfPlaneMass { argument });
            }
        }
        let scale = primal(arguments - 1);
        if !scale.im.is_zero() || scale.re <= scale.re.zero() {
            return Err(LaneError::InvalidScale);
        }
        let scaleless = (0..arguments - 1).all(|argument| primal(argument).is_zero());
        if scaleless && self.family == ScalarIntegral::DB0 {
            return Err(LaneError::UndefinedScalelessDerivative);
        }
        if self.components > 1 {
            // A scalar boundary formula alone does not define its transverse
            // derivative. Keep such lanes explicitly unsupported instead of
            // silently dropping their jets through a primal zero branch.
            let seeded = self.kinematics_seeded(point);
            let zero_mass = (momenta..arguments - 1).any(|argument| primal(argument).is_zero());
            let lightlike_massless_tangent =
                matches!(self.family, ScalarIntegral::B0 | ScalarIntegral::DB0)
                    && primal(0).is_zero()
                    && !scaleless
                    && (momenta..arguments - 1).all(|argument| {
                        !primal(argument).is_zero()
                            || point[argument * self.components..(argument + 1) * self.components]
                                .iter()
                                .all(SingleFloat::is_zero)
                    });
            if seeded && (scaleless || (zero_mass && !lightlike_massless_tangent)) {
                return Err(LaneError::UnsupportedDerivativeDomain);
            }
            if seeded && matches!(self.family, ScalarIntegral::C0 | ScalarIntegral::D0) {
                // Physical-sheet qualification uses real invariant directions.
                // Complex mass directions are two-sided only at strictly negative
                // widths; real-width boundary seeds need a separate continuation.
                let real_invariant_jets = point[..momenta * self.components]
                    .iter()
                    .all(|value| value.im.is_zero());
                let massive_neighborhood = (momenta..arguments - 1).all(|argument| {
                    primal(argument).re > primal(argument).re.zero()
                        && (!primal(argument).im.is_zero()
                            || point
                                [argument * self.components + 1..(argument + 1) * self.components]
                                .iter()
                                .all(|value| value.im.is_zero()))
                });
                if !real_invariant_jets || !massive_neighborhood {
                    return Err(LaneError::UnsupportedDerivativeDomain);
                }
            }
        }
        Ok(())
    }

    fn kinematics_seeded(&self, point: &[Complex<T>]) -> bool {
        self.components > 1
            && (0..self.family.arity() - 1).any(|argument| {
                point[argument * self.components + 1..(argument + 1) * self.components]
                    .iter()
                    .any(|value| !value.is_zero())
            })
    }

    // The symbolic program selects the Taylor chart only at zero discriminant.
    // These primal domain checks distinguish the regular pseudothreshold from
    // the genuine two-particle threshold before accepting that zero guard.
    fn regular_bubble_pseudothreshold(&self, point: &[Complex<T>]) -> bool {
        if !matches!(self.family, ScalarIntegral::B0 | ScalarIntegral::DB0) {
            return false;
        }
        let s = &point[0];
        let first = &point[self.components];
        let second = &point[2 * self.components];
        if !s.im.is_zero()
            || s.re <= s.re.zero()
            || !first.im.is_zero()
            || !second.im.is_zero()
            || first.re <= first.re.zero()
            || second.re <= second.re.zero()
        {
            return false;
        }
        let k = first.re.one() + second.re.clone() / &first.re - s.re.clone() / &first.re;
        k.is_finite() && k > k.zero()
    }
}

fn regular_bubble_finite(
    p: &[Atom],
    order: usize,
    derivative: bool,
    ordinary: &Atom,
) -> Result<Atom, MasterError> {
    let equal = bubble_neighborhood::equal_coefficients(p, order, derivative)?;
    let unequal = bubble_neighborhood::unequal_coefficients(p, order, derivative)?;
    let first_massless = bubble_neighborhood::one_massless_coefficients(p, order, derivative, 0)?;
    let second_massless = bubble_neighborhood::one_massless_coefficients(p, order, derivative, 1)?;
    let massive = Symbol::IF.call((&p[2] - &p[1], &unequal[0], &equal[0]));
    let at_zero = Symbol::IF.call((
        &p[1],
        Symbol::IF.call((&p[2], massive, &second_massless[0])),
        Symbol::IF.call((&p[2], &first_massless[0], ordinary)),
    ));
    let pseudothreshold = bubble_neighborhood::pseudothreshold_coefficients(p, order, derivative)?;
    // Reuse OneLOop's registered, precision-aware real-sign primitive. The
    // surrounding real-axis and nonzero guards make its contract explicit.
    // This also keeps constant Duals on genuine thresholds on the native
    // scalar branch, whose prescribed DB0 value is not an analytic derivative.
    let sign = symbolica::symbol!("oneloop::__olo_sign_nonnegative");
    let k = Atom::num(1) + &p[2] / &p[1] - &p[0] / &p[1];
    let mut chart = Symbol::IF.call((
        Atom::num(1) + sign.call((k,)),
        &pseudothreshold[0],
        ordinary,
    ));
    for mass in [&p[2], &p[1]] {
        chart = Symbol::IF.call((
            mass,
            Symbol::IF.call((
                mass - mass.conj(),
                ordinary,
                Symbol::IF.call((Atom::num(1) + sign.call((mass,)), chart, ordinary)),
            )),
            ordinary,
        ));
    }
    let at_nonzero = Symbol::IF.call((derivative_domain::bubble_discriminant(p), ordinary, chart));
    Ok(Symbol::IF.call((&p[0], at_nonzero, at_zero)))
}

/// Exact scalar outputs used by the reproducible Rust batch generator.
#[doc(hidden)]
pub fn exact_scalar_master_evaluator(
    family: ScalarIntegral,
) -> Result<ExpressionEvaluator<Complex<Rational>>, MasterError> {
    let (context, parameters, expressions) = expressions(family, false);
    context
        .evaluator_with_stack_reuse(&expressions, &parameters)
        .map_err(|error| MasterError::Preparation(error.to_string()))
}

fn expressions(
    family: ScalarIntegral,
    with_boundary: bool,
) -> (oneloop::OneLoopExpressions, Vec<Atom>, Vec<Atom>) {
    let context = oneloop::OneLoopExpressions::new();
    let p: Vec<_> = (0..family.arity())
        .map(|i| {
            symbolica::symbol!(format!("iregi_masters::{}_argument_{i}", family.name())).to_atom()
        })
        .collect();
    let series = match family {
        ScalarIntegral::A0 => oneloop::a0(&p[0], &p[1]),
        ScalarIntegral::B0 => oneloop::b0(&p[0], &p[1], &p[2], &p[3]),
        ScalarIntegral::DB0 => oneloop::db0(&p[0], &p[1], &p[2], &p[3]),
        ScalarIntegral::C0 => context.c0([&p[0], &p[1], &p[2]], [&p[3], &p[4], &p[5]], &p[6]),
        ScalarIntegral::D0 => context.d0(
            [&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]],
            [&p[6], &p[7], &p[8], &p[9]],
            &p[10],
        ),
    };
    let mut coefficients = series.into_coefficients();
    if matches!(family, ScalarIntegral::C0 | ScalarIntegral::D0) {
        // The integral of a scaleless scalar integrand vanishes in dimensional
        // regularization. This branch also prevents evaluating singular formulas
        // in inactive lanes. The scaleless derivative dB0 is handled separately.
        let any_nonzero = p[..p.len() - 1]
            .iter()
            .fold(Atom::num(0), |previous, value| {
                Symbol::IF.call((value, Atom::num(1), previous))
            });
        coefficients =
            coefficients.map(|coefficient| Symbol::IF.call((&any_nonzero, coefficient, 0)));
    }
    let uv = match family {
        ScalarIntegral::A0 => p[0].clone(),
        ScalarIntegral::B0 => Atom::num(1),
        _ => Atom::num(0),
    };
    let ir = &coefficients[1] - &uv;
    let mut outputs = Vec::from(coefficients);
    outputs.push(uv);
    outputs.push(ir);
    outputs.push(if with_boundary {
        derivative_domain::domain_discriminant(family, &p)
    } else {
        Atom::num(0)
    });
    (context, p, outputs)
}
