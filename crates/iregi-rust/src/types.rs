use crate::basis::Monomial;
use symbolica::domains::float::{Complex, FloatLike};

/// Independent tensor-reduction algorithms.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Algorithm {
    Ibp,
    PassarinoVeltman,
    #[default]
    Hybrid,
}

/// Integral normalization. All variants retain the separated UV/IR slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Convention {
    /// Native IREGI loop measure, including i*pi².
    #[default]
    IregiNative,
    /// OneLOop scalar normalization, with no odd-rank sign change.
    OneLoop,
    /// MadLoop normalization and (-1)^rank; combine UV and IR for its single pole.
    MadLoop,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TensorWeights {
    /// Actual Cartesian tensor components (IREGI ML5_CONVENTION=true).
    #[default]
    Components,
    /// Multiply each component by rank! / product(exponent!).
    Multinomial,
}

/// Source of Lorentz invariants, matching IREGI's explicit scalar-product interfaces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InvariantInput {
    #[default]
    FromShifts,
    /// Supply G_ij = p_i·p_j as the packed upper triangle, including its diagonal.
    /// Cartesian shifts still determine the returned tensor components.
    SuppliedScalarProducts,
}

/// Argument ranges in one input row, before its Taylor-component expansion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputLayout {
    pub shifts: std::ops::Range<usize>,
    pub scalar_products: Option<std::ops::Range<usize>>,
    pub masses: std::ops::Range<usize>,
    pub scale: usize,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReductionSettings {
    pub propagators: usize,
    pub rank: u16,
    pub powers: Vec<u16>,
    /// h in D=4+2h-2*epsilon.
    pub dimension_shift: i16,
    pub algorithm: Algorithm,
    pub convention: Convention,
    pub tensor_weights: TensorWeights,
    #[serde(default)]
    pub invariants: InvariantInput,
}

impl ReductionSettings {
    pub fn new(propagators: usize, rank: u16) -> Self {
        Self {
            propagators,
            rank,
            powers: vec![1; propagators],
            dimension_shift: 0,
            algorithm: Algorithm::Hybrid,
            convention: Convention::IregiNative,
            tensor_weights: TensorWeights::Components,
            invariants: InvariantInput::FromShifts,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReductionError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("reduction preparation: {0}")]
    Preparation(String),
    #[error("unsupported reduction: {0}")]
    Unsupported(String),
    #[error("numerical evaluation: {0}")]
    Evaluation(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneStatus {
    Success,
    InvalidInput(String),
    MasterFailure(String),
    Unstable(String),
}

/// A batch of complex numerical input components.
///
/// A row contains all shift components (propagator-major, time first), all
/// optional packed scalar products (i=0..n, j=i..n), all squared masses,
/// then the renormalization scale mu (not mu²). Each argument
/// has `components` consecutive Taylor components: one for ordinary numbers,
/// or the exact component order of the prepared Symbolica Dual shape.
#[derive(Clone, Debug)]
pub struct KinematicsBatch<T: FloatLike> {
    pub rows: usize,
    pub components: usize,
    pub values: Vec<Complex<T>>,
}

/// Tensor moments with independent lane status.
///
/// Successful data is row-major, then basis index, then UV1/IR1/IR2/finite,
/// then Taylor component. Failed lanes must be accessed through `lane`, which
/// returns their status instead of exposing initialized scratch values.
#[derive(Clone, Debug)]
pub struct BatchResult<T: FloatLike> {
    pub basis: Vec<Monomial>,
    pub components: usize,
    pub(crate) metadata: ResultMetadata,
    pub(crate) values: Vec<Complex<T>>,
    pub statuses: Vec<LaneStatus>,
    /// True when this sample was retried with the alternate reduction method.
    pub fallback_used: Vec<bool>,
    /// The lane used an exact analytic kinematic-limit profile.
    pub analytic_used: Vec<bool>,
}

/// Immutable interpretation of a tensor result's stored coefficients.
///
/// A component count alone does not identify a Dual algebra: for example,
/// a second-order univariate jet and two first derivatives both have three
/// components. Contraction checks this full metadata before using a result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultMetadata {
    convention: Convention,
    tensor_weights: TensorWeights,
    dual_shape: Option<Vec<Vec<usize>>>,
}

impl ResultMetadata {
    pub(crate) fn new(
        convention: Convention,
        tensor_weights: TensorWeights,
        dual_shape: Option<Vec<Vec<usize>>>,
    ) -> Self {
        Self {
            convention,
            tensor_weights,
            dual_shape,
        }
    }

    pub fn convention(&self) -> Convention {
        self.convention
    }
    pub fn tensor_weights(&self) -> TensorWeights {
        self.tensor_weights
    }
    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.dual_shape.as_deref()
    }
}

impl<T: FloatLike> BatchResult<T> {
    pub fn metadata(&self) -> &ResultMetadata {
        &self.metadata
    }
    pub fn rows(&self) -> usize {
        self.statuses.len()
    }
    pub fn lane(&self, row: usize) -> Result<&[Complex<T>], &LaneStatus> {
        if self.statuses[row] != LaneStatus::Success {
            return Err(&self.statuses[row]);
        }
        let width = self.basis.len() * 4 * self.components;
        Ok(&self.values[row * width..(row + 1) * width])
    }
}
