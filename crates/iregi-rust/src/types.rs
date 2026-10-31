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

/// Whether derived real scalar products follow the historical on-shell policy.
/// Supplied scalar products remain authoritative under either policy.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InputPolicy {
    #[default]
    Preserve,
    /// An exact nonnegative decimal or rational threshold avoids a binary64
    /// configuration round trip. Zero retains the historical massless 1e-3
    /// absolute rule while disabling relative matching for positive masses.
    /// N5/N6 coordinate inputs through rank6 conservatively require every
    /// demanded selected snap to preserve the original polynomial in every
    /// Taylor component. Geometry-changing snaps fail explicitly; no momentum
    /// projection is applied. Other highpoint domains remain unsupported.
    LegacyOnShell { threshold: String },
}

impl InputPolicy {
    pub fn legacy_onshell() -> Self {
        Self::LegacyOnShell {
            threshold: "1/1000000".into(),
        }
    }
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
    #[serde(default)]
    pub input_policy: InputPolicy,
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
            input_policy: InputPolicy::Preserve,
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

/// Private primal sensitivity operand, never a differentiated error bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrimalOperandKind {
    UncancelledMasters,
    NoMaster,
}

#[derive(Clone, Debug)]
pub(crate) struct PrimalOperand<T: FloatLike> {
    pub output_index: usize,
    pub amplitude: Complex<T>,
    pub kind: PrimalOperandKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrimalEnvelopeRoute {
    Ordinary,
    Analytic,
    Hybrid,
    CertifiedPlanar,
    AutomaticPlanar,
    CertifiedAffine,
    AutomaticAffine,
    CertifiedScalar,
    CertifiedTensor,
}

/// Coupled to one successful result lane. Numerical operands retain their
/// original precision; metadata records the interpretation of every slot.
#[derive(Clone, Debug)]
pub(crate) struct PrimalEnvelope<T: FloatLike> {
    pub basis: Vec<Monomial>,
    pub metadata: ResultMetadata,
    pub components: usize,
    pub working_precision: u32,
    pub accuracy_bits: u32,
    pub operands: Vec<PrimalOperand<T>>,
    pub output_primals: Vec<Complex<T>>,
    /// Dense opt-in full-Taylor sensitivity, never differentiated ABS data.
    pub component_amplitudes: Option<Vec<Complex<T>>>,
    pub output_components: Option<Vec<Complex<T>>>,
    pub routes: Vec<PrimalEnvelopeRoute>,
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
    /// True when an optional fallback stage was attempted for this sample,
    /// including preparation or certificate rejection before numerical work.
    pub fallback_used: Vec<bool>,
    /// The lane used an exact analytic kinematic-limit profile.
    pub analytic_used: Vec<bool>,
    pub(crate) primal_envelopes: Vec<Option<PrimalEnvelope<T>>>,
    pub(crate) primal_envelope_errors: Vec<Option<String>>,
    /// Per lane, where the engine measured one: estimated relative error of
    /// the lane's values (E/A over the whole array, moments weighted by
    /// mu^-rank), from the difference to the same integral with relabelled
    /// propagators. An estimate, not a bound. Empty when the engine gives none.
    pub estimated_errors: Vec<Option<f64>>,
    /// Lanes whose status is not `Success` but whose values hold the best
    /// array the engine computed: uncertified, like a COLLIER result whose
    /// accuracy flag is set (see `estimated_errors`). Empty: none.
    pub best_effort: Vec<bool>,
}

/// A lane's array with its accuracy information (`BatchResult::lane_with_estimate`).
#[derive(Clone, Copy, Debug)]
pub struct LaneEstimate<'a, T: FloatLike> {
    pub values: &'a [Complex<T>],
    /// Estimated relative error (E/A), if the engine measured one.
    pub estimated_error: Option<f64>,
    /// `true` for lanes that passed every check (status `Success`); `false`
    /// for a refused lane's best-effort array.
    pub certified: bool,
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

    /// Estimated relative error (E/A) of lane `row`, where measured.
    pub fn estimated_error(&self, row: usize) -> Option<f64> {
        self.estimated_errors.get(row).copied().flatten()
    }

    /// Whether lane `row` failed its checks but kept a best-effort array.
    pub fn is_best_effort(&self, row: usize) -> bool {
        self.best_effort.get(row).copied().unwrap_or(false)
    }

    /// The lane's array with its accuracy information: every successful lane
    /// (`certified`), and refused lanes that kept a best-effort array
    /// (`!certified`, normally with an estimated error). `None` when the lane
    /// has no array (invalid input, master failure, no evaluation).
    pub fn lane_with_estimate(&self, row: usize) -> Option<LaneEstimate<'_, T>> {
        let certified = self.statuses[row] == LaneStatus::Success;
        if !certified && !self.is_best_effort(row) {
            return None;
        }
        let width = self.basis.len() * 4 * self.components;
        Some(LaneEstimate {
            values: &self.values[row * width..(row + 1) * width],
            estimated_error: self.estimated_error(row),
            certified,
        })
    }
}

impl<T: iregi_masters::NativeFloat> BatchResult<T> {
    pub(crate) fn validated_primal_envelope(
        &self,
        row: usize,
        expected_working_precision: u32,
        expected_accuracy_bits: u32,
    ) -> Result<&PrimalEnvelope<T>, String> {
        let envelope = self.primal_envelope(row)?;
        if envelope.working_precision != expected_working_precision
            || envelope.accuracy_bits != expected_accuracy_bits
        {
            return Err("primal envelope working precision or accuracy target differs from caller".into());
        }
        Ok(envelope)
    }

    pub(crate) fn validated_component_amplitudes(
        &self,row:usize,working_precision:u32,accuracy_bits:u32,
    )->Result<&[Complex<T>],String> {
        let envelope=self.validated_primal_envelope(row,working_precision,accuracy_bits)?;
        self.component_amplitudes_from(row,envelope)
    }

    fn component_amplitudes_from<'a>(
        &self,row:usize,envelope:&'a PrimalEnvelope<T>,
    )->Result<&'a [Complex<T>],String> {
        let outputs=self.basis.len().checked_mul(4)
            .and_then(|n|n.checked_mul(self.components)).ok_or("component sensitivity size overflow")?;
        let amplitudes=envelope.component_amplitudes.as_ref()
            .ok_or("successful route has no supported full-component sensitivity transport")?;
        let saved=envelope.output_components.as_ref()
            .ok_or("full-component sensitivity is missing its coupled output snapshot")?;
        if amplitudes.len()!=outputs || saved.len()!=outputs {
            return Err("full-component sensitivity dimensions differ from result".into());
        }
        for (index,(amplitude,saved)) in amplitudes.iter().zip(saved).enumerate() {
            if !crate::stability::components::valid_amplitude(amplitude) {
                return Err("full-component sensitivity has an invalid amplitude".into());
            }
            let actual=&self.values[row*outputs+index];
            if saved.re!=actual.re || saved.im!=actual.im
                || saved.re.get_precision()!=actual.re.get_precision()
                || saved.im.get_precision()!=actual.im.get_precision()
                || saved.re.negative_sign()!=actual.re.negative_sign()
                || saved.im.negative_sign()!=actual.im.negative_sign() {
                return Err("full-component sensitivity does not describe a returned Taylor coefficient".into());
            }
        }
        Ok(amplitudes)
    }

    /// Reject missing, stale or differently interpreted private context. This
    /// accessor does not change the ordinary result's success/failure contract.
    pub(crate) fn primal_envelope(&self, row: usize) -> Result<&PrimalEnvelope<T>, String> {
        if self.statuses.get(row) != Some(&LaneStatus::Success) {
            return Err("primal envelope requires a successful lane".into());
        }
        if self.primal_envelopes.len() != self.rows()
            || self.primal_envelope_errors.len() != self.rows()
        {
            return Err("primal envelope lane inventory does not match result".into());
        }
        if let Some(reason) = &self.primal_envelope_errors[row] {
            return Err(reason.clone());
        }
        let envelope = self.primal_envelopes[row]
            .as_ref()
            .ok_or_else(|| "successful lane has no supported primal envelope".to_string())?;
        let outputs = self.basis.len().checked_mul(4)
            .ok_or_else(|| "primal envelope output size overflow".to_string())?;
        let width = outputs.checked_mul(self.components)
            .ok_or_else(|| "primal envelope component size overflow".to_string())?;
        let expected = width.checked_mul(self.rows())
            .ok_or_else(|| "primal envelope result size overflow".to_string())?;
        if self.components == 0 || self.values.len() != expected
            || envelope.basis != self.basis || envelope.metadata != self.metadata
            || envelope.components != self.components
            || envelope.working_precision == 0 || envelope.accuracy_bits == 0
            || envelope.routes.is_empty() || envelope.operands.len() != outputs
            || envelope.output_primals.len() != outputs
        {
            return Err("primal envelope basis, metadata, precision or dimensions do not match result".into());
        }
        for (index, operand) in envelope.operands.iter().enumerate() {
            if operand.output_index != index
                || !operand.amplitude.re.is_finite() || !operand.amplitude.im.is_finite()
            {
                return Err("primal envelope has an invalid output map or nonfinite amplitude".into());
            }
            let saved = &envelope.output_primals[index];
            let actual = &self.values[row * width + index * self.components];
            if saved.re != actual.re || saved.im != actual.im
                || saved.re.get_precision() != actual.re.get_precision()
                || saved.im.get_precision() != actual.im.get_precision()
                || saved.re.negative_sign() != actual.re.negative_sign()
                || saved.im.negative_sign() != actual.im.negative_sign()
            {
                return Err("primal envelope does not describe the returned coefficient".into());
            }
        }
        if envelope.component_amplitudes.is_some() || envelope.output_components.is_some() {
            self.component_amplitudes_from(row,envelope)?;
        }
        Ok(envelope)
    }

    /// Called in the same branch that scatters a successful child's values.
    /// Always erase previous context before accepting or rejecting a copy.
    pub(crate) fn inherit_primal_envelope(
        &mut self,
        row: usize,
        source: &Self,
        source_row: usize,
        route: PrimalEnvelopeRoute,
    ) {
        // Ordinary results do not retain this private context and allocate no
        // sidecar buffers. A disabled destination must not inspect or clone it.
        if self.primal_envelopes.is_empty() && self.primal_envelope_errors.is_empty() {
            return;
        }
        self.primal_envelopes[row] = None;
        self.primal_envelope_errors[row] = None;
        let copied = if source.basis != self.basis || source.metadata != self.metadata
            || source.components != self.components
        {
            Err("successful child primal envelope has incompatible result metadata".into())
        } else {
            source.primal_envelope(source_row).cloned()
        };
        match copied {
            Ok(mut envelope) => {
                envelope.routes.push(route);
                self.primal_envelopes[row] = Some(envelope);
                if let Err(reason) = self.primal_envelope(row) {
                    self.primal_envelopes[row] = None;
                    self.primal_envelope_errors[row] = Some(reason);
                }
            }
            Err(reason) => self.primal_envelope_errors[row] = Some(reason),
        }
    }
}

#[cfg(test)]
mod primal_envelope_tests {
    use super::*;
    use symbolica::domains::float::Float;

    fn fixture(rows: usize, shape: Option<Vec<Vec<usize>>>) -> BatchResult<f64> {
        let components = shape.as_ref().map_or(1, Vec::len);
        let basis = vec![Monomial([0; 4]), Monomial([0, 1, 0, 0])];
        let outputs = basis.len() * 4;
        let values: Vec<_> = (0..rows * outputs * components)
            .map(|index| Complex::new((index + 1) as f64, -(index as f64)))
            .collect();
        let metadata = ResultMetadata::new(Convention::IregiNative, TensorWeights::Components, shape);
        let envelopes = (0..rows).map(|row| Some(PrimalEnvelope {
            basis: basis.clone(), metadata: metadata.clone(), components,
            working_precision: 53, accuracy_bits: 53,
            operands: (0..outputs).map(|output_index| PrimalOperand {
                output_index,
                amplitude: Complex::new((10_000 + 100 * row + output_index) as f64, 0.),
                kind: if output_index == 0 { PrimalOperandKind::NoMaster }
                      else { PrimalOperandKind::UncancelledMasters },
            }).collect(),
            output_primals: (0..outputs)
                .map(|index| values[(row * outputs + index) * components].clone()).collect(),
            component_amplitudes:None,output_components:None,
            routes: vec![PrimalEnvelopeRoute::Ordinary],
        })).collect();
        BatchResult {
            basis, components, metadata, values,
            statuses: vec![LaneStatus::Success; rows],
            fallback_used: vec![false; rows], analytic_used: vec![false; rows],
            primal_envelopes: envelopes, primal_envelope_errors: vec![None; rows],
            estimated_errors: Vec::new(), best_effort: Vec::new(),
        }
    }

    #[test]
    fn ordinary_destination_does_not_allocate_or_inspect_private_envelopes() {
        let source = fixture(0, None);
        let mut target = fixture(1, None);
        target.primal_envelopes = Vec::new();
        target.primal_envelope_errors = Vec::new();
        let values = target.values.clone();
        // There is deliberately no source lane. A disabled destination must
        // not inspect or clone any of its private operand context.
        target.inherit_primal_envelope(0, &source, usize::MAX, PrimalEnvelopeRoute::Analytic);
        assert_eq!(target.values, values);
        assert_eq!(target.primal_envelopes.capacity(), 0);
        assert_eq!(target.primal_envelope_errors.capacity(), 0);

        let mut enabled = fixture(1, None);
        let mut missing = fixture(1, None);
        missing.primal_envelopes = Vec::new();
        missing.primal_envelope_errors = Vec::new();
        enabled.inherit_primal_envelope(0, &missing, 0, PrimalEnvelopeRoute::Analytic);
        assert!(enabled.primal_envelopes[0].is_none());
        assert!(enabled.primal_envelope_errors[0].is_some());
    }

    #[test]
    fn private_envelope_scatter_preserves_every_slot_and_route() {
        let source = fixture(3, Some(vec![vec![0, 0], vec![1, 0], vec![0, 1]]));
        let routes = [PrimalEnvelopeRoute::Analytic, PrimalEnvelopeRoute::Hybrid,
            PrimalEnvelopeRoute::CertifiedPlanar, PrimalEnvelopeRoute::AutomaticPlanar,
            PrimalEnvelopeRoute::CertifiedScalar, PrimalEnvelopeRoute::CertifiedTensor];
        for route in routes {
            let mut target = fixture(3, source.metadata.dual_shape.clone());
            let width = source.basis.len() * 4 * source.components;
            for (row, source_row) in [2, 0, 1].into_iter().enumerate() {
                target.values[row * width..(row + 1) * width]
                    .clone_from_slice(source.lane(source_row).unwrap());
                target.inherit_primal_envelope(row, &source, source_row, route);
                let got = target.primal_envelope(row).unwrap();
                assert_eq!(got.routes, vec![PrimalEnvelopeRoute::Ordinary, route]);
                assert_eq!(got.output_primals, source.primal_envelope(source_row).unwrap().output_primals);
                for (index, operand) in got.operands.iter().enumerate() {
                    assert_eq!(operand.output_index, index);
                    assert_eq!(operand.amplitude.re, (10_000 + 100 * source_row + index) as f64);
                    assert_eq!(operand.kind, source.primal_envelope(source_row).unwrap().operands[index].kind);
                }
            }
        }
    }

    #[test]
    fn private_envelope_replacement_clears_missing_failed_and_quadrature_context() {
        let source = fixture(1, None);
        let mut target = source.clone();
        let mut missing = source.clone();
        missing.primal_envelopes[0] = None;
        missing.primal_envelope_errors[0] = Some("quadrature error transport unsupported".into());
        target.inherit_primal_envelope(0, &missing, 0, PrimalEnvelopeRoute::Hybrid);
        assert!(target.primal_envelopes[0].is_none());
        assert!(target.primal_envelope(0).unwrap_err().contains("quadrature"));
        assert!(target.lane(0).is_ok(), "private context rejection must not alter ordinary status");
        target.inherit_primal_envelope(0, &source, 0, PrimalEnvelopeRoute::AutomaticPlanar);
        assert!(target.primal_envelope(0).is_ok());
        assert!(target.primal_envelope_errors[0].is_none());
        let mut failed = source.clone();
        failed.statuses[0] = LaneStatus::Unstable("fixture".into());
        target.inherit_primal_envelope(0, &failed, 0, PrimalEnvelopeRoute::Analytic);
        assert!(target.primal_envelopes[0].is_none());
        assert!(target.primal_envelope(0).unwrap_err().contains("successful"));
        let empty = fixture(0, None);
        assert!(empty.primal_envelopes.is_empty());
        assert!(empty.primal_envelope(0).is_err());
    }

    #[test]
    fn private_envelope_rejects_equal_size_different_shape_basis_and_convention() {
        let shape = Some(vec![vec![0, 0], vec![1, 0], vec![0, 1]]);
        let source = fixture(1, shape.clone());
        for change in 0..4 {
            let mut target = fixture(1, shape.clone());
            match change {
                0 => target.metadata.dual_shape = Some(vec![vec![0, 0], vec![0, 1], vec![1, 0]]),
                1 => target.metadata.convention = Convention::OneLoop,
                2 => target.metadata.tensor_weights = TensorWeights::Multinomial,
                _ => target.basis.reverse(),
            }
            target.inherit_primal_envelope(0, &source, 0, PrimalEnvelopeRoute::CertifiedPlanar);
            assert!(target.primal_envelopes[0].is_none());
            assert!(target.primal_envelope(0).is_err());
        }
    }

    #[test]
    fn private_envelope_rejects_stale_values_sparse_maps_and_nonfinite_amplitudes() {
        let valid = fixture(1, None);
        assert!(valid.validated_primal_envelope(0, 53, 53).is_ok());
        assert!(valid.validated_primal_envelope(0, 128, 53).is_err());
        assert!(valid.validated_primal_envelope(0, 53, 32).is_err());
        for change in 0..6 {
            let mut result = fixture(1, None);
            match change {
                0 => result.values[2].re += 1.,
                1 => result.primal_envelopes[0].as_mut().unwrap().operands[2].output_index = 1,
                2 => { result.primal_envelopes[0].as_mut().unwrap().operands.pop(); }
                3 => result.primal_envelopes[0].as_mut().unwrap().operands[0].amplitude.re = f64::INFINITY,
                4 => result.primal_envelope_errors.clear(),
                _ => { result.values.pop(); }
            }
            assert!(result.primal_envelope(0).is_err(), "mutation {change}");
        }
    }

    #[test]
    fn private_envelope_preserves_operand_precision_and_binds_returned_precision() {
        let value = Complex::new(Float::with_val(192, 3), Float::with_val(160, 4));
        let amplitude = Complex::new(Float::with_val(80, 1000), Float::new(96));
        let basis = vec![Monomial([0; 4])];
        let metadata = ResultMetadata::new(Convention::IregiNative, TensorWeights::Components, None);
        let envelope = PrimalEnvelope {
            basis: basis.clone(), metadata: metadata.clone(), components: 1,
            working_precision: 256, accuracy_bits: 53,
            operands: (0..4).map(|output_index| PrimalOperand {
                output_index, amplitude: amplitude.clone(), kind: PrimalOperandKind::UncancelledMasters,
            }).collect(),
            output_primals: vec![value.clone(); 4],component_amplitudes:None,output_components:None, routes: vec![PrimalEnvelopeRoute::Ordinary],
        };
        let source = BatchResult {
            basis, metadata, components: 1, values: vec![value; 4],
            statuses: vec![LaneStatus::Success], fallback_used: vec![false], analytic_used: vec![false],
            primal_envelopes: vec![Some(envelope)], primal_envelope_errors: vec![None],
            estimated_errors: Vec::new(), best_effort: Vec::new(),
        };
        let mut target = source.clone();
        target.inherit_primal_envelope(0, &source, 0, PrimalEnvelopeRoute::Analytic);
        let retained = &target.primal_envelope(0).unwrap().operands[0].amplitude;
        assert_eq!(retained.re.get_precision(), amplitude.re.get_precision());
        assert_eq!(retained.im.get_precision(), amplitude.im.get_precision());
        target.values[0].re = Float::with_val(64, 3);
        assert!(target.primal_envelope(0).unwrap_err().contains("returned coefficient"));
    }
    #[test]
    fn full_component_context_rejects_missing_stale_and_invalid_derivative_operands() {
        let shape=vec![vec![0,0],vec![1,0],vec![0,1],vec![2,0],vec![1,1],vec![0,2]];
        let mut source=fixture(1,Some(shape));
        assert!(source.validated_component_amplitudes(0,53,53).is_err());
        let snapshot=source.values.clone();
        let expected=source.basis.len()*4*source.components;
        let envelope=source.primal_envelopes[0].as_mut().unwrap();
        envelope.output_components=Some(snapshot);
        envelope.component_amplitudes=Some(vec![Complex::new(7.,0.);expected]);
        assert_eq!(source.validated_component_amplitudes(0,53,53).unwrap().len(),expected);
        for mutation in 0..5 {
            let mut bad=source.clone();
            match mutation {
                0=>bad.values[4].re+=1., // mixed derivative; primal unchanged
                1=>bad.primal_envelopes[0].as_mut().unwrap().component_amplitudes.as_mut().unwrap()[4].re=-1.,
                2=>{bad.primal_envelopes[0].as_mut().unwrap().output_components.as_mut().unwrap().pop();},
                3=>bad.primal_envelopes[0].as_mut().unwrap().component_amplitudes.as_mut().unwrap()[4].im=1.,
                _=>bad.primal_envelopes[0].as_mut().unwrap().output_components=None,
            }
            assert!(bad.validated_component_amplitudes(0,53,53).is_err(),"mutation {mutation}");
        }
        let mut destination=source.clone();
        destination.inherit_primal_envelope(0,&source,0,PrimalEnvelopeRoute::Hybrid);
        assert_eq!(destination.validated_component_amplitudes(0,53,53).unwrap(),
            source.validated_component_amplitudes(0,53,53).unwrap());
        assert_eq!(destination.primal_envelope(0).unwrap().routes,
            [PrimalEnvelopeRoute::Ordinary,PrimalEnvelopeRoute::Hybrid]);
    }

    #[test]
    fn full_component_context_binds_nonprimal_float_precision() {
        let value=Complex::new(Float::with_val(192,3),Float::new(192));
        let amplitude=Complex::new(Float::with_val(80,1000),Float::new(96));
        let basis=vec![Monomial([0;4])];
        let metadata=ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,
            Some(vec![vec![0],vec![1],vec![2]]));
        let values=vec![value.clone();12];
        let envelope=PrimalEnvelope {basis:basis.clone(),metadata:metadata.clone(),components:3,
            working_precision:256,accuracy_bits:53,
            operands:(0..4).map(|output_index|PrimalOperand {output_index,amplitude:amplitude.clone(),
                kind:PrimalOperandKind::UncancelledMasters}).collect(),
            output_primals:vec![value;4],component_amplitudes:Some(vec![amplitude;12]),
            output_components:Some(values.clone()),routes:vec![PrimalEnvelopeRoute::Ordinary]};
        let mut result=BatchResult {basis,components:3,metadata,values,statuses:vec![LaneStatus::Success],
            fallback_used:vec![false],analytic_used:vec![false],primal_envelopes:vec![Some(envelope)],
            primal_envelope_errors:vec![None],
            estimated_errors: Vec::new(), best_effort: Vec::new(),};
        let amplitudes=result.validated_component_amplitudes(0,256,53).unwrap();
        assert_eq!(amplitudes[1].re.get_precision(),80);
        result.values[1].re=Float::with_val(64,3);
        assert!(result.validated_component_amplitudes(0,256,53).unwrap_err().contains("returned Taylor"));
    }

    #[test]
    fn full_component_context_binds_signed_zero_in_primals_and_higher_jets() {
        for (component,imaginary) in [(0,false),(0,true),(1,false),(2,true)] {
            let mut source=fixture(1,Some(vec![vec![0],vec![1],vec![2]]));
            let index=3*3+component;
            if imaginary {source.values[index].im=0.;} else {source.values[index].re=0.;}
            let snapshot=source.values.clone();
            let expected=source.basis.len()*4*source.components;
            let envelope=source.primal_envelopes[0].as_mut().unwrap();
            envelope.output_primals=snapshot.chunks_exact(3).map(|jet|jet[0]).collect();
            envelope.output_components=Some(snapshot);
            envelope.component_amplitudes=Some(vec![Complex::new(1.,0.);expected]);
            assert!(source.validated_component_amplitudes(0,53,53).is_ok());
            if imaginary {source.values[index].im=-0.;} else {source.values[index].re=-0.;}
            assert!(source.validated_component_amplitudes(0,53,53).is_err(),"component {component}, imaginary {imaginary}");
        }
    }

}
