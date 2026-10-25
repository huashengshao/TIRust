//! Explicit six-point Cartesian reduction through one batched five-point child
//! evaluation. This interface accepts ranks zero through seven.
//!
//! The supported domain is unit propagator powers, D=4-2epsilon, preserved
//! Cartesian shifts, PV, native IREGI normalization and raw tensor components.
//! Singular charts and unavailable child error contexts remain lane failures.
//! This interface does not change ordinary or Hybrid reduction dispatch.
//!
//! Isolated public-API qualification covers the retained real-mass base fixture
//! with Symbolica Float128/Float256 at requested53; scaled-input lifecycle
//! evidence is Float128 only. See `validation/highpoint-six-point-api` for the
//! exact source and dependency context. Generic numeric typing does not imply
//! successful binary64 or DoubleFloat qualification.
//! Public numerical Dual support remains deferred. Rank-seven reference
//! validation uses higher-precision original IREGI IBP: its binary64 version
//! loses accuracy on the retained fixture. Qualification of particular builds,
//! numeric types and kinematic domains remains explicit in the evidence.

use iregi_masters::NativeFloat;
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

use crate::{
    Algorithm, BatchResult, ContractionPlan, Convention, InputLayout, InputPolicy,
    InvariantInput, KinematicsBatch, ReductionError, ReductionSettings, TensorWeights,
    basis::Monomial,
    thin_n6::{ThinSixReductionPlan, ThinSixWorkspace},
};

pub use crate::thin_n6::SixPointExecutionStats;

/// Immutable, explicitly selected six-point plan, currently restricted to rank
/// at most seven. All requested settings are checked before symbolic work.
#[derive(Clone, Debug)]
pub struct SixPointReductionPlan {
    inner: ThinSixReductionPlan,
}

impl SixPointReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        validate_settings(&settings)?;
        Ok(Self { inner: ThinSixReductionPlan::new(settings)? })
    }

    pub fn settings(&self) -> &ReductionSettings { self.inner.settings() }
    pub fn basis(&self) -> &[Monomial] { self.inner.basis() }
    pub fn input_layout(&self) -> &InputLayout { self.inner.input_layout() }
    /// Twenty-four shift components, six squared masses and mu (not mu squared).
    pub fn parameter_count(&self) -> usize { 31 }

    /// Prepare the existing Symbolica numerator contraction against this plan's
    /// raw native moments without exposing its private imported parameters.
    pub fn contraction_plan(&self, monomials: &[Monomial]) -> Result<ContractionPlan, ReductionError> {
        self.inner.contraction_plan(monomials)
    }
}

pub(crate) fn validate_settings(settings: &ReductionSettings) -> Result<(), ReductionError> {
    if settings.propagators != 6
        || settings.rank > 7
        || settings.powers != [1; 6]
        || settings.dimension_shift != 0
        || settings.algorithm != Algorithm::PassarinoVeltman
        || settings.invariants != InvariantInput::FromShifts
        || settings.input_policy != InputPolicy::Preserve
        || settings.convention != Convention::IregiNative
        || settings.tensor_weights != TensorWeights::Components
    {
        return Err(ReductionError::Unsupported(
            "explicit six-point reduction requires rank<=7, six unit powers, D=4-2epsilon, preserved Cartesian shifts, PV and native raw components".into(),
        ));
    }
    Ok(())
}

/// Owned scratch for ordinary numerical values. Sample lanes remain distinct
/// from Taylor components; this public constructor supports one component.
///
/// Internal five-point retries retain their actual-route primal sensitivity
/// operands. This preserves the ordinary stability model; it is not a rigorous
/// error certificate, and a child requiring unsupported quadrature-error
/// transport cannot silently become a successful parent lane.
#[derive(Clone)]
pub struct SixPointWorkspace<T: NativeFloat> {
    inner: ThinSixWorkspace<T>,
}

impl<T: NativeFloat> SixPointWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &SixPointReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Ok(Self { inner: ThinSixWorkspace::new(&plan.inner, prototype)? })
    }

    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        self.inner.set_accuracy_precision(bits)
    }
    pub fn component_count(&self) -> usize { self.inner.component_count() }
    pub fn binary_precision(&self) -> u32 { self.inner.binary_precision() }
    pub fn last_execution(&self) -> &SixPointExecutionStats { self.inner.last_execution() }
    pub fn last_diagnostics(&self) -> &[Vec<String>] { self.inner.last_diagnostics() }

    /// Return all cumulative tensor moments and all four Laurent slots. Failed
    /// lanes remain inaccessible through `BatchResult::lane`, including when a
    /// neighboring lane or a previous call succeeded.
    pub fn reduce_batch(&mut self, batch: &KinematicsBatch<T>) -> Result<BatchResult<T>, ReductionError> {
        self.inner.reduce_batch(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ReductionSettings {
        let mut settings = ReductionSettings::new(6, 3);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings
    }

    #[test]
    fn unsupported_domain_is_rejected_before_symbolic_preparation() {
        let mut variants = Vec::new();
        for rank in [8, u16::MAX] { let mut x = settings(); x.rank = rank; variants.push(x); }
        for n in [0, 5, 7] { let mut x = settings(); x.propagators = n; variants.push(x); }
        for powers in [vec![1; 5], vec![1; 7], vec![0; 6], vec![2; 6]] {
            let mut x = settings(); x.powers = powers; variants.push(x);
        }
        for shift in [-1, 1] { let mut x = settings(); x.dimension_shift = shift; variants.push(x); }
        for algorithm in [Algorithm::Ibp, Algorithm::Hybrid] {
            let mut x = settings(); x.algorithm = algorithm; variants.push(x);
        }
        let mut x = settings(); x.convention = Convention::MadLoop; variants.push(x);
        let mut x = settings(); x.tensor_weights = TensorWeights::Multinomial; variants.push(x);
        let mut x = settings(); x.invariants = InvariantInput::SuppliedScalarProducts; variants.push(x);
        let mut x = settings(); x.input_policy = InputPolicy::legacy_onshell(); variants.push(x);
        for invalid in variants {
            assert!(matches!(SixPointReductionPlan::new(invalid), Err(ReductionError::Unsupported(_))));
        }
    }

    #[test]
    fn rank_seven_passes_the_public_domain_guard() {
        let mut settings = settings();
        settings.rank = 7;
        assert!(validate_settings(&settings).is_ok());
    }
}
