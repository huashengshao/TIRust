//! Explicit seven-point scalar reduction through one packed five-point batch.
//!
//! This interface is restricted to rank zero, unit propagator powers,
//! D=4-2epsilon, preserved Cartesian shifts, PV, native IREGI normalization and
//! raw tensor components. Every one of the 21 five-point subsets is required.
//! Singular active charts, failed children and unavailable child error contexts
//! remain lane failures. Ordinary and Hybrid dispatch are unchanged.
//!
//! Current locked-Cargo builds pass focused tests and Float128/Float256 scalar
//! comparisons on one real-mass family at requested53. The retained batch
//! lifecycle stopped after six completed calls; independent-scratch and later
//! recovery steps remain unqualified. See `docs/seven-point.md` for scope.
//! Generic typing does not establish binary64, DoubleFloat, complex-mass or
//! Dual qualification. There is deliberately no public Dual constructor.

use iregi_masters::NativeFloat;
use symbolica::{domains::float::Complex, evaluate::EvaluationDomain};

use crate::{
    Algorithm, BatchResult, ContractionPlan, Convention, InputLayout, InputPolicy,
    InvariantInput, KinematicsBatch, ReductionError, ReductionSettings, TensorWeights,
    basis::Monomial,
    thin_n7::{ThinSevenReductionPlan, ThinSevenWorkspace},
};

pub use crate::thin_n7::SevenPointExecutionStats;

/// Immutable, explicitly selected seven-point scalar plan. Unsupported settings
/// are rejected before preparing symbolic geometry or child reductions.
#[derive(Clone, Debug)]
pub struct SevenPointReductionPlan {
    inner: ThinSevenReductionPlan,
}

impl SevenPointReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        validate_settings(&settings)?;
        Ok(Self { inner: ThinSevenReductionPlan::new(settings)? })
    }

    pub fn settings(&self) -> &ReductionSettings { self.inner.settings() }
    pub fn basis(&self) -> &[Monomial] { self.inner.basis() }
    pub fn input_layout(&self) -> &InputLayout { self.inner.input_layout() }
    /// Twenty-eight shift components, seven squared masses and mu (not mu²).
    pub fn parameter_count(&self) -> usize { 36 }

    /// Prepare an existing Symbolica numerator contraction against the native
    /// moments without exposing the private imported-child parameters.
    pub fn contraction_plan(&self, monomials: &[Monomial]) -> Result<ContractionPlan, ReductionError> {
        self.inner.contraction_plan(monomials)
    }
}

fn validate_settings(settings: &ReductionSettings) -> Result<(), ReductionError> {
    if settings.propagators != 7
        || settings.rank != 0
        || settings.powers != [1; 7]
        || settings.dimension_shift != 0
        || settings.algorithm != Algorithm::PassarinoVeltman
        || settings.invariants != InvariantInput::FromShifts
        || settings.input_policy != InputPolicy::Preserve
        || settings.convention != Convention::IregiNative
        || settings.tensor_weights != TensorWeights::Components
    {
        return Err(ReductionError::Unsupported(
            "explicit seven-point reduction requires rank=0, seven unit powers, D=4-2epsilon, preserved Cartesian shifts, PV and native raw components".into(),
        ));
    }
    Ok(())
}

/// Owned scratch for ordinary numerical values, with one numerical component.
///
/// All 21 five-point children are evaluated in one subset-major batch across
/// active parent lanes. Their actual-route primal sensitivity operands feed
/// the guarded parent expression. These are ordinary sensitivity estimates,
/// not rigorous error bounds or derivative error certificates. Unsupported
/// quadrature-error transport cannot silently produce a successful parent.
#[derive(Clone)]
pub struct SevenPointWorkspace<T: NativeFloat> {
    inner: ThinSevenWorkspace<T>,
}

impl<T: NativeFloat> SevenPointWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &SevenPointReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        Ok(Self { inner: ThinSevenWorkspace::new(&plan.inner, prototype)? })
    }

    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        self.inner.set_accuracy_precision(bits)
    }
    pub fn component_count(&self) -> usize { self.inner.component_count() }
    pub fn binary_precision(&self) -> u32 { self.inner.binary_precision() }
    pub fn last_execution(&self) -> &SevenPointExecutionStats { self.inner.last_execution() }
    pub fn last_diagnostics(&self) -> &[Vec<String>] { self.inner.last_diagnostics() }

    /// Return all four native Laurent slots. Failed lanes remain inaccessible
    /// through `BatchResult::lane`, including after successful previous calls.
    pub fn reduce_batch(&mut self, batch: &KinematicsBatch<T>) -> Result<BatchResult<T>, ReductionError> {
        self.inner.reduce_batch(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> ReductionSettings {
        let mut settings = ReductionSettings::new(7, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings
    }

    #[test]
    fn unsupported_domain_is_rejected_before_symbolic_preparation() {
        let mut variants = Vec::new();
        for rank in [1, 3, 7, 8] { let mut x = settings(); x.rank = rank; variants.push(x); }
        for n in [0, 6, 8] { let mut x = settings(); x.propagators = n; variants.push(x); }
        for powers in [vec![1; 6], vec![1; 8], vec![0; 7], vec![2; 7]] {
            let mut x = settings(); x.powers = powers; variants.push(x);
        }
        for shift in [-1, 1] { let mut x = settings(); x.dimension_shift = shift; variants.push(x); }
        for algorithm in [Algorithm::Ibp, Algorithm::Hybrid] {
            let mut x = settings(); x.algorithm = algorithm; variants.push(x);
        }
        for convention in [Convention::OneLoop, Convention::MadLoop] {
            let mut x = settings(); x.convention = convention; variants.push(x);
        }
        let mut x = settings(); x.tensor_weights = TensorWeights::Multinomial; variants.push(x);
        let mut x = settings(); x.invariants = InvariantInput::SuppliedScalarProducts; variants.push(x);
        let mut x = settings(); x.input_policy = InputPolicy::legacy_onshell(); variants.push(x);
        for invalid in variants {
            assert!(matches!(SevenPointReductionPlan::new(invalid), Err(ReductionError::Unsupported(_))));
        }
    }

    #[test]
    fn exact_supported_domain_passes_the_cheap_preparation_guard() {
        // This checks only the public guard; the accepted constructor has a
        // separately budgeted symbolic-geometry validation requirement.
        assert!(validate_settings(&settings()).is_ok());
    }
}
