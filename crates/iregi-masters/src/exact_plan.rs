//! Precision-independent preparation for scalar-master batch evaluators.

use crate::{
    Complex, DualNumberStructure, MasterError, MasterEvaluator, MasterExecutionPolicy, NativeFloat, ScalarIntegral,
    exact_scalar_master_evaluator, expressions, regular_bubble_finite, validate_dual_structure,
};
use symbolica::{
    domains::rational::Rational,
    evaluate::{EvaluationDomain, ExpressionEvaluator},
};

/// Exact Symbolica instructions that can be instantiated at multiple precisions.
///
/// Only rational constants and the validated, ordered Dual shape are retained.
/// Numeric constants, input samples, and mutable batch buffers belong to each
/// independently instantiated [`MasterEvaluator`]. This allows precision rescue
/// to reuse symbolic preparation without reusing rounded instructions or values.
#[derive(Clone)]
pub struct ExactMasterPlan {
    family: ScalarIntegral,
    shape: Option<Vec<Vec<usize>>>,
    exact: ExpressionEvaluator<Complex<Rational>>,
}

impl ExactMasterPlan {
    pub fn new(family: ScalarIntegral) -> Result<Self, MasterError> {
        Ok(Self {
            family,
            shape: None,
            exact: exact_scalar_master_evaluator(family)?,
        })
    }

    /// Compile the complete Dual instruction program once for this ordered shape.
    pub fn new_dual<D: DualNumberStructure>(
        family: ScalarIntegral,
        dual: D,
    ) -> Result<Self, MasterError> {
        validate_dual_structure(&dual)?;
        let shape = dual
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect();
        let (context, parameters, mut expressions) = expressions(family, true);
        if matches!(family, ScalarIntegral::B0 | ScalarIntegral::DB0) {
            expressions[0] = regular_bubble_finite(
                &parameters,
                dual.get_max_depth(),
                family == ScalarIntegral::DB0,
                &expressions[0],
            )?;
        }
        let exact = context
            .dual_evaluator(&expressions, &parameters, dual)
            .map_err(MasterError::Preparation)?;
        Ok(Self {
            family,
            shape: Some(shape),
            exact,
        })
    }

    /// Map exact coefficients directly to the supplied precision and numeric domain.
    ///
    /// This always selects Symbolica's instruction-major batch interpreter. The
    /// independently generated Rust backend is selected with
    /// [`MasterEvaluator::new_with_backend`].
    pub fn instantiate<T: NativeFloat>(
        &self,
        prototype: T,
    ) -> Result<MasterEvaluator<T>, MasterError>
    where
        Complex<T>: EvaluationDomain,
    {
        self.instantiate_with_policy(prototype, MasterExecutionPolicy::Default)
    }

    /// Instantiate with a requested policy. `Default` preserves this exact
    /// plan's Symbolica backend. The selective policy replaces only ordinary
    /// f64 A0/B0/C0; all Dual plans and other numeric domains remain Symbolica.
    pub fn instantiate_with_policy<T: NativeFloat>(
        &self,
        prototype: T,
        policy: MasterExecutionPolicy,
    ) -> Result<MasterEvaluator<T>, MasterError>
    where
        Complex<T>: EvaluationDomain,
    {
        policy.validate_available()?;
        MasterEvaluator::from_exact(
            self.family,
            prototype,
            self.shape.clone(),
            self.exact.clone(),
        )?
        .select_execution_policy(policy)
    }

    pub fn family(&self) -> ScalarIntegral {
        self.family
    }

    pub fn component_count(&self) -> usize {
        self.shape.as_ref().map_or(1, Vec::len)
    }

    pub fn dual_shape(&self) -> Option<&[Vec<usize>]> {
        self.shape.as_deref()
    }
}
