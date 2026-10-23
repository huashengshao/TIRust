//! Generalized scalar integrals from supplied scalar products alone.
//!
//! This is the numerical counterpart of IREGI's scalar-product scalar API.
//! It shares the tensor reducer's exact plans, batch executor, conventions and
//! diagnostics. No coordinate reconstruction or additional algebra is needed.

use iregi_masters::NativeFloat;
use symbolica::{
    domains::{
        dual::DualNumberStructure,
        float::{Complex, FloatLike},
    },
    evaluate::EvaluationDomain,
};

use crate::{
    BatchResult, ExecutionProfile, ExecutionStats, InvariantInput, KinematicsBatch, ReductionError,
    ReductionPlan, ReductionSettings, Workspace,
};

/// Immutable rank-zero reduction with authoritative supplied scalar products.
#[derive(Clone, Debug)]
pub struct ScalarReductionPlan {
    tensor: ReductionPlan,
    arguments: usize,
}

impl ScalarReductionPlan {
    /// Preserve powers, dimension shift, algorithm, normalization and input
    /// policy. Rank must be zero; invariants are supplied as `G_ij = p_i·p_j`.
    pub fn new(mut settings: ReductionSettings) -> Result<Self, ReductionError> {
        if settings.rank != 0 {
            return Err(ReductionError::InvalidInput(
                "a scalar reduction plan requires rank zero".into(),
            ));
        }
        if settings.propagators >= 6 {
            return Err(ReductionError::Unsupported(
                "scalar-product-only reduction with six or more propagators requires validated four-dimensional Gram constraints in every Taylor component".into(),
            ));
        }
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        let tensor = ReductionPlan::new(settings)?;
        let arguments = tensor.input_layout().scale + 1 - tensor.input_layout().shifts.len();
        Ok(Self { tensor, arguments })
    }

    pub fn tensor_plan(&self) -> &ReductionPlan {
        &self.tensor
    }

    /// Number of scalar arguments per sample, before Taylor expansion.
    pub fn argument_count(&self) -> usize {
        self.arguments
    }
}

/// Rows contain packed upper-triangular `G` (including its diagonal), complex
/// squared masses in propagator order, then the real scale `mu`, not `mu²`.
/// Each argument stores `components` consecutive Taylor coefficients, in the
/// exact order of the workspace's static or dynamic Symbolica Dual layout.
#[derive(Clone, Debug)]
pub struct ScalarKinematicsBatch<T: FloatLike> {
    pub rows: usize,
    pub components: usize,
    pub values: Vec<Complex<T>>,
}

/// Owned scalar scratch storage. Packing adds no sample-by-sample reduction:
/// the complete batch is passed once to the shared instruction-major executor.
#[derive(Clone)]
pub struct ScalarWorkspace<T: NativeFloat> {
    tensor: Workspace<T>,
    arguments: usize,
    shifts: usize,
    zero: Complex<T>,
    packed: KinematicsBatch<T>,
}

impl<T: NativeFloat> ScalarWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub fn new(plan: &ScalarReductionPlan, prototype: T) -> Result<Self, ReductionError> {
        let tensor = Workspace::new(&plan.tensor, prototype.clone())?;
        Ok(Self::with_workspace(plan, prototype, tensor))
    }

    pub fn new_dual<D: DualNumberStructure>(
        plan: &ScalarReductionPlan,
        prototype: T,
        dual: D,
    ) -> Result<Self, ReductionError> {
        let tensor = Workspace::new_dual(&plan.tensor, prototype.clone(), dual)?;
        Ok(Self::with_workspace(plan, prototype, tensor))
    }

    fn with_workspace(plan: &ScalarReductionPlan, prototype: T, tensor: Workspace<T>) -> Self {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Self {
            tensor,
            arguments: plan.arguments,
            shifts: plan.tensor.input_layout().shifts.len(),
            zero,
            packed: KinematicsBatch {
                rows: 0,
                components: 1,
                values: Vec::new(),
            },
        }
    }

    pub fn reduce_batch(
        &mut self,
        input: &ScalarKinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        if input.components != self.tensor.component_count() {
            return Err(ReductionError::InvalidInput(
                "scalar input Taylor layout does not match workspace".into(),
            ));
        }
        let overflow = || ReductionError::InvalidInput("scalar batch size overflow".into());
        let input_width = self
            .arguments
            .checked_mul(input.components)
            .ok_or_else(overflow)?;
        let expected = input_width.checked_mul(input.rows).ok_or_else(overflow)?;
        if input.values.len() != expected {
            return Err(ReductionError::InvalidInput(format!(
                "scalar batch expects {expected} values, received {}",
                input.values.len()
            )));
        }
        let padding = self
            .shifts
            .checked_mul(input.components)
            .ok_or_else(overflow)?;
        let output_width = input_width.checked_add(padding).ok_or_else(overflow)?;
        let output_len = output_width.checked_mul(input.rows).ok_or_else(overflow)?;
        self.packed.values.resize(output_len, self.zero.clone());
        for row in 0..input.rows {
            let offset = row * output_width;
            self.packed.values[offset..offset + padding].fill(self.zero.clone());
            self.packed.values[offset + padding..offset + output_width]
                .clone_from_slice(&input.values[row * input_width..(row + 1) * input_width]);
        }
        self.packed.rows = input.rows;
        self.packed.components = input.components;
        self.tensor.reduce_batch(&self.packed)
    }

    pub fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        self.tensor.set_accuracy_precision(bits)
    }

    pub fn last_execution(&self) -> ExecutionStats {
        self.tensor.last_execution()
    }
    pub fn last_diagnostics(&self) -> &[Vec<String>] {
        self.tensor.last_diagnostics()
    }
    pub fn set_profiling(&mut self, enabled: bool) {
        self.tensor.set_profiling(enabled);
    }
    pub fn last_profile(&self) -> Option<ExecutionProfile> {
        self.tensor.last_profile()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Algorithm, Convention, LaneStatus};
    use symbolica::domains::dual::HyperDual;

    fn c(x: f64) -> Complex<f64> {
        Complex::new(x, 0.)
    }

    #[test]
    fn high_point_scalar_products_require_a_validated_physical_domain() {
        for propagators in [6, 7] {
            for algorithm in [
                Algorithm::Ibp,
                Algorithm::PassarinoVeltman,
                Algorithm::Hybrid,
            ] {
                let mut settings = ReductionSettings::new(propagators, 0);
                settings.algorithm = algorithm;
                let error = ScalarReductionPlan::new(settings.clone()).unwrap_err();
                assert!(matches!(error, ReductionError::Unsupported(_)));
                assert!(error.to_string().contains("every Taylor component"));
                settings.invariants = InvariantInput::SuppliedScalarProducts;
                for rank in [0, 2] {
                    settings.rank = rank;
                    let error = ReductionPlan::new(settings.clone()).unwrap_err();
                    assert!(matches!(error, ReductionError::Unsupported(_)));
                    assert!(error.to_string().contains("every Taylor component"));
                }
            }
        }
    }

    #[test]
    fn generalized_scalar_interface_matches_mass_derivative_and_isolates_lanes() {
        let mut settings = ReductionSettings::new(1, 0);
        settings.powers = vec![2];
        settings.algorithm = Algorithm::Ibp;
        settings.convention = Convention::OneLoop;
        let plan = ScalarReductionPlan::new(settings).unwrap();
        assert_eq!(plan.argument_count(), 3);
        let mut workspace = ScalarWorkspace::new(&plan, 0_f64).unwrap();
        let input = ScalarKinematicsBatch {
            rows: 3,
            components: 1,
            values: vec![
                c(7.),
                c(2.),
                c(3.),
                c(0.),
                c(4.),
                c(-1.),
                c(-9.),
                c(5.),
                c(2.),
            ],
        };
        let result = workspace.reduce_batch(&input).unwrap();
        for (lane, mass, mu) in [(0, 2_f64, 3_f64), (2, 5., 2.)] {
            let slots = result.lane(lane).unwrap();
            assert!((slots[0].re - 1.).abs() < 1e-13);
            assert!(slots[1].re.hypot(slots[1].im) < 1e-13);
            assert!(slots[2].re.hypot(slots[2].im) < 1e-13);
            assert!((slots[3].re - (mu * mu / mass).ln()).abs() < 1e-12);
        }
        assert!(matches!(result.statuses[1], LaneStatus::InvalidInput(_)));
        assert_eq!(workspace.last_execution().reduction_rows, 2);
        let empty = workspace
            .reduce_batch(&ScalarKinematicsBatch {
                rows: 0,
                components: 1,
                values: vec![],
            })
            .unwrap();
        assert_eq!(empty.rows(), 0);
    }

    #[test]
    fn scalar_input_preserves_full_taylor_order_and_scale_derivatives() {
        let mut settings = ReductionSettings::new(1, 0);
        settings.convention = Convention::OneLoop;
        let plan = ScalarReductionPlan::new(settings).unwrap();
        let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
        let mut workspace = ScalarWorkspace::new_dual(&plan, 0_f64, shape).unwrap();
        let row = vec![
            c(17.),
            c(3.),
            c(0.),
            c(2.),
            c(1.),
            c(0.),
            c(3.),
            c(1.),
            c(0.),
        ];
        let result = workspace
            .reduce_batch(&ScalarKinematicsBatch {
                rows: 3,
                components: 3,
                values: row.repeat(3),
            })
            .unwrap();
        for lane in 0..3 {
            let slots = result.lane(lane).unwrap();
            assert_eq!(&slots[..3], &[c(2.), c(1.), c(0.)]);
            assert!((slots[10].re - ((9_f64 / 2.).ln() + 4. / 3.)).abs() < 1e-12);
            assert!((slots[11].re - (-0.25 + 2. / 3. - 2. / 9.)).abs() < 1e-12);
        }
        assert_eq!(workspace.last_execution().master_rows, 3);
    }

    #[test]
    fn scalar_products_alone_set_the_massless_bubble_invariant() {
        let mut settings = ReductionSettings::new(2, 0);
        settings.convention = Convention::OneLoop;
        let plan = ScalarReductionPlan::new(settings).unwrap();
        assert_eq!(plan.argument_count(), 6);
        let mut workspace = ScalarWorkspace::new(&plan, 0_f64).unwrap();
        let spacelike = [4_f64, 16., 9., 4., 25.];
        let values = spacelike
            .iter()
            .flat_map(|s| [c(0.), c(0.), c(-s), c(0.), c(0.), c(3.)])
            .collect();
        let result = workspace
            .reduce_batch(&ScalarKinematicsBatch {
                rows: 5,
                components: 1,
                values,
            })
            .unwrap();
        for (lane, s) in spacelike.into_iter().enumerate() {
            let slots = result.lane(lane).unwrap();
            assert!((slots[0].re - 1.).abs() < 1e-13);
            assert_eq!(slots[1], c(0.));
            assert_eq!(slots[2], c(0.));
            assert!((slots[3].re - (2. - (s / 9.).ln())).abs() < 1e-12);
            assert!(slots[3].im.abs() < 1e-13);
        }
    }
}
