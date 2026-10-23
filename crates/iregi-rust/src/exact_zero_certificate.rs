//! Shared exact full-Taylor polynomial predicates for analytic branches.
//!
//! All algebra and arithmetic reuse Symbolica's determinant/polynomial and
//! certified ball batch evaluators. This adapter only interprets the existing
//! certificate engine with a constant unit chart.

use symbolica::{
    atom::Atom,
    domains::{dual::DualNumberStructure, float::Complex, rational::Rational},
};

use crate::{
    ReductionError,
    invariant_certificate::{
        InvariantCertificate, InvariantCertificatePlan, InvariantCertificateWorkspace,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ExactZeroCertificate {
    /// Every requested polynomial vanishes in every Taylor component.
    Zero,
    Nonzero {
        constraint: usize,
        component: usize,
    },
    /// Zero-containing enclosures never establish an equality.
    Unresolved {
        working_bits: u32,
    },
}

/// Exact polynomial constraints in the supplied ordered parameter layout.
/// Preparation enforces rational polynomials in the supplied symbols, rejecting
/// functions, non-polynomial powers, unknown symbols and duplicate parameters. Construct
/// determinant constraints with Symbolica's matrix API before preparing this
/// plan; branch predicates and approximate zero tests are not certificates.
#[derive(Clone)]
pub(crate) struct ExactZeroConstraintPlan(InvariantCertificatePlan);

impl ExactZeroConstraintPlan {
    pub(crate) fn new(parameters: &[Atom], residuals: &[Atom]) -> Result<Self, ReductionError> {
        InvariantCertificatePlan::from_zero_polynomials(parameters, residuals, None).map(Self)
    }

    pub(crate) fn new_dual<D: DualNumberStructure>(
        parameters: &[Atom],
        residuals: &[Atom],
        dual: &D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(dual)
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        let shape = dual
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect();
        InvariantCertificatePlan::from_zero_polynomials(parameters, residuals, Some(shape))
            .map(Self)
    }

    pub(crate) fn instantiate(&self) -> ExactZeroConstraintWorkspace {
        ExactZeroConstraintWorkspace(self.0.instantiate())
    }
}

#[derive(Clone)]
pub(crate) struct ExactZeroConstraintWorkspace(InvariantCertificateWorkspace);

impl ExactZeroConstraintWorkspace {
    pub(crate) fn set_precision_budget(
        &mut self,
        initial_bits: u32,
        maximum_bits: u32,
    ) -> Result<(), ReductionError> {
        self.0.set_precision_budget(initial_bits, maximum_bits)
    }

    pub(crate) fn evaluated_rows(&self) -> usize {
        self.0.evaluated_rows()
    }

    pub(crate) fn passes(&self) -> usize {
        self.0.passes()
    }

    /// Lane, parameter, then ordered Taylor component. Inputs are exact
    /// represented values, converted through `ExactInvariantScalar`.
    pub(crate) fn check_batch(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
    ) -> Result<Vec<ExactZeroCertificate>, ReductionError> {
        convert_certificates(self.0.check_batch(input, rows)?)
    }

    /// Certify only the selected `(constraint, Taylor component)` pairs in
    /// each lane. Other components may be nonzero, as in a lightlike primal
    /// with a transverse derivative. Empty lane selections perform no numeric
    /// evaluation. Refinement retains the original lane's selection.
    pub(crate) fn check_selected_batch(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
        selected: &[Vec<(usize, usize)>],
    ) -> Result<Vec<ExactZeroCertificate>, ReductionError> {
        convert_certificates(self.0.check_selected_batch(input, rows, selected)?)
    }
}

fn convert_certificates(
    certificates: Vec<InvariantCertificate>,
) -> Result<Vec<ExactZeroCertificate>, ReductionError> {
    certificates
        .into_iter()
        .map(|certificate| match certificate {
            InvariantCertificate::Compatible { .. } => Ok(ExactZeroCertificate::Zero),
            InvariantCertificate::Incompatible {
                pair, component, ..
            } => Ok(ExactZeroCertificate::Nonzero {
                constraint: pair.0,
                component,
            }),
            InvariantCertificate::Unresolved { working_bits } => {
                Ok(ExactZeroCertificate::Unresolved { working_bits })
            }
            InvariantCertificate::NoRegularChart => Err(ReductionError::Evaluation(
                "internal exact-zero certificate lost its constant unit chart".into(),
            )),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::{atom::AtomCore, domains::dual::HyperDual};

    fn c(value: i64) -> Complex<Rational> {
        Complex::new(Rational::from(value), Rational::from(0))
    }

    #[test]
    fn zero_polynomial_certificate_isolates_tiny_nonzero_rows() {
        let x = Atom::var(symbolica::symbol!("iregi_zero_certificate::x"));
        let y = Atom::var(symbolica::symbol!("iregi_zero_certificate::y"));
        let plan = ExactZeroConstraintPlan::new(&[x.clone(), y.clone()], &[&x - &y]).unwrap();
        let mut workspace = plan.instantiate();
        workspace.set_precision_budget(128, 128).unwrap();
        let mut tiny = c(1);
        tiny.re += Rational::from((1, 2)).pow(300);
        let rows = [c(1), c(1), tiny, c(1), c(2), c(1)];
        assert_eq!(
            workspace.check_batch(&rows, 3).unwrap(),
            [
                ExactZeroCertificate::Zero,
                ExactZeroCertificate::Unresolved { working_bits: 128 },
                ExactZeroCertificate::Nonzero {
                    constraint: 0,
                    component: 0
                },
            ]
        );
        workspace.set_precision_budget(128, 1024).unwrap();
        assert_eq!(
            workspace.check_batch(&rows, 3).unwrap(),
            [
                ExactZeroCertificate::Zero,
                ExactZeroCertificate::Nonzero {
                    constraint: 0,
                    component: 0
                },
                ExactZeroCertificate::Nonzero {
                    constraint: 0,
                    component: 0
                },
            ]
        );
        assert!(workspace.evaluated_rows() > 3);
        assert!(workspace.passes() > 1);
        assert!(workspace.check_batch(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn zero_polynomial_certificate_keeps_ordered_mixed_dual_components() {
        let x = Atom::var(symbolica::symbol!("iregi_zero_certificate::x"));
        let y = Atom::var(symbolica::symbol!("iregi_zero_certificate::y"));
        let shape = HyperDual::<u8>::new(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
        let plan = ExactZeroConstraintPlan::new_dual(
            &[x.clone(), y.clone()],
            &[&x - &y, &x * &x - &y * &y],
            &shape,
        )
        .unwrap();
        let mut workspace = plan.instantiate();
        let good = [1, 2, 3, 4, 1, 2, 3, 4].map(c);
        let bad = [1, 2, 3, 5, 1, 2, 3, 4].map(c);
        assert_eq!(
            workspace.check_batch(&[good, bad].concat(), 2).unwrap(),
            [
                ExactZeroCertificate::Zero,
                ExactZeroCertificate::Nonzero {
                    constraint: 0,
                    component: 3
                },
            ]
        );
    }

    #[test]
    fn zero_certificate_rejects_non_polynomial_or_undeclared_inputs() {
        let x = Atom::var(symbolica::symbol!("iregi_zero_certificate::x"));
        let y = Atom::var(symbolica::symbol!("iregi_zero_certificate::y"));
        let function = symbolica::function!(
            symbolica::symbol!("iregi_zero_certificate::unknown"),
            x.clone()
        );
        for residual in [function, x.pow(-1), y.clone()] {
            assert!(ExactZeroConstraintPlan::new(&[x.clone()], &[residual]).is_err());
        }
        assert!(ExactZeroConstraintPlan::new(&[x.clone(), x.clone()], &[x.clone()]).is_err());
        assert!(ExactZeroConstraintPlan::new(&[&x + &y], &[x]).is_err());
    }

    #[test]
    fn selected_zero_components_preserve_nonzero_jets_and_refinement_lane_masks() {
        let x = Atom::var(symbolica::symbol!("iregi_zero_certificate::x"));
        let y = Atom::var(symbolica::symbol!("iregi_zero_certificate::y"));
        let shape = HyperDual::<u8>::new(vec![vec![0], vec![1]]);
        let plan = ExactZeroConstraintPlan::new_dual(
            &[x.clone(), y.clone()],
            &[&x - &y, &x * &x - &y * &y],
            &shape,
        )
        .unwrap();
        let mut workspace = plan.instantiate();
        let nonzero_jet = [c(1), c(2), c(1), c(3)];
        let mut tiny = c(1);
        tiny.re += Rational::from((1, 2)).pow(300);
        let tiny_row = [tiny, c(1), c(1), c(1)];
        let input = [
            nonzero_jet.clone(),
            tiny_row,
            nonzero_jet.clone(),
            nonzero_jet,
        ]
        .concat();
        let selected = vec![vec![(0, 1)], vec![(1, 0)], vec![(0, 0), (1, 0)], vec![]];
        workspace.set_precision_budget(128, 1024).unwrap();
        assert_eq!(
            workspace
                .check_selected_batch(&input, 4, &selected)
                .unwrap(),
            [
                ExactZeroCertificate::Nonzero {
                    constraint: 0,
                    component: 1
                },
                ExactZeroCertificate::Nonzero {
                    constraint: 1,
                    component: 0
                },
                ExactZeroCertificate::Zero,
                ExactZeroCertificate::Zero,
            ]
        );
        assert!(workspace.passes() > 1);
        assert!(workspace.evaluated_rows() > 3);
        assert!(
            workspace
                .check_selected_batch(&input, 4, &[vec![]])
                .is_err()
        );
        for invalid in [(2, 0), (0, 2)] {
            assert!(
                workspace
                    .check_selected_batch(&input, 4, &vec![vec![invalid]; 4])
                    .is_err()
            );
        }
        assert_eq!(
            workspace
                .check_selected_batch(&input, 4, &vec![vec![]; 4])
                .unwrap(),
            vec![ExactZeroCertificate::Zero; 4]
        );
        assert_eq!(workspace.evaluated_rows(), 0);
        assert_eq!(workspace.passes(), 0);
    }
}
