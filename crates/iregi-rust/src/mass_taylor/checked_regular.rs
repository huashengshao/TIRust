//! Fresh full compound-jet geometry checks before private mass-Taylor units.
//!
//! The existing exact affine guard owns all compatibility/rank arithmetic.
//! This adapter only identifies certified regular-primal lanes and maps failed
//! predicates to explicit statuses. It performs no integral evaluation.

use iregi_masters::NativeFloat;
use symbolica::domains::dual::HyperDual;

use crate::{
    CertificateExecutionStats, KinematicsBatch, LaneStatus, ReductionError,
    affine_descent_guard::{
        AffineDescentGuardPlan, AffineDescentGuardWorkspace, AffineDescentSelection,
    },
    pv::AffineDescentBudget,
};

#[derive(Clone, Debug)]
pub(super) struct CheckedRegularSelection {
    /// Physical input-lane indices, in their original order. These are the only
    /// rows eligible to enter the subsequent unit-integral numerical actor.
    pub(super) accepted: Vec<usize>,
    /// Success is a provisional geometry result only. The numerical actor must
    /// replace accepted statuses with its actual evaluation/scatter outcomes.
    pub(super) statuses: Vec<LaneStatus>,
    /// All attempted full compatibility and rank/chart/cofactor proof work.
    /// No integral-reduction batch has been executed by this selector.
    pub(super) stats: CertificateExecutionStats,
}

#[derive(Clone)]
pub(super) struct CheckedRegularUnitGuard {
    workspace: AffineDescentGuardWorkspace,
    stats: CertificateExecutionStats,
}

impl CheckedRegularUnitGuard {
    /// `shape` is the complete ordered auxiliary-mass x user Taylor algebra,
    /// including its primal. Components never become extra sample lanes.
    pub(super) fn new(shape: &[Vec<usize>]) -> Result<Self, ReductionError> {
        let dual = HyperDual::<u8>::new(shape.to_vec());
        let plan = AffineDescentGuardPlan::new_checked_supplied_dual(
            5, 3, AffineDescentBudget::default(), &dual,
        )?;
        Ok(Self { workspace: plan.instantiate(), stats: CertificateExecutionStats::default() })
    }

    pub(super) fn set_precision_budget(
        &mut self, initial: u32, maximum: u32,
    ) -> Result<(), ReductionError> {
        self.workspace.set_precision_budget(initial, maximum)
    }

    /// Harvestable even when the last selection returned a malformed-input or
    /// evaluation error. Cached workspaces never lend a previous call's count.
    pub(super) fn last_certificate_execution(&self) -> CertificateExecutionStats { self.stats }

    pub(super) fn select<T: NativeFloat>(
        &mut self, input: &KinematicsBatch<T>,
    ) -> Result<CheckedRegularSelection, ReductionError> {
        self.stats = CertificateExecutionStats { input_rows: input.rows, ..Default::default() };
        // select_batch validates the full 41-argument p/G/mass/mu input, then
        // checks ALL G/p residuals in EVERY compound Taylor component before
        // rank/chart/cofactor selection. The outer caller's proof is not reused.
        let selected = self.workspace.select_batch(input);
        self.stats.evaluated_rows = self.workspace.evaluated_rows();
        self.stats.passes = self.workspace.passes();
        let selected = selected?;
        if selected.len() != input.rows {
            return Err(ReductionError::Evaluation(
                "mass-Taylor regular-unit guard returned an incompatible lane count".into(),
            ));
        }
        let mut accepted = Vec::new();
        let mut statuses = Vec::with_capacity(input.rows);
        for (lane, selection) in selected.into_iter().enumerate() {
            let status = match selection {
                // N5/r3 has exactly one 4x4 relative-coordinate minor. Only its
                // certified nonzero primal authorizes the regular unit actor.
                AffineDescentSelection::OffAffineFamily { constraint: 0, component: 0 } => {
                    accepted.push(lane);
                    LaneStatus::Success
                }
                AffineDescentSelection::InvalidInput(reason) => LaneStatus::InvalidInput(reason),
                AffineDescentSelection::OffAffineFamily { constraint, component } => LaneStatus::Unstable(format!(
                    "mass-Taylor regular unit lacks a certified regular primal: rank constraint {constraint}, Taylor component {component} is nonzero; transverse rank-changing jets require another path"
                )),
                AffineDescentSelection::Chart { index, .. } => LaneStatus::Unstable(format!(
                    "mass-Taylor regular unit requires a nonzero primal rank-four determinant; certified planar chart {index} belongs to the checked affine path"
                )),
                AffineDescentSelection::NoRegularChart => LaneStatus::Unstable(
                    "mass-Taylor regular unit has no certified regular primal; lower-span or homogeneous geometry requires another path".into(),
                ),
                AffineDescentSelection::Unresolved { working_bits } => LaneStatus::Unstable(format!(
                    "mass-Taylor regular-unit exact geometry certificate unresolved at {working_bits} bits; input was preserved"
                )),
            };
            statuses.push(status);
        }
        Ok(CheckedRegularSelection { accepted, statuses, stats: self.stats })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::{float::{Complex, Float, SingleFloat}, rational::Rational};

    fn shape() -> Vec<Vec<usize>> {
        // The first degree is an auxiliary mass variable and the second a user
        // variable. Keep the deliberately explicit user-before-auxiliary order.
        vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    }
    fn regular_data() -> [i64; 41] {
        // p=[0,e0,e1,e2,e3], metric (+---), original upper G, m2, mu.
        [0,0,0,0, 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1,
         0,0,0,0,0, 1,0,0,0, -1,0,0,-1,0,-1, 4,9,16,25,36,3]
    }
    fn planar_data() -> [i64; 41] {
        [0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,1,
         0,0,0,0,0, -1,0,0,-1, -1,0,-1,-1,-1,-3, 4,9,16,25,36,3]
    }
    fn row<T: NativeFloat>(prototype: &T, data: &[i64; 41], components: usize) -> Vec<Complex<T>> {
        let mut row = vec![Complex::new(prototype.zero(), prototype.zero()); 41 * components];
        for (argument, &value) in data.iter().enumerate() {
            row[argument * components].re = prototype.from_rational(&Rational::from(value));
        }
        row
    }

    #[test]
    fn actual_checked_guard_isolates_good_bad_good_without_evaluating_units() {
        let mut guard = CheckedRegularUnitGuard::new(&[vec![0]]).unwrap();
        let good = row(&0_f64, &regular_data(), 1);
        let mut bad = good.clone(); bad[20].re = 1.;
        let input = KinematicsBatch { rows: 3, components: 1, values: [good.clone(), bad, good.clone()].concat() };
        let original = input.values.clone();
        let selected = guard.select(&input).unwrap();
        assert_eq!(input.values, original);
        assert_eq!(selected.accepted, [0, 2]);
        assert_eq!(selected.statuses[0], LaneStatus::Success);
        assert!(matches!(&selected.statuses[1], LaneStatus::InvalidInput(reason)
            if reason.contains("supplied-G/Cartesian") && reason.contains("component 0")));
        assert_eq!(selected.statuses[2], LaneStatus::Success);
        assert_eq!(selected.stats.input_rows, 3);
        assert_eq!(selected.stats.evaluated_rows, 5); // compatibility(3), regular rank(2)
        assert_eq!(selected.stats.passes, 2);
        assert_eq!(selected.stats.reduction_batches, 0);
        assert_eq!(guard.last_certificate_execution(), selected.stats);
        let mut independent = guard.clone();
        assert!(guard.select(&KinematicsBatch::<f64> { rows: 0, components: 1, values: vec![] }).unwrap().accepted.is_empty());
        assert_eq!(guard.last_certificate_execution(), CertificateExecutionStats::default());
        let singleton = KinematicsBatch { rows: 1, components: 1, values: good };
        assert_eq!(independent.select(&singleton).unwrap().accepted, [0]);
        assert_eq!(guard.select(&singleton).unwrap().accepted, [0]);
    }

    #[test]
    fn complete_auxiliary_mass_user_shape_rechecks_mixed_geometry_before_regular_rank() {
        let mut guard = CheckedRegularUnitGuard::new(&shape()).unwrap();
        let mut good = row(&0_f64, &regular_data(), 4);
        // An auxiliary mass seed and mixed mass/user coefficients are retained
        // independently of geometry. A common full-jet routing translation
        // leaves original relative G constant in every compound component.
        good[35 * 4 + 2].re = 1.;
        good[35 * 4 + 3].re = 5.;
        good[36 * 4 + 1].re = 2.;
        for line in 0..5 {
            good[(4 * line) * 4 + 1].re = 2.;
            good[(4 * line + 1) * 4 + 3].re = 3.;
        }
        let mut incompatible = good.clone(); incompatible[20 * 4 + 3].re = 1.;
        let input = KinematicsBatch { rows: 3, components: 4, values: [good.clone(), incompatible, good.clone()].concat() };
        let original = input.values.clone();
        let selected = guard.select(&input).unwrap();
        assert_eq!(input.values, original);
        assert_eq!(selected.accepted, [0, 2]);
        assert!(matches!(&selected.statuses[1], LaneStatus::InvalidInput(reason)
            if reason.contains("Taylor component 3")));
        assert_eq!(selected.stats.evaluated_rows, 5);
        let order = [2, 1, 0];
        let reordered = KinematicsBatch { rows: 3, components: 4, values: order.iter().flat_map(|&lane|
            input.values[lane * 164..(lane + 1) * 164].iter().cloned()).collect() };
        let actual = guard.select(&reordered).unwrap();
        assert_eq!(actual.accepted, [0, 2]);
        for (lane, &original) in order.iter().enumerate() { assert_eq!(actual.statuses[lane], selected.statuses[original]); }
        let bad_only = KinematicsBatch { rows: 1, components: 4, values: input.values[164..328].to_vec() };
        assert!(guard.select(&bad_only).unwrap().accepted.is_empty());
        assert_eq!(guard.last_certificate_execution().evaluated_rows, 1);
        assert_eq!(guard.select(&KinematicsBatch { rows: 1, components: 4, values: good }).unwrap().accepted, [0]);
    }

    #[test]
    fn planar_and_mixed_only_rank_changes_never_authorize_regular_unit_evaluation() {
        let mut guard = CheckedRegularUnitGuard::new(&shape()).unwrap();
        let planar = row(&0_f64, &planar_data(), 4);
        let mut transverse = planar.clone();
        // p4.time=aux*user preserves the full G jet in this algebra, while the
        // coordinate determinant first becomes nonzero in mixed component3.
        transverse[16 * 4 + 3].re = 1.;
        let mut singular = vec![Complex::new(0., 0.); 164];
        for mass in 35..40 { singular[mass * 4].re = 1.; }
        singular[40 * 4].re = 3.;
        let input = KinematicsBatch { rows: 3, components: 4, values: [planar, transverse, singular].concat() };
        let original = input.values.clone();
        let selected = guard.select(&input).unwrap();
        assert_eq!(input.values, original);
        assert!(selected.accepted.is_empty());
        assert!(matches!(&selected.statuses[0], LaneStatus::Unstable(reason) if reason.contains("planar chart")));
        assert!(matches!(&selected.statuses[1], LaneStatus::Unstable(reason)
            if reason.contains("Taylor component 3") && reason.contains("transverse rank-changing")));
        assert!(matches!(&selected.statuses[2], LaneStatus::Unstable(reason) if reason.contains("no certified regular primal")));
        assert_eq!(selected.stats.reduction_batches, 0);
    }

    #[test]
    fn unresolved_compatibility_and_malformed_dimensions_have_no_accepted_unit_rows() {
        let prototype = Float::new(512);
        let mut guard = CheckedRegularUnitGuard::new(&[vec![0]]).unwrap();
        let mut tiny = row(&prototype, &regular_data(), 1);
        let discrepancy = Rational::from(1) + Rational::from((1, 2)).pow(300);
        tiny[25].re = prototype.from_rational(&discrepancy); // G11=1+2^-300
        let input = KinematicsBatch { rows: 1, components: 1, values: tiny };
        guard.set_precision_budget(128, 128).unwrap();
        let selected = guard.select(&input).unwrap();
        assert!(selected.accepted.is_empty());
        assert!(matches!(&selected.statuses[0], LaneStatus::Unstable(reason) if reason.contains("unresolved at 128 bits")));
        assert_eq!(selected.stats.evaluated_rows, 1);
        guard.set_precision_budget(128, 1024).unwrap();
        let selected = guard.select(&input).unwrap();
        assert!(selected.accepted.is_empty());
        assert!(matches!(selected.statuses[0], LaneStatus::InvalidInput(_)));
        for (rows, components, values) in [
            (usize::MAX, 1, vec![]),
            (1, usize::MAX, vec![]),
            (1, 1, vec![Complex::new(0_f64, 0.); 40]),
        ] {
            assert!(guard.select(&KinematicsBatch { rows, components, values }).is_err());
            assert_eq!(guard.last_certificate_execution().input_rows, rows);
            assert_eq!(guard.last_certificate_execution().evaluated_rows, 0);
            assert_eq!(guard.last_certificate_execution().passes, 0);
            assert_eq!(guard.last_certificate_execution().reduction_batches, 0);
        }
    }
}
