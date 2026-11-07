//! The rotation copy (`DirectWorkspace::reduce_batch_with_rotation`): the
//! rotated frame's moments from the same call, bit for bit those of a
//! separate call on the rotated batch, agreeing with the primary to the
//! reduction's accuracy, with the primary's diagnostics and shared work.
use iregi_rust::{
    BatchResult, DirectReductionPlan, DirectWorkspace, InvariantInput, KinematicsBatch, LaneStatus, ReductionError,
    ReductionSettings, TensorStore, direct::EXPANSION_TOLERANCE,
};
use iregi_masters::NativeFloat;
use symbolica::domains::float::{Complex, DoubleFloat};

/// A small deterministic generator (the lanes must not depend on a crate).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Generic massive one-loop kinematics: `lanes` rows of `n` shifts (time
/// component first, a few hundred GeV), squared masses and mu = 500.
fn lanes(n: usize, lanes: usize, supplied: bool, seed: u64) -> KinematicsBatch<f64> {
    let mut rng = Lcg(seed);
    let mut values = Vec::new();
    for _ in 0..lanes {
        let shifts: Vec<[f64; 4]> = (0..n)
            .map(|i| if i == 0 { [0.0; 4] } else { [0.0, 1.0, 2.0, 3.0].map(|_| 600.0 * rng.next() - 300.0) })
            .collect();
        for shift in &shifts {
            values.extend(shift.iter().map(|&x| Complex::new(x, 0.0)));
        }
        if supplied {
            for i in 0..n {
                for j in i + 1..n {
                    let d: Vec<f64> = (0..4).map(|a| shifts[i][a] - shifts[j][a]).collect();
                    values.push(Complex::new(d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3], 0.0));
                }
            }
        }
        for _ in 0..n {
            values.push(Complex::new(1000.0 + 40000.0 * rng.next(), 0.0));
        }
        values.push(Complex::new(500.0, 0.0));
    }
    KinematicsBatch { rows: lanes, components: 1, values }
}

fn settings(n: usize, rank: u16, supplied: bool) -> ReductionSettings {
    let mut settings = ReductionSettings::new(n, rank);
    if supplied {
        settings.invariants = InvariantInput::SuppliedInvariants;
    }
    settings
}

fn lift(batch: &KinematicsBatch<f64>) -> KinematicsBatch<DoubleFloat> {
    KinematicsBatch {
        rows: batch.rows,
        components: 1,
        values: batch.values.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect(),
    }
}

/// Every lane's array and status, exactly.
fn fingerprint<T: NativeFloat>(result: &BatchResult<T>) -> (Vec<String>, String) {
    let values = (0..result.rows())
        .map(|lane| result.lane_with_estimate(lane).map_or_else(|| "refused".to_string(), |best| format!("{:?}", best.values)))
        .collect();
    (values, format!("{:?}", result.statuses))
}

/// Largest E/A between two lanes' arrays (moments weighted by mu^-rank are
/// not needed here: every lane shares mu, and the arrays are compared as a
/// whole).
fn difference<T: NativeFloat>(a: &[Complex<T>], b: &[Complex<T>]) -> f64 {
    let size = a.iter().map(|z| z.re.to_f64().hypot(z.im.to_f64())).fold(0.0f64, f64::max);
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x.re.to_f64() - y.re.to_f64()).hypot(x.im.to_f64() - y.im.to_f64()))
        .fold(0.0f64, f64::max);
    error / size
}

fn check_copy<T: NativeFloat>(
    plan: &DirectReductionPlan,
    batch: &KinematicsBatch<T>,
    rotation: usize,
    make: &dyn Fn() -> DirectWorkspace<T>,
    tolerance: f64,
) -> usize
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let mut workspace = make();
    let (primary, copy) = workspace.reduce_batch_with_rotation(batch, rotation).unwrap();
    let conditions = workspace.last_conditions().to_vec();
    // The primary and the copy are what two separate calls return.
    let mut alone = make();
    assert_eq!(fingerprint(&primary), fingerprint(&alone.reduce_batch(batch).unwrap()));
    assert_eq!(conditions, alone.last_conditions(), "diagnostics describe the primary");
    let rotated = plan.rotated_batch(batch, rotation).unwrap();
    assert_eq!(fingerprint(&copy), fingerprint(&make().reduce_batch(&rotated).unwrap()), "rotation {rotation}");
    // The same integrals: lanes and their copies agree.
    let (mut served, mut worst) = (0, 0.0f64);
    for lane in 0..batch.rows {
        if let (Ok(a), Ok(b)) = (primary.lane(lane), copy.lane(lane)) {
            let difference = difference(a, b);
            assert!(difference <= tolerance, "lane {lane}, rotation {rotation}: E/A {difference:e}");
            worst = worst.max(difference);
            served += 1;
        }
    }
    eprintln!("N{} R{} rotation {rotation}: {served}/{} lanes, worst E/A {worst:e}", plan.settings().propagators, plan.settings().rank, batch.rows);
    served
}

#[test]
fn the_rotation_copy_is_a_separate_rotated_call_from_the_same_call() {
    for (n, rank) in [(4usize, 4u16), (5, 5), (6, 4)] {
        for supplied in [false, true] {
            let plan = DirectReductionPlan::new(settings(n, rank, supplied)).unwrap();
            let batch = lanes(n, 6, supplied, 17 + n as u64);
            for rotation in [1, 2, n - 1] {
                let gates = || plan.workspace(0.0f64).unwrap();
                let served = check_copy(&plan, &batch, rotation, &gates, 1e-6);
                let choice = || {
                    let mut workspace = plan.workspace(0.0f64).unwrap();
                    workspace.set_stability_threshold(1e-15);
                    workspace.enable_estimated_method_choice(12, 1e-15);
                    workspace.set_alternative_first(Some(EXPANSION_TOLERANCE));
                    workspace
                };
                let chosen = check_copy(&plan, &batch, rotation, &choice, 1e-6);
                assert!(served + chosen > 0, "N{n} R{rank}: no lane served");
            }
            // Double-double: the same in the workspace's own precision.
            let lifted = lift(&batch);
            let double = || plan.workspace(DoubleFloat::from(0.0)).unwrap();
            assert!(check_copy(&plan, &lifted, 1, &double, 1e-14) > 0, "N{n} R{rank}: no double-double lane");
        }
    }
}

#[test]
fn the_copy_repeats_the_subsets_that_keep_their_order_through_the_store() {
    let plan = DirectReductionPlan::new(settings(6, 6, true)).unwrap();
    let batch = lanes(6, 4, true, 5);
    let store = TensorStore::new();
    let mut workspace = plan.workspace(0.0f64).unwrap();
    workspace.set_tensor_store(Some(&store));
    let (primary, copy) = workspace.reduce_batch_with_rotation(&batch, 1).unwrap();
    let stats = store.statistics();
    eprintln!("{stats:?}");
    // Every master of a subset without propagator 0 is met again in the copy.
    assert!(stats.masters_reused > 0, "{stats:?}");
    // With the store and without one (a store private to the call), and
    // against two separate calls: the same arrays.
    let (p, c) = plan.workspace(0.0f64).unwrap().reduce_batch_with_rotation(&batch, 1).unwrap();
    assert_eq!(fingerprint(&primary), fingerprint(&p));
    assert_eq!(fingerprint(&copy), fingerprint(&c));
    // Rotation by the number of propagators (or 0) is the primary itself.
    for rotation in [0, 6, 12] {
        let (p, c) = plan.workspace(0.0f64).unwrap().reduce_batch_with_rotation(&batch, rotation).unwrap();
        assert_eq!(fingerprint(&p), fingerprint(&c));
        assert_eq!(fingerprint(&p), fingerprint(&primary));
    }
    // The copy's master evaluations add to the primary's.
    let mut alone = plan.workspace(0.0f64).unwrap();
    alone.reduce_batch(&batch).unwrap();
    let mut masters = alone.last_master_rows();
    alone.reduce_batch(&plan.rotated_batch(&batch, 1).unwrap()).unwrap();
    masters += alone.last_master_rows();
    let mut both = plan.workspace(0.0f64).unwrap();
    both.reduce_batch_with_rotation(&batch, 1).unwrap();
    assert_eq!(both.last_master_rows(), masters);
    assert!(primary.statuses.iter().any(|s| *s == LaneStatus::Success), "{:?}", primary.statuses);
}

#[test]
fn rotated_copies_of_dotted_or_dual_batches_are_refused() {
    let mut settings = ReductionSettings::new(4, 2);
    settings.powers = vec![2, 1, 1, 1];
    let plan = DirectReductionPlan::new(settings).unwrap();
    let batch = lanes(4, 2, false, 3);
    assert!(matches!(plan.rotated_batch(&batch, 1), Err(ReductionError::Unsupported(_))));
    assert!(matches!(
        plan.workspace(0.0f64).unwrap().reduce_batch_with_rotation(&batch, 1),
        Err(ReductionError::Unsupported(_))
    ));
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, 2)).unwrap();
    let dual = KinematicsBatch { rows: 1, components: 2, values: batch.values.clone() };
    assert!(plan.rotated_batch(&dual, 1).is_err());
    let short = KinematicsBatch { rows: 3, components: 1, values: batch.values.clone() };
    assert!(matches!(plan.rotated_batch(&short, 1), Err(ReductionError::InvalidInput(_))));
}

#[test]
fn rotated_batches_relabel_shifts_invariants_and_masses() {
    let n = 5;
    let plan = DirectReductionPlan::new(settings(n, 2, true)).unwrap();
    let batch = lanes(n, 1, true, 9);
    let rotated = plan.rotated_batch(&batch, 2).unwrap();
    let layout = iregi_rust::InputLayout::of(plan.settings());
    let (row, new) = (&batch.values, &rotated.values);
    for k in 0..n {
        let from = (k + 2) % n;
        assert_eq!(new[4 * k..4 * k + 4], row[4 * from..4 * from + 4]);
        assert_eq!(new[layout.masses.start + k], row[layout.masses.start + from]);
    }
    let supplied = layout.scalar_products.clone().unwrap();
    for i in 0..n {
        for j in i + 1..n {
            let (a, b) = ((i + 2) % n, (j + 2) % n);
            let at = |x: usize, y: usize| supplied.start + InvariantInput::invariant_index(n, x.min(y), x.max(y));
            assert_eq!(new[at(i, j)], row[at(a, b)]);
        }
    }
    assert_eq!(new[layout.scale], row[layout.scale]);
}
