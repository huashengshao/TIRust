//! Reproducer from pyAmpliCol (m-lit lit-tir, Born-interfered g g > g g g at
//! one loop, binary64): massless triangles and boxes whose Gram systems the
//! direct engine's default gate refuses (conditioning 4e-5 .. 6e-4, every
//! loop momentum massless, external gluons lightlike). At TIRust main
//! 3558433 the binary64 a priori method choice (`enable_estimated_method_choice`,
//! pyAmpliCol's TIR backend: `Convention::OneLoop`, `TensorWeights::Components`,
//! method-choice gate 1e-15, boost 12; also at the default gate 1e-12, with
//! and without `set_alternative_first(Some(EXPANSION_TOLERANCE))`) returned
//! these lanes `Success` with values off by O(1) to O(1e3): where neither
//! expansion reached its tolerance at the largest boost, the fallback
//! alternative PV reduction divided by the modified Cayley determinant,
//! which vanishes for these IR-divergent integrals.
//!
//! The cases are the first five lanes of `fixtures/method_choice_lanes.json`
//! (one triangle R = 3, four boxes R = 4; truth: NinjaRS's generic engine at
//! 40 digits on the same binary64 inputs with the exact on-shell S-matrix,
//! lightlike legs exactly 0). The arms: invariants from the shifts (the
//! lightlike legs come out at |p^2| ~ 1e-16 .. 1e-18 of the largest
//! invariant; the binary64 lightlike rule puts them on shell) and the
//! supplied S-matrix (`InvariantInput::SuppliedInvariants`).

mod common;

use common::{Lane, batch, contract, error, lanes, reference, settings};
use iregi_rust::direct::EXPANSION_TOLERANCE;
use iregi_rust::{DirectReductionPlan, KinematicsBatch, LaneStatus};
use symbolica::domains::float::{Complex, Float, RealLike};

const FIXTURE: &str = include_str!("fixtures/method_choice_lanes.json");

fn cases() -> Vec<Lane> {
    lanes(FIXTURE).into_iter().take(5).collect()
}

/// Every small-Gram method a workspace offers, as pyAmpliCol would run it,
/// serves a case only within 1e-6 of the truth (refusing is allowed).
#[test]
fn method_choice_serves_small_gram_rows_accurately() {
    let mut failures = Vec::new();
    for (index, case) in cases().iter().enumerate() {
        for supplied in [false, true] {
            let batch = batch(case, supplied);
            let plan = DirectReductionPlan::new(settings(case, supplied)).unwrap();
            let gates = plan
                .workspace(0.0f64)
                .unwrap()
                .reduce_batch(&batch)
                .unwrap();
            println!(
                "case {index} N={} R={} supplied={supplied}: default gate {:?}; 256-bit error {:?}",
                case.n,
                case.rank,
                gates.statuses[0],
                reference(case, supplied).map(|high| error(&high, &case.truth))
            );
            let mut check = |name: String, workspace: &mut iregi_rust::DirectWorkspace<f64>| {
                let result = workspace.reduce_batch(&batch).unwrap();
                let e = result
                    .lane(0)
                    .ok()
                    .map(|array| error(&contract(&result.basis, array, case), &case.truth));
                println!("  {name}: {:?}, error {e:?}", result.statuses[0]);
                if let Some(e) = e.filter(|&e| e > 1e-6) {
                    failures.push(format!("case {index} supplied={supplied} {name}: {e:e}"));
                }
            };
            for gate in [1e-15, 1e-12] {
                for alternative_first in [None, Some(EXPANSION_TOLERANCE)] {
                    let mut workspace = plan.workspace(0.0f64).unwrap();
                    workspace.set_stability_threshold(gate);
                    workspace.enable_estimated_method_choice(12, gate);
                    workspace.set_alternative_first(alternative_first);
                    check(
                        format!(
                            "method choice, gate {gate:e}, alternative_first {alternative_first:?}"
                        ),
                        &mut workspace,
                    );
                }
            }
            for (name, mode) in [("expansion", 0), ("alternative PV", 1), ("everywhere", 2)] {
                let mut workspace = plan.workspace(0.0f64).unwrap();
                match mode {
                    0 => workspace.enable_gram_expansion(12),
                    1 => workspace.enable_alternative_pv(),
                    _ => workspace.enable_gram_expansion_everywhere(12, 1e-15),
                }
                check(name.to_string(), &mut workspace);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "small-Gram lanes served beyond 1e-6: {failures:#?}"
    );
}

/// The minimal form: the triangle of the first case (massless propagators,
/// one lightlike leg, so IR-divergent; Gram conditioning 6.3e-5; supplied
/// invariants) by the plain small-Gram expansion (`enable_gram_expansion(12)`)
/// against the 256-bit direct array, per Laurent slot `[UV1, IR1, IR2,
/// finite]` (each monomial weighted by mu^-(rank + 4 - 2N)), at ranks 0 to 3;
/// the same triangle with every propagator at m^2 = 173^2 (IR-finite, the
/// same Gram matrix) is the control. At TIRust main 3558433 ranks 0 to 2 and
/// the massive control agreed to <= 7e-9, while the massless rank-3 array
/// was off by 1.2e-2 (UV1), 7.4e3 (IR1) and 1.6e4 (finite) of the slots'
/// sizes, returned `Success`: at boost 12 neither expansion reaches its
/// tolerance and the alternative PV divided by the vanishing modified Cayley
/// determinant. That fallback is now refused, and the expansion gets the
/// boost its observed convergence rate predicts: every rank of both
/// triangles must be served, within 1e-8 of the reference.
#[test]
fn small_gram_expansion_of_an_ir_divergent_triangle() {
    let case = &cases()[0];
    let mut failures = Vec::new();
    for mass2 in [0.0, 173.0 * 173.0] {
        for rank in 0..=case.rank {
            let case = Lane {
                rank,
                masses: vec![Complex::new(mass2, 0.0); case.n],
                ..case.clone()
            };
            let values = batch(&case, true).values;
            let plan = DirectReductionPlan::new(settings(&case, true)).unwrap();
            let mut workspace = plan.workspace(0.0f64).unwrap();
            workspace.enable_gram_expansion(12);
            let result = workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: values.clone(),
                })
                .unwrap();
            let lifted: Vec<Complex<Float>> = values
                .iter()
                .map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
                .collect();
            let mut high = plan.workspace(Float::new(256)).unwrap();
            high.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
            let truth = high
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: lifted,
                })
                .unwrap();
            assert_eq!(truth.statuses[0], LaneStatus::Success);
            let truth: Vec<Complex<f64>> = truth
                .lane(0)
                .unwrap()
                .iter()
                .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
                .collect();
            let Ok(array) = result.lane(0) else {
                println!("m^2 {mass2} rank {rank}: {:?}", result.statuses[0]);
                failures.push(format!("m^2 {mass2} rank {rank}: refused"));
                continue;
            };
            let mu = case.mu2.sqrt();
            let base = 4 - 2 * case.n as i32;
            let mut slots = [(0.0f64, 0.0f64); 4];
            for (m, monomial) in result.basis.iter().enumerate() {
                let weight = mu.powi(monomial.rank() as i32 + base);
                for (slot, (error, size)) in slots.iter_mut().enumerate() {
                    let (a, b) = (array[4 * m + slot], truth[4 * m + slot]);
                    *error = error.max((a.re - b.re).hypot(a.im - b.im) / weight);
                    *size = size.max(b.re.hypot(b.im) / weight);
                }
            }
            let relative = slots.map(|(e, s)| if s > 0.0 { e / s } else { e });
            println!(
                "m^2 {mass2} rank {rank}: {:?}; [UV1, IR1, IR2, finite] error / size {relative:?}",
                result.statuses[0]
            );
            if relative.iter().any(|&e| e > 1e-8) {
                failures.push(format!("m^2 {mass2} rank {rank}: {relative:?}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "small-Gram expansion refused or beyond 1e-8: {failures:#?}"
    );
}
