use super::*;
use crate::RescueScalar;
use symbolica::domains::float::{Float, FloatLike, SingleFloat};
#[path = "certified_affine_dilation_jets.rs"]
mod dilation_jets;

fn settings(rank: u16) -> ReductionSettings {
    let mut s = ReductionSettings::new(3, rank);
    s.algorithm = Algorithm::Hybrid;
    s
}

/// Existing RescueScalar promotion preserves every represented input bit;
/// this comparison never passes through f64 or a decimal display string.
fn precision_pair(
    basis: &[Monomial],
    low: &[Complex<Float>],
    high: &[Complex<Float>],
    components: usize,
) {
    assert_eq!(low.len(), basis.len() * 4 * components);
    assert_eq!(low.len(), high.len());
    let prototype = Float::new(256);
    let tolerance = prototype.from_i64(2).pow(80).inv();
    for (moment, alpha) in basis.iter().enumerate() {
        let dimension = alpha.rank() as i32 - 2;
        let mut unit = prototype
            .from_i64(3)
            .pow(u64::from(dimension.unsigned_abs()));
        if dimension < 0 {
            unit = unit.inv();
        }
        for slot in 0..4 {
            for component in 0..components {
                let i = (4 * moment + slot) * components + component;
                let lower = Complex::new(
                    low[i].re.promote_exact(256).unwrap(),
                    low[i].im.promote_exact(256).unwrap(),
                );
                let upper = Complex::new(
                    high[i].re.promote_exact(256).unwrap(),
                    high[i].im.promote_exact(256).unwrap(),
                );
                let error = NativeFloat::hypot(
                    &(lower.re.clone() - &upper.re),
                    &(lower.im.clone() - &upper.im),
                );
                let magnitude = NativeFloat::hypot(&upper.re, &upper.im);
                let scale = if magnitude > unit {
                    magnitude
                } else {
                    unit.clone()
                };
                assert!(
                    error <= tolerance.clone() * &scale,
                    "128/256 precision pair failed: moment={moment}, native slot={slot}, Taylor component={component}"
                );
            }
        }
    }
}

/// Actual numeric tangent regression; the exact input path and homogeneity
/// response are reused from the project's Symbolica/Dualizer jet helper.
#[test]
#[ignore = "controlled full840 Dual dilation qualification after independent anchor admission"]
fn public_hybrid_rank_six_dilation_dual_checks_all840_primals_and_jets() {
    let plan = ReductionPlan::new(settings(6)).unwrap();
    let shape = vec![vec![0, 0], vec![0, 1]]; // First dilation jet, no additive mass seed.
    let base: Vec<Rational> = row(&Float::new(256), 3)
        .iter()
        .map(|z| z.re.exact_rational().unwrap())
        .collect();
    let exact = dilation_jets::input_jets(&base, 3, &shape).unwrap();
    assert_eq!(exact.len(), 32);
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let anchor=root.join("target/collinear-rank6-parameter-oracle-refinement-v3/capture-001/grids/n3-order48-identity/result.json");
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(anchor).unwrap()).unwrap();
    assert_eq!(oracle["basis"].as_array().unwrap().len(), 330);
    assert_eq!(
        oracle["native_slot_order"],
        serde_json::json!(["uv1", "ir1", "ir2", "finite"])
    );
    let reference = oracle["native_laurent"].as_array().unwrap();
    assert_eq!(reference.len(), 330);
    let mut captured = Vec::new();
    for bits in [128, 256] {
        let prototype = Float::new(bits);
        let input = KinematicsBatch {
            rows: 1,
            components: 2,
            values: exact
                .iter()
                .map(|x| Complex::new(prototype.from_rational(x), prototype.zero()))
                .collect(),
        };
        let original = input.values.clone();
        let mut workspace = Workspace::new_dual(
            &plan,
            prototype.clone(),
            HyperDual::<u8>::new(shape.clone()),
        )
        .unwrap();
        workspace.set_accuracy_precision(53).unwrap();
        let result = workspace.reduce_batch(&input).unwrap();
        assert_eq!(input.values, original);
        assert_eq!(result.components, 2);
        assert_eq!(result.rows(), 1);
        assert_eq!(result.basis.len(), 210);
        assert_eq!(result.statuses, [LaneStatus::Success]);
        assert!(result.analytic_used[0]);
        let actual = result.lane(0).unwrap();
        assert_eq!(actual.len(), 1680);
        assert_eq!(
            workspace.last_execution().input_rows,
            1,
            "Taylor components are not sample lanes"
        );
        assert_eq!(
            workspace
                .last_planar_certificate_execution()
                .reduction_batches,
            1
        );
        let tolerance = prototype.from_rational(&Rational::from((1_i64, 10_000_000_000_i64)));
        for (moment, alpha) in result.basis.iter().enumerate() {
            let response = dilation_jets::response(3, alpha.rank()).unwrap();
            let degree = &response[2][0]; // Existing exact Symbolica first-dilation response.
            assert_eq!(*degree, Rational::from(alpha.rank() as i64 + 4 - 2 * 3));
            let weight = prototype.from_rational(degree);
            let dimension = alpha.rank() as i32 - 2;
            let mut unit = prototype
                .from_i64(3)
                .pow(u64::from(dimension.unsigned_abs()));
            if dimension < 0 {
                unit = unit.inv();
            }
            for slot in 0..4 {
                let primal = Complex::new(
                    Float::parse(
                        reference[moment][slot][0].as_str().unwrap().trim(),
                        Some(bits),
                    )
                    .unwrap(),
                    Float::parse(
                        reference[moment][slot][1].as_str().unwrap().trim(),
                        Some(bits),
                    )
                    .unwrap(),
                );
                let jet = Complex::new(primal.re.clone() * &weight, primal.im.clone() * &weight);
                for (component, expected) in [primal, jet].iter().enumerate() {
                    let z = &actual[(4 * moment + slot) * 2 + component];
                    let error = NativeFloat::hypot(
                        &(z.re.clone() - &expected.re),
                        &(z.im.clone() - &expected.im),
                    );
                    let magnitude = NativeFloat::hypot(&expected.re, &expected.im);
                    let scale = if magnitude > unit {
                        magnitude
                    } else {
                        unit.clone()
                    };
                    assert!(
                        error <= tolerance.clone() * &scale,
                        "bits={bits}, moment={moment}, slot={slot}, dilation component={component}"
                    );
                }
            }
        }
        captured.push(actual.to_vec());
    }
    precision_pair(plan.basis(), &captured[0], &captured[1], 2);
}
fn context() -> CertifiedAffineChart {
    CertifiedAffineChart {
        key: crate::pv::AffineChartKey {
            coordinate_rows: vec![1],
            relative_columns: vec![0, 1],
        },
        inactive: vec![],
    }
}

#[test]
fn affine_chart_variants_preserve_proof_context_and_disable_recursive_retry() {
    let plan = ReductionPlan::new_certified_affine(settings(2), context()).unwrap();
    assert!(plan.automatic_affine_settings().is_none());
    assert!(plan.fallback.is_none());
    let derivative = plan.with_analytic_jet_order(2).unwrap();
    assert!(derivative.automatic_affine_settings().is_none());
    assert!(derivative.fallback.is_none());
    assert!(
        plan.with_zero_invariants(crate::plan::ZeroInvariantProfile {
            masses: vec![false; 3],
            distances: vec![(0, 1)]
        })
        .is_err()
    );
    assert_eq!(derivative.settings().algorithm, Algorithm::Hybrid);
    assert_eq!(derivative.parameter_count(), 16);
    assert!(derivative.leaves.iter().all(|leaf| leaf.active.len() <= 2));
    assert_eq!(derivative.basis(), plan.basis());
    let mut changed = settings(2);
    changed.dimension_shift = 1;
    assert!(ReductionPlan::new_certified_affine(changed, context()).is_err());
    let mut changed = settings(2);
    changed.powers[0] = 2;
    assert!(ReductionPlan::new_certified_affine(changed, context()).is_err());
}

fn row(prototype: &Float, mu: i64) -> Vec<Complex<Float>> {
    let mut entries = [0_i64; 16];
    entries[5] = 1;
    entries[9] = 2;
    entries[12] = 1;
    entries[13] = 2;
    entries[14] = 3;
    entries[15] = mu;
    entries
        .into_iter()
        .map(|value| Complex::new(prototype.from_i64(value), prototype.zero()))
        .collect()
}

/// Actual numerical regression. UNEXECUTED in the source proposal. Run only in
/// the root's finite controlled qualification phase, then compare all840 saved
/// slots to the independently admitted collinear parameter oracle separately.
#[test]
#[ignore = "controlled N3 numerical phase requires independent full-array anchor admission"]
fn public_hybrid_rank_six_affine_prepass_preserves_batch_lanes_and_all_moments() {
    let plan = ReductionPlan::new(settings(6)).unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let anchor=root.join("target/collinear-rank6-parameter-oracle-refinement-v3/capture-001/grids/n3-order48-identity/result.json");
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(anchor).unwrap()).unwrap();
    assert_eq!(oracle["basis"].as_array().unwrap().len(), 330);
    assert_eq!(
        oracle["native_slot_order"],
        serde_json::json!(["uv1", "ir1", "ir2", "finite"])
    );
    let reference = oracle["native_laurent"].as_array().unwrap();
    assert_eq!(reference.len(), 330);
    let mut captured = Vec::new();
    for bits in [128, 256] {
        let prototype = Float::new(bits);
        let good = row(&prototype, 3);
        let bad = row(&prototype, -3);
        let input = KinematicsBatch {
            rows: 3,
            components: 1,
            values: [good.clone(), bad, good].concat(),
        };
        let original = input.values.clone();
        let mut workspace = Workspace::new(&plan, prototype.clone()).unwrap();
        workspace.set_accuracy_precision(53).unwrap();
        let result = workspace.reduce_batch(&input).unwrap();
        assert_eq!(input.values, original);
        assert_eq!(result.basis.len(), 210);
        assert_eq!(result.components, 1);
        assert_eq!(result.rows(), 3);
        assert_eq!(result.statuses[0], LaneStatus::Success);
        assert!(
            matches!(&result.statuses[1],LaneStatus::InvalidInput(reason) if reason.contains("mu must be positive"))
        );
        assert_eq!(result.statuses[2], LaneStatus::Success);
        assert_eq!(result.lane(0).unwrap().len(), 840);
        assert!(result.lane(1).is_err());
        assert_eq!(result.lane(0).unwrap(), result.lane(2).unwrap());
        assert!(result.analytic_used[0] && result.analytic_used[2]);
        let work = workspace.last_planar_certificate_execution();
        assert_eq!(
            work.reduction_batches, 1,
            "both accepted lanes share one chart-workspace batch"
        );
        assert!(work.evaluated_rows >= 2 && work.passes > 0);
        assert!(workspace.last_execution().master_rows > 0);
        let tolerance = prototype.from_rational(&Rational::from((1_i64, 10_000_000_000_i64)));
        for (moment, alpha) in result.basis.iter().enumerate() {
            let dimension = alpha.rank() as i32 - 2;
            let mut unit = prototype
                .from_i64(3)
                .pow(u64::from(dimension.unsigned_abs()));
            if dimension < 0 {
                unit = unit.inv();
            }
            for slot in 0..4 {
                let expected = Complex::new(
                    Float::parse(
                        reference[moment][slot][0].as_str().unwrap().trim(),
                        Some(bits),
                    )
                    .unwrap(),
                    Float::parse(
                        reference[moment][slot][1].as_str().unwrap().trim(),
                        Some(bits),
                    )
                    .unwrap(),
                );
                let actual = &result.lane(0).unwrap()[4 * moment + slot];
                let difference = NativeFloat::hypot(
                    &(actual.re.clone() - &expected.re),
                    &(actual.im.clone() - &expected.im),
                );
                let magnitude = NativeFloat::hypot(&expected.re, &expected.im);
                let scale = if magnitude > unit {
                    magnitude
                } else {
                    unit.clone()
                };
                assert!(
                    difference <= tolerance.clone() * &scale,
                    "bits={bits}, moment={moment}, native slot={slot}: independent all840 anchor disagreement"
                );
            }
        }
        captured.push(result.lane(0).unwrap().to_vec());
    }
    precision_pair(plan.basis(), &captured[0], &captured[1], 1);
}
