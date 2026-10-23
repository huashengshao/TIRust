//! Independent D6 parameter oracles through the public reduction API.
//!
//! Source values and failed original-reference comparisons are retained in
//! validation/shifted-pentagon/{evidence,contours}.json. These tests must run
//! under both the interpreter and the optional native scalar backend.

use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, Convention, KinematicsBatch, LaneStatus, ReductionPlan, ReductionSettings, Workspace,
};
use symbolica::{
    domains::{
        dual::HyperDual,
        float::{Complex, DoubleFloat, Float},
        integer::Integer,
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

fn decimal(text: &str) -> Rational {
    let (whole, fraction) = text.split_once('.').unwrap();
    Rational::from((
        format!("{whole}{fraction}").parse::<Integer>().unwrap(),
        Integer::from(10).pow(fraction.len() as u64),
    ))
}

fn generic_point<T: NativeFloat>(prototype: &T, components: usize, scale: i64) -> Vec<Complex<T>> {
    [
        0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 10, 12, 15, 18, 22, 1,
    ]
    .into_iter()
    .enumerate()
    .flat_map(|(argument, value)| {
        (0..components).map(move |component| {
            Complex::new(
                prototype.from_i64(match component {
                    0 => value * if argument < 20 { scale } else { scale * scale },
                    1 if (20..25).contains(&argument) => 1,
                    _ => 0,
                }),
                prototype.zero(),
            )
        })
    })
    .collect()
}

fn finite_plan(shift: i16) -> ReductionPlan {
    let mut settings = ReductionSettings::new(5, 0);
    settings.algorithm = Algorithm::Ibp;
    settings.dimension_shift = shift;
    settings.convention = Convention::OneLoop;
    ReductionPlan::new(settings).unwrap()
}

fn point(timelike: f64, masses: [f64; 5], width: f64) -> Vec<Complex<f64>> {
    let mut values: Vec<_> = [
        [0., 0., 0., 0.],
        [timelike, 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]
    .into_iter()
    .flatten()
    .map(|value| Complex::new(value, 0.))
    .collect();
    values.extend(masses.map(|mass| Complex::new(mass, width)));
    values.push(Complex::new(1., 0.));
    values
}

fn check_generic<T: NativeFloat>(plan: &ReductionPlan, prototype: T)
where
    Complex<T>: EvaluationDomain,
{
    let mut workspace = Workspace::new(plan, prototype.clone()).unwrap();
    workspace.set_accuracy_precision(80).unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 3,
            components: 1,
            values: (1..=3)
                .flat_map(|scale| generic_point(&prototype, 1, scale))
                .collect(),
        })
        .unwrap();
    let expected = prototype.from_rational(&decimal("-0.0001771401382640081806076521310069008176"));
    let tolerance = prototype.from_i64(10).pow(23).inv();
    for lane in 0..3 {
        assert_eq!(
            result.statuses[lane],
            LaneStatus::Success,
            "{:?}",
            result.statuses
        );
        let finite = &result.lane(lane).unwrap()[3];
        let scale = prototype.from_i64((lane + 1) as i64).pow(4);
        let error =
            (finite.re.clone() * &scale - &expected).norm() + (finite.im.clone() * &scale).norm();
        assert!(
            error < tolerance,
            "bits={} error={error}",
            prototype.get_precision()
        );
    }
    assert!(workspace.last_execution().pentagon_box_rows >= 3 * 5 * 8);
}

#[test]
fn shifted_pentagon_core_preserves_doublefloat_and_float_precision() {
    let plan = finite_plan(1);
    check_generic(&plan, DoubleFloat::from(0.));
    check_generic(&plan, Float::new(128));
}

#[test]
fn shifted_pentagon_core_common_mass_dual_matches_independent_d4() {
    let plan = finite_plan(1);
    let mut workspace =
        Workspace::new_dual(&plan, 0., HyperDual::<f64>::new(vec![vec![0], vec![1]])).unwrap();
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 3,
            components: 2,
            values: (1..=3)
                .flat_map(|scale| generic_point(&0., 2, scale))
                .collect(),
        })
        .unwrap();
    for lane in 0..3 {
        assert_eq!(
            result.statuses[lane],
            LaneStatus::Success,
            "{:?}",
            result.statuses
        );
        let output = result.lane(lane).unwrap();
        let scale = (lane + 1) as f64;
        assert!((output[6].re * scale.powi(4) + 0.0001771401382640081806).abs() < 2e-14);
        assert!((output[7].re * scale.powi(6) - 0.0000236059994160983338257).abs() < 2e-14);
        assert!(
            output[6].im.abs() * scale.powi(4) < 2e-14
                && output[7].im.abs() * scale.powi(6) < 2e-14
        );
        assert!(
            output[..6]
                .iter()
                .all(|value| value.re == 0. && value.im == 0.)
        );
    }
}

#[test]
#[ignore = "second-order D0 Dual preparation is qualified separately with a bounded runtime"]
fn shifted_pentagon_core_second_order_common_mass_identity() {
    let e6 = finite_plan(1);
    let d4 = finite_plan(0);
    let shape = HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]);
    let mut e6_workspace = Workspace::new_dual(&e6, 0., shape.clone()).unwrap();
    let mut d4_workspace = Workspace::new_dual(&d4, 0., shape).unwrap();
    let input = KinematicsBatch {
        rows: 3,
        components: 3,
        values: (1..=3)
            .flat_map(|scale| generic_point(&0., 3, scale))
            .collect(),
    };
    let e6 = e6_workspace.reduce_batch(&input).unwrap();
    let d4 = d4_workspace.reduce_batch(&input).unwrap();
    for lane in 0..3 {
        assert_eq!(e6.statuses[lane], LaneStatus::Success, "{:?}", e6.statuses);
        assert_eq!(d4.statuses[lane], LaneStatus::Success, "{:?}", d4.statuses);
        let e6 = e6.lane(lane).unwrap();
        let d4 = d4.lane(lane).unwrap();
        // Symbolica uses Taylor coefficients: (k+1)E6[k+1]=-D4[k].
        for k in 0..2 {
            let factor = (k + 1) as f64;
            assert!((factor * e6[9 + k + 1].re + d4[9 + k].re).abs() < 2e-14);
            assert!((factor * e6[9 + k + 1].im + d4[9 + k].im).abs() < 2e-14);
        }
    }
}

#[test]
fn shifted_pentagon_core_preserves_independent_lanes_and_finite_poles() {
    let mut settings = ReductionSettings::new(5, 0);
    settings.algorithm = Algorithm::Ibp;
    settings.dimension_shift = 1;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    assert_eq!(plan.algorithm_used(), Algorithm::Ibp);
    let mut workspace = Workspace::new(&plan, 0.).unwrap();
    let positive = point(1., [10., 12., 15., 18., 22.], 0.);
    let complex = point(4., [1., 2., 3., 4., 5.], -5.);
    let unqualified = point(4., [1., 2., 3., 4., 5.], 0.);
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 3,
            components: 1,
            values: [positive.clone(), unqualified, complex].concat(),
        })
        .unwrap();
    assert_eq!(
        result.statuses[0],
        LaneStatus::Success,
        "{:?}",
        result.statuses
    );
    assert_eq!(
        result.statuses[2],
        LaneStatus::Success,
        "{:?}",
        result.statuses
    );
    assert!(
        matches!(&result.statuses[1], LaneStatus::MasterFailure(reason)
        if reason.contains("UnqualifiedEndpoint")),
        "{:?}",
        result.statuses
    );
    assert!(result.lane(1).is_err());
    for lane in [0, 2] {
        assert!(
            result.lane(lane).unwrap()[..3]
                .iter()
                .all(|value| value.re == 0. && value.im == 0.)
        );
    }
    let real = &result.lane(0).unwrap()[3];
    assert!(
        (real.re + 0.0001771401382640081806).abs() < 2e-14,
        "{real:?}"
    );
    assert!(real.im.abs() < 2e-14, "{real:?}");
    let complex = &result.lane(2).unwrap()[3];
    assert!(
        (complex.re - 0.00115669079120771305).abs() < 1e-13,
        "{complex:?}"
    );
    assert!(
        (complex.im + 0.000473960524532898804).abs() < 1e-13,
        "{complex:?}"
    );
    assert!(workspace.last_execution().master_family_batches >= 1);
    assert!(workspace.last_execution().master_rows >= 2);
    assert!(workspace.last_execution().pentagon_box_rows >= 2 * 5 * 8);
    let mut scaled = positive;
    for value in &mut scaled[..20] {
        value.re *= 3.;
    }
    for value in &mut scaled[20..25] {
        value.re *= 9.;
    }
    scaled[25].re *= 3.;
    let scaled = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: scaled,
        })
        .unwrap();
    assert_eq!(scaled.statuses, [LaneStatus::Success]);
    let value = &scaled.lane(0).unwrap()[3];
    assert!((value.re * 81. - real.re).abs() < 2e-14, "{value:?}");
    assert!(value.im.abs() * 81. < 2e-14, "{value:?}");
    assert!(
        workspace
            .reduce_batch(&KinematicsBatch {
                rows: 0,
                components: 1,
                values: Vec::new()
            })
            .unwrap()
            .statuses
            .is_empty()
    );
}
