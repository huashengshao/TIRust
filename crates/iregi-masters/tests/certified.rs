//! Public certificate contracts for finite D0 masters.
//!
//! At zero invariants and four equal squared masses s, the parameter integral
//! is D0 = 1/(6 s^2). Thus s=1+t has normalized Taylor coefficients
//! [1/6, -1/3, 1/2, -2/3]. These fixed analytic values are independent oracles;
//! this test performs no polynomial manipulation or numerical differentiation.

use iregi_masters::{
    Complex, Float, HyperDual, LaneError, MasterError, MasterEvaluator, MasterLane,
    NativeFloat, ScalarIntegral,
    certified::{
        AccuracyMasterBatch, AccuracyMasterEvaluator, CertificatePolicy, Formula,
        MasterAccuracyContext, SeriesStatus,
    },
};
use symbolica::domains::{float::SingleFloat, rational::Rational};

const REQUIRE: CertificatePolicy = CertificatePolicy::RequireFiniteCertificate;
const ALLOW: CertificatePolicy = CertificatePolicy::AllowOrdinaryUncertified;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn box_point(eta: f64, mass: Complex<f64>) -> Vec<Complex<f64>> {
    let mut point: Vec<_> = [0., -2., -2., 0., 0., -2.]
        .into_iter()
        .map(|coefficient| c(coefficient * eta, 0.))
        .collect();
    point.extend([mass; 4]);
    point.push(c(1., 0.));
    point
}

fn vacuum_jet<T: NativeFloat>(
    prototype: &T,
    components: usize,
    mass_seed: bool,
    scale_seed: bool,
) -> Vec<Complex<T>> {
    let zero = Complex::new(prototype.zero(), prototype.zero());
    let one = Complex::new(prototype.from_rational(&Rational::from(1)), prototype.zero());
    let mut point = vec![zero; 11 * components];
    for argument in 6..11 {
        point[argument * components] = one.clone();
    }
    if mass_seed {
        for argument in 6..10 {
            point[argument * components + 1] = one.clone();
        }
    }
    if scale_seed {
        point[10 * components + 1] = one;
    }
    point
}

fn assert_zero_poles<T: NativeFloat>(value: &MasterLane<'_, T>) {
    assert!(value.uv.iter().chain(value.ir).chain(value.ir2).chain(value.total_single)
        .all(|coefficient| coefficient.re.is_zero() && coefficient.im.is_zero()));
}

fn assert_same_f64(actual: MasterLane<'_, f64>, expected: MasterLane<'_, f64>) {
    for (actual, expected) in [actual.finite, actual.uv, actual.ir, actual.ir2, actual.total_single]
        .into_iter()
        .zip([expected.finite, expected.uv, expected.ir, expected.ir2, expected.total_single])
    {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!((actual.re.to_bits(), actual.im.to_bits()),
                (expected.re.to_bits(), expected.im.to_bits()));
        }
    }
}

fn abs(value: Rational) -> Rational {
    if value < Rational::from(0) { -value } else { value }
}

/// Verify the public rectangle and its distance budget using exact rationals.
/// If supplied, the independent oracle is real and must lie in the rectangle.
fn assert_certificate<T: NativeFloat>(
    batch: &AccuracyMasterBatch<T>,
    lane: usize,
    component: usize,
    accuracy: &MasterAccuracyContext,
    oracle: Option<Rational>,
) {
    let value = &batch.lane(lane).unwrap().finite[component];
    let bounds = batch.diagnostics()[lane].finite.as_ref().unwrap();
    assert_eq!(bounds.len(), batch.component_count());
    let bound = &bounds[component];
    assert!(bound.real_lower <= bound.real_upper);
    assert!(bound.imaginary_lower <= bound.imaginary_upper);
    assert!(bound.truncation >= Rational::from(0));
    let re = value.re.exact_rational().unwrap();
    let im = value.im.exact_rational().unwrap();
    let real_distance = abs(&re - &bound.real_lower).max(abs(&re - &bound.real_upper));
    let imaginary_distance = abs(&im - &bound.imaginary_lower)
        .max(abs(&im - &bound.imaginary_upper));
    let rectangle_distance = real_distance.pow(2) + imaginary_distance.pow(2);
    let allowed = accuracy.absolute_per_component[component].pow(2)
        .max(accuracy.relative.pow(2) * (re.pow(2) + im.pow(2)));
    assert_eq!(bound.maximum_error_squared, rectangle_distance);
    assert_eq!(bound.allowed_error_squared, allowed);
    assert!(rectangle_distance <= allowed);
    if let Some(exact) = oracle {
        assert!(bound.real_lower <= exact && exact <= bound.real_upper);
        assert!(bound.imaginary_lower <= Rational::from(0));
        assert!(Rational::from(0) <= bound.imaginary_upper);
        assert!((&re - &exact).pow(2) + im.pow(2) <= rectangle_distance);
    }
}

#[test]
fn primal_certificates_preserve_lanes_and_explicit_accuracy_contracts() {
    let vacuum = box_point(0., c(1., 0.));
    let tiny = box_point(2f64.powi(-64), c(2., -1.));
    let mut invalid_scale = vacuum.clone();
    invalid_scale[10] = c(0., 0.);
    let mut upper_mass = tiny.clone();
    upper_mass[6].im = 1.;
    let outside = box_point(2., c(2., -1.));
    let rows = [vacuum.clone(), tiny, invalid_scale, upper_mass, outside.clone()];
    let input = rows.concat();
    let preserved_input = input.clone();
    let mut ordinary = MasterEvaluator::new(ScalarIntegral::D0, 0f64).unwrap();
    let mut evaluator = AccuracyMasterEvaluator::new(ordinary.clone());
    let accuracy = MasterAccuracyContext::binary(40, 1);

    let empty = evaluator.evaluate_batch_with_accuracy(&[], 0, &accuracy, REQUIRE).unwrap();
    assert_eq!(empty.row_count(), 0);
    assert!(empty.series_execution.is_empty());
    assert!(evaluator.prepared_series_orders().is_empty());

    let required = evaluator.evaluate_batch_with_accuracy(&input, rows.len(), &accuracy, REQUIRE).unwrap();
    assert_eq!(required.row_count(), 5);
    assert_eq!(required.component_count(), 1);
    assert_eq!(required.ordinary_rows, 0);
    assert_eq!(evaluator.prepared_series_orders(), vec![2]);
    assert_eq!(required.series_execution.len(), 1);
    let (order, execution) = &required.series_execution[0];
    assert_eq!(*order, 2);
    assert_eq!((execution.input_rows, execution.active_rows), (3, 2));
    assert_eq!((execution.numeric_batch_calls, execution.enclosure_batch_calls, execution.tail_batch_calls), (1, 1, 1));
    for lane in [0, 1] {
        assert_eq!(required.diagnostics()[lane].formula, Formula::NearVacuum { order: 2 });
        assert_zero_poles(&required.lane(lane).unwrap());
        assert_certificate(&required, lane, 0, &accuracy,
            (lane == 0).then(|| Rational::from((1, 6))));
    }
    assert_eq!(required.errors()[2], Some(LaneError::InvalidScale));
    assert_eq!(required.errors()[3], Some(LaneError::UpperHalfPlaneMass { argument: 6 }));
    assert_eq!(required.errors()[4], Some(LaneError::UncertifiedFinite(None)));
    for lane in [2, 3, 4] {
        assert!(required.lane(lane).is_err());
        assert!(required.diagnostics()[lane].finite.is_none());
        assert_eq!(required.diagnostics()[lane].formula, Formula::Rejected);
    }
    for lane in [2, 3] {
        assert!(required.diagnostics()[lane].attempts.is_empty());
    }
    let attempt = &required.diagnostics()[4].attempts;
    assert_eq!(attempt.len(), 1);
    assert_eq!(attempt[0].status, SeriesStatus::OutsideAnalyticDomain);
    assert_eq!(attempt[0].rho, Some(Rational::from((3, 4))));

    let allowed = evaluator.evaluate_batch_with_accuracy(&input, rows.len(), &accuracy, ALLOW).unwrap();
    assert_eq!(allowed.ordinary_rows, 1);
    assert_eq!(evaluator.prepared_series_orders(), vec![2]);
    assert_eq!(allowed.diagnostics()[4].formula, Formula::OrdinaryUncertified);
    assert!(allowed.diagnostics()[4].finite.is_none());
    let legacy = ordinary.evaluate_batch(&outside, 1).unwrap();
    assert_same_f64(allowed.lane(4).unwrap(), legacy.lane(0).unwrap());
    for lane in [0, 1] {
        assert_same_f64(allowed.lane(lane).unwrap(), required.lane(lane).unwrap());
    }
    for lane in [2, 3] {
        assert_eq!(allowed.errors()[lane], required.errors()[lane]);
        assert!(allowed.lane(lane).is_err());
        assert!(allowed.diagnostics()[lane].finite.is_none());
    }

    let reversed_input: Vec<_> = rows.iter().rev().flatten().copied().collect();
    let reversed = evaluator.evaluate_batch_with_accuracy(&reversed_input, rows.len(), &accuracy, REQUIRE).unwrap();
    for lane in 0..rows.len() {
        let other = rows.len() - 1 - lane;
        assert_eq!(required.errors()[lane], reversed.errors()[other]);
        if let Ok(value) = required.lane(lane) {
            assert_same_f64(value, reversed.lane(other).unwrap());
        }
    }
    assert_eq!(input, preserved_input);

    // Accuracy belongs to each call; a previous result is not an acceptance cache.
    let tighter = MasterAccuracyContext::binary(44, 1);
    let tight = evaluator.evaluate_batch_with_accuracy(&vacuum, 1, &tighter, REQUIRE).unwrap();
    assert_certificate(&tight, 0, 0, &tighter, Some(Rational::from((1, 6))));
    let impossible = MasterAccuracyContext::binary(200, 1);
    let rejected = evaluator.evaluate_batch_with_accuracy(&vacuum, 1, &impossible, REQUIRE).unwrap();
    assert!(rejected.lane(0).is_err());
    assert!(rejected.diagnostics()[0].finite.is_none());
    assert_eq!(evaluator.prepared_series_orders(), vec![2, 4]);
    let mut explicit = accuracy.clone();
    explicit.requested_bits = 200; // Metadata does not replace the explicit budgets.
    explicit.relative = Rational::from(0);
    let accepted = evaluator.evaluate_batch_with_accuracy(&vacuum, 1, &explicit, REQUIRE).unwrap();
    assert_eq!(accepted.diagnostics()[0].requested_bits, 200);
    assert_certificate(&accepted, 0, 0, &explicit, Some(Rational::from((1, 6))));
    assert_same_f64(accepted.lane(0).unwrap(), required.lane(0).unwrap());

    let mut invalid_contexts = vec![MasterAccuracyContext::binary(0, 1), MasterAccuracyContext::binary(40, 2)];
    let mut nonpositive = accuracy.clone();
    nonpositive.absolute_per_component[0] = Rational::from(0);
    invalid_contexts.push(nonpositive);
    let mut negative_relative = accuracy.clone();
    negative_relative.relative = Rational::from(-1);
    invalid_contexts.push(negative_relative);
    for context in invalid_contexts {
        assert!(matches!(evaluator.evaluate_batch_with_accuracy(&[], 0, &context, REQUIRE), Err(MasterError::InvalidAccuracy(_))));
    }
    assert!(matches!(evaluator.evaluate_batch_with_accuracy(&[], 1, &accuracy, REQUIRE),
        Err(MasterError::InputLength { expected: 11, actual: 0 })));
    assert!(matches!(evaluator.evaluate_batch_with_accuracy(&[], usize::MAX, &accuracy, REQUIRE),
        Err(MasterError::SizeOverflow)));

    let mut independent = AccuracyMasterEvaluator::new(ordinary);
    assert!(independent.prepared_series_orders().is_empty());
    let other = independent.evaluate_batch_with_accuracy(&vacuum, 1, &accuracy, REQUIRE).unwrap();
    assert_same_f64(other.lane(0).unwrap(), required.lane(0).unwrap());
    assert_eq!(independent.prepared_series_orders(), vec![2]);
    assert_eq!(evaluator.prepared_series_orders(), vec![2, 4]);
}

#[test]
fn zero_primal_tail_does_not_hide_a_third_mass_derivative() {
    let shape = (0..=3).map(|degree| vec![degree]).collect::<Vec<_>>();
    let prototype = Float::new(256);
    let mass_jet = vacuum_jet(&prototype, 4, true, false);
    let scale_jet = vacuum_jet(&prototype, 4, false, true);
    let input = [mass_jet, scale_jet].concat();
    let preserved = input.clone();
    let ordinary = MasterEvaluator::new_dual(
        ScalarIntegral::D0, prototype, HyperDual::<Rational>::new(shape),
    ).unwrap();
    let mut evaluator = AccuracyMasterEvaluator::new(ordinary);
    let accuracy = MasterAccuracyContext::binary(128, 4);
    let batch = evaluator.evaluate_batch_with_accuracy(&input, 2, &accuracy, REQUIRE).unwrap();
    assert_eq!(batch.row_count(), 2);
    assert_eq!(batch.component_count(), 4);
    assert_eq!(batch.ordinary_rows, 0);
    assert_eq!(evaluator.prepared_series_orders(), vec![2, 4]);
    let mass = &batch.diagnostics()[0];
    assert_eq!(mass.formula, Formula::NearVacuum { order: 4 });
    assert_eq!(mass.attempts.len(), 2);
    assert_eq!(mass.attempts[0].order, 2);
    assert_eq!(mass.attempts[0].status, SeriesStatus::UncertifiedComponent(3));
    assert_eq!(mass.attempts[0].rho, Some(Rational::from(0)));
    assert_eq!(mass.attempts[1].order, 4);
    assert_eq!(mass.attempts[1].status, SeriesStatus::Success);
    assert_eq!(mass.attempts[1].rho, Some(Rational::from(0)));
    assert_eq!(batch.diagnostics()[1].formula, Formula::NearVacuum { order: 2 });
    assert_eq!(batch.diagnostics()[1].attempts.len(), 1);
    assert_eq!(batch.series_execution.len(), 2);
    for ((order, execution), (expected_order, expected_rows)) in batch.series_execution.iter().zip([(2, 2), (4, 1)]) {
        assert_eq!(*order, expected_order);
        assert_eq!((execution.input_rows, execution.active_rows), (expected_rows, expected_rows));
        assert_eq!((execution.numeric_batch_calls, execution.enclosure_batch_calls, execution.tail_batch_calls), (1, 1, 1));
    }
    let exact_mass = [(1, 6), (-1, 3), (1, 2), (-2, 3)];
    for (component, exact) in exact_mass.into_iter().enumerate() {
        assert_certificate(&batch, 0, component, &accuracy, Some(Rational::from(exact)));
        assert_eq!(mass.finite.as_ref().unwrap()[component].truncation, Rational::from(0));
        assert_certificate(&batch, 1, component, &accuracy,
            Some(if component == 0 { Rational::from((1, 6)) } else { Rational::from(0) }));
    }
    for lane in 0..2 {
        assert_zero_poles(&batch.lane(lane).unwrap());
    }
    assert!(batch.lane(1).unwrap().finite[1..].iter()
        .all(|value| value.re.is_zero() && value.im.is_zero()));
    assert_eq!(input, preserved);
}

#[test]
fn unsupported_native_shape_can_use_only_explicit_uncertified_fallback() {
    let shape = (0..=5).map(|degree| vec![degree]).collect::<Vec<_>>();
    let seeded = vacuum_jet(&0f64, 6, false, true);
    let mut upper = seeded.clone();
    upper[6 * 6].im = 1.;
    let constant = vacuum_jet(&0f64, 6, false, false);
    let input = [seeded, upper, constant].concat();
    let mut ordinary = MasterEvaluator::new_dual(
        ScalarIntegral::D0, 0f64, HyperDual::<Rational>::new(shape),
    ).unwrap();
    let mut evaluator = AccuracyMasterEvaluator::new(ordinary.clone());
    let accuracy = MasterAccuracyContext::binary(40, 6);
    let required = evaluator.evaluate_batch_with_accuracy(&input, 3, &accuracy, REQUIRE).unwrap();
    assert!(evaluator.prepared_series_orders().is_empty());
    assert!(required.series_execution.is_empty());
    assert_eq!(required.ordinary_rows, 0);
    for lane in [0, 2] {
        assert_eq!(required.errors()[lane], Some(LaneError::UncertifiedFinite(None)));
        assert!(required.lane(lane).is_err());
        let diagnostic = &required.diagnostics()[lane];
        assert_eq!(diagnostic.formula, Formula::Rejected);
        assert!(diagnostic.series_unavailable.is_some());
        assert!(diagnostic.finite.is_none());
        assert!(diagnostic.attempts.is_empty());
    }
    assert_eq!(required.errors()[1], Some(LaneError::UpperHalfPlaneMass { argument: 6 }));
    assert!(required.diagnostics()[1].series_unavailable.is_none());
    assert!(required.diagnostics()[1].attempts.is_empty());

    let allowed = evaluator.evaluate_batch_with_accuracy(&input, 3, &accuracy, ALLOW).unwrap();
    let legacy = ordinary.evaluate_batch(&input, 3).unwrap();
    assert_eq!(allowed.errors(), legacy.errors());
    assert_eq!(allowed.ordinary_rows, 2);
    assert!(allowed.series_execution.is_empty());
    assert!(evaluator.prepared_series_orders().is_empty());
    for lane in [0, 2] {
        let diagnostic = &allowed.diagnostics()[lane];
        assert_eq!(diagnostic.formula, Formula::OrdinaryUncertified);
        assert!(diagnostic.finite.is_none());
        assert!(diagnostic.series_unavailable.is_some());
        assert!(diagnostic.attempts.is_empty());
        let value = allowed.lane(lane).unwrap();
        assert_same_f64(value, legacy.lane(lane).unwrap());
        assert_eq!(value.finite[0].re.to_bits(), 0f64.from_rational(&Rational::from((1, 6))).to_bits());
        assert_eq!(value.finite[0].im, 0.);
        assert!(value.finite[1..].iter().all(|coefficient| coefficient.re == 0. && coefficient.im == 0.));
        assert_zero_poles(&value);
    }
    assert!(allowed.lane(1).is_err());
    assert!(allowed.diagnostics()[1].finite.is_none());
}
