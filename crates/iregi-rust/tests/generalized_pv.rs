//! Generalized mode dispatch: independent reference values and interface checks.
//! Original IREGI's generalized tensor wrapper is real-mass-only; the complex
//! tensor test below explicitly compares independent Rust PV and IBP recurrences.

use iregi_rust::{
    Algorithm, BatchResult, Convention, InvariantInput, KinematicsBatch, LaneStatus,
    PrecisionRescueWorkspace, ReductionPlan, ReductionSettings, RescueSettings,
    ScalarKinematicsBatch, ScalarReductionPlan, ScalarWorkspace, TensorWeights, Workspace,
};
use serde_json::Value;
use symbolica::domains::{
    float::{Complex, Float},
    integer::Integer,
};

const MODES: [Algorithm; 3] = [
    Algorithm::Ibp,
    Algorithm::PassarinoVeltman,
    Algorithm::Hybrid,
];

fn c(value: f64) -> Complex<f64> {
    Complex::new(value, 0.)
}

fn settings(input: &Value, mode: Algorithm) -> ReductionSettings {
    let propagators = input["propagators"].as_array().unwrap();
    let mut settings =
        ReductionSettings::new(propagators.len(), input["rank"].as_u64().unwrap() as u16);
    settings.dimension_shift = (input["dimension_shift"].as_i64().unwrap() / 2) as i16;
    settings.powers = propagators
        .iter()
        .map(|propagator| propagator["power"].as_u64().unwrap() as u16)
        .collect();
    settings.algorithm = mode;
    settings
}

fn row(input: &Value) -> Vec<Complex<f64>> {
    let propagators = input["propagators"].as_array().unwrap();
    let mut row: Vec<_> = propagators
        .iter()
        .flat_map(|propagator| {
            propagator["shift"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| c(value.as_f64().unwrap()))
        })
        .collect();
    row.extend(propagators.iter().map(|propagator| {
        Complex::new(
            propagator["mass_squared"][0].as_f64().unwrap(),
            propagator["mass_squared"][1].as_f64().unwrap(),
        )
    }));
    row.push(c(input["mu"].as_f64().unwrap()));
    row
}

fn characteristic_scale(input: &Value) -> f64 {
    input["propagators"].as_array().unwrap().iter().fold(
        input["mu"].as_f64().unwrap(),
        |scale, propagator| {
            let scale = propagator["shift"]
                .as_array()
                .unwrap()
                .iter()
                .fold(scale, |scale, value| {
                    scale.max(value.as_f64().unwrap().abs())
                });
            scale.max(
                propagator["mass_squared"][0]
                    .as_f64()
                    .unwrap()
                    .hypot(propagator["mass_squared"][1].as_f64().unwrap())
                    .sqrt(),
            )
        },
    )
}

fn compare_reference(result: &BatchResult<f64>, lane: usize, case: &Value) {
    assert_eq!(case["expected"]["stable"], true);
    let input = &case["input"];
    let actual = result
        .lane(lane)
        .unwrap_or_else(|error| panic!("{} lane {lane}: {error:?}", input["id"]));
    let expected = case["expected"]["moments"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len() * 4);
    let powers: i32 = input["propagators"]
        .as_array()
        .unwrap()
        .iter()
        .map(|propagator| propagator["power"].as_i64().unwrap() as i32)
        .sum();
    let base_dimension = 4 + input["dimension_shift"].as_i64().unwrap() as i32 - 2 * powers;
    let scale = characteristic_scale(input);
    for (index, expected) in expected.iter().enumerate() {
        let monomial = result.basis[index];
        // Exact factorial arithmetic is delegated to Symbolica, independently
        // of the core's prepared Atom convention transformation.
        let weight = if result.metadata().tensor_weights() == TensorWeights::Multinomial {
            monomial
                .0
                .into_iter()
                .fold(
                    Integer::factorial(monomial.rank() as u32),
                    |value, power| value / Integer::factorial(u32::from(power)),
                )
                .to_i64()
                .unwrap() as f64
        } else {
            1.
        };
        let natural = weight * scale.powi(base_dimension + monomial.rank() as i32);
        for slot in 0..4 {
            let re = weight * expected[slot][0].as_f64().unwrap();
            let im = weight * expected[slot][1].as_f64().unwrap();
            let value = actual[4 * index + slot];
            let error = (value.re - re).hypot(value.im - im) / (natural + re.hypot(im));
            assert!(
                error.is_finite() && error <= 1e-10,
                "{} moment {index}, slot {slot}: {value:?} expected ({re},{im}), normalized error {error}",
                input["id"]
            );
        }
    }
}

#[test]
fn supplied_products_are_authoritative_in_generalized_pv_and_scalar_interfaces() {
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../validation/generalized-pv/reference-capture.json"
    ))
    .unwrap();
    for id in ["n2_p12_h1_mode0", "n2_p21_h1_mode0", "n3_p123_h4_mode0"] {
        let case = fixtures["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["input"]["id"] == id)
            .unwrap();
        let input = &case["input"];
        let n = input["propagators"].as_array().unwrap().len();
        // These fixture vectors are (0,e_x,2e_y): their exact packed Gram
        // entries are data, independent of the deliberately different
        // Cartesian coordinates provided to the tensor interface below.
        let gram = match n {
            2 => vec![0., 0., -1.],
            3 => vec![0., 0., 0., -1., 0., -4.],
            _ => unreachable!(),
        };
        let mut scalar_row: Vec<_> = gram.into_iter().map(c).collect();
        scalar_row.extend_from_slice(&row(input)[4 * n..]);
        for mode in MODES {
            let mut configuration = settings(input, mode);
            configuration.invariants = InvariantInput::SuppliedScalarProducts;
            let plan = ReductionPlan::new(configuration.clone()).unwrap();
            let mut tensor_values = Vec::new();
            for offset in [2., 7.] {
                tensor_values.extend((0..4 * n).map(|index| c(offset + index as f64)));
                tensor_values.extend_from_slice(&scalar_row);
            }
            let mut workspace =
                PrecisionRescueWorkspace::new(&plan, 0., RescueSettings::default()).unwrap();
            let actual = workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 2,
                    components: 1,
                    values: tensor_values,
                })
                .unwrap();
            compare_reference(&actual, 0, case);
            compare_reference(&actual, 1, case);
            assert_eq!(actual.lane(0).unwrap(), actual.lane(1).unwrap());
            let scalar_plan = ScalarReductionPlan::new(configuration).unwrap();
            let mut scalar_workspace = ScalarWorkspace::new(&scalar_plan, 0.).unwrap();
            let scalar = scalar_workspace
                .reduce_batch(&ScalarKinematicsBatch {
                    rows: 2,
                    components: 1,
                    values: scalar_row.repeat(2),
                })
                .unwrap();
            compare_reference(&scalar, 0, case);
            compare_reference(&scalar, 1, case);
        }
    }
}

#[test]
fn generalized_multinomial_outputs_match_original_nonzero_mixed_moments() {
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../validation/generalized-pv/reference-mixed-capture.json"
    ))
    .unwrap();
    for case in fixtures["cases"].as_array().unwrap() {
        let input = &case["input"];
        let mode = MODES[input["mode"].as_u64().unwrap() as usize];
        let mut configuration = settings(input, mode);
        configuration.tensor_weights = TensorWeights::Multinomial;
        let plan = ReductionPlan::new(configuration).unwrap();
        let mixed = plan
            .basis()
            .iter()
            .position(|monomial| monomial.0 == [1, 1, 0, 0])
            .unwrap();
        let mixed_finite = &case["expected"]["moments"][mixed][3];
        assert!(
            mixed_finite[0]
                .as_f64()
                .unwrap()
                .hypot(mixed_finite[1].as_f64().unwrap())
                > 1e-6,
            "the fixture must expose a nontrivial multiplicity"
        );
        let mut workspace =
            PrecisionRescueWorkspace::new(&plan, 0., RescueSettings::default()).unwrap();
        let actual = workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: row(input),
            })
            .unwrap();
        assert_eq!(
            actual.metadata().tensor_weights(),
            TensorWeights::Multinomial
        );
        assert_eq!(actual.metadata().convention(), Convention::IregiNative);
        compare_reference(&actual, 0, case);
    }
}

#[test]
fn complex_generalized_scalars_match_original_in_each_rust_mode() {
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../validation/generalized-pv/reference-complex-capture.json"
    ))
    .unwrap();
    let cases = fixtures["scalar_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    for case in cases {
        let input = &case["input"];
        for mode in MODES {
            let plan = ReductionPlan::new(settings(input, mode)).unwrap();
            let mut workspace =
                PrecisionRescueWorkspace::new(&plan, 0., RescueSettings::default()).unwrap();
            let actual = workspace
                .reduce_batch(&KinematicsBatch {
                    rows: 1,
                    components: 1,
                    values: row(input),
                })
                .unwrap();
            compare_reference(&actual, 0, case);
        }
    }
}

#[test]
fn complex_generalized_tensor_batches_agree_between_pv_and_ibp() {
    let mut configuration = ReductionSettings::new(2, 2);
    configuration.powers = vec![1, 2];
    configuration.dimension_shift = 1;
    let values = [0.25, 0.5, 0.75]
        .into_iter()
        .flat_map(|width| {
            let mut row: Vec<_> = [0., 0., 0., 0., 1., 2., 0., 0.]
                .into_iter()
                .map(c)
                .collect();
            row.extend([
                Complex::new(4., -width),
                Complex::new(9., -2. * width),
                c(3.),
            ]);
            row
        })
        .collect();
    let input = KinematicsBatch {
        rows: 3,
        components: 1,
        values,
    };
    configuration.algorithm = Algorithm::Ibp;
    let ibp_plan = ReductionPlan::new(configuration.clone()).unwrap();
    let mut ibp = Workspace::new(&ibp_plan, 0.).unwrap();
    let expected = ibp.reduce_batch(&input).unwrap();
    for mode in [Algorithm::PassarinoVeltman, Algorithm::Hybrid] {
        configuration.algorithm = mode;
        let plan = ReductionPlan::new(configuration.clone()).unwrap();
        let mut workspace = Workspace::new(&plan, 0.).unwrap();
        let actual = workspace.reduce_batch(&input).unwrap();
        assert_eq!(workspace.last_execution().input_rows, 3);
        assert_eq!(workspace.last_execution().argument_rows, 3);
        assert_eq!(workspace.last_execution().reduction_rows, 3);
        assert_eq!(
            workspace.last_execution().master_rows,
            3 * plan.master_count()
        );
        assert_eq!(actual.basis, expected.basis);
        for lane in 0..3 {
            let actual_values = actual.lane(lane).unwrap();
            let expected_values = expected.lane(lane).unwrap();
            assert_eq!(actual_values.len(), actual.basis.len() * 4);
            assert_eq!(expected_values.len(), expected.basis.len() * 4);
            for (index, (left, right)) in actual_values.iter().zip(expected_values).enumerate() {
                let rank = actual.basis[index / 4].rank() as i32;
                let natural = 3_f64.powi(rank);
                assert!(
                    (left.re - right.re).hypot(left.im - right.im)
                        <= 1e-10 * (natural + right.re.hypot(right.im)),
                    "{mode:?} lane {lane}, coefficient {index}: {left:?} versus {right:?}"
                );
            }
        }
    }
}

#[test]
fn generalized_hybrid_preserves_unresolved_singular_gram_diagnostics() {
    let mut configuration = ReductionSettings::new(3, 1);
    configuration.powers = vec![1, 2, 1];
    configuration.dimension_shift = 1;
    configuration.convention = Convention::OneLoop;
    configuration.algorithm = Algorithm::PassarinoVeltman;
    let pv_plan = ReductionPlan::new(configuration.clone()).unwrap();
    configuration.algorithm = Algorithm::Hybrid;
    let hybrid_plan = ReductionPlan::new(configuration.clone()).unwrap();
    configuration.algorithm = Algorithm::Ibp;
    let ibp_plan = ReductionPlan::new(configuration).unwrap();
    // The regular lane triggers conservative finite-cancellation diagnostics
    // in binary64. Keep the same 53-bit accuracy request while increasing the
    // working precision; the singular neighbor must remain a failed lane.
    for bits in [128, 256] {
        let values = [1, 0]
            .into_iter()
            .flat_map(|transverse| {
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, transverse, 0, 4, 9, 16, 3]
                    .map(|value| Complex::new(Float::with_val(bits, value), Float::new(bits)))
            })
            .collect();
        let input = KinematicsBatch {
            rows: 2,
            components: 1,
            values,
        };
        let mut pv = Workspace::new(&pv_plan, Float::new(bits)).unwrap();
        pv.set_accuracy_precision(53).unwrap();
        let primary = pv.reduce_batch(&input).unwrap();
        let mut hybrid = Workspace::new(&hybrid_plan, Float::new(bits)).unwrap();
        hybrid.set_accuracy_precision(53).unwrap();
        let actual = hybrid.reduce_batch(&input).unwrap();
        eprintln!(
            "generalized fallback working_bits={bits}, requested_bits=53, primary={:?}, hybrid={:?}, flags={:?}, stats={:?}",
            primary.statuses,
            actual.statuses,
            actual.fallback_used,
            hybrid.last_execution()
        );
        assert_eq!(primary.statuses[0], LaneStatus::Success);
        assert!(matches!(primary.statuses[1], LaneStatus::Unstable(_)));
        assert_eq!(actual.fallback_used, [false, true]);
        assert_eq!(hybrid.last_execution().fallback_rows, 1);
        assert_eq!(actual.lane(0).unwrap(), primary.lane(0).unwrap());
        let mut ibp = Workspace::new(&ibp_plan, Float::new(bits)).unwrap();
        ibp.set_accuracy_precision(53).unwrap();
        let expected = ibp.reduce_batch(&input).unwrap();
        // In this exact collinear triangle, the inverse Cayley matrix exists
        // but sum(R^-1 * 1) is zero. Its independent IBP dimension recurrence
        // is also singular; an attempted fallback must remain a failure.
        assert!(matches!(expected.statuses[1], LaneStatus::Unstable(_)));
        assert!(matches!(&actual.statuses[1], LaneStatus::Unstable(reason)
            if reason.contains("PV reduction failed") && reason.contains("IBP fallback returned")));
        assert_eq!(hybrid.last_execution().fallback_successes, 0);
        assert!(actual.lane(1).is_err());
    }
}
