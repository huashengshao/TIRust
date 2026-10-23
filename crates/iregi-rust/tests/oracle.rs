//! Independent frozen results from the separately built original Fortran code.
use iregi_rust::{
    Algorithm, KinematicsBatch, LaneStatus, PrecisionRescueWorkspace, ReductionPlan,
    ReductionSettings, RescueSettings, Workspace,
};
use symbolica::domains::float::Complex;

#[test]
fn original_iregi_scalar_and_tensor_fixtures() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../../validation/reference/fixtures.json")).unwrap();
    let filter = std::env::var("IREGI_ORACLE_CASE").ok();
    let mut validated = 0;
    for case in fixtures["cases"].as_array().unwrap() {
        let input = &case["input"];
        let id = input["id"].as_str().unwrap();
        if filter.as_ref().is_some_and(|filter| !id.contains(filter)) {
            continue;
        }
        let propagators = input["propagators"].as_array().unwrap();
        let mut settings =
            ReductionSettings::new(propagators.len(), input["rank"].as_u64().unwrap() as u16);
        settings.algorithm = match input["mode"].as_u64().unwrap() {
            0 => Algorithm::Ibp,
            1 => Algorithm::PassarinoVeltman,
            _ => Algorithm::Hybrid,
        };
        settings.dimension_shift = (input["dimension_shift"].as_i64().unwrap() / 2) as i16;
        settings.powers = propagators
            .iter()
            .map(|p| p["power"].as_u64().unwrap() as u16)
            .collect();
        if input["api"] == "scalar" || input["api"] == "general" {
            settings.algorithm = Algorithm::Ibp;
        }
        let mut characteristic_scale = input["mu"].as_f64().unwrap();
        for propagator in propagators {
            for value in propagator["shift"].as_array().unwrap() {
                characteristic_scale = characteristic_scale.max(value.as_f64().unwrap().abs());
            }
            let mass = propagator["mass_squared"].as_array().unwrap();
            characteristic_scale = characteristic_scale.max(
                mass[0]
                    .as_f64()
                    .unwrap()
                    .hypot(mass[1].as_f64().unwrap())
                    .sqrt(),
            );
        }
        let base_dimension = 4 + 2 * i32::from(settings.dimension_shift)
            - 2 * settings.powers.iter().map(|&p| i32::from(p)).sum::<i32>();
        let mut values = Vec::new();
        for propagator in propagators {
            for value in propagator["shift"].as_array().unwrap() {
                values.push(Complex::new(value.as_f64().unwrap(), 0.0));
            }
        }
        for propagator in propagators {
            let m = propagator["mass_squared"].as_array().unwrap();
            values.push(Complex::new(m[0].as_f64().unwrap(), m[1].as_f64().unwrap()));
        }
        values.push(Complex::new(input["mu"].as_f64().unwrap(), 0.0));
        eprintln!("preparing {id}");
        let plan = ReductionPlan::new(settings).unwrap_or_else(|e| panic!("{id}: {e}"));
        let mut workspace = Workspace::new(&plan, 0.0f64).unwrap_or_else(|e| panic!("{id}: {e}"));
        let batch = KinematicsBatch {
            rows: 1,
            components: 1,
            values,
        };
        let mut result = workspace.reduce_batch(&batch).unwrap();
        if matches!(result.statuses[0], LaneStatus::Unstable(_)) {
            // A rejected binary64 result is not a successful binary64 case.
            // Qualify the explicit rescue API with two independently evaluated
            // working precisions, then retain the same reference tolerance.
            eprintln!(
                "precision rescue required for {id}: {:?}",
                result.statuses[0]
            );
            let mut rescue =
                PrecisionRescueWorkspace::new(&plan, 0.0_f64, RescueSettings::default()).unwrap();
            result = rescue.reduce_batch(&batch).unwrap();
            assert_eq!(
                rescue.last_execution().accepted_rows,
                1,
                "{id}: {:?}",
                rescue.last_rescue_statuses()
            );
        }
        let actual = result.lane(0).unwrap_or_else(|e| panic!("{id}: {e:?}"));
        let expected = case["expected"]["moments"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len() * 4, "{id}");
        for (moment, expected) in expected.iter().enumerate() {
            for pole in 0..4 {
                let expected = &expected[pole];
                let re = expected[0].as_f64().unwrap();
                let im = expected[1].as_f64().unwrap();
                let value = &actual[4 * moment + pole];
                let error = (value.re - re).hypot(value.im - im);
                let natural =
                    characteristic_scale.powi(base_dimension + plan.basis()[moment].rank() as i32);
                assert!(
                    natural.is_finite() && natural > 0.0,
                    "{id}: invalid dimensional normalization"
                );
                let scale = natural + re.hypot(im);
                assert!(
                    error <= 1e-10 * scale,
                    "{id}, moment {moment}, pole {pole}: got ({},{}), expected ({re},{im}), scaled error {}",
                    value.re,
                    value.im,
                    error / scale
                );
            }
        }
        validated += 1;
        eprintln!("validated {id}");
    }
    assert!(validated > 0, "oracle filter selected no cases");
}
