//! Explicit precision-rescue qualification against original IREGI with snapping.
//!
//! Every IBP triangle/box fixture participates, including the exact singular
//! pinched-bubble case. Primary binary64 failures remain visible in the report;
//! higher precision never changes the requested accuracy or reference tolerance.

use iregi_rust::{
    Algorithm, ExecutionStats, InputPolicy, KinematicsBatch, LaneStatus, PrecisionRescueWorkspace,
    ReductionPlan, ReductionSettings, RescueSettings, RescueStatus,
};
use serde_json::{Value, json};
use symbolica::domains::float::Complex;

const TOLERANCE: f64 = 1e-10;
const EXPECTED_CASES: usize = 8;

fn number(value: &Value) -> f64 {
    value
        .as_str()
        .map(|value| value.parse().unwrap())
        .unwrap_or_else(|| value.as_f64().unwrap())
}

fn lane_status(status: &LaneStatus) -> Value {
    match status {
        LaneStatus::Success => json!({"kind": "Success"}),
        LaneStatus::InvalidInput(reason) => json!({"kind": "InvalidInput", "reason": reason}),
        LaneStatus::MasterFailure(reason) => json!({"kind": "MasterFailure", "reason": reason}),
        LaneStatus::Unstable(reason) => json!({"kind": "Unstable", "reason": reason}),
    }
}

fn rescue_status(status: &RescueStatus) -> Value {
    match status {
        RescueStatus::NotRequested => json!({"kind": "NotRequested"}),
        RescueStatus::Converged {
            working_bits,
            verification_bits,
            primary_agreed,
        } => json!({
            "kind": "Converged", "working_bits": working_bits,
            "verification_bits": verification_bits, "primary_agreed": primary_agreed,
        }),
        RescueStatus::Rejected(reason) => json!({"kind": "Rejected", "reason": reason}),
    }
}

fn execution_stats(stats: ExecutionStats) -> Value {
    json!({
        "input_rows": stats.input_rows, "argument_rows": stats.argument_rows,
        "master_family_batches": stats.master_family_batches, "master_rows": stats.master_rows,
        "reduction_rows": stats.reduction_rows, "fallback_rows": stats.fallback_rows,
        "fallback_successes": stats.fallback_successes, "analytic_rows": stats.analytic_rows,
    })
}

fn finite_or_description(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        json!(value.to_string())
    }
}

fn settings_for(input: &Value) -> ReductionSettings {
    let lines = input["propagators"].as_array().unwrap();
    let mut settings = ReductionSettings::new(lines.len(), input["rank"].as_u64().unwrap() as u16);
    settings.algorithm = Algorithm::Ibp;
    let shift = input["dimension_shift"].as_i64().unwrap();
    assert_eq!(shift % 2, 0, "reference dimension shift must be even");
    settings.dimension_shift = (shift / 2) as i16;
    settings.powers = lines
        .iter()
        .map(|line| line["power"].as_u64().unwrap() as u16)
        .collect();
    assert_eq!(input["snap"], true);
    settings.input_policy = InputPolicy::LegacyOnShell {
        threshold: input["threshold"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| input["threshold"].to_string()),
    };
    settings
}

fn case_record(case: &Value) -> Value {
    json!({
        "id": case["input"]["id"], "input": case["input"], "reference": case["reference"],
        "passed": false, "primary_status": null, "rescue_status": null,
        "final_status": null, "maximum_normalized_error": null, "compared_coefficients": 0,
    })
}

#[test]
fn legacy_ibp_triangles_and_boxes_pass_verified_precision_rescue() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../validation/reference/snapping/reductions.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    let selected: Vec<_> = cases
        .iter()
        .filter(|case| {
            case["input"]["mode"].as_u64() == Some(0)
                && case["input"]["propagators"].as_array().unwrap().len() >= 3
        })
        .collect();
    let mut failures = Vec::new();
    if selected.len() != EXPECTED_CASES {
        failures.push(format!(
            "expected {EXPECTED_CASES} IBP triangle/box cases, found {}",
            selected.len()
        ));
    }
    let mut groups = std::collections::BTreeMap::<usize, Vec<&Value>>::new();
    for case in selected {
        groups
            .entry(case["input"]["propagators"].as_array().unwrap().len())
            .or_default()
            .push(case);
    }
    let mut report = Vec::new();
    let mut batch_reports = Vec::new();
    for (propagators, group) in groups {
        let settings = settings_for(&group[0]["input"]);
        for case in &group {
            assert_eq!(json!(settings_for(&case["input"])), json!(&settings));
        }
        eprintln!("preparing {propagators}-propagator legacy rescue batch");
        let preparation = ReductionPlan::new(settings.clone()).and_then(|plan| {
            let workspace =
                PrecisionRescueWorkspace::new(&plan, 0.0_f64, RescueSettings::default())?;
            Ok((plan, workspace))
        });
        let (plan, mut workspace) = match preparation {
            Ok(prepared) => prepared,
            Err(error) => {
                for case in group {
                    let mut record = case_record(case);
                    record["preparation_error"] = json!(error.to_string());
                    failures.push(format!("{}: preparation: {error}", case["input"]["id"]));
                    report.push(record);
                }
                continue;
            }
        };
        let mut values = Vec::new();
        let mut scales = Vec::new();
        for case in &group {
            let input = &case["input"];
            let lines = input["propagators"].as_array().unwrap();
            let mut scale = number(&input["mu"]);
            for line in lines {
                for component in line["shift"].as_array().unwrap() {
                    let component = number(component);
                    scale = scale.max(component.abs());
                    values.push(Complex::new(component, 0.0));
                }
            }
            for line in lines {
                let mass = &line["mass_squared"];
                let (re, im) = (number(&mass[0]), number(&mass[1]));
                scale = scale.max(re.hypot(im).sqrt());
                values.push(Complex::new(re, im));
            }
            values.push(Complex::new(number(&input["mu"]), 0.0));
            scales.push(scale);
        }
        let batch = KinematicsBatch {
            rows: group.len(),
            components: 1,
            values,
        };
        let before = batch.values.clone();
        let result = workspace.reduce_batch(&batch);
        let inputs_preserved = batch.values == before;
        let stats = workspace.last_execution();
        batch_reports.push(json!({
            "propagators": propagators, "rows": group.len(), "settings": settings,
            "primary": execution_stats(stats.primary),
            "working": execution_stats(stats.working),
            "verification": execution_stats(stats.verification),
            "requested_rows": stats.requested_rows, "accepted_rows": stats.accepted_rows,
            "corrected_successful_rows": stats.corrected_successful_rows,
        }));
        for (row, case) in group.iter().enumerate() {
            let id = case["input"]["id"].as_str().unwrap();
            let mut record = case_record(case);
            record["characteristic_scale"] = json!(scales[row]);
            record["inputs_preserved"] = json!(inputs_preserved);
            if let Some(status) = workspace.last_primary_statuses().get(row) {
                record["primary_status"] = lane_status(status);
            }
            if let Some(status) = workspace.last_rescue_statuses().get(row) {
                record["rescue_status"] = rescue_status(status);
            }
            let result = match &result {
                Ok(result) => result,
                Err(error) => {
                    record["evaluation_error"] = json!(error.to_string());
                    failures.push(format!("{id}: evaluation: {error}"));
                    report.push(record);
                    continue;
                }
            };
            record["final_status"] = lane_status(&result.statuses[row]);
            record["fallback_used"] = json!(result.fallback_used[row]);
            record["analytic_used"] = json!(result.analytic_used[row]);
            let actual = match result.lane(row) {
                Ok(actual) => actual,
                Err(status) => {
                    failures.push(format!("{id}: {status:?}"));
                    report.push(record);
                    continue;
                }
            };
            let expected = case["reference"]["moments"].as_array().unwrap();
            if actual.len() != expected.len() * 4 || expected.len() != plan.basis().len() {
                let error = format!(
                    "{id}: incomplete output: {} values for {} reference moments",
                    actual.len(),
                    expected.len()
                );
                record["comparison_error"] = json!(error);
                failures.push(error);
                report.push(record);
                continue;
            }
            let base_dimension = 4 + 2 * i32::from(settings.dimension_shift)
                - 2 * settings
                    .powers
                    .iter()
                    .map(|&power| i32::from(power))
                    .sum::<i32>();
            let mut maximum_error = 0.0_f64;
            let mut worst = (0, 0);
            let mut failed_coefficients = 0;
            let mut actual_moments = Vec::new();
            for (moment, poles) in expected.iter().enumerate() {
                let natural = scales[row].powi(base_dimension + plan.basis()[moment].rank() as i32);
                let mut actual_poles = Vec::new();
                for pole in 0..4 {
                    let (re, im) = (number(&poles[pole][0]), number(&poles[pole][1]));
                    let value = &actual[4 * moment + pole];
                    let error = if natural.is_finite() && natural > 0.0 {
                        (value.re - re).hypot(value.im - im) / (natural + re.hypot(im))
                    } else {
                        f64::INFINITY
                    };
                    if !error.is_finite() || error > TOLERANCE {
                        failed_coefficients += 1;
                    }
                    let comparable = if error.is_finite() {
                        error
                    } else {
                        f64::INFINITY
                    };
                    if comparable > maximum_error {
                        maximum_error = comparable;
                        worst = (moment, pole);
                    }
                    actual_poles.push(json!([
                        finite_or_description(value.re),
                        finite_or_description(value.im)
                    ]));
                }
                actual_moments.push(actual_poles);
            }
            let passed =
                failed_coefficients == 0 && inputs_preserved && case["reference"]["stable"] == true;
            record["passed"] = json!(passed);
            record["maximum_normalized_error"] = finite_or_description(maximum_error);
            record["worst_moment"] = json!(worst.0);
            record["worst_laurent_slot"] = json!(worst.1);
            record["failed_coefficients"] = json!(failed_coefficients);
            record["compared_coefficients"] = json!(actual.len());
            record["actual_moments"] = json!(actual_moments);
            if !passed {
                failures.push(format!(
                    "{id}: {failed_coefficients} coefficient failures, max error {maximum_error:e} \
                     at moment {}, Laurent {}; inputs preserved {inputs_preserved}, reference stable {}",
                    worst.0, worst.1, case["reference"]["stable"]
                ));
            }
            report.push(record);
        }
    }
    let report = json!({
        "schema_version": 1, "tolerance": TOLERANCE, "expected_cases": EXPECTED_CASES,
        "normalization": "Q=max(mu,abs(shift components),sqrt(abs(mass_squared))); error=abs(actual-reference)/(Q^(4+2*h+rank-2*sum(powers))+abs(reference))",
        "output_precision_bits": 53, "working_precision_bits": 128,
        "verification_precision_bits": 256, "accuracy_precision_bits": 53,
        "rescue_trigger": "UnstableOnly", "accuracy_override": null,
        "reference_fixture": "validation/reference/snapping/reductions.json",
        "reference_fixture_schema_version": fixture["schema_version"],
        "reference_fixture_case_count": cases.len(),
        "reference_executable_sha256": fixture["executable_sha256"],
        "reference_driver_sha256": fixture["driver_sha256"],
        "reproduction": "source scripts/env.sh; cargo test -p iregi-rust --test legacy_rescue -- --nocapture",
        "passed": failures.is_empty(), "failures": failures, "batches": batch_reports,
        "cases": report,
    });
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/legacy-snapping-rescue-validation.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    eprintln!(
        "legacy rescue validation: {} cases, {} failures; {}",
        report["cases"].as_array().unwrap().len(),
        failures.len(),
        path.display()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
