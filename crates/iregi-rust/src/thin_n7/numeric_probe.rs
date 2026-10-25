//! Private ignored numerical probe for the actual main ThinSeven implementation.
//! Public seven-point rank restrictions are unchanged.
//!
//! This deliberately performs no numerical specialization of symbolic plans.
//! Its output ledger is written before every potentially expensive stage.

#[path = "../../examples/seven_point_public_probe/domain.rs"]
mod domain;
use iregi_masters::NativeFloat;
use crate::{
    Algorithm, KinematicsBatch, LaneStatus, ReductionSettings,
};
use super::{ThinSevenReductionPlan as ReductionPlan, ThinSevenWorkspace as Workspace};
use serde_json::{Value, json};
use std::{error::Error, path::Path, time::Instant};
use symbolica::domains::{
    float::{Complex, Float, FloatLike, SingleFloat},
    integer::Integer,
    rational::Rational,
};

fn number(value: &Value) -> Result<Rational, Box<dyn Error>> {
    let text = value.as_str().ok_or("expected an exact rational string")?;
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let d = d.parse::<Integer>()?;
    if d == Integer::from(0) {
        return Err("zero rational denominator".into());
    }
    Ok(Rational::from((n.parse::<Integer>()?, d)))
}

fn write(path: &Path, record: &Value) -> Result<(), Box<dyn Error>> {
    std::fs::write(path, serde_json::to_string_pretty(record)? + "\n")?;
    Ok(())
}

fn execution_record(stats: &crate::ExecutionStats) -> Value {
    json!({
        "input_rows":stats.input_rows,"argument_rows":stats.argument_rows,
        "master_family_batches":stats.master_family_batches,"master_rows":stats.master_rows,
        "pentagon_box_rows":stats.pentagon_box_rows,"reduction_rows":stats.reduction_rows,
        "fallback_rows":stats.fallback_rows,"fallback_successes":stats.fallback_successes,
        "analytic_rows":stats.analytic_rows,
    })
}

fn run_capture() -> Result<u8, Box<dyn Error>> {
    // Each externally supervised process runs exactly this ignored test. No
    // reference path or expected coefficient is an input to the core probe.
    let input_path = std::env::var("IREGI_N7_PROBE_FIXTURE")?;
    let rank: u16 = std::env::var("IREGI_N7_PROBE_RANK")?.parse()?;
    let working_bits: u32 = std::env::var("IREGI_N7_PROBE_WORKING_BITS")?.parse()?;
    let output = std::env::var("IREGI_N7_PROBE_OUTPUT")?;
    let n = 7;
    let rows = 1;
    let requested_bits = 53;
    let algorithm = Algorithm::PassarinoVeltman;
    if !(1..=7).contains(&rank) || !matches!(working_bits, 128 | 256) {
        return Err("probe requires rank 1..=7 and working precision 128 or 256".into());
    }
    for key in [
        "IREGI_PV_PREPARATION_PROFILE",
        "IREGI_PREPARATION_PROFILE",
        "SYMBOLICA_SCOPE_PROFILE",
    ] {
        if std::env::var(key).as_deref() != Ok("1") {
            return Err(format!("profile flag {key}=1 is required").into());
        }
    }
    if Path::new(&output).exists() {
        return Err("probe output must not already exist".into());
    }
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&input_path)?)?;
    let samples = fixture["samples"].as_array().ok_or("missing samples")?;
    if samples.len() < rows {
        return Err("insufficient fixture samples".into());
    }
    let prototype = Float::new(working_bits);
    let mut values = Vec::with_capacity(rows * (5 * n + 1));
    let mut original_inputs = Vec::new();
    for sample in &samples[..rows] {
        let propagators = sample["propagators"]
            .as_array()
            .ok_or("missing propagators")?;
        if propagators.len() < n {
            return Err("insufficient fixture propagators".into());
        }
        let mut input = Vec::with_capacity(5 * n + 1);
        for propagator in &propagators[..n] {
            let shift = propagator["shift"].as_array().ok_or("missing shift")?;
            if shift.len() != 4 {
                return Err("invalid shift length".into());
            }
            for component in shift {
                input.push((number(component)?, Rational::from(0)));
            }
        }
        for propagator in &propagators[..n] {
            let mass = propagator["mass_squared"]
                .as_array()
                .ok_or("missing mass")?;
            if mass.len() != 2 || propagator["power"] != 1 {
                return Err("fixture requires complex mass pairs and unit powers".into());
            }
            input.push((number(&mass[0])?, number(&mass[1])?));
        }
        input.push((number(&sample["mu"])?, Rational::from(0)));
        for (re, im) in &input {
            let value = Complex::new(prototype.from_rational(re), prototype.from_rational(im));
            if value.re.exact_rational().as_ref() != Some(re)
                || value.im.exact_rational().as_ref() != Some(im)
            {
                return Err(
                    "fixture inputs must be represented exactly at working precision".into(),
                );
            }
            values.push(value);
        }
        original_inputs.push(
            input
                .iter()
                .map(|(re, im)| [re.to_string(), im.to_string()])
                .collect::<Vec<_>>(),
        );
    }
    let input = KinematicsBatch {
        rows,
        components: 1,
        values,
    };
    let mut settings = ReductionSettings::new(n, rank);
    settings.algorithm = algorithm;
    let mut record = json!({
        "schema_version": 1,
        "status": "preparing_plan",
        "interface": "Actual main cfg(test): ThinSevenReductionPlan::new; ThinSevenWorkspace<Float>::new; ThinSevenWorkspace::reduce_batch",
        "private_validation_overlay": false,
        "private_internal_probe": true,
        "production_n7_rank_guard_unchanged": true,
        "preparation_profiles": ["IREGI_PV_PREPARATION_PROFILE=1", "IREGI_PREPARATION_PROFILE=1", "SYMBOLICA_SCOPE_PROFILE=1"],
        "independent_reference_qualified": false,
        "public_positive_rank_support": false,
        "settings": settings,
        "working_bits": working_bits,
        "requested_bits": requested_bits,
        "rows": rows,
        "components": 1,
        "fixture": input_path,
        "input_exact_rational": original_inputs,
        "limitations": "Private main-module N7 positive-rank coordinate-input probe, ordinary component and all21mandatoryN5children, actual-route primal sensitivity operands; no public guard override, numerical specialization, reference scalars or disabled internal guards. Working precision is not a claim of achieved requested precision. Timings are diagnostic, not performance gates.",
    });
    let output = Path::new(&output);
    write(output, &record)?;
    let total = Instant::now();
    record["status"] = json!("checking_exact_domain");
    write(output, &record)?;
    let domain = domain::audit(&fixture)?;
    let accepted = domain["domain_accepted"] == true;
    record["domain_audit"] = domain;
    record["status"] = json!(if accepted {
        "preparing_plan"
    } else {
        "domain_rejected"
    });
    record["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
    write(output, &record)?;
    if !accepted {
        return Ok(2);
    }
    eprintln!("private main ThinSeven N={n} rank={rank} {algorithm:?}: preparing structural plan");
    let started = Instant::now();
    let plan = match ReductionPlan::new(settings) {
        Ok(plan) => plan,
        Err(error) => {
            record["status"] = json!("preparation_failed");
            record["error"] = json!(error.to_string());
            record["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
            write(output, &record)?;
            return Ok(2);
        }
    };
    record["diagnostic_plan_seconds"] = json!(started.elapsed().as_secs_f64());
    record["algorithm_used"] = json!(Algorithm::PassarinoVeltman);
    record["preparation_notes"] =
        json!(["Private main-module seven-point probe; native/raw; no quadrature import support"]);
    record["parameter_count"] = json!(36);
    record["input_layout"] = json!(format!("{:?}", plan.input_layout()));
    if plan.input_layout() != &(crate::InputLayout {
        shifts: 0..28, scalar_products: None, masses: 28..35, scale: 35,
    }) || plan.basis() != crate::basis::tensor_basis(rank) {
        return Err("private N7 positive-rank layout mismatch".into());
    }
    record["basis"] = json!(plan.basis());
    record["expected_layout"] = json!({
        "parent_moments":plan.basis().len(),
        "parent_complex_laurent_slots":plan.basis().len()*4,
        "child_rank":rank.saturating_sub(2),
        "child_moments":crate::basis::tensor_basis(rank.saturating_sub(2)).len(),
        "mandatory_child_subsets":21,
        "master_rows_are_rank_dependent":true,
    });
    record["status"] = json!("preparing_workspace");
    write(output, &record)?;
    eprintln!(
        "plan ready at {:.6}s; preparing workspace",
        total.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let mut workspace = match Workspace::new(&plan, prototype) {
        Ok(workspace) => workspace,
        Err(error) => {
            record["status"] = json!("workspace_failed");
            record["error"] = json!(error.to_string());
            record["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
            write(output, &record)?;
            return Ok(2);
        }
    };
    workspace.set_accuracy_precision(requested_bits)?;
    if workspace.component_count() != 1 || workspace.binary_precision() != working_bits {
        return Err("private N7 workspace component or precision mismatch".into());
    }
    record["diagnostic_workspace_seconds"] = json!(started.elapsed().as_secs_f64());
    record["status"] = json!("reducing");
    write(output, &record)?;
    eprintln!(
        "workspace ready at {:.6}s; reducing {rows} lanes",
        total.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let result = match workspace.reduce_batch(&input) {
        Ok(result) => result,
        Err(error) => {
            record["status"] = json!("reduction_failed");
            record["error"] = json!(error.to_string());
            record["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
            record["diagnostic_reduction_seconds"] = json!(started.elapsed().as_secs_f64());
            record["diagnostics"] = json!(workspace.last_diagnostics());
            record["execution_debug"] = json!(format!("{:?}", workspace.last_execution()));
            write(output, &record)?;
            return Ok(2);
        }
    };
    if result.basis.as_slice() != plan.basis()
        || result.components != 1
        || result.rows() != rows
        || result.statuses.len() != rows
        || result.fallback_used.len() != rows
        || result.analytic_used.len() != rows
        || result.metadata().convention() != crate::Convention::IregiNative
        || result.metadata().tensor_weights() != crate::TensorWeights::Components
        || result.metadata().dual_shape().is_some()
    {
        return Err("private N7 result layout or metadata mismatch".into());
    }
    record["diagnostic_reduction_seconds"] = json!(started.elapsed().as_secs_f64());
    record["lane_statuses"] = json!(
        result
            .statuses
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
    );
    record["metadata"] = json!({
        "convention":format!("{:?}", result.metadata().convention()),
        "tensor_weights":format!("{:?}", result.metadata().tensor_weights()),
        "dual_shape":result.metadata().dual_shape(),
    });
    record["fallback_used"] = json!(result.fallback_used);
    record["analytic_used"] = json!(result.analytic_used);
    record["diagnostics"] = json!(workspace.last_diagnostics());
    let mut lanes = Vec::with_capacity(rows);
    for row in 0..rows {
        match result.lane(row) {
            Ok(values) => {
                if values.len() != result.basis.len() * 4 {
                    return Err("incomplete public output array".into());
                }
                let moments = values.chunks_exact(4).map(|slots| {
                    slots.iter().map(|v| {
                        Ok(json!({
                            "real": v.re.exact_rational().ok_or("successful nonfinite real output")?.to_string(),
                            "imaginary": v.im.exact_rational().ok_or("successful nonfinite imaginary output")?.to_string(),
                            "real_precision": v.re.get_precision(),
                            "imaginary_precision": v.im.get_precision(),
                        }))
                    }).collect::<Result<Vec<_>, Box<dyn Error>>>()
                }).collect::<Result<Vec<_>, _>>()?;
                lanes.push(json!({"row": row, "status": "Success", "moments": moments}));
            }
            Err(status) => {
                lanes.push(json!({"row": row, "status": format!("{status:?}"), "moments": null}))
            }
        }
    }
    let stats = workspace.last_execution();
    record["execution"] = json!({
        "parent_input_rows":stats.parent_input_rows,"child_batch_calls":stats.child_batch_calls,
        "packed_child_rows":stats.packed_child_rows,"parent_batch_calls":stats.parent_batch_calls,
        "packed_parent_rows":stats.packed_parent_rows,
        "child_execution":execution_record(&stats.child_execution),"parent_execution":execution_record(&stats.parent_execution),
        "child_certificate_execution":format!("{:?}",stats.child_certificate_execution),
        "parent_certificate_execution":format!("{:?}",stats.parent_certificate_execution),
    });
    record["lanes"] = json!(lanes);
    record["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
    let success = result.statuses.iter().all(|s| s == &LaneStatus::Success);
    record["status"] = json!(if success {
        "all_lanes_success"
    } else {
        "lanes_rejected"
    });
    write(output, &record)?;
    eprintln!(
        "private main ThinSeven complete at {:.6}s: {:?}",
        total.elapsed().as_secs_f64(),
        result.statuses
    );
    Ok(if success { 0 } else { 2 })
}

#[test]
#[ignore = "expensive private N7 integral capture; requires explicit finite supervision and fixture/output/profile environment"]
fn capture_complete_moments_from_fixture() {
    let code = run_capture().expect("private ThinSeven capture failed");
    assert_eq!(code, 0, "private ThinSeven lane or preparation failure");
}
