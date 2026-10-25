//! Explicit six-point example with exact input/output capture.
//!
//! This deliberately performs no numerical specialization of symbolic plans.
//! Its output ledger is written before every potentially expensive stage.

use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, KinematicsBatch, LaneStatus, ReductionSettings,
    SixPointReductionPlan as ReductionPlan, SixPointWorkspace as Workspace,
};
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

fn run() -> Result<u8, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let usage =
        "six_point input.json 6 rank pv working_bits requested_bits rows output.json";
    let input_path = args.next().ok_or(usage)?;
    let n: usize = args.next().ok_or(usage)?.parse()?;
    let rank: u16 = args.next().ok_or(usage)?.parse()?;
    let algorithm = match args.next().ok_or(usage)?.as_str() {
        "ibp" => Algorithm::Ibp,
        "pv" => Algorithm::PassarinoVeltman,
        "hybrid" => Algorithm::Hybrid,
        _ => return Err(usage.into()),
    };
    let working_bits: u32 = args.next().ok_or(usage)?.parse()?;
    let requested_bits: u32 = args.next().ok_or(usage)?.parse()?;
    let rows: usize = args.next().ok_or(usage)?.parse()?;
    let output = args.next().ok_or(usage)?;
    if args.next().is_some()
        || n != 6
        || rank > 7
        || ![128, 256, 384].contains(&working_bits)
        || requested_bits < 53
        || requested_bits > working_bits
        || !(1..=3).contains(&rows)
    {
        return Err(usage.into());
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
        "interface": "SixPointReductionPlan::new; SixPointWorkspace<Float>::new; SixPointWorkspace::reduce_batch",
        "settings": settings,
        "working_bits": working_bits,
        "requested_bits": requested_bits,
        "rows": rows,
        "components": 1,
        "fixture": input_path,
        "input_exact_rational": original_inputs,
        "limitations": "Explicit six-point coordinate-input API restricted to rank<=7 with dynamic rank-lowered children, actual-route primal envelopes and fresh full-Taylor zero-cofactor masks; no numerical specialization, no reference scalars, no disabled guards. Working precision is not a claim of achieved requested precision. Timings are diagnostic, not performance gates.",
    });
    let output = Path::new(&output);
    write(output, &record)?;
    let total = Instant::now();
    eprintln!("public high point N={n} rank={rank} {algorithm:?}: preparing structural plan");
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
    record["preparation_notes"] = json!(["Explicit six-point packed-child API; rank<=7/native/raw; no quadrature import support"]);
    record["basis"] = json!(plan.basis());
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
    record["diagnostic_workspace_seconds"] = json!(started.elapsed().as_secs_f64());
    record["status"] = json!("reducing");
    write(output, &record)?;
    eprintln!(
        "workspace ready at {:.6}s; reducing {rows} lanes",
        total.elapsed().as_secs_f64()
    );
    let started = Instant::now();
    let result = workspace.reduce_batch(&input)?;
    record["diagnostic_reduction_seconds"] = json!(started.elapsed().as_secs_f64());
    record["lane_statuses"] = json!(
        result
            .statuses
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
    );
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
        "child_execution":format!("{:?}",stats.child_execution),"parent_execution":format!("{:?}",stats.parent_execution),
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
        "public high point complete at {:.6}s: {:?}",
        total.elapsed().as_secs_f64(),
        result.statuses
    );
    Ok(if success { 0 } else { 2 })
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::from(1)
        }
    }
}
