//! Lossless single-case capture for the checked supplied-G tensor interface.
//! Input uses the existing numerical CLI JSON format; no model/graph code enters
//! the reducer. Symbolica's exact Matrix API prepares G = p eta p^T.

use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, CertifiedTensorReductionPlan, CertifiedTensorWorkspace, KinematicsBatch, LaneStatus,
    ReductionSettings,
};
use serde_json::{Value, json};
use std::{error::Error, path::Path, time::Instant};
use symbolica::{
    domains::{
        float::{Complex, Float, FloatLike, SingleFloat},
        integer::Integer,
        rational::{Q, Rational},
    },
    tensors::matrix::Matrix,
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
fn write(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}
fn run() -> Result<u8, Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "checked_tensor_public case.json algorithm working_bits requested_bits output.json";
    let input_path = args.next().ok_or(usage)?;
    let algorithm = match args.next().ok_or(usage)?.as_str() {
        "pv" => Algorithm::PassarinoVeltman,
        "ibp" => Algorithm::Ibp,
        "hybrid" => Algorithm::Hybrid,
        _ => return Err(usage.into()),
    };
    let bits: u32 = args.next().ok_or(usage)?.parse()?;
    let requested: u32 = args.next().ok_or(usage)?.parse()?;
    let output_path = args.next().ok_or(usage)?;
    if args.next().is_some() || ![128, 256].contains(&bits) || requested < 53 || requested > bits {
        return Err(usage.into());
    }
    let case: Value = serde_json::from_str(&std::fs::read_to_string(&input_path)?)?;
    let propagators = case["propagators"]
        .as_array()
        .ok_or("missing propagators")?;
    if propagators.len() != 6 {
        return Err("checked capture requires exactly six propagators".into());
    }
    let rank: u16 = case["rank"].as_u64().ok_or("missing rank")?.try_into()?;
    if rank > 6 || case["snap"].as_bool().unwrap_or(false) {
        return Err("rank<=6 and preserved inputs required".into());
    }
    let mut settings = ReductionSettings::new(6, rank);
    settings.algorithm = algorithm;
    // Existing numerical CLI cases use IREGI idim = 2h. The semantic Rust
    // library stores h, with D = 4 + 2h - 2 epsilon.
    let idim = case["dimension_shift"].as_i64().unwrap_or(0);
    if idim % 2 != 0 { return Err("capture requires an even IREGI idim = 2h".into()); }
    settings.dimension_shift = (idim / 2).try_into()?;
    settings.powers = propagators
        .iter()
        .map(|p| {
            p["power"]
                .as_u64()
                .ok_or("missing positive power")
                .and_then(|v| u16::try_from(v).map_err(|_| "power overflow"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if settings.powers.contains(&0) {
        return Err("positive powers required".into());
    }
    let points = propagators
        .iter()
        .map(|p| {
            let v = p["shift"].as_array().ok_or("missing four-vector")?;
            if v.len() != 4 {
                return Err("four Cartesian components required".into());
            }
            v.iter()
                .map(number)
                .collect::<Result<Vec<_>, Box<dyn Error>>>()
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let p = Matrix::from_nested_vec(points.clone(), Q)?;
    let eta = Matrix::from_nested_vec(
        (0..4)
            .map(|i| {
                (0..4)
                    .map(|j| {
                        Rational::from(if i == j {
                            if i == 0 { 1 } else { -1 }
                        } else {
                            0
                        })
                    })
                    .collect()
            })
            .collect(),
        Q,
    )?;
    let weighted = &p * &eta;
    let gram = &weighted * &p.transpose();
    let mut exact: Vec<Complex<Rational>> = points
        .into_iter()
        .flatten()
        .map(|v| Complex::new(v, 0.into()))
        .collect();
    for i in 0..6 {
        for j in i..6 {
            exact.push(Complex::new(gram[(i, j)].clone(), 0.into()));
        }
    }
    for p in propagators {
        let m = p["mass_squared"]
            .as_array()
            .ok_or("missing complex squared mass")?;
        if m.len() != 2 {
            return Err("two squared-mass components required".into());
        }
        exact.push(Complex::new(number(&m[0])?, number(&m[1])?));
    }
    exact.push(Complex::new(number(&case["mu"])?, 0.into()));
    let prototype = Float::new(bits);
    let mut values = Vec::with_capacity(exact.len());
    for v in &exact {
        let mapped = Complex::new(
            prototype.from_rational(&v.re),
            prototype.from_rational(&v.im),
        );
        if mapped.re.exact_rational().as_ref() != Some(&v.re)
            || mapped.im.exact_rational().as_ref() != Some(&v.im)
        {
            return Err(
                "all represented inputs must match the exact case at working precision".into(),
            );
        }
        values.push(mapped);
    }
    let input = KinematicsBatch {
        rows: 1,
        components: 1,
        values,
    };
    let preserved = input.values.clone();
    let output = Path::new(&output_path);
    let mut record = json!({"schema_version":1,"status":"preparing_plan","case_id":case["id"],
        "input_file":input_path,"settings":settings,"working_bits":bits,"requested_bits":requested,
        "rows":1,"components":1,"interface":"CertifiedTensorReductionPlan/CertifiedTensorWorkspace",
        "gram_preparation":"Symbolica exact Rational Matrix p*eta*p.transpose()",
        "input_exact_rational":exact.iter().map(|v|[v.re.to_string(),v.im.to_string()]).collect::<Vec<_>>(),
        "limitations":"One fixed numerical case; working bits are not achieved accuracy. No disabled guards, numerical specialization, reference scalar substitution or performance claim."});
    write(output, &record)?;
    let total = Instant::now();
    let start = Instant::now();
    let plan = match CertifiedTensorReductionPlan::new(settings) {
        Ok(p) => p,
        Err(e) => {
            record["status"] = json!("preparation_failed");
            record["error"] = json!(e.to_string());
            write(output, &record)?;
            return Ok(2);
        }
    };
    record["diagnostic_plan_seconds"] = json!(start.elapsed().as_secs_f64());
    record["settings"] = json!(plan.settings());
    record["basis"] = json!(plan.basis());
    record["status"] = json!("preparing_workspace");
    write(output, &record)?;
    let start = Instant::now();
    let mut workspace = match CertifiedTensorWorkspace::new(&plan, prototype) {
        Ok(w) => w,
        Err(e) => {
            record["status"] = json!("workspace_failed");
            record["error"] = json!(e.to_string());
            write(output, &record)?;
            return Ok(2);
        }
    };
    workspace.set_accuracy_precision(requested)?;
    record["diagnostic_workspace_seconds"] = json!(start.elapsed().as_secs_f64());
    record["status"] = json!("reducing");
    write(output, &record)?;
    let result = workspace.reduce_batch(&input)?;
    record["input_preserved"] = json!(input.values == preserved);
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
    let cert = workspace.last_certificate_execution();
    let stats = workspace.last_execution();
    record["certificates"] = json!({"cartesian":{"input_rows":cert.cartesian.input_rows,"evaluated_rows":cert.cartesian.evaluated_rows,"passes":cert.cartesian.passes},
        "gram":{"input_rows":cert.gram.input_rows,"evaluated_rows":cert.gram.evaluated_rows,"passes":cert.gram.passes,"reduction_batches":cert.gram.reduction_batches}});
    record["execution"] = json!({"input_rows":stats.input_rows,"argument_rows":stats.argument_rows,"master_family_batches":stats.master_family_batches,"master_rows":stats.master_rows,"pentagon_box_rows":stats.pentagon_box_rows,"reduction_rows":stats.reduction_rows,"fallback_rows":stats.fallback_rows,"fallback_successes":stats.fallback_successes,"analytic_rows":stats.analytic_rows});
    let moments = match result.lane(0) {
        Ok(values) => {
            if values.len() != plan.basis().len() * 4 {
                return Err("incomplete successful tensor array".into());
            }
            Some(values.chunks_exact(4).map(|slots|slots.iter().map(|v|Ok(json!({
                "real":v.re.exact_rational().ok_or("nonfinite successful real")?.to_string(),
                "imaginary":v.im.exact_rational().ok_or("nonfinite successful imaginary")?.to_string(),
                "real_precision":v.re.get_precision(),"imaginary_precision":v.im.get_precision()})))
                .collect::<Result<Vec<_>,Box<dyn Error>>>()).collect::<Result<Vec<_>,_>>()?)
        }
        Err(_) => None,
    };
    record["lanes"] =
        json!([{"row":0,"status":format!("{:?}",result.statuses[0]),"moments":moments}]);
    record["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
    let success = result.statuses == [LaneStatus::Success] && input.values == preserved;
    record["status"] = json!(if success {
        "all_lanes_success"
    } else {
        "lanes_rejected"
    });
    write(output, &record)?;
    Ok(if success { 0 } else { 2 })
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
