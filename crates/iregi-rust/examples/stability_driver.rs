//! TIRust side of the binary64 stability study
//! (validation/benchmark/stability-20261030).
//!
//! For every point of every integral in the points file, three evaluations of
//! the complete tensor array:
//! - `default`: the public binary64 facade (`TensorReductionPlan` default,
//!   with the direct engine's automatic precision rescue and the symbolic
//!   fallback), with per-lane diagnostics;
//! - `binary64` and `binary64_r20`: the public facade with the precision
//!   rescue disabled (`PrecisionRescue::Disabled`: every lane in binary64,
//!   refusals reported), at the default accuracy request 2^-34 and at 2^-20;
//! - `direct64`: the direct engine in binary64 alone (no rescue): refused
//!   lanes are reported, not repaired;
//! - `direct256`: the direct engine at 256 bits on the same binary64 inputs
//!   (accepting conditioning down to 2^-216), a second high-precision
//!   reference besides quadruple-precision original IREGI, written with all
//!   77 significant digits.
//!
//! usage: stability_driver POINTS_JSON OUTPUT_JSONL [INTEGRAL_ID]
use std::io::Write;

use iregi_rust::{
    BatchResult, DirectReductionPlan, KinematicsBatch, PrecisionRescue, ReductionSettings,
    TensorReductionPlan, TensorWorkspace, TensorWorkspaceOptions,
};
use serde_json::{Value, json};
use symbolica::domains::float::{Complex, Float, RealLike};

/// The lane's complete array as decimal strings with every significant digit
/// of the working precision (for reference comparisons beyond binary64).
fn moments_text(result: &BatchResult<Float>, lane: usize) -> Value {
    match result.lane(lane) {
        Ok(values) => json!({
            "status": "success",
            "moments": (0..result.basis.len())
                .map(|m| (0..4)
                    .map(|slot| {
                        let z = &values[4 * m + slot];
                        [z.re.to_string(), z.im.to_string()]
                    })
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        }),
        Err(status) => json!({"status": format!("{status:?}"), "moments": Value::Null}),
    }
}

fn moments<T: RealLike>(result: &BatchResult<T>, lane: usize) -> Value {
    match result.lane(lane) {
        Ok(values) => {
            let count = result.basis.len();
            json!({
                "status": "success",
                "moments": (0..count)
                    .map(|m| (0..4)
                        .map(|slot| {
                            let z = &values[4 * m + slot];
                            [z.re.to_f64(), z.im.to_f64()]
                        })
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            })
        }
        Err(status) => json!({"status": format!("{status:?}"), "moments": Value::Null}),
    }
}

/// A binary64 input: a decimal string, parsed with correct rounding (the
/// default serde_json number parser is not always correctly rounded), or a
/// JSON number.
fn number(value: &Value) -> f64 {
    match value.as_str() {
        Some(text) => text.parse().expect("decimal binary64 input"),
        None => value.as_f64().expect("numeric input"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let points: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let mut sink = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let only = args.get(3);
    for integral in points["integrals"].as_array().unwrap() {
        let id = integral["id"].as_str().unwrap();
        if only.is_some_and(|only| only != id) {
            continue;
        }
        let n = integral["n"].as_u64().unwrap() as usize;
        let rank = integral["rank"].as_u64().unwrap() as u16;
        let mu = number(&integral["mu"]);
        let masses: Vec<f64> = integral["masses_squared"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let lanes = integral["points"].as_array().unwrap();
        let mut values = Vec::with_capacity(lanes.len() * (5 * n + 1));
        for point in lanes {
            for shift in point.as_array().unwrap() {
                values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(number(x), 0.0)));
            }
            values.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
            values.push(Complex::new(mu, 0.0));
        }
        let batch = KinematicsBatch { rows: lanes.len(), components: 1, values };
        let settings = ReductionSettings::new(n, rank);

        let plan = TensorReductionPlan::new(settings.clone())?;
        let mut facade = TensorWorkspace::new(&plan, 0.0f64)?;
        let default = facade.reduce_batch(&batch)?;
        let diagnostics = facade.last_diagnostics().to_vec();

        let options = TensorWorkspaceOptions { precision_rescue: PrecisionRescue::Disabled, ..Default::default() };
        let mut only = TensorWorkspace::new_with_options(&plan, 0.0f64, options)?;
        let binary64 = only.reduce_batch(&batch)?;
        let binary64_diagnostics = only.last_diagnostics().to_vec();
        // Requested accuracy 2^(19-39) = 2^-20.
        only.set_accuracy_precision(39)?;
        let binary64_r20 = only.reduce_batch(&batch)?;

        let direct = DirectReductionPlan::new(settings)?;
        let direct64 = direct.workspace(0.0f64)?.reduce_batch(&batch)?;

        let bits = 256;
        let mut reference = direct.workspace(Float::new(bits))?;
        reference.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))));
        let lifted = KinematicsBatch {
            rows: batch.rows,
            components: 1,
            values: batch
                .values
                .iter()
                .map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im)))
                .collect(),
        };
        let direct256 = reference.reduce_batch(&lifted)?;

        for lane in 0..batch.rows {
            let mut entry_default = moments(&default, lane);
            entry_default["fallback_used"] = json!(default.fallback_used[lane]);
            entry_default["diagnostics"] = json!(diagnostics.get(lane));
            let mut entry_binary64 = moments(&binary64, lane);
            entry_binary64["fallback_used"] = json!(binary64.fallback_used[lane]);
            entry_binary64["diagnostics"] = json!(binary64_diagnostics.get(lane));
            let mut entry_binary64_r20 = moments(&binary64_r20, lane);
            entry_binary64_r20["fallback_used"] = json!(binary64_r20.fallback_used[lane]);
            let record = json!({
                "integral": id,
                "index": lane,
                "default": entry_default,
                "binary64": entry_binary64,
                "binary64_r20": entry_binary64_r20,
                "direct64": moments(&direct64, lane),
                "direct256": moments_text(&direct256, lane),
            });
            writeln!(sink, "{}", serde_json::to_string(&record)?)?;
        }
        sink.flush()?;
        eprintln!("{id}: {} points", batch.rows);
    }
    Ok(())
}
