//! TIRust side of the S-wave quarkonium comparison (benchmarks/onia).
//!
//! Like `stability_driver`, but every lane carries its own squared masses
//! (the charm mass of a lattice point is fixed by its momenta). For every
//! integral of the input, the complete tensor array of every lane from:
//! - `default`: the public binary64 facade (`TensorReductionPlan` default:
//!   direct engine, automatic precision rescue, symbolic fallback), with the
//!   per-lane diagnostics and the time per lane of a repeated call;
//! - `binary64`: the facade with the precision rescue disabled (refused lanes
//!   reported, with their best-effort arrays);
//! - `direct256`: the direct engine at 256 bits (conditioning accepted down
//!   to 2^-216), all 77 significant digits;
//! - `symbolic256` (only with `TIRUST_ONIA_SYMBOLIC256=1`): the facade at 256
//!   bits, which runs the symbolic engine, all digits.
//!
//! usage: onia_driver INPUT_JSON OUTPUT_JSONL
//!
//! INPUT_JSON: {"integrals": [{"id", "n", "rank", "mu", "lanes": [{"shifts":
//! [[q0, q1, q2, q3], ...], "masses_squared": [...]}]}]}, numbers as decimal
//! strings (parsed with correct rounding) or JSON numbers. An optional
//! "powers": [...] per integral raises propagator powers (dotted
//! propagators, for the partial-fraction references of affine_reference.py).
use std::io::Write;
use std::time::Instant;

use iregi_rust::{
    BatchResult, DirectReductionPlan, KinematicsBatch, PrecisionRescue, ReductionSettings, TensorReductionPlan,
    TensorWorkspace, TensorWorkspaceOptions,
};
use serde_json::{Value, json};
use symbolica::domains::float::{Complex, Float, RealLike};

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
    let mut entry = match result.lane(lane) {
        Ok(values) => json!({
            "status": "success",
            "moments": values
                .chunks_exact(4)
                .map(|poles| poles.iter().map(|z| [z.re.to_f64(), z.im.to_f64()]).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        }),
        Err(status) => json!({"status": format!("{status:?}"), "moments": Value::Null}),
    };
    entry["estimated_error"] = json!(result.estimated_error(lane));
    entry["fallback_used"] = json!(result.fallback_used[lane]);
    if let Some(best) = result.lane_with_estimate(lane).filter(|best| !best.certified) {
        entry["best_effort_moments"] = json!(best
            .values
            .chunks_exact(4)
            .map(|poles| poles.iter().map(|z| [z.re.to_f64(), z.im.to_f64()]).collect::<Vec<_>>())
            .collect::<Vec<_>>());
    }
    entry
}

fn number(value: &Value) -> f64 {
    match value.as_str() {
        Some(text) => text.parse().expect("decimal binary64 input"),
        None => value.as_f64().expect("numeric input"),
    }
}

fn lift(batch: &KinematicsBatch<f64>, bits: u32) -> KinematicsBatch<Float> {
    KinematicsBatch {
        rows: batch.rows,
        components: 1,
        values: batch
            .values
            .iter()
            .map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im)))
            .collect(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let mut sink = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let symbolic256 = std::env::var("TIRUST_ONIA_SYMBOLIC256").as_deref() == Ok("1");
    for integral in input["integrals"].as_array().unwrap() {
        let id = integral["id"].as_str().unwrap();
        let n = integral["n"].as_u64().unwrap() as usize;
        let rank = integral["rank"].as_u64().unwrap() as u16;
        let mu = number(&integral["mu"]);
        let lanes = integral["lanes"].as_array().unwrap();
        let mut values = Vec::with_capacity(lanes.len() * (5 * n + 1));
        for lane in lanes {
            for shift in lane["shifts"].as_array().unwrap() {
                values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(number(x), 0.0)));
            }
            values.extend(lane["masses_squared"].as_array().unwrap().iter().map(|m| Complex::new(number(m), 0.0)));
            values.push(Complex::new(mu, 0.0));
        }
        let batch = KinematicsBatch { rows: lanes.len(), components: 1, values };
        let mut settings = ReductionSettings::new(n, rank);
        if let Some(powers) = integral.get("powers").and_then(Value::as_array) {
            settings.powers = powers.iter().map(|p| p.as_u64().expect("integer power") as u16).collect();
        }

        let plan = TensorReductionPlan::new(settings.clone())?;
        let mut facade = TensorWorkspace::new_with_options(&plan, 0.0f64, TensorWorkspaceOptions::default())?;
        let default = facade.reduce_batch(&batch)?;
        let diagnostics = facade.last_diagnostics().to_vec();
        // A second call of the same workspace, timed (indicative only: one
        // repetition, not pinned).
        let started = Instant::now();
        let _ = facade.reduce_batch(&batch)?;
        let ns_per_lane = started.elapsed().as_nanos() as f64 / batch.rows as f64;

        let options = TensorWorkspaceOptions { precision_rescue: PrecisionRescue::Disabled, ..Default::default() };
        let mut only = TensorWorkspace::new_with_options(&plan, 0.0f64, options)?;
        let binary64 = only.reduce_batch(&batch)?;
        let binary64_diagnostics = only.last_diagnostics().to_vec();

        let bits = 256;
        let direct = DirectReductionPlan::new(settings)?;
        let mut reference = direct.workspace(Float::new(bits))?;
        reference.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))));
        let lifted = lift(&batch, bits);
        let direct256 = reference.reduce_batch(&lifted)?;
        let symbolic = if symbolic256 {
            let mut workspace = TensorWorkspace::new_with_options(&plan, Float::new(bits), Default::default())?;
            Some(workspace.reduce_batch(&lifted)?)
        } else {
            None
        };

        for lane in 0..batch.rows {
            let mut entry_default = moments(&default, lane);
            entry_default["diagnostics"] = json!(diagnostics.get(lane));
            let mut entry_binary64 = moments(&binary64, lane);
            entry_binary64["diagnostics"] = json!(binary64_diagnostics.get(lane));
            let mut record = json!({
                "integral": id,
                "index": lane,
                "default": entry_default,
                "default_ns_per_lane": ns_per_lane,
                "binary64": entry_binary64,
                "direct256": moments_text(&direct256, lane),
            });
            if let Some(symbolic) = &symbolic {
                record["symbolic256"] = moments_text(symbolic, lane);
            }
            writeln!(sink, "{}", serde_json::to_string(&record)?)?;
        }
        sink.flush()?;
        eprintln!("{id}: {} lanes", batch.rows);
    }
    Ok(())
}
