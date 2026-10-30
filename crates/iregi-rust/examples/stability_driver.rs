//! TIRust side of the binary64 stability study
//! (validation/benchmark/stability-20261030).
//!
//! For every point of every integral in the points file, three evaluations of
//! the complete tensor array:
//! - `default`: the public binary64 facade (`TensorReductionPlan` default,
//!   with the direct engine's automatic precision rescue and the symbolic
//!   fallback), with per-lane diagnostics;
//! - `direct64`: the direct engine in binary64 alone (no rescue): refused
//!   lanes are reported, not repaired;
//! - `direct256`: the direct engine at 256 bits on the same binary64 inputs
//!   (accepting conditioning down to 2^-216), a second high-precision
//!   reference besides quadruple-precision original IREGI.
//!
//! usage: stability_driver POINTS_JSON OUTPUT_JSONL [INTEGRAL_ID]
use std::io::Write;

use iregi_rust::{
    BatchResult, DirectReductionPlan, KinematicsBatch, ReductionSettings, TensorReductionPlan,
    TensorWorkspace,
};
use serde_json::{Value, json};
use symbolica::domains::float::{Complex, Float, RealLike};

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
        let mu = integral["mu"].as_f64().unwrap();
        let masses: Vec<f64> = integral["masses_squared"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_f64().unwrap())
            .collect();
        let lanes = integral["points"].as_array().unwrap();
        let mut values = Vec::with_capacity(lanes.len() * (5 * n + 1));
        for point in lanes {
            for shift in point.as_array().unwrap() {
                values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(x.as_f64().unwrap(), 0.0)));
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
            let record = json!({
                "integral": id,
                "index": lane,
                "default": entry_default,
                "direct64": moments(&direct64, lane),
                "direct256": moments(&direct256, lane),
            });
            writeln!(sink, "{}", serde_json::to_string(&record)?)?;
        }
        sink.flush()?;
        eprintln!("{id}: {} points", batch.rows);
    }
    Ok(())
}
