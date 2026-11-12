//! TIRust side of the quarkonium integral families (benchmarks/onia,
//! amplitude.py): the integrals of a family evaluated together by
//! `AmplitudeWorkspace`, like `amplitude_timing`, but with per-lane masses,
//! decimal-string inputs (parsed with correct rounding) and every integral's
//! array in the output.
//!
//! usage: onia_amplitude SPEC_JSON OUTPUT_JSON [ITERATIONS]
//! SPEC_JSON: {"integrals": [{"id", "members": [i...], "rank": r}...],
//!             "lanes": [{"shifts": [[q0..q3] per parent propagator],
//!                        "masses_squared": [m^2 per parent propagator], "mu"}...]}
//! OUTPUT_JSON: {"ns_per_call": {"share": .., "separate": ..}, "iterations",
//!               "lanes", "bitwise_equal", "integrals": [{"id", "lanes":
//!               [{"status", "moments", "estimated_error"}]}]}
//! One call evaluates every integral on every lane (`reduce_batches`); the
//! arrays are those of the call with sharing on. Timing: one warm-up call,
//! then ITERATIONS (default 3) calls with sharing on and with it off.
use std::{error::Error, hint::black_box, time::Instant};

use iregi_rust::{AmplitudeWorkspace, BatchResult, KinematicsBatch, ReductionSettings, TensorReductionPlan};
use serde_json::{Value, json};
use symbolica::domains::float::Complex;

type Fallible<T> = Result<T, Box<dyn Error>>;

fn number(value: &Value) -> f64 {
    match value.as_str() {
        Some(text) => text.parse().expect("decimal binary64 input"),
        None => value.as_f64().expect("numeric input"),
    }
}

fn lane_entry(result: &BatchResult<f64>, lane: usize) -> Value {
    let mut entry = match result.lane(lane) {
        Ok(values) => json!({
            "status": "success",
            "moments": values
                .chunks_exact(4)
                .map(|poles| poles.iter().map(|z| [z.re, z.im]).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
        }),
        Err(status) => json!({"status": format!("{status:?}"), "moments": Value::Null}),
    };
    entry["estimated_error"] = json!(result.estimated_error(lane));
    entry
}

fn bits(results: &[BatchResult<f64>]) -> Vec<Option<Vec<u64>>> {
    let mut out = Vec::new();
    for result in results {
        for lane in 0..result.rows() {
            out.push(result.lane(lane).ok().map(|values| {
                values.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect()
            }));
        }
    }
    out
}

fn main() -> Fallible<()> {
    let args: Vec<String> = std::env::args().collect();
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let iterations: usize = args.get(3).map_or(Ok(3), |s| s.parse())?;
    let lanes = spec["lanes"].as_array().ok_or("lanes")?;
    let mut plans = Vec::new();
    let mut batches = Vec::new();
    let mut ids = Vec::new();
    for integral in spec["integrals"].as_array().ok_or("integrals")? {
        let members: Vec<usize> =
            integral["members"].as_array().ok_or("members")?.iter().map(|m| m.as_u64().unwrap() as usize).collect();
        let rank = integral["rank"].as_u64().ok_or("rank")? as u16;
        plans.push(TensorReductionPlan::new(ReductionSettings::new(members.len(), rank))?);
        let mut values = Vec::new();
        for lane in lanes {
            for &i in &members {
                values.extend(lane["shifts"][i].as_array().unwrap().iter().map(|x| Complex::new(number(x), 0.0)));
            }
            values.extend(members.iter().map(|&i| Complex::new(number(&lane["masses_squared"][i]), 0.0)));
            values.push(Complex::new(number(&lane["mu"]), 0.0));
        }
        batches.push(KinematicsBatch { rows: lanes.len(), components: 1, values });
        ids.push(integral["id"].clone());
    }
    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64)?;
    let mut ns = serde_json::Map::new();
    for (name, share) in [("separate", false), ("share", true)] {
        amplitude.set_subset_sharing(share);
        black_box(amplitude.reduce_batches(&batches)?);
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(amplitude.reduce_batches(&batches)?);
        }
        ns.insert(name.into(), json!(started.elapsed().as_nanos() as f64 / iterations as f64));
    }
    amplitude.set_subset_sharing(false);
    let separate = amplitude.reduce_batches(&batches)?;
    amplitude.set_subset_sharing(true);
    let shared = amplitude.reduce_batches(&batches)?;
    let output = json!({
        "ns_per_call": ns,
        "iterations": iterations,
        "lanes": lanes.len(),
        "bitwise_equal": bits(&separate) == bits(&shared),
        "integrals": ids.iter().zip(&shared).map(|(id, result)| json!({
            "id": id,
            "lanes": (0..result.rows()).map(|lane| lane_entry(result, lane)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    std::fs::write(&args[2], serde_json::to_string(&output)?)?;
    Ok(())
}
