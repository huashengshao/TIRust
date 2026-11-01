//! Cost of the a priori method choice on one benchmark lane.
//!
//! usage: method_choice_probe INPUTS_JSON CASE_ID [CALLS]
//! Times the binary64 direct workspace with and without the estimated
//! method choice and the public facade, and reports whether lane 0 used
//! the small-Gram expansion or alternative PV.
use std::time::Instant;

use iregi_rust::{DirectReductionPlan, KinematicsBatch, ReductionSettings, TensorReductionPlan, TensorWorkspace};
use serde_json::Value;
use symbolica::domains::float::Complex;

fn rational(token: &Value) -> f64 {
    let text = token.as_str().unwrap();
    match text.split_once('/') {
        Some((a, b)) => a.parse::<f64>().unwrap() / b.parse::<f64>().unwrap(),
        None => text.parse().unwrap(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let inputs: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let calls: usize = args.get(3).map_or(20, |s| s.parse().unwrap());
    let case = inputs["cases"].as_array().unwrap().iter().find(|c| c["id"].as_str() == Some(&args[2])).unwrap();
    let (n, rank) = (case["n"].as_u64().unwrap() as usize, case["rank"].as_u64().unwrap() as u16);
    let lane = &case["lanes"][0];
    let mut values = Vec::new();
    for shift in lane["shifts"].as_array().unwrap() {
        values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(rational(x), 0.0)));
    }
    for m in lane["masses_squared"].as_array().unwrap() {
        values.push(Complex::new(rational(&m[0]), rational(&m[1])));
    }
    values.push(Complex::new(rational(&lane["mu"]), 0.0));
    let batch = KinematicsBatch { rows: 1, components: 1, values };
    let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank))?;
    for estimated in [false, true] {
        let mut workspace = plan.workspace(0.0f64)?;
        if estimated {
            workspace.enable_estimated_method_choice(12, 1e-15);
        }
        let first = Instant::now();
        let result = workspace.reduce_batch(&batch)?;
        let first = first.elapsed();
        let start = Instant::now();
        for _ in 0..calls {
            std::hint::black_box(workspace.reduce_batch(&batch)?);
        }
        println!("direct estimated={estimated}: first {:?}, warm {:?}/call, status {:?}, gram lanes {:?}",
            first, start.elapsed() / calls as u32, result.statuses[0], workspace.last_gram_lanes());
    }
    // Cost of the estimate flags alone: plain and flagged, interleaved.
    let mut plain = plan.workspace(0.0f64)?;
    plain.set_stability_threshold(1e-15);
    let mut flagged = plan.workspace(0.0f64)?;
    flagged.set_stability_threshold(1e-15);
    flagged.enable_estimate_flags();
    plain.reduce_batch(&batch)?;
    flagged.reduce_batch(&batch)?;
    let (mut a, mut b) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    for _ in 0..calls {
        let start = Instant::now();
        std::hint::black_box(plain.reduce_batch(&batch)?);
        a += start.elapsed();
        let start = Instant::now();
        std::hint::black_box(flagged.reduce_batch(&batch)?);
        b += start.elapsed();
    }
    println!("direct plain {:?}/call, with estimate flags {:?}/call, flagged {:?}",
        a / calls as u32, b / calls as u32, flagged.last_estimate_lanes());
    let facade = TensorReductionPlan::new(ReductionSettings::new(n, rank))?;
    let mut workspace = TensorWorkspace::new(&facade, 0.0f64)?;
    let result = workspace.reduce_batch(&batch)?;
    if let iregi_rust::TensorEngineExecution::Direct(stats) = workspace.engine_execution() {
        println!("facade: estimated E/A {:?}, rotation failures {}, expansion rows {} (kept {})",
            result.estimated_error(0), stats.rotation_failures, stats.expansion_rows, stats.expansion_successes);
    }
    let start = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(workspace.reduce_batch(&batch)?);
    }
    println!("facade: warm {:?}/call", start.elapsed() / calls as u32);
    Ok(())
}
