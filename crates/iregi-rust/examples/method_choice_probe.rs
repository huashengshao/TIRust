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
    // ALL=1: every lane of the case as one batch (facade only).
    let all = std::env::var("ALL").is_ok();
    let lanes: Vec<&Value> = if all { case["lanes"].as_array().unwrap().iter().collect() } else { vec![&case["lanes"][0]] };
    let mut values = Vec::new();
    for lane in &lanes {
        for shift in lane["shifts"].as_array().unwrap() {
            values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(rational(x), 0.0)));
        }
        for m in lane["masses_squared"].as_array().unwrap() {
            values.push(Complex::new(rational(&m[0]), rational(&m[1])));
        }
        values.push(Complex::new(rational(&lanes[0]["mu"]), 0.0));
    }
    let batch = KinematicsBatch { rows: lanes.len(), components: 1, values };
    if all {
        let facade = TensorReductionPlan::new(ReductionSettings::new(n, rank))?;
        let mut workspace = TensorWorkspace::new(&facade, 0.0f64)?;
        if std::env::var("NORESCUE").is_ok() {
            workspace.set_precision_rescue(iregi_rust::PrecisionRescue::Disabled);
        }
        workspace.reduce_batch(&batch)?;
        let start = Instant::now();
        for _ in 0..calls {
            std::hint::black_box(workspace.reduce_batch(&batch)?);
        }
        println!("facade batch {}: warm {:?}/call", lanes.len(), start.elapsed() / calls as u32);
        if let iregi_rust::TensorEngineExecution::Direct(stats) = workspace.engine_execution() {
            println!("  stats: rotation failures {}, expansion rows {} kept {}, checked {} kept {}, fallback rows {}",
                stats.rotation_failures, stats.expansion_rows, stats.expansion_successes, stats.checked_rows, stats.checked_successes, stats.fallback_rows);
        }
        let mut reasons = std::collections::BTreeMap::new();
        for notes in workspace.last_diagnostics() {
            for note in notes {
                let key: String = note.chars().filter(|c| !c.is_ascii_digit()).take(140).collect();
                *reasons.entry(key).or_insert(0) += 1;
            }
        }
        for (reason, count) in reasons {
            println!("  {count:3} x {reason}");
        }
        if std::env::var("CHECK").is_ok() {
            use symbolica::domains::float::{Float, RealLike};
            let result = workspace.reduce_batch(&batch)?;
            let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank))?;
            let mut wide = plan.workspace(Float::new(256))?;
            wide.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
            let lifted = KinematicsBatch { rows: batch.rows, components: 1,
                values: batch.values.iter().map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im))).collect() };
            let reference = wide.reduce_batch(&lifted)?;
            let mu = batch.values[5 * n].re;
            for (lane, notes) in workspace.last_diagnostics().iter().enumerate() {
                let rescued = notes.iter().any(|note| note.contains("bits: accepted"));
                if !notes.iter().any(|note| note.contains("cyclic orders")) && !rescued && std::env::var("CHECK").as_deref() != Ok("all") { continue; }
                let (Ok(x), Ok(y)) = (result.lane(lane), reference.lane(lane)) else { continue };
                let (mut e, mut s) = (0.0f64, 0.0f64);
                for (m, monomial) in result.basis.iter().enumerate() {
                    let w = mu.powi(i32::from(monomial.0.iter().sum::<u16>()));
                    for slot in 0..4 {
                        let (u, v) = (&x[4 * m + slot], &y[4 * m + slot]);
                        e = e.max((u.re - v.re.to_f64()).hypot(u.im - v.im.to_f64()) / w);
                        s = s.max(v.re.to_f64().hypot(v.im.to_f64()) / w);
                    }
                }
                println!("  lane {lane} ({}): E/A {:.2e}", if rescued { "rescued" } else { "binary64" }, e / s);
            }
        }
        if let Some(notes) = workspace.last_diagnostics().iter().find(|n| n.len() > 1) {
            for note in notes {
                println!("  full: {}", note.chars().take(700).collect::<String>());
            }
        }
        return Ok(());
    }
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
    // One workspace alternating between the lane and its rotated copy, and
    // two workspaces (one per order).
    let width = batch.values.len();
    let mut values = Vec::with_capacity(width);
    for i in 0..n {
        let j = (i + 1) % n;
        values.extend_from_slice(&batch.values[4 * j..4 * j + 4]);
    }
    for i in 0..n {
        values.push(batch.values[4 * n + (i + 1) % n]);
    }
    values.push(batch.values[5 * n]);
    let turned = KinematicsBatch { rows: 1, components: 1, values };
    let mut one = plan.workspace(0.0f64)?;
    one.set_stability_threshold(1e-15);
    let mut other = plan.workspace(0.0f64)?;
    other.set_stability_threshold(1e-15);
    one.reduce_batch(&batch)?;
    other.reduce_batch(&turned)?;
    let (mut a, mut b) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    for _ in 0..calls {
        let start = Instant::now();
        std::hint::black_box(one.reduce_batch(&batch)?);
        std::hint::black_box(one.reduce_batch(&turned)?);
        a += start.elapsed();
        let start = Instant::now();
        std::hint::black_box(one.reduce_batch(&batch)?);
        std::hint::black_box(other.reduce_batch(&turned)?);
        b += start.elapsed();
    }
    println!("both orders: one workspace {:?}/call, two workspaces {:?}/call", a / calls as u32, b / calls as u32);
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
