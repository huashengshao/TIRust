//! Amplitude timing driver: the integrals of an amplitude evaluated together
//! by `AmplitudeWorkspace`, with per-subset sharing on or off.
//!
//! usage: amplitude_timing SPEC_JSON
//! SPEC_JSON: {"integrals": [{"members": [i...], "rank": r}...],
//!             "lanes": [{"shifts": [[q0..q3] per propagator of the N-gon],
//!                        "masses": [m^2 per propagator], "mu": mu}...]}
//! Every integral takes its members' shifts and masses from each lane (the
//! amplitude's common loop-momentum routing).
//! stdin:  BENCH share(0|1) warmup iterations  -> RESULT {ns per call, ...}
//!         FINAL  -> FINAL {bitwise comparison of sharing off and on, ...}
//!         QUIT
//! One call evaluates every integral on every lane (`reduce_batches`).
use std::{
    error::Error,
    hint::black_box,
    io::{BufRead, Write},
    time::Instant,
};

use iregi_rust::{AmplitudeWorkspace, BatchResult, KinematicsBatch, LaneStatus, ReductionSettings, TensorReductionPlan};
use serde_json::{Value, json};
use symbolica::domains::float::Complex;

type Fallible<T> = Result<T, Box<dyn Error>>;

/// FNV-1a over every lane's array (best-effort ones included), status and
/// estimate, bit for bit.
fn fingerprint(results: &[BatchResult<f64>]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |x: u64| {
        for byte in x.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for result in results {
        for lane in 0..result.rows() {
            match result.lane_with_estimate(lane) {
                Some(best) => best.values.iter().for_each(|z| {
                    eat(z.re.to_bits());
                    eat(z.im.to_bits());
                }),
                None => eat(u64::MAX),
            }
            for byte in format!("{:?}", result.statuses[lane]).bytes() {
                eat(u64::from(byte));
            }
            eat(result.estimated_errors.get(lane).copied().flatten().map_or(u64::MAX, f64::to_bits));
        }
    }
    hash
}

fn main() -> Fallible<()> {
    let args: Vec<String> = std::env::args().collect();
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let lanes = spec["lanes"].as_array().ok_or("lanes")?;
    let mut plans = Vec::new();
    let mut batches = Vec::new();
    let mut members_of = Vec::new();
    for integral in spec["integrals"].as_array().ok_or("integrals")? {
        let members: Vec<usize> = integral["members"].as_array().ok_or("members")?.iter()
            .map(|m| m.as_u64().unwrap() as usize).collect();
        let rank = integral["rank"].as_u64().ok_or("rank")? as u16;
        plans.push(TensorReductionPlan::new(ReductionSettings::new(members.len(), rank))?);
        let mut values = Vec::new();
        for lane in lanes {
            for &i in &members {
                values.extend(lane["shifts"][i].as_array().unwrap().iter().map(|x| Complex::new(x.as_f64().unwrap(), 0.0)));
            }
            values.extend(members.iter().map(|&i| Complex::new(lane["masses"][i].as_f64().unwrap(), 0.0)));
            values.push(Complex::new(lane["mu"].as_f64().unwrap(), 0.0));
        }
        batches.push(KinematicsBatch { rows: lanes.len(), components: 1, values });
        members_of.push(members);
    }
    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64)?;
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "READY TIRUST_AMPLITUDE_V1")?;
    out.flush()?;
    for line in stdin.lock().lines() {
        let line = line?;
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            [] => continue,
            ["QUIT"] => break,
            ["BENCH", share, warmup, iterations] => {
                let (share, warmup, iterations) = (*share == "1", warmup.parse::<usize>()?, iterations.parse::<usize>()?);
                amplitude.set_subset_sharing(share);
                for _ in 0..warmup {
                    black_box(amplitude.reduce_batches(black_box(&batches))?);
                }
                let started = Instant::now();
                for _ in 0..iterations {
                    black_box(amplitude.reduce_batches(black_box(&batches))?);
                }
                let ns = started.elapsed().as_nanos() as f64 / iterations as f64;
                let stats = amplitude.last_sharing();
                writeln!(out, "RESULT {}", json!({"ns_per_call": ns, "share": share, "iterations": iterations,
                    "reused": stats.reused, "computed": stats.computed}))?;
            }
            ["FINAL"] => {
                amplitude.set_subset_sharing(false);
                let off = amplitude.reduce_batches(&batches)?;
                amplitude.set_subset_sharing(true);
                let on = amplitude.reduce_batches(&batches)?;
                let stats = amplitude.last_sharing();
                let differing = off.iter().zip(&on)
                    .filter(|(a, b)| fingerprint(std::slice::from_ref(*a)) != fingerprint(std::slice::from_ref(*b)))
                    .count();
                let success = on.iter().map(|r| r.statuses.iter().filter(|s| **s == LaneStatus::Success).count()).sum::<usize>();
                let best_effort = on.iter().map(|r| (0..r.rows()).filter(|&l| r.is_best_effort(l)).count()).sum::<usize>();
                writeln!(out, "FINAL {}", json!({
                    "integrals": members_of.len(), "lanes": lanes.len(),
                    "differing_integrals": differing,
                    "fingerprint_off": format!("{:016x}", fingerprint(&off)),
                    "fingerprint_on": format!("{:016x}", fingerprint(&on)),
                    "successful_integral_lanes": success, "best_effort_integral_lanes": best_effort,
                    "reused": stats.reused, "computed": stats.computed,
                }))?;
            }
            _ => return Err(format!("invalid request: {line}").into()),
        }
        out.flush()?;
    }
    Ok(())
}
