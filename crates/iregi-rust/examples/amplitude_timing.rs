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
//! `TIRUST_ROTATION_TEST=disabled` builds the workspaces with
//! `RotationTest::Disabled` (gate-accepted lanes unchecked).
//! `TIRUST_AMPLITUDE_ENGINE=embedded` evaluates the integrals in embedder
//! mode instead (`TensorWorkspace::new_embedded`, default options), sharing
//! a `TensorStore` cleared before every call (share 1) or none (share 0).
use std::{
    error::Error,
    hint::black_box,
    io::{BufRead, Write},
    time::Instant,
};

use iregi_rust::{
    AmplitudeOptions, AmplitudeWorkspace, BatchResult, EmbedderOptions, KinematicsBatch, LaneStatus, ReductionSettings,
    RotationTest, TensorReductionPlan, TensorStore, TensorWorkspace, TensorWorkspaceOptions,
};
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

/// The amplitude's workspaces: the facade's amplitude workspace, or
/// embedder-mode workspaces with a shared `TensorStore`.
enum Evaluator {
    Facade(AmplitudeWorkspace<f64>),
    Embedded { workspaces: Vec<TensorWorkspace<f64>>, store: TensorStore, share: bool },
}

impl Evaluator {
    fn set_sharing(&mut self, enabled: bool) {
        match self {
            Self::Facade(amplitude) => amplitude.set_subset_sharing(enabled),
            Self::Embedded { workspaces, store, share } => {
                *share = enabled;
                for workspace in workspaces.iter_mut() {
                    workspace.set_tensor_store(enabled.then_some(&*store));
                }
            }
        }
    }

    fn reduce(&mut self, batches: &[KinematicsBatch<f64>]) -> Fallible<Vec<BatchResult<f64>>> {
        match self {
            Self::Facade(amplitude) => Ok(amplitude.reduce_batches(batches)?),
            Self::Embedded { workspaces, store, .. } => {
                store.clear();
                Ok(workspaces.iter_mut().zip(batches).map(|(workspace, batch)| workspace.reduce_batch(batch)).collect::<Result<Vec<_>, _>>()?)
            }
        }
    }

    /// `[reused, computed, moments_reused, moments_computed]` of the last call.
    fn stats(&self) -> [u64; 4] {
        match self {
            Self::Facade(amplitude) => {
                let stats = amplitude.last_sharing();
                [stats.reused, stats.computed, stats.moments_reused, stats.moments_computed]
            }
            Self::Embedded { store, share, .. } => {
                if !*share {
                    return [0; 4];
                }
                let stats = store.statistics();
                [stats.subsets_reused + stats.masters_reused, stats.subsets_computed + stats.masters_computed,
                    stats.moments_reused, stats.moments_computed]
            }
        }
    }
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
    let rotation_test = match std::env::var("TIRUST_ROTATION_TEST").as_deref() {
        Ok("disabled") => RotationTest::Disabled,
        _ => RotationTest::Enabled,
    };
    let options = AmplitudeOptions {
        workspace: TensorWorkspaceOptions { rotation_test, ..Default::default() },
        ..Default::default()
    };
    let mut amplitude = if std::env::var("TIRUST_AMPLITUDE_ENGINE").as_deref() == Ok("embedded") {
        let workspaces = plans
            .iter()
            .map(|plan| TensorWorkspace::new_embedded(plan, 0.0f64, EmbedderOptions::default()))
            .collect::<Result<Vec<_>, _>>()?;
        Evaluator::Embedded { workspaces, store: TensorStore::new(), share: false }
    } else {
        Evaluator::Facade(AmplitudeWorkspace::new_with_options(&plans, 0.0f64, options)?)
    };
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
                amplitude.set_sharing(share);
                for _ in 0..warmup {
                    black_box(amplitude.reduce(black_box(&batches))?);
                }
                let started = Instant::now();
                for _ in 0..iterations {
                    black_box(amplitude.reduce(black_box(&batches))?);
                }
                let ns = started.elapsed().as_nanos() as f64 / iterations as f64;
                let [reused, computed, moments_reused, moments_computed] = amplitude.stats();
                writeln!(out, "RESULT {}", json!({"ns_per_call": ns, "share": share, "iterations": iterations,
                    "reused": reused, "computed": computed,
                    "moments_reused": moments_reused, "moments_computed": moments_computed}))?;
            }
            ["FINAL"] => {
                amplitude.set_sharing(false);
                let off = amplitude.reduce(&batches)?;
                amplitude.set_sharing(true);
                let on = amplitude.reduce(&batches)?;
                let [reused, computed, moments_reused, moments_computed] = amplitude.stats();
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
                    "reused": reused, "computed": computed,
                    "moments_reused": moments_reused, "moments_computed": moments_computed,
                }))?;
            }
            _ => return Err(format!("invalid request: {line}").into()),
        }
        out.flush()?;
    }
    Ok(())
}
