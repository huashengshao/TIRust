//! How much one-loop integrals of an amplitude share (development aid).
//!
//! A colour-ordered one-loop primitive amplitude with an N-point loop has, as
//! its loop integrals, the N-gon and every integral obtained by pinching
//! propagators: every subset of at least two of its propagators, in the same
//! order and with the same shifts, at rank <= its number of propagators
//! (renormalizable vertices, Feynman gauge). This tool evaluates all of them
//! separately, as TIRust does today, on a batch of stability-study points:
//!
//! - with the public binary64 facade (default: rotation test, precision
//!   rescue), per number of propagators: wall time per call of the batch;
//! - with the bare binary64 direct engine (one pass, no checks, gate open so
//!   that every lane does the full work), the phase profile (geometry, scalar
//!   masters, recurrence, gather, output) and the number of scalar-master
//!   evaluations;
//! - the facade's execution statistics: lanes refused by the rotation test,
//!   lanes rerun at higher precision (per tier) and binary64 retry rows.
//!
//! The N-gon's own evaluation computes every subset's scalar masters and
//! coefficient recursions once; what a shared plan would add per pinched
//! integral is at least its gather and output, at most also its recurrence.
//!
//! Finally the whole amplitude through [`AmplitudeWorkspace`], with per-subset
//! sharing off and on: time per call, reused and computed subset results, and
//! a bitwise comparison of every integral's arrays and statuses.
//!
//! usage: amplitude_sharing POINTS_JSON INTEGRAL [LANES] [CALLS]
use std::time::{Duration, Instant};

use iregi_rust::{
    AmplitudeWorkspace, DirectReductionPlan, KinematicsBatch, ReductionSettings, TensorEngineExecution,
    TensorReductionPlan, TensorWorkspace,
};
use serde_json::Value;
use symbolica::domains::float::Complex;

fn subsets(n: usize) -> Vec<Vec<usize>> {
    (1u32..1 << n)
        .filter(|m| m.count_ones() >= 2)
        .map(|m| (0..n).filter(|i| m & (1 << i) != 0).collect())
        .collect()
}

fn batch(points: &[Value], members: &[usize], masses: &[f64], mu: f64) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    for point in points {
        for &i in members {
            values.extend(point[i].as_array().unwrap().iter().map(|x| Complex::new(x.as_f64().unwrap(), 0.0)));
        }
        values.extend(members.iter().map(|&i| Complex::new(masses[i], 0.0)));
        values.push(Complex::new(mu, 0.0));
    }
    KinematicsBatch { rows: points.len(), components: 1, values }
}

#[derive(Default, Clone, Copy)]
struct Phases {
    facade: Duration,
    geometry: Duration,
    masters: Duration,
    recurrence: Duration,
    gather: Duration,
    output: Duration,
    master_rows: usize,
    integrals: usize,
    rotation_failures: usize,
    rescued: [usize; 3],
    retry_rows: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let integral = data["integrals"].as_array().unwrap().iter().find(|i| i["id"] == args[2].as_str()).unwrap();
    let lanes: usize = args.get(3).map_or(32, |a| a.parse().unwrap());
    let calls: usize = args.get(4).map_or(10, |a| a.parse().unwrap());
    let n = integral["n"].as_u64().unwrap() as usize;
    let mu = integral["mu"].as_f64().unwrap();
    let masses: Vec<f64> = integral["masses_squared"].as_array().unwrap().iter().map(|m| m.as_f64().unwrap()).collect();
    let points: Vec<Value> = integral["points"].as_array().unwrap()[..lanes].to_vec();
    let mut by_size = vec![Phases::default(); n + 1];
    for members in subsets(n) {
        let k = members.len();
        let input = batch(&points, &members, &masses, mu);
        let settings = ReductionSettings::new(k, k as u16);
        let plan = TensorReductionPlan::new(settings.clone())?;
        let mut facade = TensorWorkspace::new(&plan, 0.0f64)?;
        facade.reduce_batch(&input)?;
        let started = Instant::now();
        for _ in 0..calls {
            std::hint::black_box(facade.reduce_batch(&input)?);
        }
        let entry = &mut by_size[k];
        entry.facade += started.elapsed() / calls as u32;
        if let TensorEngineExecution::Direct(stats) = facade.engine_execution() {
            entry.rotation_failures += stats.rotation_failures;
            entry.retry_rows += stats.checked_rows + stats.expansion_rows;
            for tier in &stats.precision_rescue {
                let slot = match tier.bits { 106 => 0, 256 => 1, _ => 2 };
                entry.rescued[slot] += tier.rows;
            }
        }
        let direct = DirectReductionPlan::new(settings)?;
        let mut workspace = direct.workspace(0.0f64)?;
        workspace.set_profiling(true);
        workspace.set_stability_threshold(0.0);
        workspace.reduce_batch(&input)?;
        for _ in 0..calls {
            std::hint::black_box(workspace.reduce_batch(&input)?);
            let p = workspace.last_profile();
            entry.geometry += p.geometry / calls as u32;
            entry.masters += p.masters / calls as u32;
            entry.recurrence += p.recurrence / calls as u32;
            entry.gather += p.gather / calls as u32;
            entry.output += p.output / calls as u32;
        }
        entry.master_rows += workspace.last_master_rows();
        entry.integrals += 1;
    }
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!("{} ({lanes} lanes, {calls} calls): per batch call, summed over the integrals of each size", args[2]);
    println!("size integrals  facade_ms | direct: geometry masters recurrence gather output (ms) | master evaluations");
    let mut total = Phases::default();
    for (k, e) in by_size.iter().enumerate().skip(2) {
        println!(
            "     facade lanes: rotation-refused {}, rescued at 106/256/512 bits {:?}, binary64 retry rows {}",
            e.rotation_failures, e.rescued, e.retry_rows
        );
        println!(
            "{k:4} {:9} {:10.3} | {:8.3} {:7.3} {:10.3} {:6.3} {:6.3} | {}",
            e.integrals, ms(e.facade), ms(e.geometry), ms(e.masters), ms(e.recurrence), ms(e.gather), ms(e.output), e.master_rows
        );
        total.facade += e.facade;
        total.geometry += e.geometry;
        total.masters += e.masters;
        total.recurrence += e.recurrence;
        total.gather += e.gather;
        total.output += e.output;
        total.master_rows += e.master_rows;
        total.integrals += e.integrals;
    }
    println!(
        " all {:9} {:10.3} | {:8.3} {:7.3} {:10.3} {:6.3} {:6.3} | {}",
        total.integrals, ms(total.facade), ms(total.geometry), ms(total.masters), ms(total.recurrence), ms(total.gather),
        ms(total.output), total.master_rows
    );
    let top = by_size[n];
    let pinched_gather: Duration = by_size[2..n].iter().map(|e| e.gather + e.output).sum();
    let pinched_recurrence: Duration = by_size[2..n].iter().map(|e| e.recurrence).sum();
    let separate = ms(total.geometry + total.masters + total.recurrence + total.gather + total.output);
    let top_direct = ms(top.geometry + top.masters + top.recurrence + top.gather + top.output);
    let shared_low = top_direct + ms(pinched_gather);
    let shared_high = shared_low + ms(pinched_recurrence);
    println!(
        "direct engine, one pass: separate {separate:.3} ms; shared plan estimate {shared_low:.3}..{shared_high:.3} ms \
         (factor {:.1}..{:.1}); master evaluations {} -> {}",
        separate / shared_high, separate / shared_low, total.master_rows, top.master_rows
    );
    println!(
        "facade (with checks): separate {:.3} ms, of which the {n}-point integral {:.3} ms",
        ms(total.facade), ms(top.facade)
    );
    // The whole amplitude, sharing off and on.
    let mut plans = Vec::new();
    let mut batches = Vec::new();
    for members in subsets(n) {
        let k = members.len();
        plans.push(TensorReductionPlan::new(ReductionSettings::new(k, k as u16))?);
        batches.push(batch(&points, &members, &masses, mu));
    }
    let mut amplitude = AmplitudeWorkspace::new(&plans, 0.0f64)?;
    let mut outputs = Vec::new();
    for share in [false, true] {
        amplitude.set_subset_sharing(share);
        amplitude.reduce_batches(&batches)?;
        let started = Instant::now();
        let mut last = Vec::new();
        for _ in 0..calls {
            last = amplitude.reduce_batches(&batches)?;
        }
        let stats = amplitude.last_sharing();
        println!(
            "amplitude ({} integrals), sharing {}: {:.1} ms per call; subset results reused {}, computed {}",
            amplitude.len(),
            if share { "on" } else { "off" },
            ms(started.elapsed() / calls as u32),
            stats.reused,
            stats.computed
        );
        outputs.push(last);
    }
    let bits = |result: &iregi_rust::BatchResult<f64>| -> (Vec<u64>, String) {
        let mut values = Vec::new();
        for lane in 0..result.rows() {
            match result.lane_with_estimate(lane) {
                Some(best) => values.extend(best.values.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()])),
                None => values.push(u64::MAX),
            }
        }
        (values, format!("{:?} {:?}", result.statuses, result.estimated_errors))
    };
    let differing = outputs[0].iter().zip(&outputs[1]).filter(|(a, b)| bits(a) != bits(b)).count();
    println!("integrals whose arrays, statuses or estimates differ (bitwise): {differing} of {}", outputs[0].len());
    Ok(())
}
