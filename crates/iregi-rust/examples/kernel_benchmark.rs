//! Timing-only JSONL adapter. Build with `cargo build --release --example kernel_benchmark`.
//!
//! The runner pins this process to one CPU and measures the independent Fortran
//! executable with identical input rows. Transport and symbolic preparation are
//! outside kernel timings. Profiling is a separate pass so instrumentation does
//! not inflate the production kernel comparison.
use std::{
    hint::black_box,
    io::{self, BufRead, Write},
    time::Instant,
};

use iregi_rust::{Algorithm, KinematicsBatch, ReductionPlan, ReductionSettings, Workspace};
use serde::Deserialize;
use serde_json::{Value, json};
use symbolica::domains::float::Complex;

#[derive(Deserialize)]
struct Sample {
    shifts: Vec<[f64; 4]>,
    masses: Vec<[f64; 2]>,
    mu: f64,
}

#[derive(Deserialize)]
struct Request {
    id: String,
    n: usize,
    rank: u16,
    algorithm: Algorithm,
    samples: Vec<Sample>,
    warmup: usize,
    repeats: usize,
    iterations: usize,
    #[serde(default)]
    profile: bool,
}

struct Prepared {
    key: (usize, u16, Algorithm),
    workspace: Workspace<f64>,
    degrees: Vec<usize>,
}

fn run(request: &Request, prepared: &mut Option<Prepared>) -> Result<Value, String> {
    if request.samples.is_empty() || request.repeats == 0 || request.iterations == 0 {
        return Err("samples, repeats, and iterations must be nonzero".into());
    }
    if request.n == 0 || request.n > 7 || request.rank > 7 {
        return Err("benchmark coverage is n=1..7 and rank=0..7".into());
    }
    let mut values = Vec::new();
    for sample in &request.samples {
        if sample.shifts.len() != request.n || sample.masses.len() != request.n {
            return Err("sample shape does not match n".into());
        }
        values.extend(
            sample
                .shifts
                .iter()
                .flatten()
                .map(|&x| Complex::new(x, 0.0)),
        );
        values.extend(sample.masses.iter().map(|x| Complex::new(x[0], x[1])));
        values.push(Complex::new(sample.mu, 0.0));
    }
    let batch = KinematicsBatch {
        rows: request.samples.len(),
        components: 1,
        values,
    };
    let key = (request.n, request.rank, request.algorithm);
    let reused = prepared.as_ref().is_some_and(|p| p.key == key);
    let preparation_ns = if reused {
        None
    } else {
        let start = Instant::now();
        let mut settings = ReductionSettings::new(request.n, request.rank);
        settings.algorithm = request.algorithm;
        let plan = ReductionPlan::new(settings).map_err(|e| e.to_string())?;
        let workspace = Workspace::new(&plan, 0.0f64).map_err(|e| e.to_string())?;
        *prepared = Some(Prepared {
            key,
            workspace,
            degrees: plan.basis().iter().map(|m| m.rank()).collect(),
        });
        Some(start.elapsed().as_nanos() as u64)
    };
    let prepared = prepared.as_mut().unwrap();
    prepared.workspace.set_profiling(request.profile);
    for _ in 0..request.warmup {
        black_box(
            prepared
                .workspace
                .reduce_batch(black_box(&batch))
                .map_err(|e| e.to_string())?,
        );
    }
    let mut totals = Vec::with_capacity(request.repeats);
    let mut arguments = Vec::with_capacity(request.repeats);
    let mut masters = Vec::with_capacity(request.repeats);
    let mut reduction = Vec::with_capacity(request.repeats);
    let mut instrumented_total = Vec::with_capacity(request.repeats);
    let mut every_call_successful = true;
    for _ in 0..request.repeats {
        let mut stage = [0u128; 4];
        let start = Instant::now();
        for _ in 0..request.iterations {
            let result = prepared
                .workspace
                .reduce_batch(black_box(&batch))
                .map_err(|e| e.to_string())?;
            every_call_successful &= result
                .statuses
                .iter()
                .all(|s| *s == iregi_rust::LaneStatus::Success);
            black_box(result);
            if let Some(profile) = prepared.workspace.last_profile() {
                stage[0] += profile.arguments.as_nanos();
                stage[1] += profile.scalar_masters.as_nanos();
                stage[2] += profile.reduction.as_nanos();
                stage[3] += profile.total.as_nanos();
            }
        }
        totals.push(start.elapsed().as_nanos() as f64 / request.iterations as f64);
        arguments.push(stage[0] as f64 / request.iterations as f64);
        masters.push(stage[1] as f64 / request.iterations as f64);
        reduction.push(stage[2] as f64 / request.iterations as f64);
        instrumented_total.push(stage[3] as f64 / request.iterations as f64);
    }
    // The output check is independent of timed iterations and retains every
    // tensor component and Laurent slot instead of a lossy checksum.
    let result = prepared
        .workspace
        .reduce_batch(&batch)
        .map_err(|e| e.to_string())?;
    let lanes: Vec<Value> = (0..batch.rows).map(|lane| match result.lane(lane) {
        Ok(values) => json!({
            "status": "success",
            "moments": values.chunks_exact(4).map(|pole| pole.iter().map(|v| [v.re,v.im]).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "fallback_used": result.fallback_used[lane],
            "analytic_used": result.analytic_used[lane],
        }),
        Err(error) => json!({"status": format!("{error:?}"), "moments": null}),
    }).collect();
    let stats = prepared.workspace.last_execution();
    Ok(json!({
        "id": request.id, "precision": "binary64", "plan_reused": reused,
        "master_backend": format!("{:?}", iregi_masters::default_master_backend()),
        "preparation_ns": preparation_ns, "degrees": prepared.degrees,
        "stable": every_call_successful, "lanes": lanes, "total_ns": totals,
        "profiled": request.profile,
        "profile_ns": if request.profile { json!({
            "arguments": arguments, "masters": masters, "reduction": reduction,
            "total": instrumented_total,
        }) } else { Value::Null },
        "execution": {
            "input_rows": stats.input_rows, "argument_rows": stats.argument_rows,
            "master_family_batches": stats.master_family_batches, "master_rows": stats.master_rows,
            "pentagon_box_rows": stats.pentagon_box_rows,
            "reduction_rows": stats.reduction_rows, "fallback_rows": stats.fallback_rows,
            "fallback_successes": stats.fallback_successes, "analytic_rows": stats.analytic_rows,
        },
    }))
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut prepared = None;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        if line.trim() == "QUIT" {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&request, &mut prepared)
            })) {
                Ok(Ok(value)) => value,
                Ok(Err(error)) => json!({"id":request.id,"status":"error","error":error}),
                Err(_) => {
                    prepared = None;
                    json!({"id":request.id,"status":"panic","error":"benchmark preparation or evaluation panicked"})
                }
            },
            Err(error) => json!({"status":"invalid_request","error":error.to_string()}),
        };
        writeln!(stdout, "RESULT {response}").unwrap();
        stdout.flush().unwrap();
    }
}
