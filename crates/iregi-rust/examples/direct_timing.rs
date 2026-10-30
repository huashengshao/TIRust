//! Warm full-array timing driver speaking the original-IREGI benchmark wire
//! protocol, so one runner can interleave TIRust, IREGI and COLLIER blocks.
//!
//! stdin:
//!   BENCH id n rank h mu batch warmup repeats iterations
//!   p0 p1 p2 p3 (m2re,m2im) power          (n rows per lane, lane-major)
//!   QUIT
//! stdout: `READY TIRUST_BENCHMARK_V1`, then one `RESULT {json}` per request
//! with per-block mean nanoseconds per call and the final arrays.
//!
//! The public facade selects its default binary64 engine (`plan_engine`,
//! `workspace_engine`). Plans and workspaces are cached per request shape;
//! construction and the first call are reported separately and never timed
//! as warm calls. Each block times `iterations` complete `reduce_batch` calls,
//! including result construction and destruction. Profiling is off.
use std::{
    collections::HashMap,
    error::Error,
    hint::black_box,
    io::{BufRead, Write},
    time::Instant,
};

use iregi_rust::{
    KinematicsBatch, LaneStatus, ReductionSettings, TensorEngineExecution, TensorReductionPlan,
    TensorWorkspace,
};
use serde_json::{Value, json};
use symbolica::domains::float::Complex;

type Fallible<T> = Result<T, Box<dyn Error>>;

struct Prepared {
    workspace: TensorWorkspace<f64>,
    plan_engine: String,
    plan_ns: u128,
    workspace_ns: u128,
    first_call_ns: Option<u128>,
}

fn number(token: &str) -> Fallible<f64> {
    Ok(token.parse::<f64>()?)
}

fn main() -> Fallible<()> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "READY TIRUST_BENCHMARK_V1")?;
    out.flush()?;
    let mut cache: HashMap<(usize, u16, i16, Vec<u16>), Prepared> = HashMap::new();
    while let Some(line) = lines.next() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "QUIT" {
            break;
        }
        let header: Vec<&str> = line.split_whitespace().collect();
        if header.len() != 10 || header[0] != "BENCH" {
            return Err(format!("invalid header: {line}").into());
        }
        let id = header[1].to_string();
        let n: usize = header[2].parse()?;
        let rank: u16 = header[3].parse()?;
        let h: i16 = header[4].parse()?;
        let mu = number(header[5])?;
        let batch: usize = header[6].parse()?;
        let warmup: usize = header[7].parse()?;
        let repeats: usize = header[8].parse()?;
        let iterations: usize = header[9].parse()?;
        if n == 0 || batch == 0 || repeats == 0 || iterations == 0 {
            return Err("invalid request dimensions".into());
        }
        let mut values = Vec::with_capacity(batch * (5 * n + 1));
        let mut powers: Option<Vec<u16>> = None;
        for _ in 0..batch {
            let mut shifts = Vec::with_capacity(4 * n);
            let mut masses = Vec::with_capacity(n);
            let mut lane_powers = Vec::with_capacity(n);
            for _ in 0..n {
                let row = lines.next().ok_or("missing propagator row")??;
                let cleaned = row.replace(['(', ')', ','], " ");
                let tokens: Vec<&str> = cleaned.split_whitespace().collect();
                if tokens.len() != 7 {
                    return Err(format!("invalid propagator row: {row}").into());
                }
                for token in &tokens[..4] {
                    shifts.push(Complex::new(number(token)?, 0.0));
                }
                masses.push(Complex::new(number(tokens[4])?, number(tokens[5])?));
                lane_powers.push(tokens[6].parse::<u16>()?);
            }
            match &powers {
                Some(existing) if existing != &lane_powers => {
                    return Err("powers must be common to all lanes of a request".into());
                }
                _ => powers = Some(lane_powers),
            }
            values.extend(shifts);
            values.extend(masses);
            values.push(Complex::new(mu, 0.0));
        }
        let powers = powers.unwrap();
        let batch_input = KinematicsBatch {
            rows: batch,
            components: 1,
            values,
        };
        let key = (n, rank, h, powers.clone());
        if !cache.contains_key(&key) {
            let mut settings = ReductionSettings::new(n, rank);
            settings.powers = powers.clone();
            settings.dimension_shift = h;
            let started = Instant::now();
            let plan = TensorReductionPlan::new(settings)?;
            let plan_ns = started.elapsed().as_nanos();
            let started = Instant::now();
            let workspace = TensorWorkspace::new(&plan, 0.0f64)?;
            let workspace_ns = started.elapsed().as_nanos();
            cache.insert(
                key.clone(),
                Prepared {
                    workspace,
                    plan_engine: format!("{:?}", plan.engine_kind()),
                    plan_ns,
                    workspace_ns,
                    first_call_ns: None,
                },
            );
        }
        let prepared = cache.get_mut(&key).unwrap();
        if prepared.first_call_ns.is_none() {
            let started = Instant::now();
            black_box(prepared.workspace.reduce_batch(&batch_input)?);
            prepared.first_call_ns = Some(started.elapsed().as_nanos());
        }
        for _ in 0..warmup {
            black_box(prepared.workspace.reduce_batch(black_box(&batch_input))?);
        }
        let mut times = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(prepared.workspace.reduce_batch(black_box(&batch_input))?);
            }
            times.push(started.elapsed().as_nanos() as f64 / iterations as f64);
        }
        // Untimed final call: arrays, statuses and actual execution path.
        let result = prepared.workspace.reduce_batch(&batch_input)?;
        let execution = prepared.workspace.engine_execution();
        let (fallback_rows, fallback_successes) = match &execution {
            TensorEngineExecution::Direct(stats) => (stats.fallback_rows, stats.fallback_successes),
            _ => {
                let summary = prepared.workspace.execution_summary()?;
                (summary.fallback_rows, summary.fallback_successes)
            }
        };
        let (precision_rescue, symbolic_rows) = match &execution {
            TensorEngineExecution::Direct(stats) => (
                stats
                    .precision_rescue
                    .iter()
                    .map(|tier| json!({"bits": tier.bits, "rows": tier.rows, "successes": tier.successes}))
                    .collect::<Vec<_>>(),
                Some(stats.symbolic_rows),
            ),
            _ => (Vec::new(), None),
        };
        let diagnostics = prepared.workspace.last_diagnostics().to_vec();
        let moments = result.basis.len();
        let lanes: Vec<Value> = (0..batch)
            .map(|lane| match result.lane(lane) {
                Ok(values) => json!({
                    "status": "success",
                    "fallback_used": result.fallback_used[lane],
                    "diagnostics": diagnostics.get(lane),
                    "moments": (0..moments)
                        .map(|m| (0..4)
                            .map(|slot| {
                                let z = values[4 * m + slot];
                                [z.re, z.im]
                            })
                            .collect::<Vec<_>>())
                        .collect::<Vec<_>>(),
                }),
                Err(status) => json!({
                    "status": match status {
                        LaneStatus::Unstable(_) => "unstable",
                        LaneStatus::InvalidInput(_) => "invalid",
                        LaneStatus::MasterFailure(_) => "master_failure",
                        _ => "failure",
                    },
                    "detail": format!("{status:?}"),
                    "fallback_used": result.fallback_used[lane],
                    "diagnostics": diagnostics.get(lane),
                    "moments": Value::Null,
                }),
            })
            .collect();
        let stable = result.statuses.iter().all(|s| *s == LaneStatus::Success);
        let record = json!({
            "id": id,
            "stable": stable,
            "total_ns": times,
            "plan_engine": prepared.plan_engine,
            "workspace_engine": format!("{:?}", prepared.workspace.engine_kind()),
            "plan_ns": prepared.plan_ns,
            "workspace_ns": prepared.workspace_ns,
            "first_call_ns": prepared.first_call_ns,
            "fallback_rows": fallback_rows,
            "fallback_successes": fallback_successes,
            "precision_rescue": precision_rescue,
            "symbolic_rows": symbolic_rows,
            "lanes": lanes,
        });
        writeln!(out, "RESULT {}", serde_json::to_string(&record)?)?;
        out.flush()?;
    }
    Ok(())
}
