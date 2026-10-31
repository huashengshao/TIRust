//! Reproducible, warm, full-array tensor-reduction timing for the opening report.
//!
//! Usage: opening_double request.json result.json
//! Binary64 only. Each request selects the public algorithm, master policy and
//! whether profiling is enabled after construction to preserve a thin engine.
//! Failed lanes remain inaccessible and are never represented as zero tensors.
//! Process supervision, CPU affinity, source hashes and independent numerical
//! validation belong to the external runner; this executable does not infer them.
use std::{error::Error, hint::black_box, path::Path, time::Instant};

use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use iregi_rust::{
    Algorithm, BatchResult, KinematicsBatch, LaneStatus, ReductionSettings, TensorReductionPlan,
    TensorWorkspace, TensorWorkspaceOptions,
};
use serde::Deserialize;
use serde_json::{Value, json};
use symbolica::{
    domains::{
        float::Complex,
        integer::Integer,
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

type Fallible<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Deserialize)]
struct Sample {
    shifts: Vec<[String; 4]>,
    masses_squared: Vec<[String; 2]>,
    mu: String,
}

#[derive(Clone, Deserialize)]
struct Request {
    id: String,
    n: usize,
    rank: u16,
    powers: Vec<u16>,
    dimension_shift: i16,
    algorithm: Algorithm,
    numeric_type: String,
    working_bits: u32,
    requested_bits: u32,
    samples: Vec<Sample>,
    master_policy: String,
    #[serde(default)]
    postconstruction_profiling: bool,
    warmup: usize,
    repeats: usize,
    iterations: usize,
    #[serde(default)]
    target_block_ns: Option<u64>,
    #[serde(default = "default_maximum_iterations")]
    maximum_iterations: usize,
}

fn default_maximum_iterations() -> usize {
    10_000
}

fn exact(text: &str) -> Fallible<Rational> {
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let denominator = d.parse::<Integer>()?;
    if denominator == Integer::from(0) {
        return Err("zero rational denominator".into());
    }
    Ok(Rational::from((n.parse::<Integer>()?, denominator)))
}

fn write(path: &Path, value: &Value) -> Fallible<()> {
    // A killed numerical process leaves the last complete phase record intact.
    let temporary = path.with_extension("json.partial");
    std::fs::write(&temporary, serde_json::to_string_pretty(value)? + "\n")?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn operand<T: NativeFloat>(value: &T) -> Value {
    json!({
        "exact_rational": value.exact_rational().map(|q| q.to_string()),
        "working_bits": value.get_precision(),
        "negative_sign": value.negative_sign(),
        "finite": value.is_finite(),
    })
}

fn array<T: NativeFloat>(values: &[Complex<T>]) -> Value {
    json!(
        values
            .iter()
            .map(|z| [operand(&z.re), operand(&z.im)])
            .collect::<Vec<_>>()
    )
}

fn output<T: NativeFloat>(result: &BatchResult<T>) -> Value {
    let lanes = (0..result.rows())
        .map(|lane| {
            json!({
                "lane": lane,
                "status": format!("{:?}", result.statuses[lane]),
                "fallback_used": result.fallback_used[lane],
                "analytic_used": result.analytic_used[lane],
                "values": result.lane(lane).ok().map(array),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "basis": result.basis.iter().map(|m| m.0).collect::<Vec<_>>(),
        "components": result.components,
        "laurent_order": ["UV single", "IR single", "IR double", "finite"],
        "convention": result.metadata().convention(),
        "tensor_weights": result.metadata().tensor_weights(),
        "lanes": lanes,
    })
}

fn full_success<T: NativeFloat>(result: &BatchResult<T>, rows: usize, moments: usize) -> bool {
    result.rows() == rows
        && result.basis.len() == moments
        && result.components == 1
        && result.statuses.iter().all(|s| *s == LaneStatus::Success)
        && (0..rows).all(|lane| {
            result.lane(lane).is_ok_and(|values| {
                values.len() == moments * 4
                    && values.iter().all(|z| z.re.is_finite() && z.im.is_finite())
            })
        })
}

fn same_outputs<T: NativeFloat>(left: &BatchResult<T>, right: &BatchResult<T>) -> bool {
    left.basis == right.basis
        && left.components == right.components
        && left.statuses == right.statuses
        && left.rows() == right.rows()
        && (0..left.rows()).all(|lane| match (left.lane(lane), right.lane(lane)) {
            (Ok(a), Ok(b)) => a == b,
            (Err(_), Err(_)) => true,
            _ => false,
        })
}

fn run<T: NativeFloat>(
    request: &Request,
    prototype: T,
    path: &Path,
    record: &mut Value,
) -> Fallible<()>
where
    Complex<T>: EvaluationDomain,
{
    if request.n == 0
        || request.n > 6
        || request.rank > 6
        || request.powers.len() != request.n
        || request.powers.contains(&0)
        || request.samples.is_empty()
        || request.repeats == 0
        || request.iterations == 0
        || request.maximum_iterations == 0
        || (request.target_block_ns.is_some() && request.warmup == 0)
        || request.requested_bits < 20
        || request.requested_bits > prototype.get_precision()
        || request.working_bits != prototype.get_precision()
    {
        return Err("invalid opening timing domain, shape, precision or repeat count".into());
    }
    let mut values = Vec::with_capacity(request.samples.len() * (5 * request.n + 1));
    let embed = |text: &str| -> Fallible<T> {
        let exact = exact(text)?;
        let embedded = prototype.from_rational(&exact);
        if embedded.exact_rational().as_ref() != Some(&exact) {
            return Err("this timing cohort requires exactly represented rational inputs".into());
        }
        Ok(embedded)
    };
    for sample in &request.samples {
        if sample.shifts.len() != request.n || sample.masses_squared.len() != request.n {
            return Err("sample shape disagrees with propagator count".into());
        }
        for shift in &sample.shifts {
            for component in shift {
                values.push(Complex::new(embed(component)?, prototype.zero()));
            }
        }
        for mass in &sample.masses_squared {
            values.push(Complex::new(embed(&mass[0])?, embed(&mass[1])?));
        }
        values.push(Complex::new(embed(&sample.mu)?, prototype.zero()));
    }
    let batch = KinematicsBatch {
        rows: request.samples.len(),
        components: 1,
        values,
    };
    let original_inputs = batch.values.clone();
    record["represented_inputs"] = array(&batch.values);
    let mut settings = ReductionSettings::new(request.n, request.rank);
    settings.powers = request.powers.clone();
    settings.dimension_shift = request.dimension_shift;
    settings.algorithm = request.algorithm;
    record["settings"] = json!(settings);
    record["status"] = json!("preparing_plan");
    write(path, record)?;
    let started = Instant::now();
    let plan = TensorReductionPlan::new(settings)?;
    record["plan_ns"] = json!(started.elapsed().as_nanos());
    record["plan_engine"] = json!(format!("{:?}", plan.engine_kind()));
    record["basis"] = json!(plan.basis().iter().map(|m| m.0).collect::<Vec<_>>());
    record["status"] = json!("preparing_workspace");
    write(path, record)?;
    let options = TensorWorkspaceOptions {
        master_policy: match request.master_policy.as_str() {
            "Symbolica" => MasterExecutionPolicy::Symbolica,
            "NativeF64Abc" => MasterExecutionPolicy::NativeF64Abc,
            _ => return Err("unsupported master policy".into()),
        },
        profiling: !request.postconstruction_profiling,
        ..Default::default()
    };
    let started = Instant::now();
    let mut workspace = TensorWorkspace::new_with_options(&plan, prototype, options)?;
    if request.postconstruction_profiling { workspace.set_profiling(true)?; }
    workspace.set_accuracy_precision(request.requested_bits)?;
    record["workspace_ns"] = json!(started.elapsed().as_nanos());
    record["workspace_engine"] = json!(format!("{:?}", workspace.engine_kind()));
    record["preparation_notes"] = json!(workspace.preparation_notes());
    record["status"] = json!("first_call");
    write(path, record)?;
    let started = Instant::now();
    let first = workspace.reduce_batch(black_box(&batch))?;
    record["first_call_ns"] = json!(started.elapsed().as_nanos());
    record["first_output"] = output(&first);
    record["first_diagnostics"] = json!(workspace.last_diagnostics());
    // Refusals are still timed diagnostically. Guarded lane values remain null.
    let accessible = full_success(&first, batch.rows, plan.basis().len());
    record["successful_finite_full_array"] = json!(accessible);
    record["status"] = json!("warmup");
    write(path, record)?;
    let mut warmup_times = Vec::new();
    for _ in 0..request.warmup {
        let started = Instant::now();
        let result = workspace.reduce_batch(black_box(&batch))?;
        warmup_times.push(started.elapsed().as_nanos());
        if !same_outputs(&first, &result)
        {
            record["status"] = json!("warmup_failed");
            record["failure_output"] = output(&result);
            record["timing_qualified"] = json!(false);
            write(path, record)?;
            return Ok(());
        }
        black_box(result);
    }
    let iterations = match request.target_block_ns {
        Some(target) => {
            if target == 0 {
                return Err("target_block_ns must be positive".into());
            }
            // Only the final already-warm call sets the fixed loop count. The
            // realized count is retained; no timed block adapts or drops calls.
            let estimate = warmup_times.last().copied().unwrap().max(1);
            ((u128::from(target).div_ceil(estimate)).min(request.maximum_iterations as u128)
                as usize)
                .max(1)
        }
        None => request.iterations,
    };
    record["warmup_call_ns"] = json!(warmup_times);
    record["realized_iterations_per_repeat"] = json!(iterations);
    let mut repeated = Vec::new();
    for repeat in 0..request.repeats {
        record["status"] = json!("measuring");
        record["active_repeat"] = json!(repeat);
        write(path, record)?;
        let mut calls = Vec::with_capacity(iterations);
        let mut all_successful = true;
        for _ in 0..iterations {
            // Only the public numerical call is timed. Full output/status
            // checks, profiler extraction and serialization are outside it.
            let started = Instant::now();
            let result = workspace.reduce_batch(black_box(&batch))?;
            let outer = started.elapsed().as_nanos();
            let profile = workspace
                .last_profile()
                .ok_or("profile missing from actual workspace")?;
            let masters = profile.scalar_masters.as_nanos();
            let exclusive = outer
                .checked_sub(masters)
                .ok_or("nested master time exceeds outer call duration")?;
            all_successful &= same_outputs(&first, &result);
            calls.push(json!({
                "outer_total_ns": outer,
                "scalar_masters_ns": masters,
                "tirust_exclusive_ns": exclusive,
                "profile_total_ns": profile.total.as_nanos(),
                "profile_arguments_ns": profile.arguments.as_nanos(),
                "profile_recurrence_and_stability_ns": profile.reduction.as_nanos(),
            }));
            if !all_successful {
                record["failure_output"] = output(&result);
                break;
            }
            black_box(result);
        }
        repeated.push(json!({"repeat": repeat, "calls": calls, "all_statuses_and_accessible_arrays_reproduced": all_successful}));
        record["repeated_measurements"] = json!(repeated);
        if !all_successful {
            record["status"] = json!("timed_call_failed");
            record["timing_qualified"] = json!(false);
            write(path, record)?;
            return Ok(());
        }
        write(path, record)?;
    }
    record["status"] = json!("postcheck");
    write(path, record)?;
    let result = workspace.reduce_batch(&batch)?;
    let passed = same_outputs(&first, &result)
        && batch.values == original_inputs;
    record["final_output"] = output(&result);
    record["final_diagnostics"] = json!(workspace.last_diagnostics());
    let execution = workspace.engine_execution().summary()?;
    record["execution_summary"] = json!({
        "input_rows": execution.input_rows,
        "argument_rows": execution.argument_rows,
        "master_family_batches": execution.master_family_batches,
        "master_rows": execution.master_rows,
        "pentagon_box_rows": execution.pentagon_box_rows,
        "reduction_rows": execution.reduction_rows,
        "fallback_rows": execution.fallback_rows,
        "fallback_successes": execution.fallback_successes,
        "analytic_rows": execution.analytic_rows,
    });
    record["execution_detail"] = json!(format!("{:?}", workspace.engine_execution()));
    record["all_successful_finite_full_arrays"] = json!(passed && accessible);
    record["all_outputs_exactly_reproduced"] = json!(passed);
    record["input_preserved"] = json!(batch.values == original_inputs);
    record["status"] = json!(if passed && accessible {
        "measured"
    } else if passed {
        "measured_diagnostic_refusal"
    } else {
        "postcheck_failed"
    });
    record["timing_qualified"] = json!(passed);
    write(path, record)?;
    Ok(())
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        eprintln!("usage: opening_double request.json result.json");
        std::process::exit(2);
    }
    let input = match std::fs::read_to_string(&args[0]) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let raw: Value = match serde_json::from_str(&input) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let request: Request = match serde_json::from_value(raw.clone()) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let path = Path::new(&args[1]);
    let mut record = json!({
        "schema_version": 1, "id": request.id, "request": raw,
        "interface": "TensorReductionPlan::new; TensorWorkspace::new_with_options; optional set_profiling; reduce_batch",
        "status": "starting", "timing_qualified": false,
        "independent_numerical_validation_performed_by_driver": false,
        "timing_boundary": "warm public profiled call; scalar_masters is the aggregate nested master wall time; TIRust-exclusive is outer wall minus scalar_masters; includes invariant evaluation, recurrence, routing, diagnostics, allocation and result construction. Plan/workspace/first-call, full-array checks and JSON I/O are excluded. Instrumentation remains enabled; actual selected engine, policy and algorithm are retained. Refused lanes have diagnostic timing only and no accessible coefficients.",
        "output_scope": "all cumulative Cartesian tensor moments through requested rank, all four native Laurent slots, every successful input lane",
        "precision_notice": "Working bits and the requested accuracy parameter do not certify achieved bits or a rigorous error bound.",
        "repeated_measurements": [],
    });
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Fallible<()> {
        if request.numeric_type != "f64" || request.working_bits != 53 {
            return Err("opening tables require ordinary Rust f64 at 53 working bits".into());
        }
        run(&request, 0.0_f64, path, &mut record)
    }));
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            record["failed_phase"] = record["status"].clone();
            record["status"] = json!("error");
            record["error"] = json!(error.to_string());
            record["timing_qualified"] = json!(false);
        }
        Err(_) => {
            record["failed_phase"] = record["status"].clone();
            record["status"] = json!("panic");
            record["error"] = json!("public preparation or evaluation panicked");
            record["timing_qualified"] = json!(false);
        }
    }
    if let Err(error) = write(path, &record) {
        eprintln!("{error}");
        std::process::exit(2);
    }
    if record["timing_qualified"] != true {
        std::process::exit(2);
    }
}
