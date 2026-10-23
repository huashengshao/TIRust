//! Isolated public-API precision comparison against the exact parameter oracle.
//!
//! One precision pair runs per process, so timeout/failure evidence is retained
//! independently. No snapping, input rescaling or production threshold changes.

use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, Convention, KinematicsBatch, ReductionPlan, ReductionSettings, Workspace,
};
use serde_json::{Value, json};
use std::{error::Error, path::PathBuf, process::ExitCode, time::Instant};
use symbolica::domains::{
    float::{Complex, Float, FloatLike, SingleFloat},
    integer::Integer,
    rational::Rational,
};

fn rational(text: &str) -> Result<Rational, Box<dyn Error>> {
    let (numerator, denominator) = text.split_once('/').unwrap_or((text, "1"));
    let denominator = denominator.parse::<Integer>()?;
    if denominator == Integer::from(0) {
        return Err("zero rational denominator".into());
    }
    Ok(Rational::from((numerator.parse::<Integer>()?, denominator)))
}

fn number(value: &Value) -> Result<Rational, Box<dyn Error>> {
    rational(
        value
            .as_str()
            .ok_or("oracle input must be an exact rational string")?,
    )
}

fn abs(value: &Rational) -> Rational {
    if value < &Rational::from(0) {
        -value.clone()
    } else {
        value.clone()
    }
}

fn negative_power_of_two(exponent: u32) -> Rational {
    Rational::from((Integer::from(1), Integer::from(2).pow(u64::from(exponent))))
}

fn compare(
    slots: &[Complex<Float>],
    center: &Rational,
    radius: &Rational,
    relative_budget: &Rational,
    strict_absolute_budget: &Rational,
) -> (Value, &'static str, u8) {
    let represented: Option<Vec<_>> = slots
        .iter()
        .map(|value| Some((value.re.exact_rational()?, value.im.exact_rational()?)))
        .collect();
    let Some(represented) = represented else {
        return (
            json!({"performed": false, "reason": "public Success contained a nonfinite value", "values": null}),
            "successful_nonfinite_output",
            3,
        );
    };
    let finite = &represented[3];
    let real_error = abs(&(&finite.0 - center));
    let imaginary_error = abs(&finite.1);
    let maximum_real_error = &real_error + radius;
    let minimum_real_error = if &real_error > radius {
        &real_error - radius
    } else {
        Rational::from(0)
    };
    // Bound the actual complex norm without approximate square roots. Failure
    // of an upper-bound certificate alone is never called a disagreement.
    let error_squared_upper = maximum_real_error.pow(2) + imaginary_error.pow(2);
    let error_squared_lower = minimum_real_error.pow(2) + imaginary_error.pow(2);
    let output_squared = finite.0.pow(2) + finite.1.pow(2);
    let normalization_squared = output_squared.max(Rational::from(1));
    let public_target_squared = relative_budget.pow(2) * normalization_squared;
    let error_l1_upper = maximum_real_error + imaginary_error;
    let zero_poles = represented[..3]
        .iter()
        .all(|(re, im)| re == &Rational::from(0) && im == &Rational::from(0));
    let matches_budget = zero_poles && error_squared_upper < public_target_squared;
    let definite_violation = !zero_poles || error_squared_lower > public_target_squared;
    let strict_bits = zero_poles && &error_l1_upper < strict_absolute_budget;
    let slots: Vec<_> = slots
        .iter()
        .zip(represented)
        .map(|(value, (re, im))| {
            json!({
                "real_exact_rational": re.to_string(),
                "imaginary_exact_rational": im.to_string(),
                "real_tracked_precision": value.re.get_precision(),
                "imaginary_tracked_precision": value.im.get_precision(),
            })
        })
        .collect();
    let (qualification, exit_code) = if matches_budget {
        ("passed", 0)
    } else if definite_violation {
        ("oracle_disagreement", 3)
    } else {
        ("agreement_not_certified", 4)
    };
    (
        json!({
            "performed": true,
            "slots_uv_ir_ir2_finite": slots,
            "poles_exactly_zero": zero_poles,
            "finite_complex_L1_error_upper_bound_exact_rational": error_l1_upper.to_string(),
            "finite_complex_error_squared_lower_bound_exact_rational": error_squared_lower.to_string(),
            "finite_complex_error_squared_upper_bound_exact_rational": error_squared_upper.to_string(),
            "public_mixed_error_target_squared_exact_rational": public_target_squared.to_string(),
            "within_unchanged_public_mixed_budget": matches_budget,
            "definite_budget_or_zero_pole_violation": definite_violation,
            "within_stricter_requested_relative_bits": strict_bits,
            "strict_bits_note": "Additional observation only; the public mixed target remains 2^(19-bits)*max(abs(output),1).",
        }),
        qualification,
        exit_code,
    )
}

fn run() -> Result<u8, Box<dyn Error>> {
    let started = Instant::now();
    let mut arguments = std::env::args().skip(1);
    let usage = "usage: e6_series_compare oracle.json working_bits requested_bits [output.json]";
    let oracle_path = PathBuf::from(arguments.next().ok_or(usage)?);
    let working_bits: u32 = arguments.next().ok_or(usage)?.parse()?;
    let requested_bits: u32 = arguments.next().ok_or(usage)?.parse()?;
    let output = arguments.next().map(PathBuf::from);
    if arguments.next().is_some()
        || ![128, 256, 384].contains(&working_bits)
        || ![128, 256].contains(&requested_bits)
        || requested_bits > working_bits
    {
        return Err(usage.into());
    }
    let oracle: Value = serde_json::from_str(&std::fs::read_to_string(&oracle_path)?)?;
    if oracle["status"] != "exact_oracle_generated" {
        return Err("input is not an exact parameter-series oracle".into());
    }
    let reference = oracle["approximations"]
        .as_array()
        .ok_or("missing oracle approximations")?
        .iter()
        .find(|entry| entry["order_in_eta_inclusive"] == 4)
        .ok_or("missing order-four oracle")?;
    let center = number(&reference["center_exact_rational"])?;
    let radius = number(&reference["absolute_remainder_bound_exact_rational"])?;
    if radius <= Rational::from(0) || radius >= negative_power_of_two(321) || abs(&center) <= radius
    {
        return Err("oracle does not provide the required nonzero bounded value".into());
    }

    let fixture = &oracle["fixture"];
    let shifts = fixture["momentum_shifts"]
        .as_array()
        .ok_or("missing shifts")?;
    let masses = fixture["mass_squared"].as_array().ok_or("missing masses")?;
    if shifts.len() != 5 || masses.len() != 5 {
        return Err("the oracle must describe five propagators".into());
    }
    let mut exact_input = Vec::with_capacity(26);
    for shift in shifts {
        let shift = shift.as_array().ok_or("invalid shift")?;
        if shift.len() != 4 {
            return Err("each shift must have four components".into());
        }
        exact_input.extend(shift.iter().map(number).collect::<Result<Vec<_>, _>>()?);
    }
    exact_input.extend(masses.iter().map(number).collect::<Result<Vec<_>, _>>()?);
    exact_input.push(number(&fixture["mu"])?); // public core takes mu, not mu²
    let delta = number(&fixture["delta_exact_rational"])?;
    if delta != negative_power_of_two(32)
        || exact_input[20..]
            .iter()
            .any(|value| value != &Rational::from(1))
        || exact_input[..20].iter().enumerate().any(|(index, value)| {
            value
                != &if index / 4 == index % 4 + 1 {
                    delta.clone()
                } else {
                    Rational::from(0)
                }
        })
    {
        return Err(
            "probe requires the original tiny-routing fixture with masses and mu equal to one"
                .into(),
        );
    }
    let prototype = Float::new(working_bits);
    let input = KinematicsBatch {
        rows: 1,
        components: 1,
        values: exact_input
            .iter()
            .map(|value| Complex::new(prototype.from_rational(value), prototype.zero()))
            .collect(),
    };
    // These particular dyadics are exact at every tested working precision.
    // This is an evaluation test, not a test of decimal parsing past 53 bits.
    for (exact, represented) in exact_input.iter().zip(&input.values) {
        if represented.re.exact_rational().as_ref() != Some(exact) {
            return Err("an original oracle input was rounded during construction".into());
        }
    }
    let mut settings = ReductionSettings::new(5, 0);
    settings.algorithm = Algorithm::Ibp;
    settings.dimension_shift = 1;
    settings.convention = Convention::OneLoop;
    eprintln!("E6 comparison: working={working_bits}, requested={requested_bits}; preparing plan");
    let plan_started = Instant::now();
    let plan = ReductionPlan::new(settings)?;
    let plan_seconds = plan_started.elapsed().as_secs_f64();
    eprintln!("E6 comparison: plan complete in {plan_seconds:.6}s; preparing workspace");
    let workspace_started = Instant::now();
    let mut workspace = Workspace::new(&plan, prototype)?;
    workspace.set_accuracy_precision(requested_bits)?;
    workspace.set_profiling(true);
    let workspace_seconds = workspace_started.elapsed().as_secs_f64();
    eprintln!(
        "E6 comparison: workspace complete in {workspace_seconds:.6}s; reducing original inputs"
    );
    let reduction_started = Instant::now();
    let result = workspace.reduce_batch(&input)?;
    let reduction_seconds = reduction_started.elapsed().as_secs_f64();
    eprintln!(
        "E6 comparison: reduction complete in {reduction_seconds:.6}s; status={:?}",
        result.statuses
    );

    // The original fixture's kinematic scale, and hence its natural output
    // normalization, is exactly one. Preserve the public mixed criterion;
    // separately record a stricter pure-relative requested-bit observation.
    let reference_magnitude_lower_bound = abs(&center) - &radius;
    let relative_budget = negative_power_of_two(requested_bits - 19);
    let strict_absolute_budget =
        &reference_magnitude_lower_bound * &negative_power_of_two(requested_bits);
    let (comparison, qualification, exit_code) = match result.lane(0) {
        Ok(slots) => {
            if slots.len() != 4 {
                return Err("unexpected public Laurent output layout".into());
            }
            compare(
                slots,
                &center,
                &radius,
                &relative_budget,
                &strict_absolute_budget,
            )
        }
        Err(status) => (
            json!({
                "performed": false,
                "reason": format!("{status:?}"),
                "values": null,
            }),
            "production_rejected",
            2,
        ),
    };
    let stats = workspace.last_execution();
    let profile = workspace.last_profile().unwrap();
    let report = json!({
        "schema_version": 1,
        "qualification": qualification,
        "oracle": oracle_path,
        "working_bits": working_bits,
        "requested_bits": requested_bits,
        "rows": 1,
        "components": 1,
        "algorithm": "Ibp",
        "dimension_shift": 1,
        "tensor_rank": 0,
        "convention": "OneLoop",
        "input_policy": "Preserve",
        "original_inputs_exact_rational": exact_input.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "reference_center_exact_rational": center.to_string(),
        "reference_radius_exact_rational": radius.to_string(),
        "public_relative_budget_exact_rational": relative_budget.to_string(),
        "public_mixed_budget_rule": "2^(19-requested_bits)*max(abs(output),1); the original fixture has natural normalization one",
        "strict_absolute_budget_exact_rational": strict_absolute_budget.to_string(),
        "production_status": format!("{:?}", result.statuses[0]),
        "diagnostics": workspace.last_diagnostics(),
        "comparison": comparison,
        "execution": {
            "input_rows": stats.input_rows,
            "argument_rows": stats.argument_rows,
            "master_family_batches": stats.master_family_batches,
            "master_rows": stats.master_rows,
            "pentagon_box_rows": stats.pentagon_box_rows,
            "reduction_rows": stats.reduction_rows,
            "fallback_rows": stats.fallback_rows,
            "analytic_rows": stats.analytic_rows,
        },
        "diagnostic_seconds": {
            "plan": plan_seconds,
            "workspace": workspace_seconds,
            "reduction_call": reduction_seconds,
            "arguments": profile.arguments.as_secs_f64(),
            "scalar_masters": profile.scalar_masters.as_secs_f64(),
            "reduction_kernel": profile.reduction.as_secs_f64(),
            "total_process_work": started.elapsed().as_secs_f64(),
        },
        "limits": "One special small-invariant equal-mass fixture, singleton batch, no production threshold changes or automatic precision rescue. Timing is diagnostic, not a performance benchmark. Rejected lanes expose no initialized scratch values.",
    });
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(output) = output {
        std::fs::write(output, &encoded)?;
    }
    print!("{encoded}");
    Ok(exit_code)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
