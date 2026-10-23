//! Optional qualification of a guarded D0 series; no production dispatch changes.
#[path = "support/near_vacuum_box.rs"]
mod candidate;
#[path = "support/d0_near_vacuum_oracle.rs"]
mod oracle;

use candidate::{AccuracyContext, Exact, NearVacuumPlan, Status, abs, pow2_minus, zero};
use oneloop::{EvaluationBackend, NativeFloat, PrecisionEvaluator, ScalarIntegral};
use serde_json::{Value, json};
use std::{error::Error, path::PathBuf, time::Instant};
use symbolica::{
    domains::{
        float::{Complex, DoubleFloat, Float, FloatLike, SingleFloat},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

#[derive(Clone)]
struct Fixture {
    name: &'static str,
    mass: Exact,
    eta: Rational,
    valid: bool,
}
fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            name: "tiny",
            mass: Complex::new(Rational::from(2), Rational::from(-1)),
            eta: pow2_minus(64),
            valid: true,
        },
        Fixture {
            name: "tiny_scaled",
            mass: Complex::new(Rational::from(4), Rational::from(-2)),
            eta: pow2_minus(63),
            valid: true,
        },
        Fixture {
            name: "dns_overlap",
            mass: Complex::new(Rational::from(2), Rational::from(-1)),
            eta: pow2_minus(12),
            valid: true,
        },
        Fixture {
            name: "vacuum_with_jets",
            mass: Complex::new(Rational::from(2), Rational::from(-1)),
            eta: Rational::from(0),
            valid: true,
        },
        Fixture {
            name: "outside_radius",
            mass: Complex::new(Rational::from(2), Rational::from(-1)),
            eta: Rational::from(2),
            valid: false,
        },
        Fixture {
            name: "upper_half_plane",
            mass: Complex::new(Rational::from(2), Rational::from(1)),
            eta: pow2_minus(64),
            valid: false,
        },
        Fixture {
            name: "tiny_other_mass",
            mass: Complex::new(Rational::from(3), Rational::from((-1, 2))),
            eta: pow2_minus(64),
            valid: true,
        },
    ]
}
fn exact_row(point: &Fixture, shape: &[Vec<usize>]) -> Vec<Exact> {
    let mut row = Vec::new();
    for coefficient in [0, -2, -2, 0, 0, -2] {
        for component in shape {
            let value = if component.as_slice() == [0, 0] {
                &point.eta * &Rational::from(coefficient)
            } else if component.as_slice() == [0, 1] {
                Rational::from(coefficient)
            } else {
                Rational::from(0)
            };
            row.push(Complex::new(value, Rational::from(0)));
        }
    }
    for _ in 0..4 {
        for component in shape {
            row.push(if component.as_slice() == [0, 0] {
                point.mass.clone()
            } else if component.as_slice() == [1, 0] {
                Complex::new(Rational::from(1), Rational::from(0))
            } else {
                zero()
            });
        }
    }
    row
}
fn map<T: NativeFloat>(point: &[Exact], prototype: &T) -> Vec<Complex<T>> {
    point
        .iter()
        .map(|v| {
            Complex::new(
                prototype.from_rational(&v.re),
                prototype.from_rational(&v.im),
            )
        })
        .collect()
}
fn bound_json(bound: &candidate::ComponentBound) -> Value {
    json!({"truncation_upper":bound.truncation.to_string(),"real_interval":[bound.real_lower.to_string(),bound.real_upper.to_string()],
        "imaginary_interval":[bound.imaginary_lower.to_string(),bound.imaginary_upper.to_string()],
        "maximum_error_squared":bound.maximum_error_squared.to_string(),"allowed_error_squared":bound.allowed_error_squared.to_string()})
}

fn run_numeric<T: NativeFloat>(
    plan: &NearVacuumPlan,
    prototype: T,
    requested: u32,
) -> Result<Value, Box<dyn Error>>
where
    Complex<T>: EvaluationDomain,
{
    let points = fixtures();
    let c = plan.shape.len();
    let exact: Vec<_> = points
        .iter()
        .flat_map(|point| exact_row(point, &plan.shape))
        .collect();
    let input = map(&exact, &prototype);
    // These dyadic fixtures are exactly represented by every selected domain.
    for (original, represented) in exact.iter().zip(&input) {
        assert_eq!(represented.re.exact_rational().as_ref(), Some(&original.re));
        assert_eq!(represented.im.exact_rational().as_ref(), Some(&original.im));
    }
    let mut workspace = plan.instantiate(prototype.clone());
    let accuracy = AccuracyContext::binary(requested, c);
    let mut empty = workspace.evaluate_batch(&[], 0, &accuracy)?;
    assert!(empty.is_empty());
    empty.clear();
    eprintln!(
        "near-vacuum N{}: evaluating odd batch7, components{}, requested{}",
        plan.order, c, requested
    );
    let started = Instant::now();
    let lanes = workspace.evaluate_batch(&input, points.len(), &accuracy)?;
    let evaluation_seconds = started.elapsed().as_secs_f64();
    let execution = workspace.execution.clone();
    for (point, lane) in points.iter().zip(&lanes) {
        eprintln!(
            "near-vacuum numeric lane {}: {:?}, {} component bounds",
            point.name,
            lane.status,
            lane.bounds.len()
        );
    }
    assert_eq!(execution.active_rows, 5);
    assert_eq!(execution.numeric_batch_calls, 1);
    assert_eq!(execution.enclosure_batch_calls, 1);
    assert_eq!(execution.tail_batch_calls, 1);
    let mut reports = Vec::new();
    let mut success_count = 0;
    for (index, (point, lane)) in points.iter().zip(&lanes).enumerate() {
        if !point.valid {
            assert_ne!(lane.status, Status::Success);
            assert!(lane.values.is_none());
        }
        let reference = if point.valid {
            eprintln!("near-vacuum independent oracle: {}", point.name);
            Some(oracle::build_oracle(
                &point.mass,
                &point.eta,
                &plan.shape,
                8,
            )?)
        } else {
            None
        };
        let mut comparisons = Vec::new();
        if let Some(values) = &lane.values {
            assert_eq!(lane.status, Status::Success);
            success_count += 1;
            let reference = reference.as_ref().unwrap();
            for (component, value) in values.iter().enumerate() {
                let actual = Complex::new(
                    value.re.exact_rational().unwrap(),
                    value.im.exact_rational().unwrap(),
                );
                let error = &actual - &reference.centers[component];
                let radius = &reference.radii[component];
                let upper_squared =
                    (abs(&error.re) + radius).pow(2) + (abs(&error.im) + radius).pow(2);
                let lower_squared = (abs(&error.re) - radius).max(Rational::from(0)).pow(2)
                    + (abs(&error.im) - radius).max(Rational::from(0)).pow(2);
                let allowed = accuracy.absolute_per_component[component]
                    .pow(2)
                    .max(accuracy.relative.pow(2) * (actual.re.pow(2) + actual.im.pow(2)));
                assert!(
                    upper_squared <= allowed,
                    "independent oracle cannot certify agreement on {} component{}",
                    point.name,
                    component
                );
                // Enclosures from independent truncations must overlap in
                // both Cartesian coordinates; neither is treated as exact.
                let bound = &lane.bounds[component];
                let center = &reference.centers[component];
                assert!(
                    bound.real_upper >= &center.re - radius
                        && bound.real_lower <= &center.re + radius
                );
                assert!(
                    bound.imaginary_upper >= &center.im - radius
                        && bound.imaginary_lower <= &center.im + radius
                );
                comparisons.push(json!({"component":component,"actual_re":actual.re.to_string(),"actual_im":actual.im.to_string(),
                    "oracle_re":center.re.to_string(),"oracle_im":center.im.to_string(),"oracle_radius":radius.to_string(),
                    "certified_error_squared_upper":upper_squared.to_string(),"certified_error_squared_lower":lower_squared.to_string(),"allowed_squared":allowed.to_string(),
                    "tracked_re":value.re.get_precision(),"tracked_im":value.im.get_precision()}));
            }
        } else {
            assert_ne!(lane.status, Status::Success);
        }
        reports.push(json!({"row":index,"name":point.name,"status":format!("{:?}",lane.status),
            "rho":lane.rho.as_ref().map(ToString::to_string),"bounds":lane.bounds.iter().map(bound_json).collect::<Vec<_>>(),
            "values":lane.values.as_ref().map(|_|comparisons),"oracle":reference.map(|r|r.metadata)}));
    }
    // Primal accuracy cannot authorize a component omitted by a short series.
    if plan.order == 2 && c > 1 && requested >= 100 {
        assert!(matches!(lanes[0].status, Status::UncertifiedComponent(_)));
        assert!(lanes[0].values.is_none());
    }
    if (plan.order == 4 || c == 1)
        && requested <= 128
        && requested.saturating_add(12) <= prototype.get_precision()
    {
        assert_eq!(lanes[0].status, Status::Success);
        assert_eq!(lanes[1].status, Status::Success);
        assert_eq!(lanes[3].status, Status::Success);
        assert_eq!(lanes[6].status, Status::Success);
    }
    // Reordered and singleton calls must retain both results and rejected lanes.
    let order = [6usize, 1, 5, 0, 4, 3, 2];
    let width = candidate::PHYSICAL_ARGUMENTS * c;
    let reordered: Vec<_> = order
        .iter()
        .flat_map(|&i| input[i * width..(i + 1) * width].iter().cloned())
        .collect();
    let reordered = workspace.evaluate_batch(&reordered, order.len(), &accuracy)?;
    for (result, &original) in reordered.iter().zip(&order) {
        assert_eq!(result.status, lanes[original].status);
        if let (Some(a), Some(b)) = (&result.values, &lanes[original].values) {
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.re.exact_rational(), b.re.exact_rational());
                assert_eq!(a.im.exact_rational(), b.im.exact_rational());
            }
        }
    }
    let singleton = workspace.evaluate_batch(&input[..width], 1, &accuracy)?;
    assert_eq!(singleton[0].status, lanes[0].status);
    // An unattainable request must not become a successful initialized zero.
    let impossible = workspace.evaluate_batch(
        &input[..width],
        1,
        &AccuracyContext::binary(prototype.get_precision() + 64, c),
    )?;
    assert_ne!(impossible[0].status, Status::Success);
    assert!(impossible[0].values.is_none());
    // Safe overlap uses an explicitly looser context, recorded separately.
    let loose = AccuracyContext::binary(24, c);
    let overlap = workspace.evaluate_batch(&input[2 * width..3 * width], 1, &loose)?;
    if plan.order == 4 {
        assert_eq!(overlap[0].status, Status::Success);
    }
    let mut overlap_comparisons = Vec::new();
    let oracle_overlap = oracle::build_oracle(&points[2].mass, &points[2].eta, &plan.shape, 8)?;
    if let Some(values) = &overlap[0].values {
        for (i, v) in values.iter().enumerate() {
            let e = Complex::new(
                v.re.exact_rational().unwrap(),
                v.im.exact_rational().unwrap(),
            ) - &oracle_overlap.centers[i];
            let r = &oracle_overlap.radii[i];
            let bound = (abs(&e.re) + r).pow(2) + (abs(&e.im) + r).pow(2);
            assert!(bound <= pow2_minus(24).pow(2));
            overlap_comparisons
                .push(json!({"component":i,"error_squared_upper":bound.to_string()}));
        }
    }
    Ok(
        json!({"numeric_type":std::any::type_name::<T>(),"working_bits":prototype.get_precision(),"requested_bits":requested,
        "successful_rows":success_count,"lanes":reports,"diagnostic_evaluation_seconds":evaluation_seconds,
        "batch_execution":{"input_rows":execution.input_rows,"active_rows":execution.active_rows,"numeric_batch_calls":execution.numeric_batch_calls,
          "enclosure_batch_calls":execution.enclosure_batch_calls,"tail_batch_calls":execution.tail_batch_calls,
          "estimated_exported_instruction_lane_visits":execution.estimated_exported_instruction_lane_visits,
          "method":"Symbolica BatchExpressionEvaluator executes each instruction across active lanes"},
        "empty_reorder_singleton_checks":"passed","impossible_accuracy_rejected":true,
        "overlap_requested_bits":24,"overlap_status":format!("{:?}",overlap[0].status),"overlap_components":overlap_comparisons}),
    )
}

fn dns_overlap() -> Result<Value, Box<dyn Error>> {
    let point = &fixtures()[2];
    let shape = vec![vec![0, 0]];
    let mut exact = exact_row(point, &shape);
    exact.push(Complex::new(Rational::from(1), Rational::from(0)));
    let prototype = Float::new(256);
    let input = map(&exact, &prototype);
    let mut evaluator = PrecisionEvaluator::with_binary_precision_and_backend(
        ScalarIntegral::D0,
        256,
        EvaluationBackend::Expression,
    )?;
    let mut values =
        std::array::from_fn::<_, 3, _>(|_| Complex::new(prototype.zero(), prototype.zero()));
    evaluator.evaluate(&input, &mut values)?;
    let reference = oracle::build_oracle(&point.mass, &point.eta, &shape, 8)?;
    let actual = Complex::new(
        values[0]
            .re
            .exact_rational()
            .ok_or("DNS overlap real nonfinite")?,
        values[0]
            .im
            .exact_rational()
            .ok_or("DNS overlap imag nonfinite")?,
    );
    let error = &actual - &reference.centers[0];
    let r = &reference.radii[0];
    let upper = (abs(&error.re) + r).pow(2) + (abs(&error.im) + r).pow(2);
    assert!(upper <= pow2_minus(24).pow(2), "safe DNS overlap disagrees");
    assert!(values[1..].iter().all(|v| v.re.is_zero() && v.im.is_zero()));
    Ok(
        json!({"eta":point.eta.to_string(),"working_bits":256,"absolute_comparison_bits":24,"re":actual.re.to_string(),"im":actual.im.to_string(),"oracle_error_squared_upper":upper.to_string(),"tracked_re":values[0].re.get_precision(),"tracked_im":values[0].im.get_precision()}),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut args = std::env::args().skip(1);
    let usage = "usage: near_vacuum_validate order(2|4) shape(primal|mixed) numeric(f64|dd|float128|float256) requested_bits [output.json]";
    let order = args.next().ok_or(usage)?.parse()?;
    let shape_name = args.next().ok_or(usage)?;
    let kind = args.next().ok_or(usage)?;
    let requested = args.next().ok_or(usage)?.parse()?;
    let output = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err(usage.into());
    }
    let shape = match shape_name.as_str() {
        "primal" => vec![vec![0, 0]],
        "mixed" => vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![1, 1],
            vec![2, 0],
            vec![0, 2],
        ],
        _ => return Err(usage.into()),
    };
    eprintln!("near-vacuum preparing generic N{order} program, shape={shape_name}");
    let prepare = Instant::now();
    let plan = NearVacuumPlan::new(order, shape)?;
    let preparation_seconds = prepare.elapsed().as_secs_f64();
    eprintln!(
        "near-vacuum preparation complete {preparation_seconds:.6}s, {} instructions",
        plan.instructions
    );
    let numeric = match kind.as_str() {
        "f64" => run_numeric(&plan, 0.0f64, requested)?,
        "dd" => run_numeric(&plan, DoubleFloat::from(0.0), requested)?,
        "float128" => run_numeric(&plan, Float::new(128), requested)?,
        "float256" => run_numeric(&plan, Float::new(256), requested)?,
        _ => return Err(usage.into()),
    };
    let dns = dns_overlap()?;
    let report = json!({"status":"validation_candidate_checks_passed","order":order,"shape":plan.shape,"numeric":numeric,"ordinary_dns_overlap":dns,
        "diagnostic_seconds":{"preparation":preparation_seconds,"total":started.elapsed().as_secs_f64()},
        "scope":"Validation-only N2/N4 finite box series. Production masters/reduction untouched. Every successful component certified by existing Symbolica ball arithmetic plus full-jet truncation majorant; independently checked against order8 parameter oracle. DNS overlap at explicitly24-bit absolute budget. Timings diagnostic only."});
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = output {
        std::fs::write(path, &encoded)?;
    }
    print!("{encoded}");
    Ok(())
}
