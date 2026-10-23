//! Unequal masses distinguished beyond binary64, with independent mixed jets.
#[path = "support/near_vacuum_box.rs"]
mod candidate;
#[path = "support/d0_near_vacuum_oracle.rs"]
mod oracle;

use candidate::{AccuracyContext, Exact, NearVacuumPlan, Status, abs, pow2_minus, zero};
use oneloop::NativeFloat;
use serde_json::json;
use std::{error::Error, time::Instant};
use symbolica::{
    domains::{
        float::{Complex, DoubleFloat, Float},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
};

const OFFSETS: [i64; 4] = [-1, 0, 1, 2];
fn exact_row(mass: &Exact, eta: &Rational) -> Vec<Exact> {
    let mut result = Vec::new();
    for coefficient in [0, -2, -2, 0, 0, -2] {
        result.extend([
            Complex::new(eta * &Rational::from(coefficient), Rational::from(0)),
            zero(),
            Complex::new(Rational::from(coefficient), Rational::from(0)),
            zero(),
            zero(),
            zero(),
        ]);
    }
    for coefficient in OFFSETS {
        result.extend([
            Complex::new(
                &mass.re + &(eta * &Rational::from(coefficient)),
                mass.im.clone(),
            ),
            Complex::new(Rational::from(1), Rational::from(0)),
            Complex::new(Rational::from(coefficient), Rational::from(0)),
            zero(),
            zero(),
            zero(),
        ]);
    }
    result
}

fn run<T: NativeFloat>(prototype: T, requested: u32) -> Result<serde_json::Value, Box<dyn Error>>
where
    Complex<T>: EvaluationDomain,
{
    let shape = vec![
        vec![0, 0],
        vec![1, 0],
        vec![0, 1],
        vec![1, 1],
        vec![2, 0],
        vec![0, 2],
    ];
    let plan = NearVacuumPlan::new(4, shape.clone())?;
    let points = [
        (
            Complex::new(Rational::from(2), Rational::from(-1)),
            pow2_minus(64),
        ),
        (
            Complex::new(Rational::from(4), Rational::from(-2)),
            pow2_minus(63),
        ),
        (
            Complex::new(Rational::from(2), Rational::from(0)),
            pow2_minus(64),
        ),
    ];
    let exact: Vec<_> = points.iter().flat_map(|(m, e)| exact_row(m, e)).collect();
    let input: Vec<_> = exact
        .iter()
        .map(|x| {
            Complex::new(
                prototype.from_rational(&x.re),
                prototype.from_rational(&x.im),
            )
        })
        .collect();
    for (original, value) in exact.iter().zip(&input) {
        assert_eq!(Some(&original.re), value.re.exact_rational().as_ref());
        assert_eq!(Some(&original.im), value.im.exact_rational().as_ref());
    }
    for row in input.chunks_exact(60) {
        assert_ne!(row[36].re.exact_rational(), row[42].re.exact_rational());
    }
    let accuracy = AccuracyContext::binary(requested, 6);
    let mut workspace = plan.instantiate(prototype.clone());
    let lanes = workspace.evaluate_batch(&input, 3, &accuracy)?;
    assert_eq!(workspace.execution.active_rows, 3);
    assert_eq!(workspace.execution.numeric_batch_calls, 1);
    let mut rows = Vec::new();
    for (lane, ((mass, eta), result)) in points.iter().zip(lanes).enumerate() {
        assert_eq!(result.status, Status::Success, "unequal mass lane{lane}");
        let reference = oracle::build_unequal_mass_oracle(mass, eta, &shape, 8)?;
        let mut values = Vec::new();
        for (component, value) in result.values.as_ref().unwrap().iter().enumerate() {
            let actual = Complex::new(
                value.re.exact_rational().unwrap(),
                value.im.exact_rational().unwrap(),
            );
            let difference = &actual - &reference.centers[component];
            let radius = &reference.radii[component];
            let error_upper =
                (abs(&difference.re) + radius).pow(2) + (abs(&difference.im) + radius).pow(2);
            let allowed = accuracy.absolute_per_component[component]
                .pow(2)
                .max(accuracy.relative.pow(2) * (actual.re.pow(2) + actual.im.pow(2)));
            assert!(
                error_upper <= allowed,
                "unequal-mass oracle cannot certify component{component}"
            );
            let center = &reference.centers[component];
            let bound = &result.bounds[component];
            assert!(
                bound.real_upper >= &center.re - radius && bound.real_lower <= &center.re + radius
            );
            assert!(
                bound.imaginary_upper >= &center.im - radius
                    && bound.imaginary_lower <= &center.im + radius
            );
            assert!(bound.maximum_error_squared <= bound.allowed_error_squared);
            values.push(json!({"component":component,"real":actual.re.to_string(),"imaginary":actual.im.to_string(),
                "oracle_error_squared_upper":error_upper.to_string(),"allowed_error_squared":allowed.to_string(),
                "certificate_error_squared_upper":bound.maximum_error_squared.to_string()}));
        }
        rows.push(
            json!({"lane":lane,"status":"Success","values":values,"oracle":reference.metadata}),
        );
    }
    Ok(
        json!({"status":"unequal_mass_checks_passed","numeric_type":std::any::type_name::<T>(),"working_bits":prototype.get_precision(),"requested_bits":requested,
        "mass_offsets":OFFSETS,"input_preservation":"All represented values equal exact rational fixtures; all neighboring masses remain distinct; minimum relative separation is beyond53bits.",
        "shape":shape,"rows":rows,"numeric_batch_calls":workspace.execution.numeric_batch_calls,
        "scope":"Validation-only N4 series. Independent N8 oracle expands around common M; candidate freezes first physical mass M-eta. Both include all common-mass and eta jets. No production changes."}),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut args = std::env::args().skip(1);
    let kind = args.next().ok_or("expected dd|float128|float256")?;
    let requested = args.next().ok_or("expected requestedbits")?.parse()?;
    let output = args.next();
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let mut report = match kind.as_str() {
        "dd" => run(DoubleFloat::from(0.), requested)?,
        "float128" => run(Float::new(128), requested)?,
        "float256" => run(Float::new(256), requested)?,
        _ => return Err("expected dd|float128|float256".into()),
    };
    report["diagnostic_seconds"] = json!(started.elapsed().as_secs_f64());
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = output {
        std::fs::write(path, &encoded)?;
    }
    print!("{encoded}");
    Ok(())
}
