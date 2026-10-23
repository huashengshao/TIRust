//! Exact vacuum qualification with heterogeneous higher and mixed input jets.
#[path = "support/near_vacuum_box.rs"]
mod candidate;
#[path = "support/d0_near_vacuum_oracle.rs"]
mod oracle;

use candidate::{AccuracyContext, Exact, NearVacuumPlan, Status, abs, zero};
use oneloop::NativeFloat;
use serde_json::json;
use std::{error::Error, time::Instant};
use symbolica::domains::{
    dual::{DualNumber, DualNumberStructure, HyperDual},
    float::{Complex, Float, SingleFloat},
    rational::Rational,
};

symbolica::create_hyperdual_from_components!(
    StaticJets,
    [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [3, 0]]
);

fn fixture(mass: &Exact) -> Vec<Exact> {
    let mut input = Vec::new();
    for argument in 0..10i64 {
        let complex_jet = if argument >= 6 && mass.im != Rational::from(0) {
            Rational::from((argument - 5, 32))
        } else {
            Rational::from(0)
        };
        input.extend([
            if argument < 6 { zero() } else { mass.clone() },
            Complex::new(Rational::from((argument + 1, 16)), complex_jet.clone()),
            Complex::new(Rational::from((2 - argument, 32)), -complex_jet.clone()),
            Complex::new(
                Rational::from((argument - 3, 64)),
                complex_jet / Rational::from(4),
            ),
            Complex::new(Rational::from((argument + 2, 128)), Rational::from(0)),
            Complex::new(Rational::from((-argument - 1, 128)), Rational::from(0)),
            Complex::new(Rational::from((argument - 4, 256)), Rational::from(0)),
        ]);
    }
    input
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut args = std::env::args().skip(1);
    let order: usize = args.next().ok_or("expected order2 or4")?.parse()?;
    let output = args.next();
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let static_shape = StaticJets::<Rational> {
        values: std::array::from_fn(|_| Rational::from(0)),
    };
    let shape: Vec<_> = static_shape
        .get_shape()
        .iter()
        .map(|x| x.to_vec())
        .collect();
    let dynamic_shape = HyperDual::<Rational>::new(shape.clone());
    assert_eq!(static_shape.get_shape(), dynamic_shape.get_shape());
    eprintln!("heterogeneous vacuum: preparing N{order} static-shape plan");
    let plan = NearVacuumPlan::from_dual(order, &static_shape)?;
    let prototype = Float::new(256);
    let masses = [
        Complex::new(Rational::from(2), Rational::from(-1)),
        Complex::new(Rational::from(4), Rational::from(-2)),
        Complex::new(Rational::from(2), Rational::from(0)),
    ];
    let fixtures: Vec<_> = masses.iter().map(fixture).collect();
    let input: Vec<_> = fixtures
        .iter()
        .flatten()
        .map(|v| {
            Complex::new(
                prototype.from_rational(&v.re),
                prototype.from_rational(&v.im),
            )
        })
        .collect();
    // Use actual static and dynamic carriers as component containers; neither
    // carrier's derivative component is ever counted as a sample lane.
    for row in &fixtures {
        for components in row.chunks_exact(shape.len()) {
            let carrier = StaticJets {
                values: std::array::from_fn(|i| components[i].clone()),
            };
            let mut dynamic = HyperDual::<Exact>::new(shape.clone());
            dynamic.values_mut().clone_from_slice(carrier.values());
            assert_eq!(carrier.values(), dynamic.values());
        }
    }
    let mut workspace = plan.instantiate(prototype);
    let accuracy = AccuracyContext::binary(128, shape.len());
    eprintln!("heterogeneous vacuum: evaluating three lanes and exact derivative oracles");
    let results = workspace.evaluate_batch(&input, masses.len(), &accuracy)?;
    assert_eq!(workspace.execution.active_rows, 3);
    assert_eq!(workspace.execution.numeric_batch_calls, 1);
    let mut rows = Vec::new();
    for (lane, ((mass, fixture), result)) in masses.iter().zip(&fixtures).zip(results).enumerate() {
        let reference = oracle::build_vacuum_jet_oracle(mass, &shape, fixture)?;
        assert!(reference.radii.iter().all(|r| *r == Rational::from(0)));
        let mut values = Vec::new();
        if order == 2 {
            assert_eq!(result.status, Status::UncertifiedComponent(6));
            assert!(result.values.is_none());
            assert!(result.bounds[6].truncation > Rational::from(0));
            assert!(
                result.bounds[..6]
                    .iter()
                    .all(|b| b.truncation == Rational::from(0))
            );
        } else {
            assert_eq!(result.status, Status::Success);
            for (component, value) in result.values.as_ref().unwrap().iter().enumerate() {
                let actual = Complex::new(
                    value.re.exact_rational().unwrap(),
                    value.im.exact_rational().unwrap(),
                );
                let center = &reference.centers[component];
                let error = &actual - center;
                let squared = error.re.pow(2) + error.im.pow(2);
                let allowed = accuracy.absolute_per_component[component]
                    .pow(2)
                    .max(accuracy.relative.pow(2) * (actual.re.pow(2) + actual.im.pow(2)));
                assert!(squared <= allowed);
                let bound = &result.bounds[component];
                assert!(bound.real_lower <= center.re && bound.real_upper >= center.re);
                assert!(bound.imaginary_lower <= center.im && bound.imaginary_upper >= center.im);
                assert_eq!(bound.truncation, Rational::from(0));
                values.push(json!({"component":component,"real":actual.re.to_string(),"imaginary":actual.im.to_string(),"exact_error_squared":squared.to_string(),"allowed_error_squared":allowed.to_string()}));
            }
        }
        // Preserve evidence for the omitted degree-three coefficient even when
        // N2 deliberately exposes no candidate output values.
        let third = &reference.centers[6];
        assert!(abs(&third.re) + abs(&third.im) > Rational::from(0));
        rows.push(json!({"lane":lane,"status":format!("{:?}",result.status),"values":if result.values.is_some(){Some(values)}else{None},
            "tail_bounds":result.bounds.iter().map(|b|b.truncation.to_string()).collect::<Vec<_>>(),"oracle":reference.metadata}));
    }
    let report = json!({"status":"heterogeneous_vacuum_checks_passed","order":order,"working_bits":256,"requested_bits":128,"shape":shape,
        "rows":rows,"static_dynamic_component_container_equivalence":true,"batch_rows":3,"numeric_batch_calls":workspace.execution.numeric_batch_calls,
        "scope":"Validation only; common primal masses, heterogeneous complex mass jets and real invariant jets through mixed degree2 and pure degree3. N2 must reject omitted third derivative at exact vacuum; N4 must enclose independent exact differentiated-integrand oracle. No production dispatch change.",
        "diagnostic_seconds":started.elapsed().as_secs_f64()});
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = output {
        std::fs::write(path, &encoded)?;
    }
    print!("{encoded}");
    Ok(())
}
