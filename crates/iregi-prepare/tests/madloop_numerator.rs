//! Independent HELAS numerator comparison, before integration or counterterms.
use std::collections::HashMap;

use feynkit_graph::symbols;
use iregi_prepare::{NumeratorSource, PrepareOptions, prepare_dot};
use serde_json::{Value, json};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, Float},
    function, symbol,
};

fn number(value: &Value) -> f64 {
    value.as_str().unwrap().parse().unwrap()
}

fn complex(value: &Value) -> Complex<f64> {
    Complex::new(number(&value[0]), number(&value[1]))
}

fn close(actual: Complex<f64>, expected: Complex<f64>) {
    let error = (actual.re - expected.re).hypot(actual.im - expected.im);
    let scale = expected.re.hypot(expected.im).max(1.0);
    assert!(
        error < 2e-11 * scale,
        "actual={actual:?}, expected={expected:?}, relative error={}",
        error / scale
    );
}

#[test]
fn feynkit_top_triangle_matches_independent_madloop_numerator() {
    check_numerator(
        "ggh",
        include_str!("../../../validation/physical/ggh-numerator-reference.json"),
        include_str!("../../../examples/graphs/ggh_top_triangle.dot"),
        2,
        -0.5,
    );
}

#[test]
fn feynkit_top_box_matches_independent_madloop_numerator() {
    check_numerator(
        "gghh",
        include_str!("../../../validation/physical/madloop/fixtures/gghh-helicity-1.json"),
        include_str!("../../../examples/graphs/gghh_top_box.dot"),
        2,
        -0.5,
    );
}

#[test]
fn feynkit_top_pentagon_matches_independent_madloop_numerator() {
    check_numerator(
        "gghhh",
        include_str!("../../../validation/physical/madloop/fixtures/gghhh-helicity-1.json"),
        include_str!("../../../examples/graphs/gghhh_top_pentagon.dot"),
        2,
        -0.5,
    );
}

#[test]
fn feynkit_massless_up_box_matches_independent_madloop_numerator() {
    check_numerator(
        "gggg-up",
        include_str!("../../../validation/physical/madloop/fixtures/gggg-up-helicity-1.json"),
        include_str!("../../../examples/graphs/gggg_up_box.dot"),
        4,
        -0.125,
    );
}

fn check_numerator(
    case: &str,
    reference_json: &str,
    dot: &str,
    color_axes: usize,
    color_factor: f64,
) {
    let reference: Value = serde_json::from_str(reference_json).unwrap();
    let record = reference.get("record").unwrap_or_else(|| {
        reference["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|record| record["group"] == reference["selected_feynkit_group"])
            .unwrap()
    });
    let prepared = prepare_dot(
        include_str!("../../../fixtures/models/sm.json"),
        dot,
        &PrepareOptions {
            source: NumeratorSource::FeynmanRules,
            external_wavefunctions: true,
            component: Some(vec![0; color_axes]),
        },
    )
    .unwrap();
    let metadata = &reference["metadata"];
    // MadLoop may project or perturb the requested point before evaluating it.
    let external_momenta = metadata
        .get("evaluated_external_momenta")
        .unwrap_or(&metadata["external_momenta"]);
    let mut bindings = HashMap::new();
    bindings.insert(Atom::var(symbol!("UFO::ZERO")), Complex::new(0.0, 0.0));
    if case == "gggg-up" {
        // The reference model restricts the up quark to be massless.
        bindings.insert(Atom::var(symbol!("UFO::MU")), Complex::new(0.0, 0.0));
    }
    bindings.insert(
        Atom::var(symbol!("UFO::MT")),
        Complex::new(number(&metadata["parameters"][0]), 0.0),
    );
    bindings.insert(
        Atom::var(symbol!("UFO::G")),
        Complex::new(number(&metadata["parameters"][2]), 0.0),
    );
    // The two models encode the same Yukawa vertex as -i*yt/sqrt(2).
    if let Some(yukawa) = metadata["couplings"].get(3) {
        let value = (-Atom::num(number(yukawa)) * symbolica::parse!("sqrt(2)"))
            .evaluate(&HashMap::<Atom, Complex<f64>>::new())
            .unwrap();
        bindings.insert(Atom::var(symbol!("UFO::yt")), value);
    }
    for (basis, edge) in prepared.external_edges.iter().enumerate() {
        let leg = prepared
            .external_legs
            .iter()
            .find(|leg| leg.edge_id == *edge)
            .unwrap();
        let physical_index: usize = leg
            .name
            .strip_prefix('p')
            .unwrap()
            .parse::<usize>()
            .unwrap()
            - 1;
        for mu in 0..4 {
            let component = function!(symbol!("spenso::cind"), Atom::num(mu));
            bindings.insert(
                function!(symbols::external_momentum(), Atom::num(basis), component),
                Complex::new(number(&external_momenta[physical_index][mu]), 0.0),
            );
        }
    }
    for leg in prepared
        .external_legs
        .iter()
        .filter(|leg| leg.particle_pdg == 21)
    {
        let physical_index = leg.name.strip_prefix('p').unwrap();
        for mu in 0..4 {
            let component = function!(symbol!("spenso::cind"), Atom::num(mu));
            let value = complex(&metadata["wavefunctions"][format!("{physical_index}_{}", mu + 1)]);
            bindings.insert(
                function!(
                    if leg.incoming {
                        symbols::epsilon()
                    } else {
                        symbols::epsilonbar()
                    },
                    Atom::num(leg.edge_id),
                    component
                ),
                value,
            );
        }
    }
    let output_path = std::env::var("IREGI_PHYSICAL_INPUT_DIR")
        .ok()
        .map(|directory| std::path::PathBuf::from(directory).join(format!("{case}-cli-input.json")))
        .or_else(|| {
            if case == "ggh" {
                std::env::var("IREGI_GGH_CLI_INPUT").ok().map(Into::into)
            } else {
                None
            }
        });
    if let Some(path) = output_path {
        let parameters: serde_json::Map<String, Value> = bindings
            .iter()
            .map(|(symbol, value)| {
                (
                    symbol.to_canonical_string(),
                    json!([format!("{:.17e}", value.re), format!("{:.17e}", value.im)]),
                )
            })
            .collect();
        let propagators: Vec<Value> = prepared.propagators.iter().map(|propagator| {
            let shift: [String; 4] = std::array::from_fn(|mu| {
                let expression: Atom = propagator.external_coefficients.iter().enumerate().map(|(basis, coefficient)| {
                    Atom::num(i64::from(propagator.loop_coefficient) * i64::from(*coefficient))
                        * function!(symbols::external_momentum(), Atom::num(basis), function!(symbol!("spenso::cind"), Atom::num(mu)))
                }).sum();
                let value = expression.evaluate(&bindings).unwrap();
                assert_eq!(value.im, 0.0);
                format!("{:.17e}", value.re)
            });
            let mass = propagator.mass_squared.evaluate(&bindings).unwrap();
            json!({"edge_id":propagator.edge_id,"shift":shift,"mass_squared":[format!("{:.17e}",mass.re),format!("{:.17e}",mass.im)],"power":1})
        }).collect();
        let input = json!({"id":format!("{case}_helicity_1_equal_color_0"),"mu":record["mu"],"rank":prepared.numerator.rank(),"propagators":propagators,"parameters":parameters});
        std::fs::write(path, serde_json::to_string_pretty(&input).unwrap() + "\n").unwrap();
    }
    let x: [Atom; 4] =
        std::array::from_fn(|i| Atom::var(symbol!(&format!("iregi_validation::x{i}"))));
    let ml_polynomial: Atom = reference["coefficient_indices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(record["numerator"].as_array().unwrap())
        .map(|(indices, coefficient)| {
            let coefficient = complex(coefficient);
            indices.as_array().unwrap().iter().fold(
                Atom::num(coefficient.re) + Atom::i() * Atom::num(coefficient.im),
                |term, index| term * &x[index.as_u64().unwrap() as usize],
            )
        })
        .sum();
    let probes: Vec<[Complex<f64>; 4]> = vec![
        std::array::from_fn(|i| complex(&reference["probe_momentum"][i])),
        [
            Complex::new(11.0, 2.0),
            Complex::new(17.0, -3.0),
            Complex::new(-29.0, 5.0),
            Complex::new(31.0, 7.0),
        ],
        [
            Complex::new(13.0, 0.0),
            Complex::new(-7.0, 0.0),
            Complex::new(19.0, 0.0),
            Complex::new(5.0, 0.0),
        ],
    ];
    let expression = prepared.numerator.expression();
    for (probe_index, probe) in probes.into_iter().enumerate() {
        let mut reference_bindings = HashMap::new();
        for mu in 0..4 {
            reference_bindings.insert(x[mu].clone(), probe[mu]);
            let p1 = Complex::new(number(&external_momenta[0][mu]), 0.0);
            // MadLoop polynomial variable x=p1-q, with q on the FeynKit a->b edge.
            bindings.insert(prepared.numerator.variables[mu].clone(), p1 - probe[mu]);
        }
        let expected = ml_polynomial.evaluate(&reference_bindings).unwrap();
        if probe_index == 0 {
            close(expected, complex(&record["probe_value"]));
        }
        // Use 128-bit Symbolica evaluation so this numerator comparison is
        // insensitive to binary64 summation order in expanded expressions.
        let precise_bindings: HashMap<_, _> = bindings
            .iter()
            .map(|(atom, value)| {
                (
                    atom.clone(),
                    Complex::new(
                        Float::with_val(128, value.re),
                        Float::with_val(128, value.im),
                    ),
                )
            })
            .collect();
        let actual = expression
            .evaluate_with_prec(&precise_bindings, 128)
            .map(|value| value.to_f64())
            .unwrap_or_else(|error| panic!("{error}: {expression}"));
        // Closed fermion sign times the fixed gluon-color trace.
        close(actual, expected * Complex::new(color_factor, 0.0));
    }
}
