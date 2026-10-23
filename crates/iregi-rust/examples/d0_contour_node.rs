//! Isolated DNS-box diagnostic against an independent exact parameter series.
//! Compile separately against the baseline and private helper-only experiment.

use oneloop::{EvaluationBackend, NativeFloat, PrecisionEvaluator, ScalarIntegral};
use serde_json::json;
use std::{error::Error, path::PathBuf, sync::Arc, time::Instant};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        float::{Complex, Float, FloatLike, SingleFloat},
        integer::Integer,
        rational::{Q, Rational},
    },
    id::Replacement,
    poly::PolyVariable,
    tensors::matrix::Matrix,
};

fn fraction(bits: u64) -> Rational {
    Rational::from((Integer::from(1), Integer::from(2).pow(bits)))
}

fn field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: None,
    }
}

fn oracle() -> Result<(Complex<Rational>, Rational, serde_json::Value), Box<dyn Error>> {
    let z_symbol = symbolica::symbol!("iregi_d0_node::z");
    let z = Atom::var(z_symbol);
    let x: Vec<_> = (0..4)
        .map(|i| Atom::var(symbolica::symbol!(format!("iregi_d0_node::x{i}"))))
        .collect();
    let u_symbols: Vec<_> = (0..3)
        .map(|i| symbolica::symbol!(format!("iregi_d0_node::u{i}")))
        .collect();
    let u: Vec<_> = u_symbols.iter().copied().map(Atom::var).collect();

    // Retain q1..q4 from the original pentagon; pinch q0. Symbolica computes
    // every unit squared distance and the independent Symanzik polynomial.
    let routing: Vec<[Atom; 4]> = (0..4)
        .map(|i| std::array::from_fn(|mu| Atom::num(i64::from(i == mu))))
        .collect();
    let metric = [1, -1, -1, -1];
    let mut p = Atom::num(0);
    for i in 0..4 {
        for j in i + 1..4 {
            let distance: Atom = (0..4)
                .map(|mu| Atom::num(metric[mu]) * (&routing[i][mu] - &routing[j][mu]).pow(2))
                .sum();
            p -= &x[i] * &x[j] * distance;
        }
    }
    let expected = Atom::num(2) * (&x[1] * &x[2] + &x[1] * &x[3] + &x[2] * &x[3]);
    assert!((&p - expected).expand().is_zero());
    let mut remaining = Atom::num(1);
    let mut simplex = Vec::new();
    for coordinate in &u {
        simplex.push(&remaining * coordinate);
        remaining *= Atom::num(1) - coordinate;
    }
    simplex.push(remaining);
    let jacobian = Matrix::from_nested_vec(
        simplex[..3]
            .iter()
            .map(|coordinate| {
                u_symbols
                    .iter()
                    .map(|&symbol| coordinate.derivative(symbol))
                    .collect()
            })
            .collect(),
        field(),
    )
    .map_err(|error| format!("Jacobian construction: {error:?}"))?
    .det()
    .map_err(|error| format!("Jacobian determinant: {error:?}"))?;
    let replacements: Vec<_> = x
        .iter()
        .zip(&simplex)
        .map(|(x, value)| Replacement::new(x.clone(), value.clone()))
        .collect();
    let polynomial = p.replace_multiple(replacements.iter());
    let series = (Atom::num(1) / (Atom::num(1) + &z * polynomial).pow(2))
        .series(z_symbol, 0, 4)?
        .to_atom();
    let variables = Arc::new(
        std::iter::once(z.clone())
            .chain(u)
            .map(PolyVariable::try_from)
            .collect::<Result<Vec<_>, _>>()?,
    );
    let mut integrated = (jacobian * series)
        .expand()
        .try_to_polynomial::<_, u16>(&Q, Some(variables))?;
    for variable in 1..4 {
        let antiderivative = integrated.integrate(variable);
        integrated = &antiderivative.replace(variable, &Rational::from(1))
            - &antiderivative.replace(variable, &Rational::from(0));
    }
    assert_eq!(integrated.get_constant(), Rational::from((1, 6)));
    assert_eq!(
        integrated.coefficient(&[1, 0, 0, 0]),
        Some(Rational::from((-1, 10)))
    );
    assert!((1..4).all(|i| integrated.degree(i) == 0));
    let eta = fraction(64);
    let mass = Complex::new(Rational::from(2), Rational::from(-1));
    let zero = Complex::new(Rational::from(0), Rational::from(0));
    let ratio = Complex::new(eta.clone(), Rational::from(0)) / &mass;
    let center = integrated.evaluate(
        |value| Complex::new(value.clone(), Rational::from(0)),
        &[ratio, zero.clone(), zero.clone(), zero],
    ) / mass.pow(2);

    // 0<=P<=1, |M|>=2 and |M|^2=5. The tail of (1-r)^-2,
    // divided by the simplex volume denominator6 and |M|^2, encloses the
    // complex error. Symbolica verifies the generating-function identity.
    let generating = Atom::num(1) / (Atom::num(1) - &z).pow(2);
    let tail = &generating - generating.series(z_symbol, 0, 4)?.to_atom();
    let closed = z.pow(5) * (Atom::num(6) - Atom::num(5) * &z) / (Atom::num(1) - &z).pow(2);
    assert!((tail - &closed).together().is_zero());
    let radius_atom = (closed / Atom::num(30))
        .replace_multiple([Replacement::new(z, Atom::num(&eta / &Rational::from(2)))].iter());
    let radius_poly = radius_atom.try_to_polynomial::<_, u16>(&Q, None)?;
    assert!(radius_poly.is_constant());
    let radius = radius_poly.get_constant();
    assert!(radius > Rational::from(0) && radius < fraction(327));
    let metadata = json!({
        "method": "Independent Symanzik parameter series, Symbolica expansion and polynomial integration",
        "series_order_inclusive": 4,
        "integrated_series_in_z": integrated.to_expression().to_string(),
        "z": "eta/M",
        "center_real_exact_rational": center.re.to_string(),
        "center_imaginary_exact_rational": center.im.to_string(),
        "absolute_complex_remainder_bound_exact_rational": radius.to_string(),
        "remainder_proof": "0<=P<=1, r=eta/2, |M|^2=5: radius=r^5*(6-5*r)/(30*(1-r)^2)<2^-327",
    });
    Ok((center, radius, metadata))
}

fn run() -> Result<u8, Box<dyn Error>> {
    let started = Instant::now();
    let mut args = std::env::args().skip(1);
    let label = args
        .next()
        .ok_or("usage: d0_contour_node label [output.json]")?;
    let output = args.next().map(PathBuf::from);
    if args.next().is_some() {
        return Err("usage: d0_contour_node label [output.json]".into());
    }
    eprintln!("D0 node {label}: generating independent exact series");
    let oracle_started = Instant::now();
    let (center, radius, reference) = oracle()?;
    let oracle_seconds = oracle_started.elapsed().as_secs_f64();
    eprintln!(
        "D0 node {label}: oracle complete in {oracle_seconds:.6}s; preparing expression evaluator"
    );
    let preparation_started = Instant::now();
    let bits = 256;
    let mut evaluator = PrecisionEvaluator::with_binary_precision_and_backend(
        ScalarIntegral::D0,
        bits,
        EvaluationBackend::Expression,
    )?;
    let preparation_seconds = preparation_started.elapsed().as_secs_f64();
    let eta = fraction(64);
    let mut exact: Vec<_> = [0, -2, -2, 0, 0, -2]
        .into_iter()
        .map(|n| Complex::new(&eta * &Rational::from(n), Rational::from(0)))
        .collect();
    exact.extend((0..4).map(|_| Complex::new(Rational::from(2), Rational::from(-1))));
    exact.push(Complex::new(Rational::from(1), Rational::from(0)));
    let prototype = Float::new(bits);
    let input: Vec<_> = exact
        .iter()
        .map(|v| {
            Complex::new(
                prototype.from_rational(&v.re),
                prototype.from_rational(&v.im),
            )
        })
        .collect();
    for (given, represented) in exact.iter().zip(&input) {
        assert_eq!(represented.re.exact_rational().as_ref(), Some(&given.re));
        assert_eq!(represented.im.exact_rational().as_ref(), Some(&given.im));
    }
    let mut values =
        std::array::from_fn::<_, 3, _>(|_| Complex::new(prototype.zero(), prototype.zero()));
    eprintln!(
        "D0 node {label}: preparation complete in {preparation_seconds:.6}s; evaluating one row"
    );
    let evaluation_started = Instant::now();
    let evaluated = evaluator.evaluate(&input, &mut values);
    let evaluation_seconds = evaluation_started.elapsed().as_secs_f64();
    let represented: Option<Vec<_>> = values
        .iter()
        .map(|value| {
            Some(Complex::new(
                value.re.exact_rational()?,
                value.im.exact_rational()?,
            ))
        })
        .collect();
    let budget = fraction(109);
    let (status, comparison, exit_code) = match (evaluated, represented) {
        (Err(error), _) => (
            "evaluation_error",
            json!({"error": error, "values": null}),
            2,
        ),
        (Ok(()), None) => ("nonfinite_output", json!({"values": null}), 2),
        (Ok(()), Some(exact_values)) => {
            let difference = &exact_values[0] - &center;
            let distance_squared = difference.re.pow(2) + difference.im.pow(2);
            let zero_poles = exact_values[1..]
                .iter()
                .all(|v| v.re == Rational::from(0) && v.im == Rational::from(0));
            let passed = zero_poles && distance_squared < (&budget - &radius).pow(2);
            let disproved = !zero_poles || distance_squared > (&budget + &radius).pow(2);
            let slots: Vec<_> = values
                .iter()
                .zip(&exact_values)
                .map(|(value, exact)| {
                    json!({
                        "real_exact_rational": exact.re.to_string(),
                        "imaginary_exact_rational": exact.im.to_string(),
                        "real_tracked_precision": value.re.get_precision(),
                        "imaginary_tracked_precision": value.im.get_precision(),
                    })
                })
                .collect();
            let state = if passed {
                "passed_node_absolute_budget"
            } else if disproved {
                "node_absolute_budget_violated"
            } else {
                "node_budget_inconclusive"
            };
            (
                state,
                json!({
                    "slots_finite_single_double": slots,
                    "poles_exactly_zero": zero_poles,
                    "distance_to_series_center_squared_exact_rational": distance_squared.to_string(),
                    "absolute_budget_exact_rational": budget.to_string(),
                    "certified_below_budget": passed,
                    "proven_budget_violation": disproved,
                }),
                if passed {
                    0
                } else if disproved {
                    3
                } else {
                    4
                },
            )
        }
    };
    let report = json!({
        "schema_version": 1,
        "label": label,
        "status": status,
        "working_bits": bits,
        "rows": 1,
        "backend": "Expression",
        "input_order": "p1_squared,p2_squared,p3_squared,p4_squared,s12,s23,m1_squared,m2_squared,m3_squared,m4_squared,mu_squared",
        "inputs_exact_rational": exact.iter().map(|v| json!({"re":v.re.to_string(),"im":v.im.to_string()})).collect::<Vec<_>>(),
        "contour_node": {"x": "1/2", "t": "1-i", "mass_squared": "2-i", "scope": "Representative rational contour point; not an abscissa of the production even-order Gauss rules"},
        "reference": reference,
        "comparison": comparison,
        "diagnostic_seconds": {"oracle":oracle_seconds,"preparation":preparation_seconds,"evaluation":evaluation_seconds,"total_process_work":started.elapsed().as_secs_f64()},
        "limits": "One scalar primal Float256 evaluation; fixed absolute2^-109 diagnostic target, not a public E6 mixed-tolerance or Dual qualification. No snapping, no E6 geometry or quadrature. Candidate changes are confined to a private archive.",
    });
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(output) = output {
        std::fs::write(output, &encoded)?;
    }
    print!("{encoded}");
    Ok(exit_code)
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
