//! Exact, bounded D6 parameter-integral oracle for optional validation.
//!
//! This example deliberately does not call the reduction or scalar masters.
//! Symbolica constructs the Symanzik polynomial, Taylor series, Jacobian,
//! polynomial antiderivatives and exact endpoint substitutions. See
//! `validation/shifted-pentagon/series-oracle.md` for the remainder proof.

use serde_json::json;
use std::{error::Error, path::PathBuf, sync::Arc, time::Instant};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        integer::Integer,
        rational::{Q, Rational},
    },
    id::Replacement,
    poly::PolyVariable,
    tensors::matrix::Matrix,
};

const DELTA_EXPONENT: u64 = 32;
const SERIES_ORDER: u32 = 4;
const METRIC: [i64; 4] = [1, -1, -1, -1];

fn exact_field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: None,
    }
}

fn binary_fraction(exponent: u64) -> Rational {
    Rational::from((Integer::from(1), Integer::from(2).pow(exponent)))
}

fn exact_constant(expression: &Atom) -> Result<Rational, Box<dyn Error>> {
    let polynomial = expression.try_to_polynomial::<_, u16>(&Q, None)?;
    if !polynomial.is_constant() {
        return Err(format!("expected an exact rational constant, got {expression}").into());
    }
    Ok(polynomial.get_constant())
}

fn main() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let mut arguments = std::env::args_os().skip(1);
    let output = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err("usage: e6_parameter_series [output.json]".into());
    }

    let eta_symbol = symbolica::symbol!("iregi_e6_series::eta");
    let eta = Atom::var(eta_symbol);
    let x: Vec<_> = (0..5)
        .map(|i| Atom::var(symbolica::symbol!(format!("iregi_e6_series::x{i}"))))
        .collect();
    let cube_symbols: Vec<_> = (0..4)
        .map(|i| symbolica::symbol!(format!("iregi_e6_series::u{i}")))
        .collect();
    let cube: Vec<_> = cube_symbols.iter().copied().map(Atom::var).collect();

    // q_i = delta * e_(i-1), i=1..4; q_0=0. The formal expansion
    // parameter eta=delta^2 multiplies every squared distance.
    let routing: Vec<[Atom; 4]> = (0..5)
        .map(|i| std::array::from_fn(|mu| Atom::num(i64::from(i == mu + 1))))
        .collect();
    let unit_distances: Vec<Vec<Atom>> = (0..5)
        .map(|i| {
            (0..5)
                .map(|j| {
                    (0..4)
                        .map(|mu| {
                            Atom::num(METRIC[mu]) * (&routing[i][mu] - &routing[j][mu]).pow(2)
                        })
                        .sum::<Atom>()
                        .expand()
                })
                .collect()
        })
        .collect();
    // One-loop Symanzik formula, independently of any reduction recurrence:
    // U=sum_i x_i, F=U*sum_i x_i*m_i^2-sum_(i<j) x_i*x_j*s_ij.
    let first_symanzik: Atom = x.iter().cloned().sum();
    let mut second_symanzik = first_symanzik.pow(2); // all m_i^2=1
    for i in 0..5 {
        for j in i + 1..5 {
            second_symanzik -= &eta * &x[i] * &x[j] * &unit_distances[i][j];
        }
    }

    let gram: Vec<Vec<Atom>> = (1..5)
        .map(|i| {
            (1..5)
                .map(|j| {
                    (0..4)
                        .map(|mu| {
                            Atom::num(2 * METRIC[mu]) * &eta * &routing[i][mu] * &routing[j][mu]
                        })
                        .sum()
                })
                .collect()
        })
        .collect();
    let gram_determinant = Matrix::from_nested_vec(gram, exact_field())
        .map_err(|error| format!("Gram construction: {error:?}"))?
        .det()
        .map_err(|error| format!("Gram determinant: {error:?}"))?;
    assert!(
        (&gram_determinant + Atom::num(16) * eta.pow(4))
            .expand()
            .is_zero()
    );

    // Standard stick breaking, with its Jacobian derived by Symbolica.
    let mut remaining = Atom::num(1);
    let mut simplex = Vec::with_capacity(5);
    for coordinate in &cube {
        simplex.push(&remaining * coordinate);
        remaining *= Atom::num(1) - coordinate;
    }
    simplex.push(remaining);
    assert!(
        (simplex.iter().cloned().sum::<Atom>() - 1)
            .expand()
            .is_zero()
    );
    let jacobian_matrix: Vec<Vec<Atom>> = simplex[..4]
        .iter()
        .map(|coordinate| {
            cube_symbols
                .iter()
                .map(|&symbol| coordinate.derivative(symbol))
                .collect()
        })
        .collect();
    let jacobian = Matrix::from_nested_vec(jacobian_matrix, exact_field())
        .map_err(|error| format!("Jacobian construction: {error:?}"))?
        .det()
        .map_err(|error| format!("Jacobian determinant: {error:?}"))?;
    let expected_jacobian = (Atom::num(1) - &cube[0]).pow(3)
        * (Atom::num(1) - &cube[1]).pow(2)
        * (Atom::num(1) - &cube[2]);
    assert!((&jacobian - expected_jacobian).expand().is_zero());
    let replacements: Vec<_> = x
        .iter()
        .zip(&simplex)
        .map(|(x, coordinate)| Replacement::new(x.clone(), coordinate.clone()))
        .collect();
    let f = second_symanzik
        .replace_multiple(replacements.iter())
        .expand();
    let mut q_bound = &x[1] * (Atom::num(1) - &x[1]);
    for coordinate in &x[2..] {
        q_bound -= coordinate * (Atom::num(1) - coordinate);
    }
    let q_bound = q_bound.replace_multiple(replacements.iter());
    assert!((&f - (Atom::num(1) - &eta * q_bound)).expand().is_zero());
    let variables = Arc::new(
        std::iter::once(eta.clone())
            .chain(cube.iter().cloned())
            .map(PolyVariable::try_from)
            .collect::<Result<Vec<_>, _>>()?,
    );
    let eta_value = binary_fraction(2 * DELTA_EXPONENT);
    let mut evaluation_point = vec![Rational::from(0); 5];
    evaluation_point[0] = eta_value.clone();
    let mut approximations = Vec::new();
    let generating_function = Atom::num(1) / (Atom::num(1) - &eta).pow(2);

    for order in 0..=SERIES_ORDER {
        let series = (-Atom::num(1) / f.pow(2))
            .series(eta_symbol, 0, order)?
            .to_atom();
        let mut integral = (&jacobian * series)
            .expand()
            .try_to_polynomial::<_, u16>(&Q, Some(variables.clone()))?;
        for variable in 1..5 {
            let antiderivative = integral.integrate(variable);
            integral = &antiderivative.replace(variable, &Rational::from(1))
                - &antiderivative.replace(variable, &Rational::from(0));
        }
        assert!((1..5).all(|variable| integral.degree(variable) == 0));
        assert_eq!(integral.get_constant(), Rational::from((-1, 24)));
        if order >= 1 {
            assert_eq!(
                integral.coefficient(&[1, 0, 0, 0, 0]),
                Some(Rational::from((1, 45)))
            );
        }
        let center = integral.replace_all(&evaluation_point);

        // Uniform |Q|<=1 bounds the omitted generating-function tail.
        // Symbolica verifies the closed form before its exact substitution.
        let tail =
            &generating_function - generating_function.series(eta_symbol, 0, order)?.to_atom();
        let tail_closed_form = eta.pow(i64::from(order) + 1)
            * (Atom::num(order + 2) - Atom::num(order + 1) * &eta)
            / (Atom::num(1) - &eta).pow(2);
        assert!((tail - &tail_closed_form).together().is_zero());
        let radius_expression = (tail_closed_form / Atom::num(24))
            .replace_multiple([Replacement::new(eta.clone(), Atom::num(eta_value.clone()))].iter());
        let radius = exact_constant(&radius_expression)?;
        assert!(radius > Rational::from(0));
        let proves_absolute_256_bits = radius < binary_fraction(256);
        if order == SERIES_ORDER {
            assert!(proves_absolute_256_bits);
            assert!(radius < binary_fraction(321));
        }
        approximations.push(json!({
            "order_in_eta_inclusive": order,
            "integrated_series": integral.to_expression().to_string(),
            "center_exact_rational": center.to_string(),
            "absolute_remainder_bound_exact_rational": radius.to_string(),
            "lower_bound_exact_rational": (&center - &radius).to_string(),
            "upper_bound_exact_rational": (&center + &radius).to_string(),
            "proves_absolute_error_less_than_2_pow_minus_256": proves_absolute_256_bits,
        }));
    }
    let actual_gram = exact_constant(
        &gram_determinant
            .replace_multiple([Replacement::new(eta, Atom::num(eta_value.clone()))].iter()),
    )?;
    assert_eq!(actual_gram, -binary_fraction(8 * DELTA_EXPONENT - 4));
    let actual_shifts: Vec<Vec<String>> = routing
        .iter()
        .map(|row| {
            row.iter()
                .map(|component| {
                    exact_constant(&(Atom::num(binary_fraction(DELTA_EXPONENT)) * component))
                        .map(|value| value.to_string())
                })
                .collect()
        })
        .collect::<Result<_, _>>()?;
    let mut actual_distances = Vec::new();
    for i in 0..5 {
        for j in i + 1..5 {
            actual_distances.push(
                exact_constant(&(Atom::num(eta_value.clone()) * &unit_distances[i][j]))?
                    .to_string(),
            );
        }
    }
    assert_eq!(
        actual_distances
            .iter()
            .filter(|value| *value == "0")
            .count(),
        3
    );
    let report = json!({
        "schema_version": 1,
        "status": "exact_oracle_generated",
        "qualification": "Independent exact rational D6 parameter-series oracle with an analytic uniform remainder bound; no production reduction or scalar-master evaluation.",
        "normalization": "J1 = - integral_simplex F^-2 dx; native OneLoop slots [UV,IR,IR2,finite]=[0,0,0,J1]",
        "fixture": {
            "mass_squared": ["1", "1", "1", "1", "1"],
            "mu": "1",
            "mu_squared": "1",
            "momentum_shifts": actual_shifts,
            "squared_distances_lexicographic_i_less_than_j": actual_distances,
            "routing": "q0=0; q1..q4=delta times the four Cartesian basis vectors",
            "metric": METRIC,
            "delta_exact_rational": binary_fraction(DELTA_EXPONENT).to_string(),
            "eta_exact_rational": eta_value.to_string(),
            "doubled_external_gram_determinant_exact_rational": actual_gram.to_string(),
            "domain": "Full external Gram rank, equal masses, positive F; three pair distances are exactly zero. This is a special family, not a generic-point qualification. Its dyadic inputs are individually representable in binary64, so it does not test parsing beyond 53 bits.",
        },
        "proof": {
            "first_symanzik": first_symanzik.to_string(),
            "second_symanzik": second_symanzik.to_string(),
            "cube_jacobian": jacobian.to_string(),
            "uniform_bound": "F=1-eta*Q, Q=x1*(1-x1)-sum_(i=2..4) xi*(1-xi), |Q|<=sum_(i=1..4) xi<=1, so F>=1-eta>0",
            "remainder": "eta^(N+1)*((N+2)-(N+1)*eta)/(24*(1-eta)^2)",
            "final_absolute_remainder_less_than": "2^-321",
            "proof_notes": "validation/shifted-pentagon/series-oracle.md",
        },
        "approximations": approximations,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "limits": "No production agreement, generic E6 coverage, physical contour coverage or performance claim. Exact rational bounds do not infer unverified digits for the existing generic-point quadrature oracle.",
    });
    let encoded = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(output) = output {
        std::fs::write(output, &encoded)?;
    }
    print!("{encoded}");
    Ok(())
}
