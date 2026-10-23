//! Adapted from FeynKit's integral-family partial-fraction tests. Explicit
//! independent symbols replace the graph-layer Kinematics construction.

use std::collections::BTreeMap;

use feynkit_integrals::{AffineFamily, IntegralFamilyError};
use symbolica::{
    atom::{Atom, AtomCore},
    parse,
};

fn verify(
    denominators: Vec<Atom>,
    powers: Vec<i32>,
    variables: Vec<Atom>,
) -> Vec<(Atom, Vec<i32>)> {
    let family = AffineFamily::new(variables.clone(), denominators.clone()).unwrap();
    let terms = family.partial_fraction(&powers, 10_000).unwrap();
    let expression = |powers: &[i32]| {
        denominators
            .iter()
            .zip(powers)
            .map(|(d, p)| d.clone().pow(-i64::from(*p)))
            .product::<Atom>()
    };
    let mut reconstructed = Atom::Zero;
    for (coefficient, term_powers) in &terms {
        assert_eq!(term_powers.len(), denominators.len());
        for variable in &variables {
            assert!(!coefficient.contains(variable.as_view()));
        }
        let active = denominators
            .iter()
            .zip(term_powers)
            .filter_map(|(d, p)| (*p > 0).then_some(d.clone()))
            .collect();
        assert!(
            AffineFamily::new(variables.clone(), active)
                .unwrap()
                .is_independent()
        );
        reconstructed += coefficient * expression(term_powers);
    }
    assert!((reconstructed - expression(&powers)).together().is_zero());
    terms
}

#[test]
fn mass_shifts_repeated_and_constant_propagators_preserve_the_rational_function() {
    let x = parse!("apart_mass::x");
    verify(
        vec![x.clone(), &x - parse!("m2")],
        vec![2, 3],
        vec![x.clone()],
    );
    verify(
        vec![x.clone(), Atom::num(2) * &x, &x - parse!("m2")],
        vec![1, 2, 1],
        vec![x.clone()],
    );
    verify(
        vec![Atom::num(7), x.clone(), x.clone()],
        vec![2, 1, 1],
        vec![x.clone()],
    );
    let terms = verify(vec![Atom::num(7), x.clone()], vec![2, 1], vec![x]);
    assert_eq!(terms, vec![(Atom::one() / Atom::num(49), vec![0, 1])]);
}

#[test]
fn homogeneous_and_inhomogeneous_multivariate_relations_preserve_negative_powers() {
    let x = parse!("apart_multi::x");
    let y = parse!("apart_multi::y");
    for constant in [Atom::Zero, parse!("s")] {
        verify(
            vec![x.clone(), y.clone(), &x + &y + constant, &x - &y],
            vec![2, 1, 2, 1],
            vec![x.clone(), y.clone()],
        );
    }
    verify(
        vec![x.clone(), y.clone(), &x + &y],
        vec![-2, 1, 1],
        vec![x, y],
    );
}

#[test]
fn budgets_invalid_powers_and_overflow_return_errors_without_partial_results() {
    let x = parse!("apart_limit::x");
    let family = AffineFamily::new(vec![x.clone()], vec![x.clone(), &x - 1]).unwrap();
    assert!(matches!(
        family.partial_fraction(&[1], 10),
        Err(IntegralFamilyError::InvalidPowers)
    ));
    assert!(matches!(
        family.partial_fraction(&[2, 2], 1),
        Err(IntegralFamilyError::PartialFractionLimit(1))
    ));
    assert!(matches!(
        family.partial_fraction(&[1, 1], 0),
        Err(IntegralFamilyError::PartialFractionLimit(0))
    ));
    let repeated = AffineFamily::new(vec![x.clone()], vec![x.clone(), x.clone()]).unwrap();
    assert!(matches!(
        repeated.partial_fraction(&[1, i32::MAX], 100),
        Err(IntegralFamilyError::PowerOverflow)
    ));
    let zero = AffineFamily::new(vec![x], vec![Atom::Zero]).unwrap();
    assert!(zero.partial_fraction(&[1], 10).is_err());
    assert_eq!(
        zero.partial_fraction(&[0], 10).unwrap(),
        vec![(Atom::one(), vec![0])]
    );
}

#[test]
fn fractional_propagator_coefficients_reconstruct_exactly() {
    let x = parse!("apart_fraction::x");
    verify(
        vec![&x / 2 - parse!("m2"), &x / 3 - parse!("M2")],
        vec![2, 1],
        vec![x],
    );
}

#[test]
fn gaussian_homogeneous_relations_and_raised_powers_reconstruct_exactly() {
    let x = parse!("apart_gaussian::x");
    let y = parse!("apart_gaussian::y");
    let imaginary = parse!("𝑖");
    let first = &x + &imaginary * &y;
    let second = &imaginary * &x - &y;
    for powers in [vec![1, 2, 1], vec![-2, 1, 2]] {
        verify(
            vec![
                first.clone(),
                second.clone(),
                &first - (Atom::one() + &imaginary),
            ],
            powers,
            vec![x.clone(), y.clone()],
        );
    }
}

#[test]
fn expanded_tiny_symbolic_coefficients_keep_original_order_and_weights() {
    let x = parse!("apart_exact::x");
    let y = parse!("apart_exact::y");
    for q in [
        parse!("(apart_exact::a+apart_exact::b)/10^1000"),
        parse!("(apart_exact::a+apart_exact::b)*10^1000"),
        parse!("(apart_exact::a+apart_exact::b)/(apart_exact::c+apart_exact::d)"),
        parse!("sin(apart_exact::a)+cos(apart_exact::b)"),
        parse!("𝑖"),
        parse!("(apart_exact::a+𝑖*apart_exact::b)/10^1000"),
        parse!("(sin(apart_exact::a)+𝑖*cos(apart_exact::b))/(3-𝑖)"),
    ] {
        let denominators = [&x + &q * &y - 1, &x - 2, &y - 3];
        let constant = Atom::one() + Atom::num(3) * &q;
        let expected = BTreeMap::from([
            (vec![0, 1, 1], Atom::one() / &constant),
            (vec![1, 0, 1], -Atom::one() / &constant),
            (vec![1, 1, 0], -&q / &constant),
        ]);
        for expanded in [false, true] {
            let input = denominators
                .iter()
                .map(|d| if expanded { d.expand() } else { d.clone() })
                .collect();
            let terms = verify(input, vec![1, 1, 1], vec![x.clone(), y.clone()]);
            assert_eq!(terms.len(), 3, "expanded={expanded}");
            for (coefficient, powers) in terms {
                assert!((coefficient - &expected[&powers]).together().is_zero());
            }
        }
    }
}

#[test]
fn constructor_rejects_duplicate_or_ambiguous_variables() {
    let x = parse!("apart_variables::x");
    for variables in [
        vec![],
        vec![x.clone(), x.clone()],
        vec![parse!("apart_variables::f(1)")],
        vec![x.pow(2)],
        vec![Atom::num(1)],
    ] {
        assert!(matches!(
            AffineFamily::new(variables, vec![x.clone()]),
            Err(IntegralFamilyError::InvalidVariables)
        ));
    }
    let empty = AffineFamily::new(vec![x.clone()], vec![]).unwrap();
    assert_eq!(empty.variables(), &[x]);
    assert_eq!(empty.rank(), 0);
    assert!(empty.denominators().is_empty());
    assert_eq!(
        empty.partial_fraction(&[], 1).unwrap(),
        vec![(Atom::one(), vec![])]
    );
}

#[test]
fn tiny_hidden_dependence_and_nonlinear_terms_are_rejected_exactly() {
    let x = parse!("apart_guard::x");
    let y = parse!("apart_guard::y");
    let hidden = symbolica::symbol!("apart_guard::hidden");
    for scale in [
        parse!("(apart_guard::a+apart_guard::b)/10^1000"),
        parse!("(apart_guard::a+𝑖*apart_guard::b)/10^1000"),
        parse!("(apart_guard::a+apart_guard::b)*10^1000"),
    ] {
        for bad in [x.pow(2), x.pow(-1), hidden.call(&x)] {
            for expanded in [false, true] {
                let bad = &scale * &bad;
                let bad = if expanded { bad.expand() } else { bad };
                assert!(matches!(
                    AffineFamily::new(vec![x.clone(), y.clone()], vec![&y + bad]),
                    Err(IntegralFamilyError::InvalidBasis(_))
                ));
            }
        }
    }
}

#[test]
fn floating_coefficients_are_not_silently_converted_to_exact_rationals() {
    let x = parse!("apart_float::x");
    assert!(matches!(
        AffineFamily::new(vec![x.clone()], vec![&x + Atom::num(0.5)]),
        Err(IntegralFamilyError::InvalidBasis(_))
    ));
}
