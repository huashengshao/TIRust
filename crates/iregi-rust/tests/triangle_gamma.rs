//! Independent exact oracle for the massless triangle with one off-shell leg.
//!
//! This formula follows directly from Euler's Dirichlet simplex integral. For
//! powers `(a,b,c)`, `n=a+b+c`, `D=4+2h-2epsilon`, and `R02=R20=s` with all
//! other Cayley entries zero, its unnormalized Feynman-parameter expression is
//!
//! ```text
//! (-1)^n (-s)^(D/2-n)
//!   Gamma(n-D/2) Gamma(D/2-b-c) Gamma(D/2-a-b)
//!   / [Gamma(a) Gamma(c) Gamma(D-n)].
//! ```
//!
//! Dividing by the unit-power `D=4-2epsilon` triangle removes the common
//! dimensional-regularization normalization. This derivation shares neither
//! the IBP recurrences nor their Laurent arithmetic. Symbolica evaluates the
//! Gamma functions and Laurent series; the test implements no Gamma recurrence
//! or polynomial/series arithmetic of its own. The finite C0 coefficient stays
//! an independent symbol, so agreement also checks its exact coefficient.

use std::collections::BTreeSet;

use iregi_rust::ibp::{ScalarCompiler, ScalarRequest};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::integer::Integer,
    id::Replacement,
    transcendental::TranscendentalFunctions,
};

fn gamma_ratio(powers: [u16; 3], h: i16, s: &Atom, eps: &Atom) -> Atom {
    let [a, b, c] = powers.map(i32::from);
    let h = i32::from(h);
    let total = a + b + c;
    let sign = if (total - 3) % 2 == 0 { 1 } else { -1 };
    Atom::num(sign) * (-s).pow(i64::from(3 + h - total)) * (Atom::num(total - 2 - h) + eps).gamma()
        / (Atom::num(1) + eps).gamma()
        * (Atom::num(2 + h - b - c) - eps).gamma()
        / (-eps).gamma()
        * (Atom::num(2 + h - a - b) - eps).gamma()
        / (-eps).gamma()
        * (Atom::num(1) - Atom::num(2) * eps).gamma()
        / (Atom::num(4 + 2 * h - total) - Atom::num(2) * eps).gamma()
        / Atom::num(Integer::factorial((a - 1) as u32))
        / Atom::num(Integer::factorial((c - 1) as u32))
}

fn verify(cases: impl IntoIterator<Item = ([u16; 3], i16)>) -> usize {
    let s = Atom::var(symbolica::symbol!("triangle_gamma_oracle::s"));
    // L=log(-s/mu^2), with the same analytic branch as the scalar masters.
    let log = Atom::var(symbolica::symbol!("triangle_gamma_oracle::L"));
    let finite = Atom::var(symbolica::symbol!("triangle_gamma_oracle::F"));
    let eps_symbol = symbolica::symbol!("triangle_gamma_oracle::eps");
    let eps = Atom::var(eps_symbol);
    let mut r = vec![vec![Atom::new(); 3]; 3];
    r[0][2] = s.clone();
    r[2][0] = s.clone();
    let mut compiler = ScalarCompiler::new(r).unwrap();
    let c0 = Atom::num(1) / (&s * eps.pow(2)) - &log / (&s * &eps) + &finite;
    let mut count = 0;
    for (powers, h) in cases {
        let actual = compiler
            .compile(ScalarRequest {
                powers: powers.to_vec(),
                dimension_shift: h,
            })
            .unwrap_or_else(|error| panic!("request {powers:?}, h={h}: {error}"));
        let replacements: Vec<_> = compiler
            .leaves
            .iter()
            .flat_map(|leaf| {
                let values = match leaf.active.as_slice() {
                    [0, 2] => [Atom::num(1), Atom::new(), Atom::new(), Atom::num(2) - &log],
                    [0, 1, 2] => [Atom::new(), -&log / &s, Atom::num(1) / &s, finite.clone()],
                    active => panic!("unexpected master {active:?}"),
                };
                leaf.slots
                    .iter()
                    .cloned()
                    .zip(values)
                    .map(|(symbol, value)| Replacement::new(symbol, value))
            })
            .collect();
        let actual = actual.map(|slot| slot.replace_multiple(replacements.iter()));
        // The ratio has no negative epsilon powers, so C0 through epsilon^0
        // suffices even for raised propagators and lower dimensions.
        let series = (gamma_ratio(powers, h, &s, &eps) * &c0)
            .series(eps_symbol, 0, 0)
            .unwrap_or_else(|error| panic!("Gamma series {powers:?}, h={h}: {error:?}"));
        let pole1 = series.coefficient((-1).into()).unwrap();
        let pole2 = series.coefficient((-2).into()).unwrap();
        let finite = series.coefficient(0.into()).unwrap();
        // A UV pole requires n-h-2<=0. Then both endpoint Gamma arguments
        // at epsilon=0 are strictly positive, so there can be no IR pole.
        // Otherwise Gamma(n-D/2) is regular and all poles are infrared.
        let ultraviolet = i32::from(powers.iter().copied().sum::<u16>()) - i32::from(h) - 2 <= 0;
        let expected = if ultraviolet {
            assert!(pole2.together().is_zero(), "unexpected UV double pole");
            [pole1, Atom::new(), Atom::new(), finite]
        } else {
            [Atom::new(), pole1, pole2, finite]
        };
        for slot in 0..4 {
            let difference = (&actual[slot] - &expected[slot]).together();
            assert!(
                difference.is_zero(),
                "powers={powers:?}, h={h}, slot={slot}: actual={}, expected={}, difference={difference}",
                actual[slot],
                expected[slot]
            );
        }
        count += 1;
    }
    eprintln!("Gamma oracle validated {count} requests and all native Laurent slots");
    count
}

#[test]
fn every_triangle_tensor_coefficient_through_rank_seven_matches_gamma_formula() {
    let mut cases = BTreeSet::new();
    // Every coefficient in Davydychev's representation, including every
    // metric-pair count and every allocation of momentum-vector indices.
    for rank in 0_u16..=7 {
        for pairs in 0..=rank / 2 {
            let degree = rank - 2 * pairs;
            for a in 0..=degree {
                for b in 0..=degree - a {
                    cases.insert(([a + 1, b + 1, degree - a - b + 1], (rank - pairs) as i16));
                }
            }
        }
    }
    assert_eq!(verify(cases), 200);
}

#[test]
fn generalized_triangle_powers_and_dimensions_match_gamma_formula() {
    let mut cases = BTreeSet::new();
    for h in -3..=0 {
        for powers in [
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2],
            [2, 2, 1],
            [2, 3, 2],
        ] {
            cases.insert((powers, h));
        }
    }
    for h in 1..=4 {
        for powers in [[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 3, 2]] {
            cases.insert((powers, h));
        }
    }
    assert_eq!(verify(cases), 40);
}
