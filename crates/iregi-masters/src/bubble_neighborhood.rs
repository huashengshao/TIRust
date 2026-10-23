//! Analytic massive B0 neighborhoods about p²=0.
//!
//! Inputs are `[p², m0², m1², mu²]`; outputs are `[finite, single, double]`.
//! Callers select these formulas only when p² has zero primal part. The equal
//! and unequal functions require two nonzero primal masses; the one-massless
//! function requires its zero mass to vanish in every Dual component.
//! Equal/unequal refers to primal squared masses. All unconstrained supplied
//! Dual components remain in the expressions.
//! `order` is the maximal total nilpotent degree of the prepared Dual shape.

use crate::MasterError;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::integer::Integer,
    id::Replacement,
};

fn preparation(message: impl Into<String>) -> MasterError {
    MasterError::Preparation(message.into())
}

fn degree(order: usize, derivative: bool) -> Result<usize, MasterError> {
    let degree = order
        .checked_add(usize::from(derivative))
        .ok_or(MasterError::SizeOverflow)?;
    u32::try_from(degree).map_err(|_| MasterError::SizeOverflow)?;
    Ok(degree)
}

fn factorial(n: usize) -> Result<Atom, MasterError> {
    Ok(Atom::num(Integer::factorial(
        u32::try_from(n).map_err(|_| MasterError::SizeOverflow)?,
    )))
}

fn parameters() -> [Atom; 4] {
    std::array::from_fn(|i| Atom::var(symbolica::symbol!(format!("iregi_master_bubble::p{i}"))))
}

fn finish(
    mut finite: Atom,
    symbols: &[Atom; 4],
    supplied: &[Atom],
    derivative: bool,
) -> Result<[Atom; 3], MasterError> {
    if supplied.len() != 4 {
        return Err(preparation(
            "a massive lightlike bubble needs four parameters",
        ));
    }
    if derivative {
        finite = finite.derivative(symbolica::symbol!("iregi_master_bubble::p0"));
    }
    let replacements: Vec<_> = symbols
        .iter()
        .zip(supplied)
        .map(|(a, b)| Replacement::new(a.clone(), b.clone()))
        .collect();
    Ok([
        finite.replace_multiple(replacements.iter()),
        Atom::num(if derivative { 0 } else { 1 }),
        Atom::new(),
    ])
}

/// Regular discriminant-zero chart for two positive real primal masses.
///
/// The caller must require nonzero primal `s`, zero primal Kallen discriminant,
/// and `m0²+m1²-s > 0`. The opposite sign is the genuine two-particle threshold
/// and must not select this chart. All supplied nilpotent directions are kept.
pub(crate) fn pseudothreshold_coefficients(
    supplied: &[Atom],
    order: usize,
    derivative: bool,
) -> Result<[Atom; 3], MasterError> {
    let degree = degree(order, derivative)?;
    let p = parameters();
    // Divide first, before forming the discriminant or its square root. This
    // avoids squared-mass underflow for otherwise regular tiny-scale inputs.
    let momentum = &p[0] / &p[1];
    let mass_ratio = &p[2] / &p[1];
    let k = Atom::num(1) + &mass_ratio - &momentum;
    let lambda_symbol = symbolica::symbol!("iregi_master_bubble::lambda");
    let lambda = Atom::var(lambda_symbol);
    let z_symbol = symbolica::symbol!("iregi_master_bubble::atanh_argument");
    let z = Atom::var(z_symbol);
    // Expand the analytic expression first, then substitute into the resulting
    // even polynomial. Expanding the composed square-root expression directly
    // would ask the series engine to differentiate sqrt at zero. Symbolica
    // normalizes the even powers after substitution; no coefficients or power
    // reductions are implemented here.
    let series_depth = degree.checked_mul(2).ok_or(MasterError::SizeOverflow)?;
    let regular = (&z * symbolica::transcendental::atanh().call((&z,)))
        .series(
            z_symbol,
            0,
            u32::try_from(series_depth).map_err(|_| MasterError::SizeOverflow)?,
        )
        .map_err(|error| preparation(format!("pseudothreshold bubble series: {error}")))?
        .to_atom();
    let regular =
        &k / &momentum * regular.replace_multiple([Replacement::new(z, lambda.sqrt() / &k)].iter());
    let discriminant = &k * &k - Atom::num(4) * &mass_ratio;
    let regular = regular.replace_multiple([Replacement::new(lambda, discriminant)].iter());
    // Both arguments are positive on this chart. Separate their logarithms
    // so a representable pair with an unrepresentable ratio remains usable.
    let finite = Atom::num(2) - p[1].log()
        + p[3].log()
        + (&mass_ratio - Atom::num(1) + &momentum) / (Atom::num(2) * &momentum)
            * (&p[1] / &p[2]).log()
        + regular;
    finish(finite, &p, supplied, derivative)
}

/// Equal-mass primal neighborhood, including transverse mass and momentum jets.
pub(crate) fn equal_coefficients(
    supplied: &[Atom],
    order: usize,
    derivative: bool,
) -> Result<[Atom; 3], MasterError> {
    let degree = degree(order, derivative)?;
    let p = parameters();
    let delta = &p[2] - &p[1];
    let a0 = oneloop::a0(&p[1], &p[3]).into_coefficients();
    let mut finite = &a0[0] / &p[1] - Atom::num(1);
    // Expand -int_0^1 log(M+delta*x-s*x*(1-x))dx. The Dirichlet/Beta
    // coefficient is (-1)^b (a+b-1)!(a+b)!/[b!(2a+b+1)! M^(a+b)].
    // This is an exact Taylor formula, not a small nonzero-s approximation.
    for total in 1..=degree {
        for momentum_order in 0..=total {
            let mass_order = total - momentum_order;
            let denominator_order = total
                .checked_add(momentum_order)
                .and_then(|v| v.checked_add(1))
                .ok_or(MasterError::SizeOverflow)?;
            let mut coefficient = Atom::num(if mass_order % 2 == 0 { 1 } else { -1 })
                * factorial(total - 1)?
                * factorial(total)?
                / factorial(mass_order)?
                / factorial(denominator_order)?
                / p[1].pow(i64::try_from(total).map_err(|_| MasterError::SizeOverflow)?);
            if momentum_order != 0 {
                coefficient = coefficient
                    * p[0]
                        .pow(i64::try_from(momentum_order).map_err(|_| MasterError::SizeOverflow)?);
            }
            if mass_order != 0 {
                coefficient = coefficient
                    * delta.pow(i64::try_from(mass_order).map_err(|_| MasterError::SizeOverflow)?);
            }
            finite += coefficient;
        }
    }
    finish(finite, &p, supplied, derivative)
}

/// Unequal-mass primal neighborhood, retaining arbitrary mass and scale jets.
pub(crate) fn unequal_coefficients(
    supplied: &[Atom],
    order: usize,
    derivative: bool,
) -> Result<[Atom; 3], MasterError> {
    let degree = degree(order, derivative)?;
    let p = parameters();
    let eps_symbol = symbolica::symbol!("iregi_master_bubble::epsilon");
    let eps = Atom::var(eps_symbol);
    let m0_symbol = symbolica::symbol!("iregi_master_bubble::p1");
    let m1_symbol = symbolica::symbol!("iregi_master_bubble::p2");
    let formal_logs = [(&p[1] / &p[3]).log(), (&p[2] / &p[3]).log()];
    let physical_logs = [1usize, 2].map(|i| {
        let a0 = oneloop::a0(&p[i], &p[3]).into_coefficients();
        Atom::num(1) - &a0[0] / &p[i]
    });
    let logarithm_replacements = [
        Replacement::new(formal_logs[0].clone(), physical_logs[0].clone()),
        Replacement::new(formal_logs[1].clone(), physical_logs[1].clone()),
    ];
    let mut finite = Atom::new();
    for momentum_order in 0..=degree {
        // J_a(A0(M))=(-M)^a/prod_{k=2}^{a+1}(k-epsilon)*A0(M).
        // Symbolica performs the Laurent truncation. For a>=1, the mixed
        // mass derivatives below kill the polynomial UV residue exactly.
        let shifted: Vec<_> = [1usize, 2]
            .into_iter()
            .enumerate()
            .map(|(log_index, i)| {
                let mut factor = Atom::num(1);
                for step in 1..=momentum_order {
                    factor = factor * (-&p[i]) / (Atom::num(step as u64 + 1) - &eps);
                }
                let a0 = &p[i] * (Atom::num(1) - &formal_logs[log_index]) + &p[i] / &eps;
                let series = (factor * a0)
                    .series(eps_symbol, 0, 0)
                    .map_err(|e| preparation(format!("shifted tadpole series: {e}")))?;
                series
                    .coefficient(0.into())
                    .ok_or_else(|| preparation("shifted tadpole finite term is absent"))
            })
            .collect::<Result<_, _>>()?;
        let mut coefficient = (&shifted[0] - &shifted[1]) / (&p[1] - &p[2]);
        // Feynman-parameter differentiation gives
        // [s^a]B0(s)=1/a! d_m0^a d_m1^a J_a(B0(0)).
        for _ in 0..momentum_order {
            coefficient = coefficient.derivative(m0_symbol).derivative(m1_symbol);
        }
        coefficient = (coefficient / factorial(momentum_order)?).together();
        coefficient = coefficient.replace_multiple(logarithm_replacements.iter());
        if momentum_order != 0 {
            coefficient = coefficient
                * p[0].pow(i64::try_from(momentum_order).map_err(|_| MasterError::SizeOverflow)?);
        }
        finite += coefficient;
    }
    finish(finite, &p, supplied, derivative)
}

/// One exactly massless propagator, with `massless_index` zero or one.
/// Its complete Dual value must be zero; transverse mass derivatives at this
/// boundary are nonanalytic. Momentum, the other mass, and scale may carry jets.
pub(crate) fn one_massless_coefficients(
    supplied: &[Atom],
    order: usize,
    derivative: bool,
    massless_index: usize,
) -> Result<[Atom; 3], MasterError> {
    let massive_position = match massless_index {
        0 => 2,
        1 => 1,
        _ => return Err(preparation("massless bubble index must be zero or one")),
    };
    let degree = degree(order, derivative)?;
    let p = parameters();
    let mass = &p[massive_position];
    let a0 = oneloop::a0(mass, &p[3]).into_coefficients();
    let mut finite = &a0[0] / mass;
    // The remaining parameter integral is
    // -int_0^1 log(1-s*x/M)dx = sum_{a>=1}(s/M)^a/[a(a+1)].
    for momentum_order in 1..=degree {
        let n = i64::try_from(momentum_order).map_err(|_| MasterError::SizeOverflow)?;
        finite += (&p[0] / mass).pow(n) / Atom::num(n) / Atom::num(n + 1);
    }
    finish(finite, &p, supplied, derivative)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equal(actual: Atom, expected: Atom) {
        assert!(
            (&actual - &expected).together().is_zero(),
            "actual {actual}, expected {expected}"
        );
    }

    #[test]
    fn equal_mass_mixed_jet_matches_feynman_parameter_moments() {
        let s_symbol = symbolica::symbol!("bubble_neighborhood_test::s");
        let d_symbol = symbolica::symbol!("bubble_neighborhood_test::d");
        let s = Atom::var(s_symbol);
        let delta = Atom::var(d_symbol);
        let p = [s.clone(), Atom::num(3), Atom::num(3) + &delta, Atom::num(1)];
        let value = equal_coefficients(&p, 2, false).unwrap();
        let expected = -Atom::num(3).log() + &s / Atom::num(18) - &delta / Atom::num(6)
            + s.pow(2) / Atom::num(540)
            - &s * &delta / Atom::num(108)
            + delta.pow(2) / Atom::num(54);
        equal(value[0].clone(), expected);
        assert_eq!(value[1], Atom::num(1));
        assert!(value[2].is_zero());
        let derivative = equal_coefficients(&p, 1, true).unwrap();
        equal(
            derivative[0].clone(),
            Atom::num(1) / Atom::num(18) + &s / Atom::num(270) - &delta / Atom::num(108),
        );
        assert!(derivative[1].is_zero());
    }

    #[test]
    fn unequal_mass_momentum_jet_matches_exact_parameter_integral() {
        let s = Atom::var(symbolica::symbol!("bubble_neighborhood_test::unequal_s"));
        let p = [s.clone(), Atom::num(2), Atom::num(5), Atom::num(1)];
        let value = unequal_coefficients(&p, 1, false).unwrap();
        let slope = Atom::num(7) / Atom::num(18)
            - Atom::num(10) * (Atom::num(5).log() - Atom::num(2).log()) / Atom::num(27);
        let base = Atom::num(1)
            - (Atom::num(5) * Atom::num(5).log() - Atom::num(2) * Atom::num(2).log())
                / Atom::num(3);
        equal(value[0].clone(), base + &s * &slope);
        equal(unequal_coefficients(&p, 0, true).unwrap()[0].clone(), slope);
    }

    #[test]
    fn literal_lightlike_equal_mass_input_keeps_constant_term() {
        let p = [Atom::new(), Atom::num(3), Atom::num(3), Atom::num(1)];
        equal(
            equal_coefficients(&p, 2, false).unwrap()[0].clone(),
            -Atom::num(3).log(),
        );
        equal(
            equal_coefficients(&p, 1, true).unwrap()[0].clone(),
            Atom::num(1) / Atom::num(18),
        );
    }

    #[test]
    fn one_massless_tangent_jet_keeps_momentum_derivatives() {
        let s = Atom::var(symbolica::symbol!("bubble_neighborhood_test::massless_s"));
        for (p, index) in [
            ([s.clone(), Atom::new(), Atom::num(3), Atom::num(1)], 0),
            ([s.clone(), Atom::num(3), Atom::new(), Atom::num(1)], 1),
        ] {
            equal(
                one_massless_coefficients(&p, 2, false, index).unwrap()[0].clone(),
                Atom::num(1) - Atom::num(3).log() + &s / Atom::num(6) + s.pow(2) / Atom::num(54),
            );
            equal(
                one_massless_coefficients(&p, 1, true, index).unwrap()[0].clone(),
                Atom::num(1) / Atom::num(6) + &s / Atom::num(27),
            );
        }
    }

    #[test]
    fn pseudothreshold_jet_matches_regular_parameter_integral() {
        // At s=1, m0²=4, m1²=9 the Feynman denominator is (x+2)².
        // Its defining logarithmic integral has the following finite term
        // and momentum derivative, without a singular discriminant limit.
        let s_symbol = symbolica::symbol!("bubble_neighborhood_test::pseudo_s");
        let s = Atom::var(s_symbol);
        let p = [s.clone(), Atom::num(4), Atom::num(9), Atom::num(1)];
        let at_point = [Replacement::new(s, Atom::num(1))];
        let logarithm = (Atom::num(4) / Atom::num(9)).log();
        let base = Atom::num(2) - Atom::num(4).log() + Atom::num(3) * &logarithm;
        let slope = -Atom::num(2) - Atom::num(5) * &logarithm / Atom::num(2);
        let value = pseudothreshold_coefficients(&p, 2, false).unwrap();
        equal(value[0].replace_multiple(at_point.iter()), base);
        equal(
            value[0]
                .derivative(s_symbol)
                .replace_multiple(at_point.iter()),
            slope.clone(),
        );
        equal(
            pseudothreshold_coefficients(&p, 1, true).unwrap()[0].replace_multiple(at_point.iter()),
            slope,
        );
    }
}
