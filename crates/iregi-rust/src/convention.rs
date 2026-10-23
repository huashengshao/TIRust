//! Exact changes of integral convention, prepared by Symbolica.
//!
//! Laurent multiplication and truncation are delegated to Symbolica. The UV
//! and IR sources are expanded separately so a scaleless integral does not
//! lose its cancelling single poles.

use std::sync::LazyLock;

use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::integer::Integer,
    id::Replacement,
};

use crate::{Convention, ReductionSettings, TensorWeights, basis::Monomial};

struct NativeMap {
    values: [Atom; 4],
    a: Atom,
    b: Atom,
    expressions: [Atom; 4],
}

static NATIVE_MAP: LazyLock<NativeMap> = LazyLock::new(|| {
    let eps_symbol = symbolica::symbol!("iregi_internal::convention_eps");
    let eps = Atom::var(eps_symbol);
    let a = Atom::var(symbolica::symbol!("iregi_internal::convention_a"));
    let b = Atom::var(symbolica::symbol!("iregi_internal::convention_b"));
    let values: [Atom; 4] = std::array::from_fn(|slot| {
        Atom::var(symbolica::symbol!(format!(
            "iregi_internal::convention_value{slot}"
        )))
    });
    // Higher epsilon orders cannot contribute through the finite term of a
    // one-loop integral. Keep the exponential unexpanded for Symbolica.
    let factor = Symbol::EXP.call((-&a * &eps - &b * eps.pow(2),));
    let uv = &values[0] / &eps;
    let ir = &values[1] / &eps + &values[2] / eps.pow(2);
    let uv_series = (&factor * &uv)
        .series(eps_symbol, 0, 0)
        .expect("the universal UV normalization has a Laurent series");
    let ir_series = (&factor * &ir)
        .series(eps_symbol, 0, 0)
        .expect("the universal IR normalization has a Laurent series");
    let total_series = (&factor * (&uv + &ir + &values[3]))
        .series(eps_symbol, 0, 0)
        .expect("the universal normalization has a Laurent series");
    let expressions = [
        uv_series
            .coefficient((-1).into())
            .expect("UV pole is known"),
        ir_series
            .coefficient((-1).into())
            .expect("IR single pole is known"),
        ir_series
            .coefficient((-2).into())
            .expect("IR double pole is known"),
        total_series
            .coefficient(0.into())
            .expect("finite coefficient is known"),
    ];
    NativeMap {
        values,
        a,
        b,
        expressions,
    }
});

/// Convert internal OneLOop-normalized raw components to the selected API
/// convention. Input and output retain `[UV1, IR1, IR2, finite]` ordering.
pub(crate) fn convert_convention(
    value: &[Atom; 4],
    monomial: Monomial,
    settings: &ReductionSettings,
) -> [Atom; 4] {
    let mut result = match settings.convention {
        Convention::OneLoop => value.clone(),
        Convention::MadLoop => {
            let sign = Atom::num(if monomial.rank() % 2 == 0 { 1 } else { -1 });
            value.each_ref().map(|coefficient| coefficient * &sign)
        }
        Convention::IregiNative => {
            let pi = Atom::var(Symbol::PI);
            let a = Atom::from(symbolica::transcendental::euler_gamma()) + Symbol::LOG.call((&pi,));
            let b = pi.pow(2) / Atom::num(12);
            let replacements: Vec<_> = NATIVE_MAP
                .values
                .iter()
                .zip(value)
                .map(|(symbol, value)| Replacement::new(symbol.clone(), value.clone()))
                .chain([
                    Replacement::new(NATIVE_MAP.a.clone(), a),
                    Replacement::new(NATIVE_MAP.b.clone(), b),
                ])
                .collect();
            let normalization =
                symbolica::parse!("𝑖") * pi.pow(2 + i64::from(settings.dimension_shift));
            NATIVE_MAP
                .expressions
                .each_ref()
                .map(|expression| expression.replace_multiple(replacements.iter()) * &normalization)
        }
    };
    if settings.tensor_weights == TensorWeights::Multinomial {
        let weight = multinomial(monomial);
        for coefficient in &mut result {
            *coefficient = &*coefficient * &weight;
        }
    }
    result
}

pub(crate) fn multinomial(monomial: Monomial) -> Atom {
    let rank: u32 = monomial.0.iter().map(|&power| u32::from(power)).sum();
    monomial
        .0
        .into_iter()
        .fold(Atom::num(Integer::factorial(rank)), |coefficient, power| {
            coefficient / Atom::num(Integer::factorial(u32::from(power)))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaleless_poles_survive_native_conversion_without_finite_remainder() {
        let settings = ReductionSettings::new(2, 0);
        let result = convert_convention(
            &[Atom::num(1), Atom::num(-1), Atom::Zero, Atom::Zero],
            Monomial([0; 4]),
            &settings,
        );
        let normalization = symbolica::parse!("𝑖") * Atom::var(Symbol::PI).pow(2);
        assert!((&result[0] - &normalization).together().is_zero());
        assert!((&result[1] + &normalization).together().is_zero());
        assert!(result[2].is_zero());
        assert!(result[3].is_zero());
    }

    #[test]
    fn madloop_odd_sign_and_tensor_multiplicity_are_independent() {
        let mut settings = ReductionSettings::new(2, 3);
        settings.convention = Convention::MadLoop;
        settings.tensor_weights = TensorWeights::Multinomial;
        let values = [1, 2, 3, 4].map(Atom::num);
        let result = convert_convention(&values, Monomial([2, 1, 0, 0]), &settings);
        for (actual, expected) in result.iter().zip([-3, -6, -9, -12]) {
            assert_eq!(actual, &Atom::num(expected));
        }
    }
}
