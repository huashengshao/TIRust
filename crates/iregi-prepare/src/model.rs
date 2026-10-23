//! Compatibility with FeynKit's exact quantum-number JSON deserializer.
//!
//! `serde_json/arbitrary_precision` preserves numeric tokens but presents decimal
//! numbers to untagged enum deserializers as maps. Adapt only the three fields
//! using FeynKit's `quantum_number` deserializer to its supported exact strings.

use feynkit_model::Model;
use serde_json::Value;
use symbolica::domains::float::{Float, SingleFloat};

use crate::PrepareError;

pub(crate) fn from_json(input: &str) -> Result<Model, PrepareError> {
    let normalized = normalize_quantum_numbers(input)?;
    Model::from_json(&normalized).map_err(|error| PrepareError::Model(error.to_string()))
}

fn normalize_quantum_numbers(input: &str) -> Result<String, PrepareError> {
    let mut value: Value =
        serde_json::from_str(input).map_err(|error| PrepareError::Model(error.to_string()))?;
    if let Some(particles) = value.get_mut("particles").and_then(Value::as_array_mut) {
        for (index, particle) in particles.iter_mut().enumerate() {
            for field in ["charge", "y_charge", "y_charge_right"] {
                let Some(entry) = particle.get_mut(field) else {
                    continue;
                };
                let Some(number) = entry.as_number() else {
                    continue;
                };
                // Preserve upstream's integer and unsigned-integer variants.
                if number.is_i64() || number.is_u64() {
                    continue;
                }
                let token = number.to_string();
                let invalid = || {
                    PrepareError::Model(format!(
                        "particles[{index}].{field}: fractional or large floating-point quantum numbers require an exact string, e.g. \"2/3\""
                    ))
                };
                // Four bits per decimal character plus a guard exceed the input
                // significand's precision. In the allowed integer range this
                // prevents a fractional decimal from rounding to an integer.
                let bits = token
                    .len()
                    .checked_mul(4)
                    .and_then(|n| n.checked_add(64))
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(invalid)?;
                let parsed = Float::parse(&token, Some(bits)).map_err(|_| invalid())?;
                let limit = 9_007_199_254_740_991_i64;
                if !parsed.is_finite()
                    || parsed < Float::with_val(bits, -limit)
                    || parsed > Float::with_val(bits, limit)
                {
                    return Err(invalid());
                }
                // Reject nonzero sub-unit values before rational conversion so
                // enormous negative exponents never allocate huge denominators.
                if parsed > Float::with_val(bits, -1)
                    && parsed < Float::with_val(bits, 1)
                    && (!parsed.is_zero()
                        || token
                            .split(['e', 'E'])
                            .next()
                            .unwrap_or("")
                            .bytes()
                            .any(|digit| matches!(digit, b'1'..=b'9')))
                {
                    return Err(invalid());
                }
                let exact = parsed.try_to_rational().ok_or_else(invalid)?;
                if !exact.is_integer() {
                    return Err(invalid());
                }
                *entry = Value::String(exact.to_string());
            }
        }
    }
    serde_json::to_string(&value).map_err(|error| PrepareError::Model(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_boundary_accepts_integral_decimal_quantum_numbers() {
        for (token, expected) in [
            ("0.0", "0"),
            ("-0e+10", "0"),
            ("-1.0", "-1"),
            ("2E0", "2"),
            ("9.007199254740991e15", "9007199254740991"),
            ("-9007199254740991.000", "-9007199254740991"),
        ] {
            let input = format!(
                r#"{{"particles":[{{"charge":{token},"y_charge":{token},"y_charge_right":{token}}}]}}"#
            );
            let normalized: Value =
                serde_json::from_str(&normalize_quantum_numbers(&input).unwrap()).unwrap();
            for field in ["charge", "y_charge", "y_charge_right"] {
                assert_eq!(
                    normalized["particles"][0][field], expected,
                    "{field}: {token}"
                );
            }
        }
    }

    #[test]
    fn model_boundary_rejects_fractional_and_large_decimal_quantum_numbers() {
        for token in [
            "0.5",
            "1.00000000000000000000000000000000000001",
            "9007199254740990.999999999999999999999999999999999",
            "9.007199254740992e15",
            "1e-1000",
            "1e-9999999999",
            "1e9999999999",
        ] {
            for field in ["charge", "y_charge", "y_charge_right"] {
                let input = format!(r#"{{"particles":[{{"{field}":{token}}}]}}"#);
                assert!(
                    normalize_quantum_numbers(&input).is_err(),
                    "{field}: {token}"
                );
            }
        }
    }

    #[test]
    fn model_boundary_preserves_other_numeric_tokens_and_exact_strings() {
        let input = r#"{"particles":[{"pdg_code":25,"charge":"2/3","y_charge":null,"y_charge_right":1}],"parameters":[{"value":[1.0000000000000000000000000000000001,1e-1000]}]}"#;
        let before: Value = serde_json::from_str(input).unwrap();
        let after: Value =
            serde_json::from_str(&normalize_quantum_numbers(input).unwrap()).unwrap();
        assert_eq!(before, after);
        assert_eq!(
            after["parameters"][0]["value"][0].to_string(),
            "1.0000000000000000000000000000000001"
        );
    }

    #[test]
    fn model_boundary_loads_both_pinned_models() {
        from_json(include_str!("../../../fixtures/models/phi3.json")).unwrap();
        from_json(include_str!("../../../fixtures/models/sm.json")).unwrap();
    }
}
