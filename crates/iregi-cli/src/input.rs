//! Exact CLI input parsing shared by standalone execution and validation drivers.
use iregi_masters::{Complex, Float, NativeFloat};
use iregi_rust::{
    Algorithm, Convention, InputPolicy, InvariantInput, KinematicsBatch, ReductionSettings,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use symbolica::{
    atom::Atom,
    domains::{float::SingleFloat, rational::Rational},
};
type CliResult<T> = Result<T, String>;

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum Literal {
    Text(String),
    Number(serde_json::Number),
}
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ComplexLiteral {
    Pair([Literal; 2]),
    Real(Literal),
}
#[derive(Clone, Debug, Deserialize)]
pub struct Propagator {
    pub shift: [Literal; 4],
    pub mass_squared: ComplexLiteral,
    #[serde(default = "unit_power")]
    pub power: u16,
    #[serde(default)]
    pub edge_id: Option<usize>,
}
fn unit_power() -> u16 {
    1
}
#[derive(Clone, Debug, Deserialize)]
pub struct Sample {
    #[serde(default)]
    pub id: String,
    pub propagators: Vec<Propagator>,
    /// Packed upper triangle p_i.p_j, row-major including the diagonal.
    #[serde(default)]
    pub scalar_products: Option<Vec<Literal>>,
    pub mu: Literal,
    /// Full even offset: D=4+dimension_shift-2*epsilon.
    #[serde(default)]
    pub dimension_shift: i16,
    #[serde(default)]
    pub rank: Option<u16>,
    #[serde(default)]
    pub mode: Option<u8>,
    #[serde(default)]
    pub snap: bool,
    /// Exact threshold for opt-in legacy snapping; defaults to 1/1000000.
    #[serde(default)]
    pub threshold: Option<Literal>,
    #[serde(default)]
    pub parameters: BTreeMap<String, ComplexLiteral>,
}
#[derive(Deserialize)]
#[serde(untagged)]
pub enum InputDocument {
    Batch { samples: Vec<Sample> },
    Single(Sample),
}

pub fn settings(
    sample: &Sample,
    rank: u16,
    algorithm: Algorithm,
    convention: Convention,
) -> CliResult<ReductionSettings> {
    if sample.dimension_shift % 2 != 0 {
        return Err("dimension_shift must be even: D=4+dimension_shift-2*epsilon".into());
    }
    let mut settings = ReductionSettings::new(sample.propagators.len(), rank);
    if sample.snap {
        settings.input_policy = match &sample.threshold {
            Some(Literal::Text(threshold)) => InputPolicy::LegacyOnShell {
                threshold: threshold.clone(),
            },
            // arbitrary_precision retains the complete JSON token, including
            // digits beyond binary64; the core validates and parses it exactly.
            Some(Literal::Number(threshold)) => InputPolicy::LegacyOnShell {
                threshold: threshold.to_string(),
            },
            None => InputPolicy::legacy_onshell(),
        };
    }
    if let Some(products) = &sample.scalar_products {
        let n = sample.propagators.len();
        let expected = n
            .checked_add(1)
            .and_then(|m| n.checked_mul(m))
            .map(|m| m / 2)
            .ok_or("too many scalar-product entries")?;
        if products.len() != expected {
            return Err(format!(
                "sample {} supplies {} scalar products; expected {} packed upper-triangle entries",
                sample.id,
                products.len(),
                expected
            ));
        }
        settings.invariants = InvariantInput::SuppliedScalarProducts;
    }
    settings.powers = sample.propagators.iter().map(|p| p.power).collect();
    settings.dimension_shift = sample.dimension_shift / 2;
    settings.algorithm = algorithm;
    settings.convention = convention;
    Ok(settings)
}
pub fn settings_key(
    sample: &Sample,
    rank: u16,
    algorithm: Algorithm,
    convention: Convention,
) -> CliResult<String> {
    Ok(serde_json::to_string(&settings(sample, rank, algorithm, convention)?).unwrap())
}
/// Symbolica rounds decimal strings directly at the destination precision.
/// Rational expressions remain exact until conversion; neither uses binary64.
pub fn parse_real<T: NativeFloat>(
    literal: &Literal,
    prototype: &T,
    allow_numbers: bool,
) -> CliResult<T> {
    let text=match literal {
        Literal::Text(text)=>text.clone(),Literal::Number(n) if allow_numbers=>n.to_string(),
        Literal::Number(_)=>return Err("quote numerical input values as strings; bare JSON numbers are accepted only by compare for binary64 reference data".into()),
    };
    let rational = if text.contains('/') {
        let expression = Atom::parse(&text, "iregi_cli", Default::default())
            .map_err(|e| format!("invalid rational {text:?}: {e}"))?;
        Rational::try_from(&expression).map_err(|e| format!("invalid rational {text:?}: {e}"))?
    } else {
        let parsed = Float::parse(&text, Some(prototype.get_precision()))
            .map_err(|e| format!("invalid decimal {text:?}: {e}"))?;
        if parsed.is_zero() && parsed.is_sign_negative() {
            return Ok(-prototype.zero());
        }
        parsed
            .try_to_rational()
            .ok_or_else(|| format!("input {text:?} must be finite"))?
    };
    let value = prototype.from_rational(&rational);
    if !value.is_finite() {
        return Err(format!("input {text:?} exceeds the selected numeric range"));
    }
    Ok(value)
}
pub fn parse_complex<T: NativeFloat>(
    literal: &ComplexLiteral,
    prototype: &T,
    allow_numbers: bool,
) -> CliResult<Complex<T>> {
    match literal {
        ComplexLiteral::Real(x) => Ok(Complex::new(
            parse_real(x, prototype, allow_numbers)?,
            prototype.zero(),
        )),
        ComplexLiteral::Pair([re, im]) => Ok(Complex::new(
            parse_real(re, prototype, allow_numbers)?,
            parse_real(im, prototype, allow_numbers)?,
        )),
    }
}
pub fn pack<T: NativeFloat>(
    samples: &[Sample],
    prototype: &T,
    allow_numbers: bool,
) -> CliResult<KinematicsBatch<T>> {
    let mut values = Vec::new();
    for sample in samples {
        for p in &sample.propagators {
            for x in &p.shift {
                values.push(Complex::new(
                    parse_real(x, prototype, allow_numbers)?,
                    prototype.zero(),
                ));
            }
        }
        if let Some(products) = &sample.scalar_products {
            for product in products {
                values.push(Complex::new(
                    parse_real(product, prototype, allow_numbers)?,
                    prototype.zero(),
                ));
            }
        }
        for p in &sample.propagators {
            values.push(parse_complex(&p.mass_squared, prototype, allow_numbers)?);
        }
        values.push(Complex::new(
            parse_real(&sample.mu, prototype, allow_numbers)?,
            prototype.zero(),
        ));
    }
    Ok(KinematicsBatch {
        rows: samples.len(),
        components: 1,
        values,
    })
}
pub fn format_complex<T: NativeFloat>(v: &Complex<T>, bits: u32) -> [String; 2] {
    let digits = (u64::from(bits) * 30103 / 100000 + 3) as usize;
    [
        format!("{:.*e}", digits, v.re),
        format!("{:.*e}", digits, v.im),
    ]
}

pub fn characteristic_scale<T: NativeFloat>(sample: &Sample, prototype: &T) -> CliResult<T> {
    let mut scale = parse_real(&sample.mu, prototype, true)?;
    for propagator in &sample.propagators {
        for component in &propagator.shift {
            let value = parse_real(component, prototype, true)?;
            let magnitude = NativeFloat::hypot(&value, &prototype.zero());
            if magnitude > scale {
                scale = magnitude;
            }
        }
        let mass_squared = parse_complex(&propagator.mass_squared, prototype, true)?;
        let mass_scale = NativeFloat::hypot(&mass_squared.re, &mass_squared.im).sqrt();
        if mass_scale > scale {
            scale = mass_scale;
        }
    }
    if !scale.is_finite() || scale <= prototype.zero() {
        return Err("invalid characteristic scale for reference comparison".into());
    }
    Ok(scale)
}
