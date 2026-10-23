//! Exact preparation of IREGI's optional, order-dependent on-shell policy.
//!
//! This is applied at individual `scalarprod` call sites. It is deliberately
//! not a transformation of the input momenta or a shared Gram matrix: the
//! original operation does not preserve bilinearity. The supplied-G interface
//! bypasses this module, just as IREGI's `*2` routines bypass `scalarprod`.
//!
//! Symbolica owns every expression, comparison predicate and Dual branch.
//! `IF` predicates inspect primal values; a selected mass branch carries the
//! complete jet of `Re(m²)`, while a selected zero branch has a zero jet.

use std::collections::HashMap;

use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{integer::Integer, rational::Rational},
};

use crate::ReductionError;

/// An active strict-comparison residual used to diagnose snapping boundaries.
///
/// Evaluate alongside the invariant expressions. An inactive comparison yields
/// one, an active boundary yields zero, and nonfinite values indicate that the
/// selected precision cannot safely classify that branch. Only the primal
/// component is a diagnostic; derivatives of predicates have no meaning.
#[derive(Clone, Debug)]
pub struct SnapDiagnostic {
    pub label: String,
    /// Evaluate this diagnostic only on paths that demand this predicate alias.
    pub owner: Atom,
    pub expression: Atom,
}

/// Dependency-ordered Symbolica expressions for the legacy snapping policy.
///
/// Construct once with masses in the original input order, before any routing
/// reversal or subtopology selection. The same reference list applies to all
/// scalar-product queries and all pinched subtopologies.
#[derive(Clone, Debug)]
pub struct LegacySnapping {
    threshold: Atom,
    real_masses: Vec<Atom>,
    retained: Vec<Atom>,
    first_reference: Atom,
    has_references: Atom,
    aliases: Vec<(Atom, Atom)>,
    diagnostics: Vec<SnapDiagnostic>,
    queries: HashMap<Atom, Atom>,
}

impl LegacySnapping {
    pub fn new(masses: &[Atom], threshold: &str) -> Result<Self, ReductionError> {
        let threshold = exact_threshold(threshold)?;
        let mut result = Self {
            threshold,
            real_masses: Vec::with_capacity(masses.len()),
            retained: Vec::with_capacity(masses.len()),
            first_reference: Atom::num(1),
            has_references: Atom::Zero,
            aliases: Vec::new(),
            diagnostics: Vec::new(),
            queries: HashMap::new(),
        };
        for (i, mass) in masses.iter().enumerate() {
            let real = result.alias((mass + Symbol::CONJ.call((mass,))) / Atom::num(2));
            result.real_masses.push(real.clone());
            let mut retained = result.less_than(
                result.threshold.clone(),
                guarded_abs(&real),
                Atom::num(1),
                format!("reference mass {i}: abs(Re(m²)) > threshold"),
            );
            for j in 0..i {
                let active = both(&retained, &result.retained[j]);
                // The signed denominator is intentional. It reproduces the
                // original behavior for negative squared reference masses.
                let relative = guarded_abs(&(&real - &result.real_masses[j])) / &real;
                let duplicate = result.less_than(
                    relative,
                    result.threshold.clone(),
                    active,
                    format!("reference mass {i}: duplicate of reference {j}"),
                );
                retained = result.alias(choose(&duplicate, &Atom::Zero, &retained));
            }
            result.retained.push(retained);
        }
        for i in (0..result.retained.len()).rev() {
            result.first_reference = result.alias(choose(
                &result.retained[i],
                &result.real_masses[i],
                &result.first_reference,
            ));
            result.has_references = result.alias(choose(
                &result.retained[i],
                &Atom::num(1),
                &result.has_references,
            ));
        }
        Ok(result)
    }

    /// Prepare one real scalar-product call, preserving its full selected jet.
    /// Identical raw expressions share the same prepared query.
    pub fn snap_scalar_product(&mut self, raw: Atom, label: impl Into<String>) -> Atom {
        if let Some(value) = self.queries.get(&raw) {
            return value.clone();
        }
        let label = label.into();
        let input = self.alias(raw.clone());
        let relative_zero = self.less_than(
            guarded_abs(&(&input / &self.first_reference)),
            self.threshold.clone(),
            self.has_references.clone(),
            format!("{label}: relative zero threshold"),
        );
        let absolute_zero = self.less_than(
            guarded_abs(&input),
            Atom::num(Rational::from((1, 1000))),
            not(&self.has_references),
            format!("{label}: massless absolute zero threshold"),
        );
        let zero = self.alias(choose(&self.has_references, &relative_zero, &absolute_zero));
        let mut matched = zero.clone();
        let mut value = input.clone();
        for i in 0..self.real_masses.len() {
            let active = both(&self.retained[i], &not(&matched));
            let relative = guarded_abs(&(&input - &self.real_masses[i])) / &self.real_masses[i];
            let close = self.less_than(
                relative,
                self.threshold.clone(),
                active,
                format!("{label}: reference mass {i} threshold"),
            );
            value = self.alias(choose(&close, &self.real_masses[i], &value));
            matched = self.alias(choose(&close, &Atom::num(1), &matched));
        }
        let value = self.alias(choose(&zero, &Atom::Zero, &value));
        self.queries.insert(raw, value.clone());
        value
    }

    pub fn aliases(&self) -> &[(Atom, Atom)] {
        &self.aliases
    }

    pub fn diagnostics(&self) -> &[SnapDiagnostic] {
        &self.diagnostics
    }

    fn alias(&mut self, value: Atom) -> Atom {
        let symbol = Atom::var(symbolica::symbol!(format!(
            "iregi_snap::node_{}",
            self.aliases.len()
        )));
        self.aliases.push((symbol.clone(), value));
        symbol
    }

    fn less_than(&mut self, left: Atom, right: Atom, active: Atom, label: String) -> Atom {
        let difference = self.alias(left - right);
        // Symbolica has a nonzero IF predicate rather than a relational
        // builtin. For a finite real d, abs(d)-d is nonzero exactly for d<0.
        // Guard ABS at d=0 because Symbolica's generic Dual ABS rule contains
        // abs(d)/d. The chosen zero branch also enforces strict comparison.
        let negative = choose(
            &difference,
            &(Symbol::ABS.call((&difference,)) - &difference),
            &Atom::Zero,
        );
        let predicate = self.alias(choose(
            &active,
            &choose(&negative, &Atom::num(1), &Atom::Zero),
            &Atom::Zero,
        ));
        self.diagnostics.push(SnapDiagnostic {
            label,
            owner: predicate.clone(),
            expression: choose(&active, &difference, &Atom::num(1)),
        });
        predicate
    }
}

fn choose(condition: &Atom, yes: &Atom, no: &Atom) -> Atom {
    Symbol::IF.call((condition, yes, no))
}

fn guarded_abs(value: &Atom) -> Atom {
    choose(value, &Symbol::ABS.call((value,)), &Atom::Zero)
}

fn not(value: &Atom) -> Atom {
    choose(value, &Atom::Zero, &Atom::num(1))
}

fn both(left: &Atom, right: &Atom) -> Atom {
    choose(left, right, &Atom::Zero)
}

/// Parse a decimal/scientific literal or an exact rational without introducing
/// a floating-point coefficient. Decimal token handling is lexical only;
/// Integer parsing and all powers/arithmetic remain Symbolica operations.
pub(crate) fn exact_threshold(text: &str) -> Result<Atom, ReductionError> {
    let text = text.trim();
    let invalid = |reason: &str| {
        ReductionError::InvalidInput(format!(
            "invalid legacy on-shell threshold {text:?}: {reason}"
        ))
    };
    if text.is_empty() || text.len() > 4096 {
        return Err(invalid(
            "expected a nonnegative decimal or rational literal of at most 4096 characters",
        ));
    }
    let atom = if text.contains('/') {
        let mut parts = text.split('/');
        let numerator = parts
            .next()
            .unwrap()
            .trim()
            .parse::<Integer>()
            .map_err(|_| invalid("expected an integer rational numerator"))?;
        let denominator = parts
            .next()
            .unwrap()
            .trim()
            .parse::<Integer>()
            .map_err(|_| invalid("expected an integer rational denominator"))?;
        if parts.next().is_some() || denominator.is_zero() {
            return Err(invalid("expected one nonzero rational denominator"));
        }
        Atom::num(Rational::from((numerator, denominator)))
    } else {
        let mut scientific = text.split(['e', 'E']);
        let mantissa = scientific.next().unwrap();
        let exponent = scientific
            .next()
            .map(|value| {
                value
                    .parse::<i64>()
                    .map_err(|_| invalid("invalid decimal exponent"))
            })
            .transpose()?
            .unwrap_or(0);
        if scientific.next().is_some() || exponent.unsigned_abs() > 65536 {
            return Err(invalid("decimal exponent must be between -65536 and 65536"));
        }
        let (sign, mantissa) = if let Some(rest) = mantissa.strip_prefix('-') {
            ("-", rest)
        } else {
            ("", mantissa.strip_prefix('+').unwrap_or(mantissa))
        };
        let mut decimal = mantissa.split('.');
        let whole = decimal.next().unwrap();
        let fractional = decimal.next().unwrap_or("");
        if decimal.next().is_some()
            || whole.is_empty() && fractional.is_empty()
            || !whole
                .bytes()
                .chain(fractional.bytes())
                .all(|byte| byte.is_ascii_digit())
        {
            return Err(invalid("invalid decimal mantissa"));
        }
        let digits = format!("{sign}{whole}{fractional}");
        let integer = digits
            .parse::<Integer>()
            .map_err(|_| invalid("invalid integer digits"))?;
        Atom::num(integer) * Atom::num(10).pow(exponent - fractional.len() as i64)
    };
    let rational =
        Rational::try_from(&atom).map_err(|_| invalid("threshold must be exact and real"))?;
    if rational < Rational::zero() {
        return Err(invalid("threshold must be nonnegative"));
    }
    Ok(Atom::num(rational))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Algorithm, InputPolicy, InvariantInput, KinematicsBatch, LaneStatus,
        PrecisionRescueWorkspace, ReductionPlan, ReductionSettings, RescueSettings, RescueStatus,
        Workspace,
    };
    use iregi_masters::NativeFloat;
    use symbolica::{
        domains::{
            dual::HyperDual,
            float::{Complex, DoubleFloat, Float},
        },
        evaluate::{Dualizer, EvaluationDomain, ExpressionEvaluator},
    };

    fn symbols() -> [Atom; 3] {
        ["x", "m0", "m1"]
            .map(|name| Atom::var(symbolica::symbol!(format!("iregi_snap_test::{name}"))))
    }

    fn evaluate<T: NativeFloat>(
        plan: &LegacySnapping,
        outputs: &[Atom],
        parameters: &[Atom],
        input: &[Complex<T>],
        rows: usize,
        prototype: &T,
        shape: Option<Vec<Vec<usize>>>,
    ) -> Vec<Complex<T>>
    where
        Complex<T>: EvaluationDomain,
    {
        let components = shape.as_ref().map_or(1, Vec::len);
        let mut exact: ExpressionEvaluator<Complex<Rational>> =
            Atom::evaluator_multiple(outputs, parameters)
                .add_aliases(plan.aliases().iter().cloned())
                .unwrap()
                .direct_translation(true)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
        if let Some(shape) = shape {
            exact = exact
                .vectorize(&Dualizer::new(
                    HyperDual::<Complex<Rational>>::new(shape),
                    Vec::new(),
                ))
                .unwrap();
        }
        let mut evaluator = exact
            .map_coeff_with_prec(
                &|value| {
                    Complex::new(
                        prototype.from_rational(&value.re),
                        prototype.from_rational(&value.im),
                    )
                },
                prototype.get_precision(),
            )
            .into_batch_evaluator();
        let mut output = vec![
            Complex::new(prototype.zero(), prototype.zero());
            rows * outputs.len() * components
        ];
        evaluator.evaluate_batch(input, &mut output, rows).unwrap();
        output
    }

    #[test]
    fn exact_decimal_thresholds_do_not_round_through_float() {
        for text in ["1e-6", "0.000001", "+.0000010", "1/1000000"] {
            assert_eq!(
                exact_threshold(text).unwrap(),
                Atom::num(Rational::from((1, 1_000_000)))
            );
        }
        assert_eq!(
            exact_threshold(".125").unwrap(),
            Atom::num(Rational::from((1, 8)))
        );
        let detailed = exact_threshold("1.000000000000000000001e-6").unwrap();
        assert_ne!(detailed, exact_threshold("1e-6").unwrap());
        for zero in ["0", "-0.000", "+0e-100", "0/7"] {
            assert_eq!(exact_threshold(zero).unwrap(), Atom::Zero);
        }
        for invalid in [
            "", "-1e-6", "NaN", "inf", "1e", "1e1e1", "1.0/2", "0.1.2", "1/0", "1/2/3",
        ] {
            assert!(exact_threshold(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn zero_threshold_retains_massless_absolute_policy() {
        let parameters = symbols();
        let mut plan = LegacySnapping::new(&parameters[1..], "0").unwrap();
        let output = plan.snap_scalar_product(parameters[0].clone(), "x");
        let rows = [
            [1e-12, 8., 0.],
            [8. + 1e-10, 8., 0.],
            [0.0005, 0., 0.],
            [0.001, 0., 0.],
        ];
        let input: Vec<_> = rows
            .iter()
            .flatten()
            .map(|&x| Complex::new(x, 0.))
            .collect();
        let results = evaluate(&plan, &[output], &parameters, &input, rows.len(), &0., None);
        for (result, expected) in results.iter().zip([1e-12, 8. + 1e-10, 0., 0.001]) {
            assert_eq!(*result, Complex::new(expected, 0.));
        }
    }

    #[test]
    fn legacy_batched_tensor_arrays_match_original_iregi() {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../validation/reference/snapping/reductions.json"
        ))
        .unwrap();
        let cases = fixtures["cases"].as_array().unwrap();
        let mut report = Vec::new();
        let mut failures = Vec::new();
        let mut primary_successes = 0;
        let mut verified_rescues = 0;
        for (mode, algorithm) in [
            Algorithm::Ibp,
            Algorithm::PassarinoVeltman,
            Algorithm::Hybrid,
        ]
        .into_iter()
        .enumerate()
        {
            for propagators in 2..=4 {
                let selected: Vec<_> = cases
                    .iter()
                    .filter(|case| {
                        case["input"]["mode"].as_u64() == Some(mode as u64)
                            && case["input"]["propagators"].as_array().unwrap().len() == propagators
                    })
                    .collect();
                assert_eq!(selected.len(), if propagators == 2 { 2 } else { 4 });
                let mut settings = ReductionSettings::new(propagators, 2);
                settings.algorithm = algorithm;
                settings.input_policy = InputPolicy::legacy_onshell();
                let plan = ReductionPlan::new(settings).unwrap();
                let mut workspace =
                    PrecisionRescueWorkspace::new(&plan, 0., RescueSettings::default()).unwrap();
                let mut input = Vec::new();
                for case in &selected {
                    let sample = &case["input"];
                    let lines = sample["propagators"].as_array().unwrap();
                    for line in lines {
                        input.extend(
                            line["shift"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| Complex::new(v.as_f64().unwrap(), 0.)),
                        );
                    }
                    for line in lines {
                        input.push(Complex::new(
                            line["mass_squared"][0].as_f64().unwrap(),
                            line["mass_squared"][1].as_f64().unwrap(),
                        ));
                    }
                    input.push(Complex::new(sample["mu"].as_f64().unwrap(), 0.));
                }
                let batch = KinematicsBatch {
                    rows: selected.len(),
                    components: 1,
                    values: input,
                };
                let input_before = batch.values.clone();
                let result = workspace.reduce_batch(&batch).unwrap();
                assert_eq!(batch.values, input_before);
                for (row, case) in selected.iter().enumerate() {
                    let id = case["input"]["id"].as_str().unwrap();
                    let primary = &workspace.last_primary_statuses()[row];
                    let rescue = &workspace.last_rescue_statuses()[row];
                    let status_contract = match (primary, rescue) {
                        (LaneStatus::Success, RescueStatus::NotRequested) => {
                            primary_successes += 1;
                            true
                        }
                        (
                            LaneStatus::Unstable(_),
                            RescueStatus::Converged {
                                working_bits: 128,
                                verification_bits: 256,
                                primary_agreed: None,
                            },
                        ) => {
                            verified_rescues += 1;
                            true
                        }
                        _ => false,
                    };
                    let mut record = serde_json::json!({
                        "id": id, "passed": false,
                        "status": format!("{:?}", result.statuses[row]),
                        "primary_status": format!("{primary:?}"),
                        "rescue_status": format!("{rescue:?}"),
                        "status_contract": status_contract,
                    });
                    if !status_contract {
                        failures.push(format!(
                            "{id}: invalid primary/rescue status combination {primary:?}/{rescue:?}"
                        ));
                    }
                    let actual = match result.lane(row) {
                        Ok(actual) => actual,
                        Err(_) => {
                            let status = format!("{:?}", result.statuses[row]);
                            failures.push(format!("{id}: {status}"));
                            report.push(record);
                            continue;
                        }
                    };
                    let expected = case["reference"]["moments"].as_array().unwrap();
                    if expected.len() != plan.basis().len() || actual.len() != expected.len() * 4 {
                        failures.push(format!("{id}: incomplete tensor/Laurent output"));
                        report.push(record);
                        continue;
                    }
                    let mut characteristic_scale = case["input"]["mu"].as_f64().unwrap();
                    for propagator in case["input"]["propagators"].as_array().unwrap() {
                        for component in propagator["shift"].as_array().unwrap() {
                            characteristic_scale =
                                characteristic_scale.max(component.as_f64().unwrap().abs());
                        }
                        let mass = &propagator["mass_squared"];
                        characteristic_scale = characteristic_scale.max(
                            mass[0]
                                .as_f64()
                                .unwrap()
                                .hypot(mass[1].as_f64().unwrap())
                                .sqrt(),
                        );
                    }
                    let mut max_error = 0_f64;
                    let mut worst = (0, 0);
                    for (moment, reference) in expected.iter().enumerate() {
                        let dimension =
                            4 + plan.basis()[moment].rank() as i32 - 2 * propagators as i32;
                        let scale = characteristic_scale.powi(dimension);
                        for pole in 0..4 {
                            let real = reference[pole][0].as_f64().unwrap();
                            let imag = reference[pole][1].as_f64().unwrap();
                            let delta =
                                actual[moment * 4 + pole].clone() - Complex::new(real, imag);
                            let error = delta.re.hypot(delta.im) / (scale + real.hypot(imag));
                            if error > max_error || !error.is_finite() {
                                max_error = error;
                                worst = (moment, pole);
                            }
                        }
                    }
                    let passed = max_error.is_finite()
                        && max_error <= 1e-10
                        && status_contract
                        && case["reference"]["stable"] == true;
                    if !passed {
                        failures.push(format!(
                            "{id}, moment {}, Laurent {}: normalized error {max_error:e}",
                            worst.0, worst.1
                        ));
                    }
                    record["passed"] = serde_json::json!(passed);
                    record["maximum_normalized_error"] = serde_json::json!(max_error);
                    record["worst_moment"] = serde_json::json!(worst.0);
                    record["worst_laurent_slot"] = serde_json::json!(worst.1);
                    record["compared_coefficients"] = serde_json::json!(actual.len());
                    record["actual_moments"] = serde_json::json!(
                        actual
                            .chunks_exact(4)
                            .map(|poles| poles
                                .iter()
                                .map(|value| [value.re, value.im])
                                .collect::<Vec<_>>())
                            .collect::<Vec<_>>()
                    );
                    report.push(record);
                }
            }
        }
        let report = serde_json::json!({"schema_version": 2, "tolerance": 1e-10, "normalization": "Q=max(mu,abs(shift components),sqrt(abs(mass_squared))); error/(Q^dimension+abs(reference))", "reference_executable_sha256": fixtures["executable_sha256"], "accuracy_precision_bits": 53, "working_precision_bits": 128, "verification_precision_bits": 256, "rescue_trigger": "UnstableOnly", "primary_successes": primary_successes, "verified_rescues": verified_rescues, "cases": report});
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/legacy-snapping-tensor-qualified-validation.json");
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        eprintln!(
            "legacy tensor validation: {} cases, {} failures; {}",
            cases.len(),
            failures.len(),
            path.display()
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn supplied_products_bypass_legacy_policy_and_unused_threshold() {
        let mut settings = ReductionSettings::new(2, 1);
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        let preserved = ReductionPlan::new(settings.clone()).unwrap();
        settings.input_policy = InputPolicy::LegacyOnShell {
            threshold: "1e-1000".into(),
        };
        let legacy = ReductionPlan::new(settings).unwrap();
        assert!(legacy.snapping_threshold().is_none());
        assert!(legacy.snapping_diagnostics().is_empty());
        let values = [
            0.,
            0.,
            0.,
            0.,
            2.0000001,
            0.,
            0.,
            0.,
            0.,
            0.,
            4.00000040000001,
            4.,
            9.,
            3.,
        ]
        .map(|value| Complex::new(value, 0.))
        .to_vec();
        let batch = KinematicsBatch {
            rows: 1,
            components: 1,
            values,
        };
        let expected = Workspace::new(&preserved, 0.)
            .unwrap()
            .reduce_batch(&batch)
            .unwrap();
        let actual = Workspace::new(&legacy, 0.)
            .unwrap()
            .reduce_batch(&batch)
            .unwrap();
        assert_eq!(actual.lane(0).unwrap(), expected.lane(0).unwrap());
    }

    #[test]
    fn unqualified_higher_point_legacy_policy_is_explicitly_unsupported() {
        let mut settings = ReductionSettings::new(5, 0);
        settings.input_policy = InputPolicy::legacy_onshell();
        assert!(matches!(
            ReductionPlan::new(settings),
            Err(ReductionError::Unsupported(_))
        ));
    }

    #[test]
    fn strict_boundaries_and_massless_absolute_rule_match_fortran() {
        let parameters = symbols();
        let mut plan = LegacySnapping::new(&parameters[1..], "1/8").unwrap();
        let snapped = plan.snap_scalar_product(parameters[0].clone(), "x");
        let mut outputs = vec![snapped];
        outputs.extend(plan.diagnostics().iter().map(|d| d.expression.clone()));
        let rows = [
            [1. - f64::EPSILON / 2., 8., 0.],
            [1., 8., 0.],
            [7., 8., 0.],
            [7.5, 8., 0.],
            [9., 8., 0.],
            [0.0005, 0., 0.],
            [0.001, 0., 0.],
            [-0.001, 0., 0.],
        ];
        let input: Vec<_> = rows
            .iter()
            .flatten()
            .map(|&x| Complex::new(x, 0.))
            .collect();
        let result = evaluate(&plan, &outputs, &parameters, &input, rows.len(), &0., None);
        for (i, expected) in [0., 1., 7., 8., 9., 0., 0.001, -0.001]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                result[i * outputs.len()],
                Complex::new(expected, 0.),
                "row {i}"
            );
        }
        for row in [1, 2, 4] {
            assert!(
                result[row * outputs.len() + 1..(row + 1) * outputs.len()]
                    .iter()
                    .any(|x| x.re == 0.)
            );
        }
        assert!(result.iter().all(|x| x.re.is_finite() && x.im.is_finite()));
    }

    #[test]
    fn ordered_deduplication_and_negative_reference_quirks_match_fortran() {
        let parameters = symbols();
        let mut plan = LegacySnapping::new(&parameters[1..], "1e-6").unwrap();
        let output = plan.snap_scalar_product(parameters[0].clone(), "x");
        let rows = [
            [1e-5, 4., 100.],
            [1e-5, 100., 4.],
            [123., 4., -9.],
            [123., -4., 9.],
            [4.000006, 4., 4.000003],
        ];
        let input: Vec<_> = rows
            .iter()
            .flatten()
            .map(|&x| Complex::new(x, 0.))
            .collect();
        let result = evaluate(&plan, &[output], &parameters, &input, rows.len(), &0., None);
        for (actual, expected) in result.iter().zip([1e-5, 0., 123., -4., 4.000006]) {
            assert_eq!(actual, &Complex::new(expected, 0.));
        }
    }

    #[test]
    fn gram_cross_products_cannot_be_reconstructed_from_snapped_distances() {
        let x = symbols()[0].clone();
        let mut plan = LegacySnapping::new(&[Atom::num(4)], "1e-6").unwrap();
        // p=(1,0,0,0), q=(x,1,0,0), exactly the isolated Fortran probe.
        let direct = plan.snap_scalar_product(x.clone(), "p.q");
        let p2 = plan.snap_scalar_product(Atom::num(1), "p²");
        let q2 = plan.snap_scalar_product(x.pow(2) - Atom::num(1), "q²");
        let distance =
            plan.snap_scalar_product((&x - Atom::num(1)).pow(2) - Atom::num(1), "(p-q)²");
        let reconstructed = (p2 + q2 - distance) / Atom::num(2);
        let output = evaluate(
            &plan,
            &[direct, reconstructed],
            &[x],
            &[Complex::new(4.000001, 0.)],
            1,
            &0.,
            None,
        );
        assert_eq!(output[0].re, 4.);
        assert!((output[1].re - 4.000001).abs() < 2e-14);
        assert_ne!(output[0], output[1]);
    }

    #[test]
    fn dual_jets_follow_selected_zero_mass_and_unsnapped_branches() {
        let parameters = symbols();
        let mut plan = LegacySnapping::new(&parameters[1..], "1/8").unwrap();
        let snapped = plan.snap_scalar_product(parameters[0].clone(), "x");
        let mut outputs = vec![snapped];
        outputs.extend(plan.diagnostics().iter().map(|d| d.expression.clone()));
        let mut input = Vec::new();
        for value in [0.5, 7.5, 10., 7.] {
            input.extend([value, 30., 17.].map(|x| Complex::new(x, 0.)));
            input.extend([
                Complex::new(8., -1.),
                Complex::new(2., -3.),
                Complex::new(3., 5.),
            ]);
            input.extend([Complex::new(0., 0.); 3]);
        }
        let result = evaluate(
            &plan,
            &outputs,
            &parameters,
            &input,
            4,
            &0.,
            Some(vec![vec![0], vec![1], vec![2]]),
        );
        for (row, expected) in [[0., 0., 0.], [8., 2., 3.], [10., 30., 17.], [7., 30., 17.]]
            .iter()
            .enumerate()
        {
            let begin = row * outputs.len() * 3;
            assert_eq!(
                &result[begin..begin + 3],
                &expected.map(|x| Complex::new(x, 0.))
            );
        }
        let last = &result[3 * outputs.len() * 3..];
        assert!(last[3..].chunks_exact(3).any(|jet| jet[0].re == 0.));
        assert!(
            result
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }

    fn precision_boundary<T: NativeFloat>(prototype: T)
    where
        Complex<T>: EvaluationDomain,
    {
        let parameters = symbols();
        let mut plan = LegacySnapping::new(&parameters[1..], "1/8").unwrap();
        let output = plan.snap_scalar_product(parameters[0].clone(), "x");
        let tiny = prototype.from_rational(&Rational::from((
            Integer::from(1),
            Integer::from(2).pow(90),
        )));
        let low = prototype.one() - &tiny;
        let high = prototype.one() + &tiny;
        let input: Vec<_> = [low, high.clone()]
            .into_iter()
            .flat_map(|x| [x, prototype.from_i64(8), prototype.zero()])
            .map(|x| Complex::new(x, prototype.zero()))
            .collect();
        let result = evaluate(&plan, &[output], &parameters, &input, 2, &prototype, None);
        assert_eq!(result[0].re, prototype.zero());
        assert_eq!(result[1].re, high);
        assert_ne!(result[1].re, prototype.one());
    }

    #[test]
    fn higher_precision_branch_choices_preserve_sub_binary64_inputs() {
        precision_boundary(DoubleFloat::from(0.));
        precision_boundary(Float::with_val(256, 0));
    }
}
