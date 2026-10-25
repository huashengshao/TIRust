//! Saved-ledger driver for certified planar generalized scalars.
//! Input preparation and the independent Symanzik witnesses belong to this
//! example; the core receives only complete numerical batches and settings.
use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, CertifiedPlanarReductionPlan, CertifiedPlanarWorkspace, KinematicsBatch, LaneStatus,
    ReductionSettings,
};
use serde_json::{Value, json};
use std::{error::Error, path::Path, time::Instant};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        dual::HyperDual,
        float::{Complex, Float, FloatLike, SingleFloat},
        integer::Integer,
        rational::{Q, Rational},
    },
    id::Replacement,
    tensors::matrix::Matrix,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn rational(value: &Value) -> Result<Rational> {
    let text = value
        .as_str()
        .ok_or("kinematic values must be exact rational strings")?;
    let (a, b) = text.split_once('/').unwrap_or((text, "1"));
    let denominator = b.parse::<Integer>()?;
    if denominator == Integer::from(0) {
        return Err("zero denominator".into());
    }
    Ok(Rational::from((a.parse::<Integer>()?, denominator)))
}
fn constant(expression: &Atom) -> Result<Rational> {
    let polynomial = expression.try_to_polynomial::<_, u16>(&Q, None)?;
    if !polynomial.is_constant() {
        return Err(format!("nonconstant prepared input: {expression}").into());
    }
    Ok(polynomial.get_constant())
}
fn write(path: &Path, value: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}
fn at(expression: &Atom, parameters: &[Atom], values: &[Rational]) -> Result<Rational> {
    let replacements = parameters
        .iter()
        .zip(values)
        .map(|(p, v)| Replacement::new(p.clone(), Atom::num(v.clone())))
        .collect::<Vec<_>>();
    constant(&expression.replace_multiple(replacements.iter()))
}
fn base_row(fixture: &Value) -> Result<Vec<Atom>> {
    let propagators = fixture["propagators"]
        .as_array()
        .ok_or("missing propagators")?;
    if propagators.len() != 5 {
        return Err("requires five propagators".into());
    }
    let mut row = Vec::new();
    for p in propagators {
        let shift = p["shift"].as_array().ok_or("missing shift")?;
        if shift.len() != 4 {
            return Err("requires four Cartesian components".into());
        }
        for value in shift {
            row.push(Atom::num(rational(value)?));
        }
    }
    for p in propagators {
        row.push(Atom::num(rational(&p["mass_squared"])?));
    }
    row.push(Atom::num(rational(&fixture["mu"])?));
    Ok(row)
}
fn transformed(base: &[Atom], scale: Rational, translate: bool) -> Vec<Atom> {
    let scale = Atom::num(scale);
    let mut row = base.to_vec();
    let translation = [
        Rational::from(0),
        Rational::from((1, 4)),
        Rational::from((-1, 8)),
        Rational::from((3, 8)),
    ];
    for i in 0..5 {
        for mu in 0..4 {
            row[4 * i + mu] = &base[4 * i + mu] * &scale;
            if translate {
                row[4 * i + mu] += &Atom::num(translation[mu].clone());
            }
        }
    }
    for i in 20..25 {
        row[i] = &base[i] * scale.pow(2);
    }
    row[25] = &base[25] * scale;
    row
}

/// The documented FeynKit one-loop U/F formula, with all manipulation and
/// Jacobian/differentiation delegated to Symbolica, never reduction relations.
fn oracle_preparation(base: &[Atom]) -> Result<Value> {
    let x = (0..5)
        .map(|i| Atom::var(symbolica::symbol!(format!("planar_scalar_oracle::x{i}"))))
        .collect::<Vec<_>>();
    let cube_symbols = (0..4)
        .map(|i| symbolica::symbol!(format!("planar_scalar_oracle::u{i}")))
        .collect::<Vec<_>>();
    let cube = cube_symbols
        .iter()
        .copied()
        .map(Atom::var)
        .collect::<Vec<_>>();
    let mass_symbols = (0..5)
        .map(|i| symbolica::symbol!(format!("planar_scalar_oracle::m{i}")))
        .collect::<Vec<_>>();
    let masses = mass_symbols
        .iter()
        .copied()
        .map(Atom::var)
        .collect::<Vec<_>>();
    let u = x.iter().cloned().sum::<Atom>();
    let mut f = &u * x.iter().zip(&masses).map(|(x, m)| x * m).sum::<Atom>();
    for i in 0..5 {
        for j in i + 1..5 {
            let distance = (0..4)
                .map(|mu| {
                    Atom::num(if mu == 0 { 1 } else { -1 })
                        * (&base[4 * i + mu] - &base[4 * j + mu]).pow(2)
                })
                .sum::<Atom>();
            f -= &x[i] * &x[j] * distance;
        }
    }
    for i in 0..5 {
        if (f.derivative(mass_symbols[i]) - &u * &x[i]).expand() != Atom::new() {
            return Err("Symanzik mass derivative identity failed".into());
        }
    }
    let mut remaining = Atom::num(1);
    let mut simplex = Vec::new();
    for coordinate in &cube {
        simplex.push(&remaining * coordinate);
        remaining *= Atom::num(1) - coordinate;
    }
    simplex.push(remaining);
    if !(simplex.iter().cloned().sum::<Atom>() - 1)
        .expand()
        .is_zero()
    {
        return Err("simplex partition failed".into());
    }
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: None,
    };
    let jacobian = Matrix::from_nested_vec(
        simplex[..4]
            .iter()
            .map(|s| cube_symbols.iter().map(|&v| s.derivative(v)).collect())
            .collect(),
        field,
    )
    .map_err(|e| format!("Jacobian matrix: {e:?}"))?
    .det()
    .map_err(|e| format!("Jacobian determinant: {e:?}"))?;
    let mass_replacements = masses
        .iter()
        .zip(&base[20..25])
        .map(|(m, v)| Replacement::new(m.clone(), v.clone()))
        .collect::<Vec<_>>();
    let replacements = x
        .iter()
        .zip(&simplex)
        .map(|(x, v)| Replacement::new(x.clone(), v.clone()))
        .collect::<Vec<_>>();
    let cube_f = f
        .replace_multiple(mass_replacements.iter())
        .replace_multiple(replacements.iter())
        .expand();
    // Check the analytic one-dot parameter weights by Symbolica differentiation.
    let unit_d4 = Atom::num(-2) / f.pow(3);
    let unit_d6 = Atom::num(-1) / f.pow(2);
    for i in 0..5 {
        if !(unit_d4.derivative(mass_symbols[i]) - Atom::num(6) * &u * &x[i] / f.pow(4))
            .together()
            .is_zero()
            || !(unit_d6.derivative(mass_symbols[i]) - Atom::num(2) * &u * &x[i] / f.pow(3))
                .together()
                .is_zero()
        {
            return Err("one-dot defining-integral derivative check failed".into());
        }
    }
    let samples = [
        [(1, 2), (1, 3), (1, 4), (1, 5)],
        [(1, 7), (2, 5), (3, 8), (4, 9)],
        [(3, 4), (1, 6), (2, 7), (3, 10)],
    ];
    let evaluations=samples.iter().map(|sample| {
        let values=sample.iter().map(|&(a,b)|Rational::from((a,b))).collect::<Vec<_>>();
        Ok(json!({"cube":values.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "simplex":simplex.iter().map(|e|at(e,&cube,&values).map(|v|v.to_string())).collect::<Result<Vec<_>>>()?,
            "F":at(&cube_f,&cube,&values)?.to_string(),"jacobian":at(&jacobian,&cube,&values)?.to_string()}))
    }).collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"schema_version":1,"algebra_provider":"Symbolica","cube_variable_order":["u","v","w","z"],
        "simplex_variable_order":["x0","x1","x2","x3","x4"],"F_expression":cube_f.to_string(),
        "jacobian_expression":jacobian.to_string(),"evaluations":evaluations,
        "mass_derivative_checks":"Symbolica dJ0/dm_i²=+6*x_i/F^4 and dJ1/dm_i²=+2*x_i/F^3 on U=1, all five positions",
        "source_formula":"FeynKit IntegralFamily::symanzik one-loop U*sum(x*m²)-sum(x_i*x_j*s_ij)"}),
    )
}

fn input_values(values: &[Complex<Float>]) -> Result<Value> {
    Ok(json!(
        values
            .iter()
            .map(|z| {
                Ok(
                    json!({"real": z.re.exact_rational().ok_or("nonfinite input")?.to_string(),
            "imaginary": z.im.exact_rational().ok_or("nonfinite input")?.to_string(),
            "real_precision": z.re.get_precision(), "imaginary_precision": z.im.get_precision()}),
                )
            })
            .collect::<Result<Vec<_>>>()?
    ))
}

fn capture(
    fixture: &Value,
    h: i16,
    dot: Option<usize>,
    mode: &str,
    bits: u32,
    accuracy: u32,
    record: &mut Value,
    output: &Path,
) -> Result<bool> {
    let started = Instant::now();
    let prototype = Float::new(bits);
    let base = base_row(fixture)?;
    record["oracle_preparation"] = oracle_preparation(&base)?;
    let t_symbol = symbolica::symbol!("planar_scalar_inputs::t");
    let t = Atom::var(t_symbol);
    let (rows, labels, invalid, components) = match mode {
        "scalar" => {
            let mut bad = base.clone();
            bad[16] += &Atom::num(Rational::from((1, 16)));
            (
                vec![
                    base.clone(),
                    transformed(&base, Rational::from(2), false),
                    bad,
                    transformed(&base, Rational::from(1), true),
                    base.clone(),
                ],
                vec![
                    "base".to_string(),
                    "scaled_by_two".into(),
                    "nonplanar_invalid".into(),
                    "translated".into(),
                    "base_repeat".into(),
                ],
                Some(2),
                1,
            )
        }
        "mass-jets" => {
            let mut rows = (0..5)
                .map(|i| {
                    let mut row = base.clone();
                    row[20 + i] += &t;
                    row
                })
                .collect::<Vec<_>>();
            let mut bad = rows[0].clone();
            bad[16] += &t;
            rows.push(bad);
            rows.push(rows[0].clone());
            let mut labels = (0..5)
                .map(|i| format!("mass_squared_{i}_jet"))
                .collect::<Vec<_>>();
            labels.extend(["transverse_invalid".into(), "mass_0_repeat".into()]);
            (rows, labels, Some(5), 2)
        }
        "finite-difference" => {
            let mut rows = vec![base.clone()];
            let mut labels = vec!["base".to_string()];
            for i in 0..5 {
                for exponent in [12, 16] {
                    for sign in [-1, 1] {
                        let step =
                            Rational::from((Integer::from(sign), Integer::from(2).pow(exponent)));
                        let mut row = base.clone();
                        row[20 + i] += &Atom::num(step);
                        rows.push(row);
                        labels.push(format!("mass_{i}_step_2^-{exponent}_{sign}"));
                    }
                }
            }
            (rows, labels, None, 1)
        }
        _ => return Err("unknown capture mode".into()),
    };
    let mut values = Vec::new();
    let mut exact_rows = Vec::new();
    for row in &rows {
        let mut exact = Vec::new();
        for expression in row {
            for component in 0..components {
                let e = if component == 0 {
                    expression.clone()
                } else {
                    expression.derivative(t_symbol)
                };
                let value = constant(&e.replace(t.clone()).with(Atom::num(0)))?;
                let numeric = prototype.from_rational(&value);
                if numeric.exact_rational().as_ref() != Some(&value) {
                    return Err("input representation lost precision".into());
                }
                exact.push([value.to_string(), "0".to_string()]);
                values.push(Complex::new(numeric, prototype.zero()));
            }
        }
        exact_rows.push(exact);
    }
    let shape = if components == 2 {
        Some(vec![vec![0], vec![1]])
    } else {
        None
    };
    let mut settings = ReductionSettings::new(5, 0);
    settings.dimension_shift = h;
    if let Some(i) = dot {
        settings.powers[i] = 2;
    }
    settings.algorithm = if h == 0 && dot.is_none() {
        Algorithm::PassarinoVeltman
    } else {
        Algorithm::Ibp
    };
    record["settings"] = json!(settings);
    record["input_exact_rational"] = json!(exact_rows);
    record["row_labels"] = json!(labels);
    record["components"] = json!(components);
    record["dual_shape"] = json!(shape);
    record["expected_invalid_lane"] = json!(invalid);
    record["input_jet_provider"] = json!(
        "Symbolica derivative and exact substitution; reduction/master jets use the core Symbolica Dualizer"
    );
    record["status"] = json!("preparing_plan");
    write(output, record)?;
    let plan = CertifiedPlanarReductionPlan::new(settings)?;
    record["basis"] = json!(plan.basis());
    record["status"] = json!("preparing_workspace");
    write(output, record)?;
    let mut workspace = match &shape {
        Some(shape) => CertifiedPlanarWorkspace::new_dual(
            &plan,
            prototype,
            HyperDual::<u8>::new(shape.clone()),
        )?,
        None => CertifiedPlanarWorkspace::new(&plan, prototype)?,
    };
    workspace.set_accuracy_precision(accuracy)?;
    workspace.set_profiling(true);
    record["status"] = json!("reducing");
    write(output, record)?;
    let input = KinematicsBatch {
        rows: rows.len(),
        components,
        values,
    };
    let before = input_values(&input.values)?;
    record["input_before_numeric"] = before.clone();
    write(output, record)?;
    let result = workspace.reduce_batch(&input)?;
    let after = input_values(&input.values)?;
    let input_preserved = before == after;
    record["input_after_numeric"] = after;
    record["input_values_and_precision_preserved"] = json!(input_preserved);
    record["signed_zero_preservation_checked"] = json!(false);
    record["lane_statuses"] = json!(
        result
            .statuses
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
    );
    record["diagnostics"] = json!(workspace.last_diagnostics());
    record["fallback_used"] = json!(result.fallback_used);
    record["analytic_used"] = json!(result.analytic_used);
    let charts=workspace.last_diagnostics().iter().map(|diagnostics|diagnostics.iter().find_map(|message|
        message.strip_prefix("exact full-Taylor planar identity with regular constant-one chart omitting coordinate row ").and_then(|value|value.parse::<usize>().ok()))).collect::<Vec<_>>();
    record["actual_chart_omitted_rows_from_diagnostics"] = json!(charts);
    let mut lanes = Vec::new();
    for row in 0..rows.len() {
        match result.lane(row) {
            Ok(values) => {
                if values.len() != 4 * components {
                    return Err("incomplete scalar Laurent array".into());
                }
                let slots=values.chunks_exact(components).map(|jet|jet.iter().map(|z|Ok(json!({
                "real":z.re.exact_rational().ok_or("nonfinite successful real component")?.to_string(),
                "imaginary":z.im.exact_rational().ok_or("nonfinite successful imaginary component")?.to_string(),
                "real_precision":z.re.get_precision(),"imaginary_precision":z.im.get_precision()}))).collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
                lanes.push(json!({"row":row,"status":"Success","slots":slots}));
            }
            Err(status) => {
                lanes.push(json!({"row":row,"status":format!("{status:?}"),"slots":null}))
            }
        }
    }
    record["lanes"] = json!(lanes);
    let stats = workspace.last_execution();
    let certificates = workspace.last_certificate_execution();
    record["execution"] = json!({"input_rows":stats.input_rows,"argument_rows":stats.argument_rows,"master_family_batches":stats.master_family_batches,
        "master_rows":stats.master_rows,"pentagon_box_rows":stats.pentagon_box_rows,"reduction_rows":stats.reduction_rows,
        "fallback_rows":stats.fallback_rows,"fallback_successes":stats.fallback_successes,"analytic_rows":stats.analytic_rows});
    record["certificate_execution"] = json!({"input_rows":certificates.input_rows,"evaluated_rows":certificates.evaluated_rows,
        "passes":certificates.passes,"reduction_batches":certificates.reduction_batches});
    record["diagnostic_seconds"] = json!(started.elapsed().as_secs_f64());
    let success = result.statuses.iter().enumerate().all(|(lane, status)| {
        if Some(lane) == invalid {
            matches!(status, LaneStatus::InvalidInput(_))
        } else {
            status == &LaneStatus::Success
        }
    }) && input_preserved
        && stats.pentagon_box_rows == 0
        && charts.iter().enumerate().all(|(lane, chart)| {
            if Some(lane) == invalid {
                chart.is_none()
            } else {
                chart.is_some()
            }
        });
    record["status"] = json!(if success {
        "completed_with_expected_lane_controls"
    } else {
        "failed_or_incomplete"
    });
    record["genuine_E6_evaluated"] = json!(stats.pentagon_box_rows != 0);
    Ok(success)
}
fn run() -> Result<u8> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 7 {
        return Err("usage: certified_planar_scalar fixture.json h dot-index-or-none scalar|mass-jets|finite-difference working_bits requested_bits output.json".into());
    }
    let h = args[1].parse::<i16>()?;
    let dot = if args[2] == "none" {
        None
    } else {
        Some(args[2].parse::<usize>()?)
    };
    let bits = args[4].parse::<u32>()?;
    let accuracy = args[5].parse::<u32>()?;
    if !(0..=1).contains(&h)
        || dot.is_some_and(|i| i >= 5)
        || ![128, 256].contains(&bits)
        || accuracy < 53
        || accuracy > bits
        || (args[3] != "scalar" && dot.is_some())
    {
        return Err("unsupported finite capture request".into());
    }
    let output = Path::new(&args[6]);
    if output.exists() {
        return Err("preserve existing capture, no overwrite".into());
    }
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(&args[0])?)?;
    let mut record = json!({"schema_version":1,"status":"preparing","interface":"CertifiedPlanarReductionPlan and CertifiedPlanarWorkspace",
        "fixture":args[0],"dimension_shift":h,"dot":dot,"mode":args[3],"working_bits":bits,"requested_bits":accuracy,
        "native_slots":["uv1","ir1","ir2","finite"],"normalization":"IregiNative i*pi^(2+h), separate Laurent slots; no pole projection",
        "timings_are_performance_evidence":false,"full_generalized_domain_qualified":false});
    write(output, &record)?;
    let success = match capture(
        &fixture,
        h,
        dot,
        &args[3],
        bits,
        accuracy,
        &mut record,
        output,
    ) {
        Ok(success) => success,
        Err(error) => {
            record["status"] = json!("failed_or_incomplete");
            record["error"] = json!(error.to_string());
            false
        }
    };
    write(output, &record)?;
    Ok(if success { 0 } else { 2 })
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(code) => std::process::ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::from(2)
        }
    }
}
