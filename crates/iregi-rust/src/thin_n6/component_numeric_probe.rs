//! Private ignored highpoint Dual array probe; never a public API promotion.
//! Input jets, independent response columns and finite points reuse the reviewed
//! Symbolica helper. No reference values are supplied to the reduction engine.
#[path = "component_probe_jets.rs"]
mod jets;

use crate::thin_n6::{ThinSixReductionPlan, ThinSixWorkspace};
use crate::thin_n7::{ThinSevenReductionPlan, ThinSevenWorkspace};
use crate::{
    Algorithm, BatchResult, CertificateExecutionStats, ExecutionStats, KinematicsBatch, LaneStatus,
    ReductionError, ReductionSettings,
};
use iregi_masters::NativeFloat;
use serde_json::{Value, json};
use std::{error::Error, path::Path};
use symbolica::domains::{
    dual::HyperDual,
    float::{Complex, Float, FloatLike, SingleFloat},
    integer::Integer,
    rational::Rational,
};
type Fallible<T> = Result<T, Box<dyn Error>>;

fn rational(value: &Value) -> Fallible<Rational> {
    let text = value
        .as_str()
        .ok_or("fixture numbers must be exact rational strings")?;
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let d = d.parse::<Integer>()?;
    if d == Integer::from(0) {
        return Err("zero denominator".into());
    }
    Ok(Rational::from((n.parse::<Integer>()?, d)))
}
fn base(path: &Path, n: usize) -> Fallible<Vec<Rational>> {
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let sample = &fixture["samples"][0];
    let propagators = sample["propagators"]
        .as_array()
        .ok_or("missing propagators")?;
    if propagators.len() < n {
        return Err("insufficient fixture propagators".into());
    }
    let mut row = Vec::new();
    for p in &propagators[..n] {
        let shift = p["shift"].as_array().ok_or("missing shift")?;
        if shift.len() != 4 {
            return Err("Cartesian shift width differs from four".into());
        }
        for x in shift {
            row.push(rational(x)?);
        }
    }
    for p in &propagators[..n] {
        let mass = p["mass_squared"]
            .as_array()
            .ok_or("missing complex mass pair")?;
        if mass.len() != 2 || rational(&mass[1])? != Rational::from(0) || p["power"] != 1 {
            return Err("private retained cohort requires real masses and unit powers".into());
        }
        row.push(rational(&mass[0])?);
    }
    row.push(rational(&sample["mu"])?);
    Ok(row)
}
fn operand<T: NativeFloat>(x: &T) -> Value {
    json!({"rational":x.exact_rational().map(|q|q.to_string()),"precision":x.get_precision(),
        "negative_sign":x.negative_sign(),"finite":x.is_finite()})
}
fn represented<T: NativeFloat>(values: &[Complex<T>]) -> Value {
    json!(
        values
            .iter()
            .map(|z| json!([operand(&z.re), operand(&z.im)]))
            .collect::<Vec<_>>()
    )
}
fn convert(prototype: &Float, input: &[Rational]) -> Fallible<Vec<Complex<Float>>> {
    input
        .iter()
        .map(|q| {
            let re = prototype.from_rational(q);
            if re.exact_rational().as_ref() != Some(q) {
                return Err("input rounding changed exact cohort".into());
            }
            Ok(Complex::new(re, prototype.zero()))
        })
        .collect()
}
fn write(path: &Path, value: &Value) -> Fallible<()> {
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    Ok(())
}
fn counters(s: ExecutionStats) -> Value {
    json!({"input_rows":s.input_rows,"argument_rows":s.argument_rows,"master_rows":s.master_rows,
        "master_family_batches":s.master_family_batches,"pentagon_box_rows":s.pentagon_box_rows,
        "reduction_rows":s.reduction_rows,"fallback_rows":s.fallback_rows,
        "fallback_successes":s.fallback_successes,"analytic_rows":s.analytic_rows})
}
fn certificate(s: CertificateExecutionStats) -> Value {
    json!({"input_rows":s.input_rows,"evaluated_rows":s.evaluated_rows,"passes":s.passes,
        "reduction_batches":s.reduction_batches})
}
fn component_context(result: &BatchResult<Float>, row: usize, bits: u32, accuracy: u32) -> Value {
    // Capture rejected context as well as accepted context. The raw optional
    // sidecars are evidence, never an admission substitute for the validators.
    let raw = result.primal_envelopes.get(row).and_then(Option::as_ref).map(|envelope| {
        json!({"working_precision":envelope.working_precision,"accuracy_bits":envelope.accuracy_bits,
            "components":envelope.components,"basis":envelope.basis.iter().map(|x|x.0).collect::<Vec<_>>(),
            "metadata":{"convention":envelope.metadata.convention(),"tensor_weights":envelope.metadata.tensor_weights(),
                "dual_shape":envelope.metadata.dual_shape()},
            "route_history":envelope.routes.iter().map(|x|format!("{x:?}")).collect::<Vec<_>>(),
            "output_primals":represented(&envelope.output_primals),
            "output_components":envelope.output_components.as_deref().map(represented),
            "component_amplitudes":envelope.component_amplitudes.as_deref().map(represented),
            "primal_operands":envelope.operands.iter().map(|x|json!({"output_index":x.output_index,
                "kind":format!("{:?}",x.kind),"amplitude":represented(std::slice::from_ref(&x.amplitude))})).collect::<Vec<_>>()})
    });
    let mut record = json!({"raw_envelope":raw,
        "raw_error":result.primal_envelope_errors.get(row),
        "heuristic_only":true,"rigorous_derivative_error_bound":false});
    if result.lane(row).is_err() {
        record["status"] = json!("unavailable_for_failed_lane");
    } else if let Err(error) = result.validated_primal_envelope(row, bits, accuracy) {
        record["status"] = json!("invalid_context");
        record["error"] = json!(error);
    } else if result.components > 1 {
        match result.validated_component_amplitudes(row, bits, accuracy) {
            Ok(_) => record["status"] = json!("validated_full_component_context"),
            Err(error) => {
                record["status"] = json!("invalid_component_context");
                record["error"] = json!(error);
            }
        }
    } else {
        record["status"] = json!("validated_primal_context");
    }
    record
}
fn describe_result(result: &BatchResult<Float>, bits: u32, accuracy: u32) -> Fallible<Value> {
    let mut lanes = Vec::new();
    for row in 0..result.rows() {
        let values = match result.lane(row) {
            Ok(values) => {
                if values.len() != result.basis.len() * 4 * result.components
                    || values
                        .iter()
                        .any(|z| z.re.exact_rational().is_none() || z.im.exact_rational().is_none())
                {
                    return Err("successful array is incomplete or nonfinite".into());
                }
                represented(values)
            }
            Err(_) => Value::Null,
        };
        lanes.push(
            json!({"row":row,"status":format!("{:?}",result.statuses[row]),"values":values,
            "fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row],
            "component_context":component_context(result,row,bits,accuracy)}),
        );
    }
    Ok(json!({"rows":result.rows(),"components":result.components,
        "basis":result.basis.iter().map(|x|x.0).collect::<Vec<_>>(),
        "metadata":{"convention":result.metadata().convention(),"tensor_weights":result.metadata().tensor_weights(),
            "dual_shape":result.metadata().dual_shape()},"lanes":lanes}))
}

enum Plan {
    Six(ThinSixReductionPlan),
    Seven(ThinSevenReductionPlan),
}
enum Engine {
    Six(ThinSixWorkspace<Float>),
    Seven(ThinSevenWorkspace<Float>),
}
impl Plan {
    fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        if settings.propagators == 6 {
            ThinSixReductionPlan::new(settings).map(Self::Six)
        } else {
            ThinSevenReductionPlan::new(settings).map(Self::Seven)
        }
    }
    fn workspace(
        &self,
        prototype: Float,
        shape: Option<Vec<Vec<usize>>>,
    ) -> Result<Engine, ReductionError> {
        match (self, shape) {
            (Self::Six(plan), Some(shape)) => {
                ThinSixWorkspace::new_dual(plan, prototype, HyperDual::<u8>::new(shape))
                    .map(Engine::Six)
            }
            (Self::Seven(plan), Some(shape)) => {
                ThinSevenWorkspace::new_dual(plan, prototype, HyperDual::<u8>::new(shape))
                    .map(Engine::Seven)
            }
            (Self::Six(plan), None) => ThinSixWorkspace::new(plan, prototype).map(Engine::Six),
            (Self::Seven(plan), None) => {
                ThinSevenWorkspace::new(plan, prototype).map(Engine::Seven)
            }
        }
    }
}
impl Engine {
    fn accuracy(&mut self, bits: u32) -> Result<(), ReductionError> {
        match self {
            Self::Six(w) => w.set_accuracy_precision(bits),
            Self::Seven(w) => w.set_accuracy_precision(bits),
        }
    }
    fn reduce(
        &mut self,
        batch: &KinematicsBatch<Float>,
    ) -> Result<BatchResult<Float>, ReductionError> {
        match self {
            Self::Six(w) => w.reduce_batch(batch),
            Self::Seven(w) => w.reduce_batch(batch),
        }
    }
    fn diagnostics(&self) -> &[Vec<String>] {
        match self {
            Self::Six(w) => w.last_diagnostics(),
            Self::Seven(w) => w.last_diagnostics(),
        }
    }
    fn child(&self) -> Option<&BatchResult<Float>> {
        match self {
            Self::Six(w) => w.component_probe_children.as_deref(),
            Self::Seven(w) => w.component_probe_children.as_deref(),
        }
    }
    fn stats(&self) -> Value {
        let (inputs, child_calls, child_rows, parent_calls, parent_rows, child, parent, cc, pc) =
            match self {
                Self::Six(w) => {
                    let s = w.last_execution();
                    (
                        s.parent_input_rows,
                        s.child_batch_calls,
                        s.packed_child_rows,
                        s.parent_batch_calls,
                        s.packed_parent_rows,
                        s.child_execution,
                        s.parent_execution,
                        s.child_certificate_execution,
                        s.parent_certificate_execution,
                    )
                }
                Self::Seven(w) => {
                    let s = w.last_execution();
                    (
                        s.parent_input_rows,
                        s.child_batch_calls,
                        s.packed_child_rows,
                        s.parent_batch_calls,
                        s.packed_parent_rows,
                        s.child_execution,
                        s.parent_execution,
                        s.child_certificate_execution,
                        s.parent_certificate_execution,
                    )
                }
            };
        json!({"logical_parent_rows":inputs,"child_batch_calls":child_calls,"packed_child_rows":child_rows,
            "parent_batch_calls":parent_calls,"packed_parent_rows":parent_rows,"child":counters(child),
            "parent":counters(parent),"child_certificates":certificate(cc),"parent_certificates":certificate(pc)})
    }
}
fn call(
    engine: &mut Engine,
    rows: &[Vec<Complex<Float>>],
    components: usize,
    bits: u32,
    accuracy: u32,
    expected: &[&str],
) -> Fallible<Value> {
    let input = KinematicsBatch {
        rows: rows.len(),
        components,
        values: rows.iter().flatten().cloned().collect(),
    };
    let before = represented(&input.values);
    let result = engine.reduce(&input);
    let after = represented(&input.values);
    let mut record = json!({"input_before":before,"input_after":after,"input_preserved":before==after,
        "execution":engine.stats(),"diagnostics":engine.diagnostics(),"expected_lane_statuses":expected,
        "actual_packed_children":engine.child().map(|child|describe_result(child,bits,accuracy)).transpose()?});
    match result {
        Err(error) => {
            record["status"] = json!("batch_error");
            record["error"] = json!(error.to_string());
            record["result"] = Value::Null;
            record["expected_outcomes"] = json!(false);
        }
        Ok(result) => {
            let good = result.rows() == expected.len()
                && expected
                    .iter()
                    .enumerate()
                    .all(|(row, status)| match *status {
                        "Success" => {
                            result.statuses[row] == LaneStatus::Success
                                && result
                                    .validated_primal_envelope(row, bits, accuracy)
                                    .is_ok()
                                && (components == 1
                                    || result
                                        .validated_component_amplitudes(row, bits, accuracy)
                                        .is_ok())
                        }
                        "InvalidInput" => {
                            matches!(result.statuses[row], LaneStatus::InvalidInput(_))
                        }
                        _ => false,
                    });
            record["status"] = json!("captured");
            record["expected_outcomes"] = json!(good);
            record["result"] = describe_result(&result, bits, accuracy)?;
        }
    }
    Ok(record)
}
fn save_call(
    record: &mut Value,
    output: &Path,
    name: &str,
    run: impl FnOnce() -> Fallible<Value>,
) -> Fallible<()> {
    record["status"] = json!("evaluating_batch");
    record["active_call"] = json!(name);
    write(output, record)?;
    let mut value = run()?;
    value["name"] = json!(name);
    let good = value["status"] == "captured"
        && value["expected_outcomes"] == true
        && value["input_preserved"] == true;
    record["calls"].as_array_mut().unwrap().push(value);
    write(output, record)?;
    if good {
        Ok(())
    } else {
        Err("unexpected capture outcome; later fixed calls remain unrun".into())
    }
}
fn run() -> Fallible<u8> {
    let fixture = std::env::var("IREGI_HIGHPOINT_DUAL_FIXTURE")?;
    let n: usize = std::env::var("IREGI_HIGHPOINT_DUAL_N")?.parse()?;
    let rank: u16 = std::env::var("IREGI_HIGHPOINT_DUAL_RANK")?.parse()?;
    let bits: u32 = std::env::var("IREGI_HIGHPOINT_DUAL_WORKING_BITS")?.parse()?;
    let accuracy: u32 = std::env::var("IREGI_HIGHPOINT_DUAL_REQUESTED_BITS")?.parse()?;
    let profile = std::env::var("IREGI_HIGHPOINT_DUAL_PROFILE")?;
    let output = std::env::var("IREGI_HIGHPOINT_DUAL_OUTPUT")?;
    if ![6, 7].contains(&n)
        || rank > 7
        || ![128, 256, 384].contains(&bits)
        || accuracy < 53
        || accuracy > bits
        || !matches!(
            profile.as_str(),
            "plain" | "mass" | "dilation" | "rectangular" | "mixed" | "fd-mass"
        )
    {
        return Err("unsupported private finite probe selection".into());
    }
    let output = Path::new(&output);
    if output.exists() {
        return Err("preserve existing attempt; no overwrite".into());
    }
    let mut settings = ReductionSettings::new(n, rank);
    settings.algorithm = Algorithm::PassarinoVeltman;
    let shape = match profile.as_str() {
        "mass" => Some(jets::FIRST.map(Vec::from).to_vec()),
        "dilation" => Some(vec![vec![0, 0], vec![0, 1], vec![0, 2]]),
        "rectangular" => Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]),
        "mixed" => Some(jets::MIXED.map(Vec::from).to_vec()),
        _ => None,
    };
    let components = shape.as_ref().map_or(1, Vec::len);
    let mut record = json!({"schema_version":1,"status":"preparing_inputs","calls":[],"settings":settings,
        "fixture":fixture,"working_bits":bits,"requested_bits":accuracy,"profile":profile,"ordered_shape":shape,
        "interface":"Private cfg(test) ThinSix/ThinSeven, no public wrapper or facade Dual dispatch change",
        "native_slots":["uv1","ir1","ir2","finite"],"sample_lanes_distinct_from_taylor_components":true,
        "public_Dual_qualified":false,"numerically_qualified":false,"performance_qualified":false,
        "limits":"Retained real unit-power ordinary PV families only. Full-component operands remain heuristic estimates. FD ordinary reductions are independent of AD but not independent integration; external original-quad or defining-integral anchors are required. Unsupported E6 error transport remains failure. No silent algorithm or input projection."});
    write(output, &record)?;
    let outcome = (|| -> Fallible<()> {
        let prototype = Float::new(bits);
        let base = base(Path::new(&fixture), n)?;
        let mut exact = Vec::new();
        if profile == "fd-mass" {
            let mut points = vec![(Rational::from(0), Rational::from(0))];
            let mut labels = vec!["base".to_string()];
            for exponent in [12, 18] {
                for sign in [-1_i64, 1] {
                    points.push((Rational::from((sign, 1_i64 << exponent)), Rational::from(0)));
                    labels.push(format!("mass_squared_0_step_2^-{exponent}_{sign}"));
                }
            }
            exact = jets::points(&base, n, &points)?;
            record["row_labels"] = json!(labels);
        } else {
            exact.push(match &shape {
                Some(shape) => jets::input_jets(&base, n, shape)?,
                None => base.clone(),
            });
        }
        record["input_exact_rational"] = json!(
            exact
                .iter()
                .map(|r| r.iter().map(ToString::to_string).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        // These columns always use the complete mixed response order. Profiles
        // select rows explicitly; columns are ordinary unit/dot0/triple0 anchors.
        record["independent_response_contract"] = json!({
            "ordered_shape":jets::MIXED,"columns":["unit","dot0","triple0"],
            "profile_rows":match profile.as_str() {
                "mass"=>vec![0usize,1],"dilation"=>vec![0usize,2,5],
                "rectangular"=>vec![0usize,1,2,4],"mixed"=>vec![0usize,1,2,3,4,5],
                _=>vec![0usize]
            },"degree_index":"total degree of the returned tensor monomial"});
        record["independent_response_columns"] = json!(
            (0..=rank as usize)
                .map(|r| jets::response(n, r))
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .map(|rows| rows
                    .iter()
                    .map(|r| r.iter().map(ToString::to_string).collect::<Vec<_>>())
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
        record["status"] = json!("preparing_plan");
        write(output, &record)?;
        let plan = Plan::new(settings)?;
        record["status"] = json!("preparing_workspace");
        write(output, &record)?;
        let mut engine = plan.workspace(prototype.clone(), shape.clone())?;
        engine.accuracy(accuracy)?;
        let rows = exact
            .iter()
            .map(|r| convert(&prototype, r))
            .collect::<Fallible<Vec<_>>>()?;
        if profile == "fd-mass" {
            let expected = vec!["Success"; rows.len()];
            save_call(&mut record, output, "ordinary_first_mass_fd5", || {
                call(&mut engine, &rows, 1, bits, accuracy, &expected)
            })?;
        } else {
            let mut invalid = rows[0].clone();
            for v in &mut invalid[5 * n * components..(5 * n + 1) * components] {
                *v = Complex::new(prototype.zero(), prototype.zero());
            }
            let odd = vec![rows[0].clone(), invalid, rows[0].clone()];
            save_call(&mut record, output, "odd_repeated_with_invalid", || {
                call(
                    &mut engine,
                    &odd,
                    components,
                    bits,
                    accuracy,
                    &["Success", "InvalidInput", "Success"],
                )
            })?;
            save_call(&mut record, output, "empty", || {
                call(&mut engine, &[], components, bits, accuracy, &[])
            })?;
            save_call(&mut record, output, "singleton_recovery", || {
                call(&mut engine, &rows, components, bits, accuracy, &["Success"])
            })?;
        }
        Ok(())
    })();
    match outcome {
        Ok(()) => {
            record["status"] = json!("complete_capture_not_yet_qualified");
            write(output, &record)?;
            Ok(0)
        }
        Err(error) => {
            record["status"] = json!("failed_or_incomplete");
            record["error"] = json!(error.to_string());
            write(output, &record)?;
            Ok(2)
        }
    }
}

#[test]
#[ignore = "source-only proposal: requires a separately reviewed finite supervisor and root numerical lease"]
fn capture_highpoint_arrays_from_fixture() {
    assert_eq!(
        run().expect("private highpoint capture setup"),
        0,
        "private highpoint capture failed; retain its ledger and stderr"
    );
}
