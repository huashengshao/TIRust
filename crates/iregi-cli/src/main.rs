//! Standalone graph preparation, tensor reduction and independent comparison.
use clap::{Args, Parser, Subcommand, ValueEnum};
use iregi_masters::{Complex, DoubleFloat, Float, NativeFloat};
use iregi_prepare::{NumeratorSource, PrepareOptions, PreparedGraph, prepare_dot};
use iregi_rust::{
    Algorithm, ContractionPlan, ContractionWorkspace, Convention, InvariantInput, KinematicsBatch,
    NumeratorBatch, ReductionPlan, ReductionSettings, Workspace, basis::Monomial,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf, process::ExitCode, time::Instant};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::rational::Rational,
    evaluate::{EvaluationDomain, ExpressionEvaluator},
};

type CliResult<T> = Result<T, String>;

#[derive(Parser)]
#[command(
    name = "iregi-rust",
    version,
    about = "Batched one-loop tensor reduction with Symbolica",
    long_about = "Prepare FeynKit DOT numerators, reduce explicit propagator kinematics, and compare against independently captured IREGI fixtures. Numerical inputs and outputs use decimal strings to preserve the selected precision."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Extract exact numerator coefficients and propagator routing from a DOT graph.
    Prepare(GraphArgs),
    /// Reduce a JSON sample or batch of explicit propagator kinematics.
    Reduce(ReduceArgs),
    /// Prepare a DOT numerator, reduce matching explicit kinematics, and contract it.
    Run(RunArgs),
    /// Compare every Laurent tensor component against captured IREGI fixtures.
    Compare(CompareArgs),
}
#[derive(Args)]
struct GraphArgs {
    /// FeynKit DOT file.
    graph: PathBuf,
    /// FeynKit model JSON snapshot.
    #[arg(long)]
    model: PathBuf,
    /// Derive the numerator by applying model Feynman rules.
    #[arg(long)]
    feynman_rules: bool,
    /// Include symbolic external wavefunctions.
    #[arg(long)]
    external_wavefunctions: bool,
    /// Select residual open tensor axes, e.g. --component 0,0.
    #[arg(long, value_delimiter = ',')]
    component: Option<Vec<usize>>,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Precision {
    F64,
    Dd,
    Float,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Method {
    Ibp,
    Pv,
    Hybrid,
}
impl From<Method> for Algorithm {
    fn from(v: Method) -> Self {
        match v {
            Method::Ibp => Self::Ibp,
            Method::Pv => Self::PassarinoVeltman,
            Method::Hybrid => Self::Hybrid,
        }
    }
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Normalization {
    Native,
    Oneloop,
    Madloop,
}
impl From<Normalization> for Convention {
    fn from(v: Normalization) -> Self {
        match v {
            Normalization::Native => Self::IregiNative,
            Normalization::Oneloop => Self::OneLoop,
            Normalization::Madloop => Self::MadLoop,
        }
    }
}
#[derive(Args)]
struct NumericArgs {
    /// binary64, Symbolica DoubleFloat, or arbitrary precision.
    #[arg(long, value_enum, default_value = "f64")]
    precision: Precision,
    /// Binary working precision for --precision float.
    #[arg(long,default_value_t=128,value_parser=clap::value_parser!(u32).range(2..))]
    bits: u32,
}
#[derive(Args)]
struct ReduceArgs {
    /// JSON sample or {"samples":[...]}; quote numerical kinematic values.
    #[arg(long)]
    input: PathBuf,
    #[command(flatten)]
    numeric: NumericArgs,
    #[arg(long, value_enum, default_value = "hybrid")]
    algorithm: Method,
    /// Maximum rank; defaults to the graph degree, input rank, or zero.
    #[arg(long)]
    rank: Option<u16>,
    #[arg(long, value_enum, default_value = "native")]
    convention: Normalization,
}
#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    graph: GraphArgs,
    #[command(flatten)]
    reduction: ReduceArgs,
}
#[derive(Args)]
struct CompareArgs {
    #[arg(long, default_value = "validation/reference/fixtures.json")]
    fixtures: PathBuf,
    #[command(flatten)]
    numeric: NumericArgs,
    /// Override the method specified in each fixture.
    #[arg(long, value_enum)]
    algorithm: Option<Method>,
    /// Select IDs containing this substring; may be repeated.
    #[arg(long = "case")]
    cases: Vec<String>,
    /// Maximum dimensionally normalized mixed absolute/relative error.
    #[arg(long, default_value = "1e-10")]
    tolerance: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum Literal {
    Text(String),
    Number(serde_json::Number),
}
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum ComplexLiteral {
    Pair([Literal; 2]),
    Real(Literal),
}
#[derive(Clone, Debug, Deserialize)]
struct Propagator {
    shift: [Literal; 4],
    mass_squared: ComplexLiteral,
    #[serde(default = "unit_power")]
    power: u16,
    #[serde(default)]
    edge_id: Option<usize>,
}
fn unit_power() -> u16 {
    1
}
#[derive(Clone, Debug, Deserialize)]
struct Sample {
    #[serde(default)]
    id: String,
    propagators: Vec<Propagator>,
    /// Packed upper triangle p_i.p_j, row-major including the diagonal.
    #[serde(default)]
    scalar_products: Option<Vec<Literal>>,
    mu: Literal,
    /// Full even offset: D=4+dimension_shift-2*epsilon.
    #[serde(default)]
    dimension_shift: i16,
    #[serde(default)]
    rank: Option<u16>,
    #[serde(default)]
    mode: Option<u8>,
    #[serde(default)]
    snap: bool,
    #[serde(default)]
    parameters: BTreeMap<String, ComplexLiteral>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum InputDocument {
    Batch { samples: Vec<Sample> },
    Single(Sample),
}
#[derive(Deserialize)]
struct Fixtures {
    schema_version: u32,
    cases: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    input: Sample,
    expected: Expected,
    #[serde(default)]
    reference_repaired: bool,
}
#[derive(Deserialize)]
struct Expected {
    stable: bool,
    moments: Vec<Vec<[f64; 2]>>,
}

fn main() -> ExitCode {
    match execute(Cli::parse()) {
        Ok((value, success)) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).expect("serializable output")
            );
            if success {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(error) => {
            eprintln!("iregi-rust: {error}");
            ExitCode::from(2)
        }
    }
}
fn execute(cli: Cli) -> CliResult<(Value, bool)> {
    match cli.command {
        Command::Prepare(args) => Ok((graph_json(&prepare(&args)?), true)),
        Command::Reduce(args) => dispatch_reduction(&args, None),
        Command::Run(args) => {
            let graph = prepare(&args.graph)?;
            dispatch_reduction(&args.reduction, Some(&graph))
        }
        Command::Compare(args) => match args.numeric.precision {
            Precision::F64 => compare(&args, 0f64),
            Precision::Dd => compare(&args, DoubleFloat::from(0.)),
            Precision::Float => compare(&args, Float::new(args.numeric.bits)),
        },
    }
}
fn read(path: &PathBuf) -> CliResult<String> {
    fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}
fn prepare(args: &GraphArgs) -> CliResult<PreparedGraph> {
    prepare_dot(
        &read(&args.model)?,
        &read(&args.graph)?,
        &PrepareOptions {
            source: if args.feynman_rules {
                NumeratorSource::FeynmanRules
            } else {
                NumeratorSource::Supplied
            },
            external_wavefunctions: args.external_wavefunctions,
            component: args.component.clone(),
        },
    )
    .map_err(|e| e.to_string())
}
fn graph_json(graph: &PreparedGraph) -> Value {
    json!({"schema_version":1,"name":graph.name,"rank":graph.numerator.rank(),
        "loop_variables":graph.numerator.variables.each_ref().map(|a|a.to_canonical_string()),
        "numerator":graph.numerator.expression().to_canonical_string(),
        "terms":graph.numerator.terms.iter().map(|t|json!({"exponents":t.exponents,"coefficient":t.coefficient.to_canonical_string()})).collect::<Vec<_>>(),
        "external_edges":graph.external_edges,
        "external_legs":graph.external_legs.iter().map(|leg|json!({"edge_id":leg.edge_id,"index":leg.index,"name":leg.name,"incoming":leg.incoming,"particle_pdg":leg.particle_pdg})).collect::<Vec<_>>(),
        "propagators":graph.propagators.iter().map(|p|json!({"edge_id":p.edge_id,"particle_pdg":p.particle_pdg,
            "loop_coefficient":p.loop_coefficient,"external_coefficients":p.external_coefficients,"mass_squared":p.mass_squared.to_canonical_string()})).collect::<Vec<_>>(),
        "routing_contract":"run requires this propagator order; canonicalize each s*q+r denominator to q+s*r when s=-1"})
}
fn dispatch_reduction(
    args: &ReduceArgs,
    graph: Option<&PreparedGraph>,
) -> CliResult<(Value, bool)> {
    let document: InputDocument =
        serde_json::from_str(&read(&args.input)?).map_err(|e| format!("input JSON: {e}"))?;
    let mut samples = match document {
        InputDocument::Batch { samples } => samples,
        InputDocument::Single(s) => vec![s],
    };
    if samples.is_empty() {
        return Err("input must contain at least one sample".into());
    }
    for (index, sample) in samples.iter_mut().enumerate() {
        if sample.id.is_empty() {
            sample.id = format!("sample-{index}");
        }
    }
    if let Some(graph) = graph {
        for sample in &samples {
            if sample.propagators.len() != graph.propagators.len() {
                return Err(format!(
                    "sample {} has {} propagators; graph requires {}",
                    sample.id,
                    sample.propagators.len(),
                    graph.propagators.len()
                ));
            }
            for (input, routing) in sample.propagators.iter().zip(&graph.propagators) {
                if input.edge_id.is_some_and(|edge| edge != routing.edge_id) {
                    return Err(format!(
                        "sample {} propagator is out of graph order; expected edge {}",
                        sample.id, routing.edge_id
                    ));
                }
                if routing.loop_coefficient.abs() != 1 {
                    return Err("run requires propagator loop coefficients +1 or -1".into());
                }
            }
        }
    }
    let numerator_rank = graph
        .map(|g| {
            u16::try_from(g.numerator.rank()).map_err(|_| "numerator rank exceeds u16".to_owned())
        })
        .transpose()?;
    let rank = args
        .rank
        .or(numerator_rank)
        .or(samples[0].rank)
        .unwrap_or(0);
    if args.rank.is_none()
        && samples
            .iter()
            .any(|sample| sample.rank.is_some_and(|declared| declared != rank))
    {
        return Err("batch ranks differ from the inferred rank; use --rank to request one common maximum rank".into());
    }
    if numerator_rank.is_some_and(|required| rank < required) {
        return Err("requested rank is below the numerator degree".into());
    }
    let settings = settings(
        &samples[0],
        rank,
        args.algorithm.into(),
        args.convention.into(),
    )?;
    let key = serde_json::to_string(&settings).unwrap();
    for sample in &samples {
        if settings_key(sample, rank, args.algorithm.into(), args.convention.into())? != key {
            return Err("batch samples must share topology, powers, dimension shift, and scalar-product input convention".into());
        }
    }
    match args.numeric.precision {
        Precision::F64 => reduce(&samples, settings, 0f64, graph),
        Precision::Dd => reduce(&samples, settings, DoubleFloat::from(0.), graph),
        Precision::Float => reduce(&samples, settings, Float::new(args.numeric.bits), graph),
    }
}
fn settings(
    sample: &Sample,
    rank: u16,
    algorithm: Algorithm,
    convention: Convention,
) -> CliResult<ReductionSettings> {
    if sample.snap {
        return Err(format!(
            "sample {} requests legacy snapping; this CLI preserves kinematics",
            sample.id
        ));
    }
    if sample.dimension_shift % 2 != 0 {
        return Err("dimension_shift must be even: D=4+dimension_shift-2*epsilon".into());
    }
    let mut settings = ReductionSettings::new(sample.propagators.len(), rank);
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
fn settings_key(
    sample: &Sample,
    rank: u16,
    algorithm: Algorithm,
    convention: Convention,
) -> CliResult<String> {
    Ok(serde_json::to_string(&settings(sample, rank, algorithm, convention)?).unwrap())
}
/// Symbolica rounds decimal strings directly at the destination precision.
/// Rational expressions remain exact until conversion; neither uses binary64.
fn parse_real<T: NativeFloat>(
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
        Float::parse(&text, Some(prototype.get_precision()))
            .map_err(|e| format!("invalid decimal {text:?}: {e}"))?
            .try_to_rational()
            .ok_or_else(|| format!("input {text:?} must be finite"))?
    };
    let value = prototype.from_rational(&rational);
    if !value.is_finite() {
        return Err(format!("input {text:?} exceeds the selected numeric range"));
    }
    Ok(value)
}
fn parse_complex<T: NativeFloat>(
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
fn pack<T: NativeFloat>(
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
fn format_complex<T: NativeFloat>(v: &Complex<T>, bits: u32) -> [String; 2] {
    let digits = (u64::from(bits) * 30103 / 100000 + 3) as usize;
    [
        format!("{:.*e}", digits, v.re),
        format!("{:.*e}", digits, v.im),
    ]
}

fn characteristic_scale<T: NativeFloat>(sample: &Sample, prototype: &T) -> CliResult<T> {
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

fn reduce<T: NativeFloat>(
    samples: &[Sample],
    settings: ReductionSettings,
    prototype: T,
    graph: Option<&PreparedGraph>,
) -> CliResult<(Value, bool)>
where
    Complex<T>: EvaluationDomain,
{
    let batch = pack(samples, &prototype, false)?;
    let started = Instant::now();
    let plan = ReductionPlan::new(settings.clone()).map_err(|e| e.to_string())?;
    let mut workspace = Workspace::new(&plan, prototype.clone()).map_err(|e| e.to_string())?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;
    let reduction_seconds = started.elapsed().as_secs_f64();
    let contracted = if let Some(graph) = graph {
        Some(contract(graph, samples, &plan, &result, &prototype)?)
    } else {
        None
    };
    let mut all_success = true;
    let rows: Vec<_> = samples
        .iter()
        .enumerate()
        .map(|(row, sample)| match result.lane(row) {
            Ok(values) => {
                let moments: Vec<_> = values
                    .chunks_exact(4)
                    .map(|poles| {
                        poles
                            .iter()
                            .map(|v| format_complex(v, prototype.get_precision()))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let mut value = json!({"id":sample.id,"status":"success","moments":moments,"fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row]});
                if let Some(contracted) = &contracted {
                    match &contracted[row] {
                        Some(Ok(poles)) => value["integral"] = json!(poles.iter().map(|v| format_complex(v, prototype.get_precision())).collect::<Vec<_>>()),
                        Some(Err(reason)) => {
                            all_success = false;
                            value["status"] = json!("failed");
                            value["reason"] = json!(reason);
                        }
                        None => unreachable!("successful reduction lane was contracted"),
                    }
                }
                value
            }
            Err(status) => {
                all_success = false;
                json!({"id":sample.id,"status":"failed","reason":format!("{status:?}"),"fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row]})
            }
        })
        .collect();
    let stats = workspace.last_execution();
    let mut output = json!({"schema_version":1,"precision_bits":prototype.get_precision(),"settings":settings,
        "algorithm_requested":settings.algorithm,"algorithm_used":plan.algorithm_used(),"preparation_notes":plan.preparation_notes(),
        "basis":result.basis,"poles":["uv1","ir1","ir2","finite"],"samples":rows,
        "timing":{"plan_and_workspace_seconds":preparation_seconds,"reduction_seconds":reduction_seconds,
            "note":"single observed execution; includes Rust masters, excludes file I/O and graph preparation; not a statistically qualified benchmark"},
        "execution":{"input_rows":stats.input_rows,"argument_rows":stats.argument_rows,"master_family_batches":stats.master_family_batches,
            "master_rows":stats.master_rows,"reduction_rows":stats.reduction_rows,
            "fallback_rows":stats.fallback_rows,"fallback_successes":stats.fallback_successes,"analytic_rows":stats.analytic_rows}});
    if let Some(graph) = graph {
        output["graph"] = graph_json(graph);
        output["kinematics_contract"] = json!(
            "explicit canonical (q+p_i) propagators in reported graph edge order; external routing is reported but not inferred from external momenta"
        );
    }
    Ok((output, all_success))
}

fn contract<T: NativeFloat>(
    graph: &PreparedGraph,
    samples: &[Sample],
    plan: &ReductionPlan,
    result: &iregi_rust::BatchResult<T>,
    prototype: &T,
) -> CliResult<Vec<Option<CliResult<Vec<Complex<T>>>>>>
where
    Complex<T>: EvaluationDomain,
{
    let names: Vec<_> = samples[0].parameters.keys().cloned().collect();
    for sample in samples {
        if sample.parameters.keys().cloned().collect::<Vec<_>>() != names {
            return Err("all samples must bind the same numerator parameter names".into());
        }
    }
    let parameters = names
        .iter()
        .map(|n| Atom::parse(n, "iregi_cli", Default::default()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("numerator parameter: {e}"))?;
    let expressions: Vec<_> = graph
        .numerator
        .terms
        .iter()
        .map(|term| term.coefficient.clone())
        .collect();
    let width = expressions.len();
    let mut input = Vec::new();
    let mut valid = Vec::new();
    for (row, sample) in samples.iter().enumerate() {
        if result.lane(row).is_ok() {
            valid.push(row);
            for name in &names {
                input.push(parse_complex(&sample.parameters[name], prototype, false)?);
            }
        }
    }
    if valid.is_empty() {
        return Ok(vec![None; samples.len()]);
    }
    let mut coefficients =
        vec![Complex::new(prototype.zero(), prototype.zero()); samples.len() * width];
    if width != 0 {
        let exact: ExpressionEvaluator<Complex<Rational>> = Atom::evaluator_multiple(&expressions, &parameters)
            .direct_translation(true).horner_iterations(0).cores(1).build()
            .map_err(|e| format!("numerator parameters incomplete or unsupported: {e}; bind remaining coefficient symbols/functions in the input parameters map"))?;
        let mut evaluator = exact
            .map_coeff_with_prec(
                &|v| {
                    Complex::new(
                        prototype.from_rational(&v.re),
                        prototype.from_rational(&v.im),
                    )
                },
                prototype.get_precision(),
            )
            .into_batch_evaluator();
        let mut output =
            vec![Complex::new(prototype.zero(), prototype.zero()); valid.len() * width];
        evaluator
            .evaluate_batch(&input, &mut output, valid.len())
            .map_err(|e| format!("numerator coefficient evaluation: {e}"))?;
        for (compact, &row) in valid.iter().enumerate() {
            coefficients[row * width..(row + 1) * width]
                .clone_from_slice(&output[compact * width..(compact + 1) * width]);
        }
    }
    let support: Vec<_> = graph
        .numerator
        .terms
        .iter()
        .map(|term| Monomial(term.exponents))
        .collect();
    let contraction_plan = ContractionPlan::new(plan, &support).map_err(|e| e.to_string())?;
    let mut workspace = ContractionWorkspace::new(&contraction_plan, prototype.clone())
        .map_err(|e| e.to_string())?;
    let contracted = workspace
        .contract_batch(
            result,
            &NumeratorBatch {
                rows: samples.len(),
                components: 1,
                values: coefficients,
            },
        )
        .map_err(|e| e.to_string())?;
    Ok((0..samples.len())
        .map(|row| {
            result.lane(row).ok().map(|_| {
                contracted
                    .lane(row)
                    .map(|poles| poles.to_vec())
                    .map_err(|status| format!("numerator contraction: {status:?}"))
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::float::SingleFloat;

    #[test]
    fn decimal_input_keeps_bits_beyond_binary64() {
        let literal = Literal::Text("1.0000000000000000000000001".into());
        let wide = parse_real(&literal, &Float::new(160), false).unwrap();
        let double = parse_real(&literal, &DoubleFloat::from(0.), false).unwrap();
        assert!(wide > Float::with_val(160, 1));
        assert!(double > DoubleFloat::from(1.));
        assert_eq!(parse_real(&literal, &0f64, false).unwrap(), 1.);
    }

    #[test]
    fn rationals_and_number_policy_are_explicit() {
        let third = parse_real(&Literal::Text("1/3".into()), &Float::new(160), false).unwrap();
        assert_eq!(
            third,
            Float::new(160).from_rational(&Rational::from((1, 3)))
        );
        let number = Literal::Number(serde_json::Number::from(3));
        assert!(parse_real(&number, &0f64, false).is_err());
        assert_eq!(parse_real(&number, &0f64, true).unwrap(), 3.);
        assert!(parse_real(&Literal::Text("NaN".into()), &0f64, false).is_err());
    }

    #[test]
    fn reference_json_numbers_preserve_original_decimal_digits() {
        let text = "1.0000000000000000000000001";
        let literal: Literal = serde_json::from_str(text).unwrap();
        let Literal::Number(number) = &literal else {
            panic!("expected a JSON number");
        };
        assert_eq!(number.to_string(), text);
        let precise = parse_real(&literal, &Float::new(160), true).unwrap();
        assert!(precise > Float::with_val(160, 1));
        assert!(parse_real(&literal, &Float::new(160), false).is_err());
    }

    #[test]
    fn full_dimension_offset_maps_to_core_half_shift() {
        let sample:Sample=serde_json::from_str(r#"{"propagators":[{"shift":["0","0","0","0"],"mass_squared":"4"}],"mu":"3","dimension_shift":2}"#).unwrap();
        assert_eq!(
            settings(&sample, 0, Algorithm::Ibp, Convention::IregiNative)
                .unwrap()
                .dimension_shift,
            1
        );
        let mut odd = sample;
        odd.dimension_shift = 1;
        assert!(settings(&odd, 0, Algorithm::Ibp, Convention::IregiNative).is_err());
    }

    #[test]
    fn supplied_scalar_products_keep_precision_and_packing() {
        let sample: Sample = serde_json::from_str(r#"{"propagators":[{"shift":["0","0","0","0"],"mass_squared":"4"}],"scalar_products":["1/3"],"mu":"3"}"#).unwrap();
        assert_eq!(
            settings(&sample, 0, Algorithm::Ibp, Convention::IregiNative)
                .unwrap()
                .invariants,
            InvariantInput::SuppliedScalarProducts
        );
        let packed = pack(&[sample.clone()], &Float::new(160), false).unwrap();
        assert_eq!(packed.values.len(), 7);
        assert_eq!(
            packed.values[4].re,
            Float::new(160).from_rational(&Rational::from((1, 3)))
        );
        assert_eq!(packed.values[5].re, Float::with_val(160, 4));
        let mut wrong = sample;
        wrong.scalar_products = Some(Vec::new());
        assert!(settings(&wrong, 0, Algorithm::Ibp, Convention::IregiNative).is_err());
    }

    #[test]
    fn comparison_scale_accounts_for_kinematics_beyond_mu() {
        let sample: Sample = serde_json::from_str(
            r#"{"propagators":[{"shift":["0","1","0","0"],"mass_squared":"9"}],"mu":"0.001"}"#,
        )
        .unwrap();
        assert_eq!(characteristic_scale(&sample, &0f64).unwrap(), 3.);
        assert_eq!(
            characteristic_scale(&sample, &Float::new(160)).unwrap(),
            Float::with_val(160, 3)
        );
    }
}

fn compare<T: NativeFloat>(args: &CompareArgs, prototype: T) -> CliResult<(Value, bool)>
where
    Complex<T>: EvaluationDomain,
{
    let fixtures: Fixtures =
        serde_json::from_str(&read(&args.fixtures)?).map_err(|e| format!("fixture JSON: {e}"))?;
    if fixtures.schema_version != 1 {
        return Err("unsupported fixture schema version".into());
    }
    let selected: Vec<_> = fixtures
        .cases
        .into_iter()
        .filter(|f| args.cases.is_empty() || args.cases.iter().any(|s| f.input.id.contains(s)))
        .collect();
    if selected.is_empty() {
        return Err("no fixtures matched".into());
    }
    let tolerance = parse_real(&Literal::Text(args.tolerance.clone()), &prototype, false)?;
    if tolerance <= prototype.zero() {
        return Err("tolerance must be positive".into());
    }
    let mut groups = BTreeMap::<String, (ReductionSettings, Vec<Fixture>)>::new();
    for fixture in selected {
        let algorithm = if let Some(m) = args.algorithm {
            m.into()
        } else {
            match fixture.input.mode.unwrap_or(2) {
                0 => Algorithm::Ibp,
                1 => Algorithm::PassarinoVeltman,
                2 => Algorithm::Hybrid,
                n => return Err(format!("unknown fixture mode {n}")),
            }
        };
        let settings = settings(
            &fixture.input,
            fixture.input.rank.unwrap_or(0),
            algorithm,
            Convention::IregiNative,
        )?;
        let key = serde_json::to_string(&settings).unwrap();
        groups
            .entry(key)
            .or_insert_with(|| (settings, Vec::new()))
            .1
            .push(fixture);
    }
    let mut rows = Vec::new();
    let mut failures = 0;
    for (_, (settings, fixtures)) in groups {
        let samples: Vec<_> = fixtures.iter().map(|f| f.input.clone()).collect();
        let attempt: CliResult<_> = (|| {
            let plan = ReductionPlan::new(settings.clone()).map_err(|e| e.to_string())?;
            let mut workspace =
                Workspace::new(&plan, prototype.clone()).map_err(|e| e.to_string())?;
            let result = workspace
                .reduce_batch(&pack(&samples, &prototype, true)?)
                .map_err(|e| e.to_string())?;
            Ok((
                result,
                plan.algorithm_used(),
                plan.preparation_notes().to_vec(),
            ))
        })();
        match attempt {
            Err(error) => {
                failures += fixtures.len();
                for fixture in &fixtures {
                    rows.push(json!({"id":fixture.input.id,"status":"failed","reason":error,"algorithm_requested":settings.algorithm}));
                }
            }
            Ok((result, algorithm_used, preparation_notes)) => {
                for (row, fixture) in fixtures.iter().enumerate() {
                    let evaluation = (|| -> CliResult<(T, usize)> {
                        if !fixture.expected.stable {
                            return Err("reference fixture is unstable".into());
                        }
                        let actual = result.lane(row).map_err(|s| format!("{s:?}"))?;
                        if fixture.expected.moments.len() * 4 != actual.len()
                            || fixture.expected.moments.iter().any(|p| p.len() != 4)
                        {
                            return Err("reference tensor shape differs".into());
                        }
                        let characteristic = characteristic_scale(&fixture.input, &prototype)?;
                        let powers: i32 = fixture
                            .input
                            .propagators
                            .iter()
                            .map(|p| i32::from(p.power))
                            .sum();
                        let mut maximum = prototype.zero();
                        let mut worst = 0;
                        for (index, (got, expected)) in actual
                            .iter()
                            .zip(fixture.expected.moments.iter().flatten())
                            .enumerate()
                        {
                            // Reference coefficients are intentionally binary64 data.
                            let expected = Complex::new(
                                prototype
                                    .from_rational(&Float::with_val(53, expected[0]).to_rational()),
                                prototype
                                    .from_rational(&Float::with_val(53, expected[1]).to_rational()),
                            );
                            let dimension = 4
                                + i32::from(fixture.input.dimension_shift)
                                + result.basis[index / 4].rank() as i32
                                - 2 * powers;
                            let scale = if dimension >= 0 {
                                characteristic.pow(dimension as u64)
                            } else {
                                characteristic.pow((-dimension) as u64).inv()
                            };
                            if !scale.is_finite() || scale <= prototype.zero() {
                                return Err("comparison normalization is outside the selected numeric range; increase precision".into());
                            }
                            let difference = (got.clone() - &expected) / &scale;
                            let expected = expected / &scale;
                            if [&difference.re, &difference.im, &expected.re, &expected.im]
                                .iter()
                                .any(|v| !v.is_finite())
                            {
                                return Err(
                                    "nonfinite normalized comparison; increase precision".into()
                                );
                            }
                            let norm = NativeFloat::hypot(&expected.re, &expected.im);
                            if !norm.is_finite() {
                                return Err("nonfinite comparison norm; increase precision".into());
                            }
                            let denominator = if norm > prototype.one() {
                                norm
                            } else {
                                prototype.one()
                            };
                            let error =
                                NativeFloat::hypot(&difference.re, &difference.im) / denominator;
                            if !error.is_finite() {
                                return Err("nonfinite comparison error; increase precision".into());
                            }
                            if error > maximum {
                                maximum = error;
                                worst = index;
                            }
                        }
                        Ok((maximum, worst))
                    })();
                    match evaluation {
                        Ok((error, worst)) => {
                            let passed = error <= tolerance;
                            if !passed {
                                failures += 1;
                            }
                            rows.push(json!({"id":fixture.input.id,"status":if passed{"passed"}else{"failed"},"max_normalized_error":format!("{error:e}"),
                            "algorithm_requested":settings.algorithm,"algorithm_used":algorithm_used,"preparation_notes":preparation_notes,
                            "fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row],
                            "worst_moment":worst/4,"worst_pole":worst%4,"reference_repaired":fixture.reference_repaired}));
                        }
                        Err(reason) => {
                            failures += 1;
                            rows.push(
                                json!({"id":fixture.input.id,"status":"failed","reason":reason,"algorithm_requested":settings.algorithm,
                                "algorithm_used":algorithm_used,"preparation_notes":preparation_notes,"fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row]}),
                            );
                        }
                    }
                }
            }
        }
    }
    rows.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Ok((
        json!({"schema_version":1,"precision_bits":prototype.get_precision(),"tolerance":args.tolerance,
        "error_definition":"abs(actual-reference)/Q^dimension divided by max(1,abs(reference/Q^dimension)); Q=max(mu,abs(shift components),sqrt(abs(mass_squared)))",
        "cases":rows,"failed":failures,"passed":rows.len()-failures,
        "reference":"captured original IREGI binary64 values; no reference process transport is timed"}),
        failures == 0,
    ))
}
