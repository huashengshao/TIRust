//! Standalone graph preparation, tensor reduction and independent comparison.
use clap::{Args, Parser, Subcommand, ValueEnum};
use iregi_cli::input::*;
use iregi_masters::{Complex, DoubleFloat, Float, NativeFloat};
use iregi_prepare::{NumeratorSource, PrepareOptions, PreparedGraph, prepare_dot};
use iregi_rust::{
    Algorithm, BatchResult, CertificateExecutionStats, ContractionWorkspace, Convention,
    ExecutionStats, KinematicsBatch, NumeratorBatch, ReductionSettings, RescueScalar,
    RescueSettings, RescueStatus, RescueTrigger, TensorEngineExecution, TensorEngineKind,
    TensorReductionPlan, TensorRescueWorkspace, TensorWorkspace,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf, process::ExitCode, time::Instant};
use symbolica::{atom::AtomCore, evaluate::EvaluationDomain};

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
    /// Check supplied scalar products against Cartesian shifts with the six-point tensor API.
    #[arg(long)]
    check_scalar_products: bool,
    /// binary64, Symbolica DoubleFloat, or arbitrary precision.
    #[arg(long, value_enum, default_value = "f64")]
    precision: Precision,
    /// Binary working precision for --precision float.
    #[arg(long,default_value_t=128,value_parser=clap::value_parser!(u32).range(2..))]
    bits: u32,
    /// Accuracy parameter N for the core tolerance 2^(19-N); defaults to output precision.
    #[arg(long,value_parser=clap::value_parser!(u32).range(1..))]
    accuracy_bits: Option<u32>,
    /// Verify every successful or unstable tensor lane at two higher precisions.
    #[arg(long)]
    rescue: bool,
    /// First rescue precision; defaults to the next power of two above output bits, at least 128.
    #[arg(long,requires="rescue",value_parser=clap::value_parser!(u32).range(2..))]
    rescue_bits: Option<u32>,
    /// Independent verification precision; defaults to twice --rescue-bits.
    #[arg(long,requires="rescue",value_parser=clap::value_parser!(u32).range(2..))]
    verification_bits: Option<u32>,
    /// Compute every lane at the selected precision only: lanes the binary64
    /// direct engine refuses are reported unstable instead of being rerun at
    /// 106, 256 or 512 bits.
    #[arg(long, conflicts_with = "rescue")]
    no_precision_rescue: bool,
    /// COLLIER-style UV scale mu_UV^2 (default: each lane's mu^2).
    #[arg(long)]
    mu_uv2: Option<f64>,
    /// COLLIER-style IR scale mu_IR^2 (default: each lane's mu^2).
    #[arg(long)]
    mu_ir2: Option<f64>,
    /// Numerical value of the UV pole added to the finite slot (COLLIER Delta_UV).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    delta_uv: f64,
    /// Numerical values of the IR poles (COLLIER Delta_IR^(1), Delta_IR^(2)).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    delta_ir1: f64,
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    delta_ir2: f64,
    /// COLLIER's c(eps) normalization of the IR double pole (requires --convention one-loop).
    #[arg(long)]
    collier_normalization: bool,
}

#[derive(Clone, Copy)]
struct NumericPolicy {
    output_bits: u32,
    accuracy_bits: u32,
    rescue: Option<RescueSettings>,
    precision_rescue: iregi_rust::PrecisionRescue,
    regulators: iregi_rust::Regulators,
}

impl NumericArgs {
    fn resolve(&self, output_bits: u32) -> CliResult<NumericPolicy> {
        let accuracy_bits = self.accuracy_bits.unwrap_or(output_bits);
        if accuracy_bits == 0 || accuracy_bits > output_bits {
            return Err("--accuracy-bits must be positive and no greater than the selected output precision".into());
        }
        let rescue = if self.rescue {
            let working_bits = if let Some(bits) = self.rescue_bits {
                bits
            } else {
                output_bits
                    .checked_add(1)
                    .and_then(|bits| bits.max(128).checked_next_power_of_two())
                    .ok_or("output precision is too large to select default rescue precisions")?
            };
            let verification_bits = match self.verification_bits {
                Some(bits) => bits,
                None => working_bits.checked_mul(2).ok_or(
                    "rescue precision is too large to select a default verification precision",
                )?,
            };
            if working_bits <= output_bits || verification_bits <= working_bits {
                return Err(
                    "--rescue requires output bits < --rescue-bits < --verification-bits".into(),
                );
            }
            Some(RescueSettings {
                working_bits,
                verification_bits,
                accuracy_bits: Some(accuracy_bits),
                trigger: RescueTrigger::VerifyAllSuccessful,
            })
        } else {
            if self.rescue_bits.is_some() || self.verification_bits.is_some() {
                return Err("--rescue-bits and --verification-bits require --rescue".into());
            }
            None
        };
        Ok(NumericPolicy {
            output_bits,
            accuracy_bits,
            rescue,
            precision_rescue: if self.no_precision_rescue {
                iregi_rust::PrecisionRescue::Disabled
            } else {
                iregi_rust::PrecisionRescue::Automatic
            },
            regulators: iregi_rust::Regulators {
                mu_uv_squared: self.mu_uv2,
                mu_ir_squared: self.mu_ir2,
                delta_uv: self.delta_uv,
                delta_ir1: self.delta_ir1,
                delta_ir2: self.delta_ir2,
                normalization: if self.collier_normalization {
                    iregi_rust::RegulatorNormalization::Collier
                } else {
                    iregi_rust::RegulatorNormalization::Convention
                },
            },
        })
    }
}

impl NumericPolicy {
    fn json(&self) -> Value {
        json!({"output_bits":self.output_bits,"accuracy_bits":self.accuracy_bits,
            "tolerance_definition":"2^(19-accuracy_bits); a numerical convergence target, not a certified error bound",
            "rescue":self.rescue.map(|settings|json!({"trigger":"verify_all_successful",
                "working_bits":settings.working_bits,"verification_bits":settings.verification_bits})),
            "verification_scope":if self.rescue.is_some(){"tensor_moments"}else{"none"},
            "input_promotion":"exactly promotes the already parsed output-domain input; does not recover decimal digits rounded during input parsing",
            "precision_rescue":format!("{:?}",self.precision_rescue),
            "regulators":self.regulators,
            "scope":"tensor coefficients at the exactly represented input; numerator evaluation and contraction use output precision and are not independently verified by tensor rescue"})
    }
}

enum NumericalWorkspace<T: RescueScalar> {
    Ordinary(TensorWorkspace<T>),
    Verified {
        workspace: TensorRescueWorkspace<T>,
        algorithm_used: Algorithm,
        preparation_notes: Vec<String>,
        regulators: iregi_rust::Regulators,
    },
}

impl<T: RescueScalar> NumericalWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    fn new(plan: &TensorReductionPlan, prototype: T, policy: NumericPolicy) -> CliResult<Self> {
        if let Some(settings) = policy.rescue {
            let workspace = plan
                .rescue_workspace(prototype, settings)
                .map_err(|e| e.to_string())?;
            let algorithm_used = workspace.algorithm_used();
            let preparation_notes = workspace.preparation_notes().to_vec();
            Ok(Self::Verified {
                workspace,
                algorithm_used,
                preparation_notes,
                regulators: policy.regulators,
            })
        } else {
            let mut workspace = TensorWorkspace::new(plan, prototype).map_err(|e| e.to_string())?;
            workspace
                .set_accuracy_precision(policy.accuracy_bits)
                .map_err(|e| e.to_string())?;
            workspace.set_regulators(policy.regulators);
            workspace.set_precision_rescue(policy.precision_rescue);
            Ok(Self::Ordinary(workspace))
        }
    }

    fn reduce_batch(&mut self, batch: &KinematicsBatch<T>) -> CliResult<BatchResult<T>> {
        match self {
            Self::Ordinary(workspace) => workspace.reduce_batch(batch),
            Self::Verified { workspace, regulators, .. } => workspace.reduce_batch(batch).and_then(|mut result| {
                regulators.apply(batch, &mut result)?;
                Ok(result)
            }),
        }
        .map_err(|e| e.to_string())
    }

    fn execution(&self) -> CliResult<Value> {
        Ok(execution_json(match self {
            Self::Ordinary(workspace) => {
                workspace.execution_summary().map_err(|e| e.to_string())?
            }
            Self::Verified { workspace, .. } => workspace.last_execution().primary,
        }))
    }

    fn engine_execution(&self) -> Value {
        match self {
            Self::Ordinary(workspace) => engine_execution_json(workspace.engine_execution()),
            Self::Verified { workspace, .. } => json!({
                "engine":format!("{:?}", workspace.engine_kind()),
                "execution":execution_json(workspace.last_execution().primary),
                "primary":workspace.primary_engine_execution().map(engine_execution_json),
                "working":workspace.working_engine_execution().map(engine_execution_json),
                "verification":workspace.verification_engine_execution().map(engine_execution_json),
                "certificate_scope":"available certificate counters are retained in the actual tagged stage; the legacy generic rescue does not expose them"
            }),
        }
    }

    fn algorithm_used(&self, plan: &TensorReductionPlan) -> CliResult<Algorithm> {
        match self {
            Self::Ordinary(workspace) if workspace.engine_kind() == TensorEngineKind::Generic => {
                Ok(plan
                    .generic_plan()
                    .map_err(|e| e.to_string())?
                    .algorithm_used())
            }
            // Specialized wrappers preserve their validated requested algorithm.
            // The tagged engine/counters describe their actual nested work.
            Self::Ordinary(_) => Ok(plan.settings().algorithm),
            Self::Verified { algorithm_used, .. } => Ok(*algorithm_used),
        }
    }

    fn preparation_notes(&self) -> &[String] {
        match self {
            Self::Ordinary(workspace) => workspace.preparation_notes(),
            Self::Verified {
                preparation_notes, ..
            } => preparation_notes,
        }
    }

    fn diagnostics(&self) -> Value {
        match self {
            Self::Ordinary(workspace) => json!(workspace.last_diagnostics()),
            Self::Verified { workspace, .. } => json!(workspace.last_diagnostics()),
        }
    }

    fn rescue_execution(&self) -> Value {
        match self {
            Self::Ordinary(_) => json!({"enabled":false}),
            Self::Verified { workspace, .. } => {
                let stats = workspace.last_execution();
                json!({"enabled":true,"requested_rows":stats.requested_rows,"accepted_rows":stats.accepted_rows,
                    "corrected_successful_rows":stats.corrected_successful_rows,
                    "working":execution_json(stats.working),"verification":execution_json(stats.verification)})
            }
        }
    }

    fn lane_rescue(&self, row: usize) -> Value {
        let Self::Verified { workspace, .. } = self else {
            return json!({"status":"not_requested"});
        };
        let mut value = match &workspace.last_rescue_statuses()[row] {
            RescueStatus::NotRequested => json!({"status":"not_requested"}),
            RescueStatus::Converged {
                working_bits,
                verification_bits,
                primary_agreed,
            } => {
                json!({"status":"converged","working_bits":working_bits,"verification_bits":verification_bits,"primary_agreed":primary_agreed})
            }
            RescueStatus::Rejected(reason) => json!({"status":"rejected","reason":reason}),
        };
        value["primary_status"] = json!(format!("{:?}", workspace.last_primary_statuses()[row]));
        value
    }
}

fn execution_json(stats: ExecutionStats) -> Value {
    json!({"input_rows":stats.input_rows,"argument_rows":stats.argument_rows,"master_family_batches":stats.master_family_batches,
        "master_rows":stats.master_rows,"pentagon_box_rows":stats.pentagon_box_rows,"reduction_rows":stats.reduction_rows,
        "fallback_rows":stats.fallback_rows,"fallback_successes":stats.fallback_successes,"analytic_rows":stats.analytic_rows})
}

fn certificate_execution_json(stats: CertificateExecutionStats) -> Value {
    json!({"input_rows":stats.input_rows,"evaluated_rows":stats.evaluated_rows,
        "passes":stats.passes,"reduction_batches":stats.reduction_batches})
}

fn engine_execution_json(detail: TensorEngineExecution) -> Value {
    match detail {
        TensorEngineExecution::Direct(stats) => json!({
            "engine":"Direct",
            "execution":TensorEngineExecution::Direct(stats.clone()).summary().ok().map(execution_json),
            "input_rows":stats.input_rows,"master_rows":stats.master_rows,
            "direct_successes":stats.direct_successes,"fallback_rows":stats.fallback_rows,
            "fallback_successes":stats.fallback_successes,
            "precision_rescue":stats.precision_rescue.iter().map(|tier| json!({
                "bits":tier.bits,"rows":tier.rows,"successes":tier.successes})).collect::<Vec<_>>(),
            "symbolic_rows":stats.symbolic_rows,
            "rotation_rows":stats.rotation_rows,"rotation_failures":stats.rotation_failures,
            "checked_rows":stats.checked_rows,"checked_successes":stats.checked_successes,
            "expansion_rows":stats.expansion_rows,"expansion_successes":stats.expansion_successes,
            "fallback_preparation_error":stats.fallback_preparation_error,
            "fallback":stats.fallback.map(|child| engine_execution_json(*child)),
        }),
        TensorEngineExecution::Generic {
            execution,
            certificate_execution,
        } => json!({
            "engine":"Generic","execution":execution_json(execution),
            "certificate_execution":certificate_execution_json(certificate_execution)
        }),
        TensorEngineExecution::LegacyNoOp {
            input_rows, accepted_rows, rejected_rows, query_count,
            certificate_execution, child, child_preparation_attempts, child_preparation_error,
        } => json!({
            "engine":"LegacyNoOp", "input_rows":input_rows,
            "accepted_rows":accepted_rows,"rejected_rows":rejected_rows,
            "query_count":query_count,
            "certificate_execution":certificate_execution_json(certificate_execution),
            "child":child.map(|value|engine_execution_json(*value)),
            "child_preparation_attempts":child_preparation_attempts,
            "child_preparation_error":child_preparation_error
        }),
        TensorEngineExecution::SixPoint(stats) => json!({
            "engine":"SixPoint","parent_input_rows":stats.parent_input_rows,
            "child_batch_calls":stats.child_batch_calls,"packed_child_rows":stats.packed_child_rows,
            "parent_batch_calls":stats.parent_batch_calls,"packed_parent_rows":stats.packed_parent_rows,
            "child_execution":execution_json(stats.child_execution),
            "parent_execution":execution_json(stats.parent_execution),
            "child_certificate_execution":certificate_execution_json(stats.child_certificate_execution),
            "parent_certificate_execution":certificate_execution_json(stats.parent_certificate_execution)
        }),
        TensorEngineExecution::MassTaylor(stats) => json!({
            "engine":"MassTaylor", "input_rows":stats.input_rows,
            "certificate_execution":certificate_execution_json(stats.certificate_execution),
            "user_components":stats.user_components,
            "internal_components":stats.internal_components,
            "unit_execution":execution_json(stats.unit_execution),
            "projection_execution":execution_json(stats.projection_execution),
            "unit_algorithm_used":format!("{:?}",stats.unit_algorithm_used),
            "requested_algorithm":format!("{:?}",stats.requested_algorithm)
        }),
        TensorEngineExecution::AffinePower(stats) => json!({
            "engine":"AffinePower", "input_rows":stats.input_rows,
            "certificate_execution":certificate_execution_json(stats.certificate),
            "profiles":stats.profiles, "dag_nodes":stats.dag_nodes, "dag_edges":stats.dag_edges,
            "packed_child_rows":stats.packed_child_rows,
            "child_batch_calls":stats.child_batch_calls,
            "parent_batch_calls":stats.parent_batch_calls, "parent_rows":stats.parent_rows,
            "quadrature_work_rows":stats.quadrature_work_rows,
            "child_execution":execution_json(stats.child_execution),
            "parent_execution":execution_json(stats.parent_execution),
            "child_methods":stats.child_methods.iter().map(|method| json!({
                "propagators":method.propagators,"powers":method.powers,
                "requested":format!("{:?}",method.requested),
                "actual":method.actual.map(|value|format!("{value:?}")),
                "rows":method.rows,"notes":method.notes
            })).collect::<Vec<_>>()
        }),
        TensorEngineExecution::CertifiedTensor {
            execution,
            certificate_execution,
        } => json!({
            "engine":"CertifiedTensor", "execution":execution_json(execution),
            "certificate_execution":{
                "cartesian":certificate_execution_json(certificate_execution.cartesian),
                "gram":certificate_execution_json(certificate_execution.gram)
            }
        }),
        TensorEngineExecution::SixPointHybrid {
            primary,
            fallback,
            fallback_certificate_execution,
            fallback_rows,
            fallback_successes,
            fallback_batches,
            fallback_preparation_attempts,
            fallback_preparation_error,
        } => json!({
            "engine":"SixPointHybrid",
            "primary":engine_execution_json(TensorEngineExecution::SixPoint(primary)),
            "fallback":execution_json(fallback),
            "fallback_certificate_execution":certificate_execution_json(fallback_certificate_execution),
            "fallback_rows":fallback_rows,"fallback_successes":fallback_successes,
            "fallback_batches":fallback_batches,
            "fallback_preparation_attempts":fallback_preparation_attempts,
            "fallback_preparation_error":fallback_preparation_error
        }),
        TensorEngineExecution::SevenPoint(stats) => json!({
            "engine":"SevenPoint","parent_input_rows":stats.parent_input_rows,
            "child_batch_calls":stats.child_batch_calls,"packed_child_rows":stats.packed_child_rows,
            "parent_batch_calls":stats.parent_batch_calls,"packed_parent_rows":stats.packed_parent_rows,
            "child_execution":execution_json(stats.child_execution),
            "parent_execution":execution_json(stats.parent_execution),
            "child_certificate_execution":certificate_execution_json(stats.child_certificate_execution),
            "parent_certificate_execution":certificate_execution_json(stats.parent_certificate_execution)
        }),
    }
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
#[derive(Deserialize)]
struct Fixtures {
    schema_version: u32,
    cases: Vec<Fixture>,
}
#[derive(Deserialize)]
struct Fixture {
    input: Sample,
    #[serde(alias = "reference")]
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
            return Err("batch samples must share topology, powers, dimension shift, scalar-product input convention, and snapping policy/threshold".into());
        }
    }
    match args.numeric.precision {
        Precision::F64 => reduce(&samples, settings, 0f64, graph, &args.numeric),
        Precision::Dd => reduce(
            &samples,
            settings,
            DoubleFloat::from(0.),
            graph,
            &args.numeric,
        ),
        Precision::Float => reduce(
            &samples,
            settings,
            Float::new(args.numeric.bits),
            graph,
            &args.numeric,
        ),
    }
}
fn tensor_plan(
    settings: ReductionSettings,
    numeric: &NumericArgs,
) -> CliResult<TensorReductionPlan> {
    let plan = if numeric.check_scalar_products {
        if settings.invariants != iregi_rust::InvariantInput::SuppliedScalarProducts {
            return Err("--check-scalar-products requires scalar_products in the input".into());
        }
        TensorReductionPlan::new_checked_cartesian(settings)
    } else {
        TensorReductionPlan::new(settings)
    };
    plan.map_err(|error| error.to_string())
}

fn reduce<T: RescueScalar>(
    samples: &[Sample],
    settings: ReductionSettings,
    prototype: T,
    graph: Option<&PreparedGraph>,
    numeric: &NumericArgs,
) -> CliResult<(Value, bool)>
where
    Complex<T>: EvaluationDomain,
{
    let policy = numeric.resolve(prototype.get_precision())?;
    let batch = pack(samples, &prototype, false)?;
    let started = Instant::now();
    let plan = tensor_plan(settings.clone(), numeric)?;
    let mut workspace = NumericalWorkspace::new(&plan, prototype.clone(), policy)?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let result = workspace.reduce_batch(&batch)?;
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
        .map(|(row, sample)| {
            let mut value = match result.lane(row) {
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
                let mut value = json!({"id":sample.id,"status":"failed","reason":format!("{status:?}"),"fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row]});
                // A refused lane's best uncertified array (COLLIER-style flag).
                if let Some(lane) = result.lane_with_estimate(row) {
                    value["best_effort_moments"] = json!(lane
                        .values
                        .chunks_exact(4)
                        .map(|poles| poles.iter().map(|v| format_complex(v, prototype.get_precision())).collect::<Vec<_>>())
                        .collect::<Vec<_>>());
                }
                value
            }
            };
            value["estimated_error"] = json!(result.estimated_error(row));
            value["precision_rescue"] = workspace.lane_rescue(row);
            value
        })
        .collect();
    let mut output = json!({"schema_version":1,"precision_bits":prototype.get_precision(),"settings":settings,
        "input_policy":settings.input_policy,
        "precision_control":policy.json(),
        "algorithm_requested":settings.algorithm,"algorithm_used":workspace.algorithm_used(&plan)?,"preparation_notes":workspace.preparation_notes(),
        "basis":result.basis,"poles":["uv1","ir1","ir2","finite"],"samples":rows,
        "timing":{"plan_and_workspace_seconds":preparation_seconds,"reduction_seconds":reduction_seconds,
            "note":"one reduction workspace call, including enabled precision rescue and Rust masters; excludes file I/O, graph preparation, numerator evaluation and contraction; not a statistically qualified benchmark"},
        "execution":workspace.execution()?,"engine_execution":workspace.engine_execution(),
        "diagnostics":workspace.diagnostics(),"rescue_execution":workspace.rescue_execution()});
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
    plan: &TensorReductionPlan,
    result: &iregi_rust::BatchResult<T>,
    prototype: &T,
) -> CliResult<Vec<Option<CliResult<Vec<Complex<T>>>>>>
where
    Complex<T>: EvaluationDomain,
{
    let names: Vec<_> = samples[0].parameters.keys().cloned().collect();
    if samples
        .iter()
        .any(|sample| sample.parameters.keys().ne(names.iter()))
    {
        return Err("all samples must bind the same numerator parameter names".into());
    }
    let valid: Vec<_> = (0..samples.len())
        .filter(|&row| result.lane(row).is_ok())
        .collect();
    if valid.is_empty() {
        return Ok(vec![None; samples.len()]);
    }
    let coefficient_plan = iregi_cli::numerator::CoefficientPlan::new(graph, names)?;
    let mut coefficient_workspace =
        iregi_cli::numerator::CoefficientWorkspace::new(&coefficient_plan, prototype.clone());
    let compact_samples: Vec<_> = valid.iter().map(|&row| samples[row].clone()).collect();
    let input = coefficient_plan.pack(&compact_samples, prototype)?;
    let output = coefficient_workspace.evaluate(&input, valid.len())?;
    let width = coefficient_plan.support().len();
    let mut coefficients =
        vec![Complex::new(prototype.zero(), prototype.zero()); samples.len() * width];
    for (compact, &row) in valid.iter().enumerate() {
        coefficients[row * width..(row + 1) * width]
            .clone_from_slice(&output.values[compact * width..(compact + 1) * width]);
    }
    let contraction_plan = plan
        .contraction_plan(coefficient_plan.support())
        .map_err(|e| e.to_string())?;
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
    use iregi_rust::{InputPolicy, InvariantInput};
    use symbolica::domains::{float::SingleFloat, rational::Rational};

    fn numeric_options(options: &[&str]) -> NumericArgs {
        let mut arguments = vec!["iregi-rust", "reduce", "--input", "unused.json"];
        arguments.extend_from_slice(options);
        let Command::Reduce(args) = Cli::try_parse_from(arguments).unwrap().command else {
            unreachable!()
        };
        args.numeric
    }

    #[test]
    fn facade_cli_keeps_default_hybrid_and_real_failed_lane() {
        let valid: Sample = serde_json::from_str(
            r#"{"id":"valid","propagators":[{"shift":["0","0","0","0"],"mass_squared":"4"}],"mu":"3"}"#,
        ).unwrap();
        let mut invalid = valid.clone();
        invalid.id = "invalid-scale".into();
        invalid.mu = Literal::Text("0".into());
        let Command::Reduce(parsed) =
            Cli::try_parse_from(["iregi-rust", "reduce", "--input", "unused.json"])
                .unwrap()
                .command
        else {
            unreachable!()
        };
        assert!(matches!(parsed.algorithm, Method::Hybrid));
        let settings =
            settings(&valid, 0, parsed.algorithm.into(), parsed.convention.into()).unwrap();
        let (output, success) =
            reduce(&[valid, invalid], settings, 0f64, None, &parsed.numeric).unwrap();
        assert!(!success);
        assert_eq!(output["algorithm_requested"], "Hybrid");
        assert_eq!(output["engine_execution"]["engine"], "Direct");
        assert_eq!(output["execution"], output["engine_execution"]["execution"]);
        assert_eq!(output["execution"]["input_rows"], 2);
        assert!(output["execution"]["master_rows"].as_u64().unwrap() > 0);
        assert_eq!(output["samples"][0]["status"], "success");
        assert_eq!(output["samples"][0]["moments"].as_array().unwrap().len(), 1);
        assert_eq!(output["samples"][1]["status"], "failed");
        assert!(output["samples"][1]["moments"].is_null());
        assert_eq!(
            output["samples"][1]["reason"],
            "InvalidInput(\"mu must be positive and real\")"
        );
        assert_eq!(output["diagnostics"].as_array().unwrap().len(), 2);
        assert_eq!(output["precision_control"]["accuracy_bits"], 53);
    }

    #[test]
    fn facade_cli_rescue_retains_actual_generic_primary_metadata() {
        let sample: Sample = serde_json::from_str(
            r#"{"id":"massive","propagators":[{"shift":["0","0","0","0"],"mass_squared":"4"}],"mu":"3"}"#,
        ).unwrap();
        let settings = settings(
            &sample,
            0,
            Algorithm::PassarinoVeltman,
            Convention::IregiNative,
        )
        .unwrap();
        let (output, success) = reduce(
            &[sample],
            settings,
            0f64,
            None,
            &numeric_options(&["--rescue", "--accuracy-bits", "53"]),
        )
        .unwrap();
        assert!(success);
        assert_eq!(output["engine_execution"]["engine"], "Generic");
        assert_eq!(output["execution"], output["engine_execution"]["execution"]);
        assert_eq!(output["algorithm_used"], "PassarinoVeltman");
        assert!(
            output["preparation_notes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|note| note.as_str().unwrap().contains("precision rescue"))
        );
        assert!(output["engine_execution"]["certificate_execution"].is_null());
        assert_eq!(output["diagnostics"].as_array().unwrap().len(), 1);
        // The existing generic rescue exposes primary lane diagnostics but
        // does not expose tagged stage/certificate execution details.
        for stage in ["primary", "working", "verification"] {
            assert!(output["engine_execution"][stage].is_null());
        }
        assert_eq!(
            output["samples"][0]["precision_rescue"]["status"],
            "converged"
        );
        assert_eq!(output["rescue_execution"]["requested_rows"], 1);
        assert_eq!(output["rescue_execution"]["accepted_rows"], 1);
        assert!(
            output["rescue_execution"]["working"]["master_rows"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            output["rescue_execution"]["verification"]["master_rows"]
                .as_u64()
                .unwrap()
                > 0
        );
    }

    #[test]
    fn precision_policy_defaults_preserve_behavior_and_scale_rescue() {
        let plain = numeric_options(&[]).resolve(53).unwrap();
        assert_eq!(plain.accuracy_bits, 53);
        assert!(plain.rescue.is_none());
        let binary64 = numeric_options(&["--rescue"]).resolve(53).unwrap();
        assert_eq!(binary64.rescue.unwrap().working_bits, 128);
        assert_eq!(binary64.rescue.unwrap().verification_bits, 256);
        assert_eq!(
            binary64.rescue.unwrap().trigger,
            RescueTrigger::VerifyAllSuccessful
        );
        let high = numeric_options(&["--rescue", "--accuracy-bits", "53"])
            .resolve(256)
            .unwrap();
        assert_eq!(high.accuracy_bits, 53);
        assert_eq!(high.rescue.unwrap().working_bits, 512);
        assert_eq!(high.rescue.unwrap().verification_bits, 1024);
        let explicit = numeric_options(&[
            "--rescue",
            "--rescue-bits",
            "192",
            "--verification-bits",
            "384",
        ])
        .resolve(160)
        .unwrap();
        assert_eq!(explicit.rescue.unwrap().working_bits, 192);
        assert_eq!(explicit.rescue.unwrap().verification_bits, 384);
    }

    #[test]
    fn precision_policy_rejects_inconsistent_accuracy_and_verification() {
        assert!(
            numeric_options(&["--accuracy-bits", "54"])
                .resolve(53)
                .is_err()
        );
        assert!(
            numeric_options(&["--rescue", "--rescue-bits", "128"])
                .resolve(128)
                .is_err()
        );
        assert!(
            numeric_options(&[
                "--rescue",
                "--rescue-bits",
                "128",
                "--verification-bits",
                "128"
            ])
            .resolve(53)
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "iregi-rust",
                "reduce",
                "--input",
                "unused.json",
                "--rescue-bits",
                "128"
            ])
            .is_err()
        );
        assert!(numeric_options(&["--rescue"]).resolve(u32::MAX).is_err());
    }

    #[test]
    fn decimal_negative_zero_survives_input_conversion() {
        for text in ["-0", "-0.0", "-0e20"] {
            let literal = Literal::Text(text.into());
            assert!(
                parse_real(&literal, &0f64, false)
                    .unwrap()
                    .is_sign_negative()
            );
            assert!(
                parse_real(&literal, &DoubleFloat::from(0.), false)
                    .unwrap()
                    .into_inner()
                    .hi()
                    .is_sign_negative()
            );
            assert!(
                parse_real(&literal, &Float::new(160), false)
                    .unwrap()
                    .is_sign_negative()
            );
        }
    }

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
    fn legacy_policy_defaults_and_threshold_tokens_are_forwarded_exactly() {
        let mut sample: Sample = serde_json::from_str(
            r#"{"propagators":[{"shift":["0","0","0","0"],"mass_squared":"4"}],"mu":"3"}"#,
        )
        .unwrap();
        let make_settings =
            |sample: &Sample| settings(sample, 0, Algorithm::Ibp, Convention::IregiNative).unwrap();
        assert_eq!(make_settings(&sample).input_policy, InputPolicy::Preserve);
        sample.snap = true;
        assert_eq!(
            make_settings(&sample).input_policy,
            InputPolicy::legacy_onshell()
        );
        for threshold in ["0", "1/8", "1.0000000000000000000000001e-6"] {
            sample.threshold = Some(Literal::Text(threshold.into()));
            let configuration = make_settings(&sample);
            assert_eq!(
                configuration.input_policy,
                InputPolicy::LegacyOnShell {
                    threshold: threshold.into()
                }
            );
            assert_eq!(
                serde_json::to_value(configuration).unwrap()["input_policy"]["LegacyOnShell"]["threshold"],
                threshold
            );
        }
        let threshold = "0.0000010000000000000000000000001";
        let document: InputDocument = serde_json::from_str(&format!(
            r#"{{"propagators":[{{"shift":["0","0","0","0"],"mass_squared":"4"}}],"mu":"3","snap":true,"threshold":{threshold}}}"#,
        )).unwrap();
        let InputDocument::Single(sample) = document else {
            panic!("expected one sample")
        };
        assert_eq!(
            make_settings(&sample).input_policy,
            InputPolicy::LegacyOnShell {
                threshold: threshold.into()
            }
        );
    }

    #[test]
    fn unsupported_or_invalid_legacy_configuration_is_rejected_by_core() {
        let mut sample: Sample = serde_json::from_str(
            r#"{"propagators":[{"shift":["0","0","0","0"],"mass_squared":"4"}],"mu":"3","snap":true,"threshold":"-1e-6"}"#,
        ).unwrap();
        assert!(matches!(
            TensorReductionPlan::new(
                settings(&sample, 0, Algorithm::Ibp, Convention::IregiNative).unwrap()
            ),
            Err(iregi_rust::ReductionError::InvalidInput(_))
        ));
        sample.threshold = None;
        // The conservative checked Legacy path now admits N5/N6. Exercise
        // the still-unsupported N7 boundary without changing any kinematics.
        sample.propagators = vec![sample.propagators[0].clone(); 7];
        assert!(
            matches!(TensorReductionPlan::new(settings(&sample, 0, Algorithm::Ibp, Convention::IregiNative).unwrap()),
            Err(iregi_rust::ReductionError::Unsupported(reason)) if reason.contains("N=5 or 6"))
        );
    }

    #[test]
    fn checked_scalar_products_are_explicit_and_preserve_the_guarded_layout() {
        let options = numeric_options(&["--check-scalar-products"]);
        let mut settings = ReductionSettings::new(6, 1);
        assert!(
            tensor_plan(settings.clone(), &options)
                .unwrap_err()
                .contains("requires scalar_products")
        );
        settings.invariants = iregi_rust::InvariantInput::SuppliedScalarProducts;
        assert!(tensor_plan(settings.clone(), &numeric_options(&[])).is_err());
        let checked = tensor_plan(settings.clone(), &options).unwrap();
        assert_eq!(checked.engine_kind(), TensorEngineKind::CertifiedTensor);
        assert_eq!(checked.settings().algorithm, settings.algorithm);
        assert_eq!(checked.settings().invariants, settings.invariants);
        assert_eq!(checked.input_layout().scalar_products, Some(24..45));
        assert_eq!(checked.parameter_count(), 52);
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

fn compare<T: RescueScalar>(args: &CompareArgs, prototype: T) -> CliResult<(Value, bool)>
where
    Complex<T>: EvaluationDomain,
{
    let policy = args.numeric.resolve(prototype.get_precision())?;
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
            let plan = tensor_plan(settings.clone(), &args.numeric)?;
            let mut workspace = NumericalWorkspace::new(&plan, prototype.clone(), policy)?;
            let result = workspace.reduce_batch(&pack(&samples, &prototype, true)?)?;
            Ok((
                result,
                workspace.algorithm_used(&plan)?,
                workspace.preparation_notes().to_vec(),
                (0..samples.len())
                    .map(|row| workspace.lane_rescue(row))
                    .collect::<Vec<_>>(),
                workspace.execution()?,
                workspace.engine_execution(),
                workspace.diagnostics(),
                workspace.rescue_execution(),
            ))
        })();
        match attempt {
            Err(error) => {
                failures += fixtures.len();
                for fixture in &fixtures {
                    rows.push(json!({"id":fixture.input.id,"status":"failed","reason":error,"algorithm_requested":settings.algorithm,"input_policy":settings.input_policy}));
                }
            }
            Ok((
                result,
                algorithm_used,
                preparation_notes,
                rescue_diagnostics,
                execution,
                engine_execution,
                diagnostics,
                rescue_execution,
            )) => {
                let execution_ids: Vec<_> = fixtures
                    .iter()
                    .map(|fixture| fixture.input.id.as_str())
                    .collect();
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
                            "input_policy":settings.input_policy,
                            "fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row],
                            "precision_rescue":rescue_diagnostics[row],"execution":execution,"engine_execution":engine_execution,
                            "diagnostics":diagnostics.as_array().map(|rows|&rows[row]),"rescue_execution":rescue_execution,
                            "execution_scope":"settings_batch","execution_batch_ids":execution_ids,
                            "worst_moment":worst/4,"worst_pole":worst%4,"reference_repaired":fixture.reference_repaired}));
                        }
                        Err(reason) => {
                            failures += 1;
                            rows.push(
                                json!({"id":fixture.input.id,"status":"failed","reason":reason,"algorithm_requested":settings.algorithm,
                                "algorithm_used":algorithm_used,"preparation_notes":preparation_notes,"fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row],
                                "input_policy":settings.input_policy,
                                "precision_rescue":rescue_diagnostics[row],"execution":execution,"engine_execution":engine_execution,
                            "diagnostics":diagnostics.as_array().map(|rows|&rows[row]),"rescue_execution":rescue_execution,
                                "execution_scope":"settings_batch","execution_batch_ids":execution_ids}),
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
        "precision_control":policy.json(),
        "error_definition":"abs(actual-reference)/Q^dimension divided by max(1,abs(reference/Q^dimension)); Q=max(mu,abs(shift components),sqrt(abs(mass_squared)))",
        "cases":rows,"failed":failures,"passed":rows.len()-failures,
        "reference":"captured original IREGI binary64 values; no reference process transport is timed"}),
        failures == 0,
    ))
}
