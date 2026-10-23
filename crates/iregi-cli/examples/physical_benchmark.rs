//! In-process physical benchmark adapter; JSON transport is outside every timer.
use std::{
    hint::black_box,
    io::{self, BufRead, Write},
    time::Instant,
};

use iregi_cli::{
    input::{InputDocument, Sample, format_complex, pack, settings, settings_key},
    numerator::{CoefficientPlan, CoefficientWorkspace},
};
use iregi_masters::{Complex, DoubleFloat, Float, NativeFloat};
use iregi_prepare::{NumeratorSource, PrepareOptions, PreparedGraph, prepare_dot};
use iregi_rust::{
    Algorithm, BatchResult, ContractionPlan, ContractionResult, ContractionWorkspace, Convention,
    KinematicsBatch, LaneStatus, NumeratorBatch, ReductionPlan, Workspace,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use symbolica::evaluate::EvaluationDomain;

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Precision {
    F64,
    Dd,
    Float,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Stage {
    GraphPreparation,
    EvaluatorPreparation,
    Numerator,
    Reduction,
    Contraction,
    NumericalEndToEnd,
    RepreparedEndToEnd,
}

#[derive(Deserialize)]
struct Configuration {
    id: String,
    model: String,
    dot: String,
    component: Vec<usize>,
    input: InputDocument,
    precision: Precision,
    bits: u32,
    accuracy_bits: u32,
    algorithm: Algorithm,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Prepare {
        configuration: Configuration,
    },
    Measure {
        id: String,
        stage: Stage,
        warmup: usize,
        iterations: usize,
        repeats: usize,
        #[serde(default)]
        profile: bool,
    },
}

struct Source {
    id: String,
    model: String,
    dot: String,
    options: PrepareOptions,
    samples: Vec<Sample>,
    accuracy_bits: u32,
    algorithm: Algorithm,
}

impl Source {
    fn graph(&self) -> Result<PreparedGraph, String> {
        prepare_dot(&self.model, &self.dot, &self.options).map_err(|error| error.to_string())
    }
}

struct Pipeline<T: NativeFloat> {
    graph: PreparedGraph,
    plan: ReductionPlan,
    reduction: Workspace<T>,
    coefficients: CoefficientWorkspace<T>,
    contraction: ContractionWorkspace<T>,
    kinematics: KinematicsBatch<T>,
    parameters: Vec<Complex<T>>,
    prototype: T,
}

fn successful(statuses: &[LaneStatus]) -> Result<(), String> {
    for (lane, status) in statuses.iter().enumerate() {
        if *status != LaneStatus::Success {
            return Err(format!("lane {lane}: {status:?}"));
        }
    }
    Ok(())
}

impl<T: NativeFloat> Pipeline<T>
where
    Complex<T>: EvaluationDomain,
{
    fn new(source: &Source, graph: PreparedGraph, prototype: T) -> Result<Self, String> {
        if source.samples.is_empty() {
            return Err("physical benchmark requires at least one sample".into());
        }
        if source.accuracy_bits == 0 || source.accuracy_bits > prototype.get_precision() {
            return Err(
                "accuracy_bits must be positive and no greater than working precision".into(),
            );
        }
        let rank =
            u16::try_from(graph.numerator.rank()).map_err(|_| "numerator rank exceeds u16")?;
        let configuration = settings(
            &source.samples[0],
            rank,
            source.algorithm,
            Convention::OneLoop,
        )?;
        let key = serde_json::to_string(&configuration).map_err(|error| error.to_string())?;
        for sample in &source.samples {
            if sample.rank.is_some_and(|declared| declared != rank) {
                return Err("declared physical rank differs from prepared polynomial".into());
            }
            if sample.propagators.len() != graph.propagators.len() {
                return Err("physical propagator count differs from graph".into());
            }
            for (input, routing) in sample.propagators.iter().zip(&graph.propagators) {
                if input.edge_id.is_some_and(|edge| edge != routing.edge_id)
                    || routing.loop_coefficient.abs() != 1
                {
                    return Err(
                        "physical propagator routing/order does not match prepared graph".into(),
                    );
                }
            }
            if settings_key(sample, rank, source.algorithm, Convention::OneLoop)? != key {
                return Err("physical batch requires one common reduction configuration".into());
            }
        }
        let coefficients = CoefficientPlan::new(
            &graph,
            source.samples[0].parameters.keys().cloned().collect(),
        )?;
        let parameters = coefficients.pack(&source.samples, &prototype)?;
        let kinematics = pack(&source.samples, &prototype, false)?;
        let plan = ReductionPlan::new(configuration).map_err(|error| error.to_string())?;
        let mut reduction =
            Workspace::new(&plan, prototype.clone()).map_err(|error| error.to_string())?;
        reduction
            .set_accuracy_precision(source.accuracy_bits)
            .map_err(|error| error.to_string())?;
        let contraction_plan = ContractionPlan::new(&plan, coefficients.support())
            .map_err(|error| error.to_string())?;
        let contraction = ContractionWorkspace::new(&contraction_plan, prototype.clone())
            .map_err(|error| error.to_string())?;
        Ok(Self {
            graph,
            plan,
            reduction,
            coefficients: CoefficientWorkspace::new(&coefficients, prototype.clone()),
            contraction,
            kinematics,
            parameters,
            prototype,
        })
    }

    fn numerator(&mut self) -> Result<NumeratorBatch<T>, String> {
        self.coefficients
            .evaluate(black_box(&self.parameters), self.kinematics.rows)
    }

    fn reduce(&mut self) -> Result<BatchResult<T>, String> {
        let result = self
            .reduction
            .reduce_batch(black_box(&self.kinematics))
            .map_err(|error| error.to_string())?;
        successful(&result.statuses)?;
        Ok(result)
    }

    fn contract(
        &mut self,
        moments: &BatchResult<T>,
        numerator: &NumeratorBatch<T>,
    ) -> Result<ContractionResult<T>, String> {
        let result = self
            .contraction
            .contract_batch(black_box(moments), black_box(numerator))
            .map_err(|error| error.to_string())?;
        successful(&result.statuses)?;
        Ok(result)
    }

    fn evaluate(&mut self) -> Result<(BatchResult<T>, ContractionResult<T>), String> {
        let numerator = self.numerator()?;
        let moments = self.reduce()?;
        let integral = self.contract(&moments, &numerator)?;
        Ok((moments, integral))
    }

    fn output(&mut self, source: &Source) -> Result<Value, String> {
        let (moments, integral) = self.evaluate()?;
        let bits = self.prototype.get_precision();
        let rows: Vec<_> = (0..self.kinematics.rows).map(|row| json!({
            "id": source.samples[row].id, "status":"success",
            "moments":moments.lane(row).unwrap().chunks_exact(4).map(|poles|poles.iter().map(|value|format_complex(value,bits)).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "integral":integral.lane(row).unwrap().iter().map(|value|format_complex(value,bits)).collect::<Vec<_>>(),
            "fallback_used":moments.fallback_used[row],"analytic_used":moments.analytic_used[row]
        })).collect();
        let stats = self.reduction.last_execution();
        Ok(
            json!({"precision_bits":bits,"settings":self.plan.settings(),"basis":moments.basis,
            "algorithm_requested":source.algorithm,"algorithm_used":self.plan.algorithm_used(),
            "preparation_notes":self.plan.preparation_notes(),"poles":["uv1","ir1","ir2","finite"],
            "graph":{"name":self.graph.name},"samples":rows,
            "precision_control":{"output_bits":bits,"accuracy_bits":source.accuracy_bits,"verification_scope":"none"},
            "kinematics_contract":"explicit canonical (q+p_i) propagators in reported graph edge order",
            "execution":{"input_rows":stats.input_rows,"master_rows":stats.master_rows,
                "master_family_batches":stats.master_family_batches,"pentagon_box_rows":stats.pentagon_box_rows,
                "fallback_rows":stats.fallback_rows,"analytic_rows":stats.analytic_rows}}),
        )
    }
}

struct Prepared<T: NativeFloat> {
    source: Source,
    pipeline: Pipeline<T>,
    numerator: NumeratorBatch<T>,
    moments: BatchResult<T>,
}

impl<T: NativeFloat> Prepared<T>
where
    Complex<T>: EvaluationDomain,
{
    fn new(source: Source, prototype: T) -> Result<(Self, Value), String> {
        let first = Instant::now();
        let start = Instant::now();
        let graph = source.graph()?;
        let graph_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let mut pipeline = Pipeline::new(&source, graph, prototype)?;
        let evaluator_ns = start.elapsed().as_nanos();
        let start = Instant::now();
        let (moments, integral) = pipeline.evaluate()?;
        let first_evaluation_ns = start.elapsed().as_nanos();
        black_box(integral);
        let first_end_to_end_ns = first.elapsed().as_nanos();
        let numerator = pipeline.numerator()?;
        let output = pipeline.output(&source)?;
        let response = json!({"id":source.id,"status":"prepared","output":output,
            "first_observed_ns":{"graph_preparation":graph_ns,"evaluator_preparation":evaluator_ns,
                "numerical_end_to_end":first_evaluation_ns,"end_to_end":first_end_to_end_ns},
            "master_backend":format!("{:?}",iregi_masters::default_master_backend()),
            "preparation_qualification":"first observed within this persistent process; global Symbolica and structural caches may already be warm; excludes file I/O, JSON transport and output serialization"});
        Ok((
            Self {
                source,
                pipeline,
                numerator,
                moments,
            },
            response,
        ))
    }

    fn once(&mut self, stage: Stage) -> Result<(), String> {
        match stage {
            Stage::GraphPreparation => {
                black_box(self.source.graph()?);
            }
            Stage::EvaluatorPreparation => {
                black_box(Pipeline::new(
                    &self.source,
                    self.pipeline.graph.clone(),
                    self.pipeline.prototype.clone(),
                )?);
            }
            Stage::Numerator => {
                let numerator = self.pipeline.numerator()?;
                if numerator
                    .values
                    .iter()
                    .any(|value| !value.re.is_finite() || !value.im.is_finite())
                {
                    return Err("nonfinite numerator coefficient".into());
                }
                black_box(numerator);
            }
            Stage::Reduction => {
                black_box(self.pipeline.reduce()?);
            }
            Stage::Contraction => {
                black_box(self.pipeline.contract(&self.moments, &self.numerator)?);
            }
            Stage::NumericalEndToEnd => {
                black_box(self.pipeline.evaluate()?);
            }
            Stage::RepreparedEndToEnd => {
                let graph = self.source.graph()?;
                let mut pipeline =
                    Pipeline::new(&self.source, graph, self.pipeline.prototype.clone())?;
                black_box(pipeline.evaluate()?);
            }
        }
        Ok(())
    }

    /// Validate the same preparation path as the timed stage. Re-evaluating the
    /// retained pipeline alone would miss a fresh-object/cache discrepancy in
    /// graph or workspace preparation. This extra execution is outside timers.
    fn postcheck(&mut self, stage: Stage) -> Result<Value, String> {
        match stage {
            Stage::GraphPreparation | Stage::RepreparedEndToEnd => {
                let graph = self.source.graph()?;
                let mut pipeline =
                    Pipeline::new(&self.source, graph, self.pipeline.prototype.clone())?;
                pipeline.output(&self.source)
            }
            Stage::EvaluatorPreparation => {
                let mut pipeline = Pipeline::new(
                    &self.source,
                    self.pipeline.graph.clone(),
                    self.pipeline.prototype.clone(),
                )?;
                pipeline.output(&self.source)
            }
            Stage::Numerator | Stage::Reduction | Stage::Contraction | Stage::NumericalEndToEnd => {
                self.pipeline.output(&self.source)
            }
        }
    }

    fn measure(
        &mut self,
        id: &str,
        stage: Stage,
        warmup: usize,
        iterations: usize,
        repeats: usize,
        profile: bool,
    ) -> Result<Value, String> {
        if id != self.source.id || iterations == 0 || repeats == 0 {
            return Err("invalid id or zero iterations/repeats".into());
        }
        if profile && !matches!(stage, Stage::Reduction | Stage::NumericalEndToEnd) {
            return Err(
                "profiling is supported only for reduction and numerical_end_to_end".into(),
            );
        }
        self.pipeline.reduction.set_profiling(profile);
        let mut durations = Vec::new();
        let mut profiles = Vec::new();
        for index in 0..warmup {
            if let Err(error) = self.once(stage) {
                return Ok(
                    json!({"id":id,"status":"warmup_failed","warmup_index":index,"error":error,"total_ns":durations}),
                );
            }
        }
        for block in 0..repeats {
            let mut components = [0u128; 4];
            let started = Instant::now();
            for iteration in 0..iterations {
                if let Err(error) = self.once(stage) {
                    return Ok(
                        json!({"id":id,"status":"timed_failure","block":block,"iteration":iteration,
                        "failed_block_elapsed_ns":started.elapsed().as_nanos(),"error":error,"total_ns":durations,"profile_ns":profiles}),
                    );
                }
                if let Some(value) = self.pipeline.reduction.last_profile() {
                    components[0] += value.arguments.as_nanos();
                    components[1] += value.scalar_masters.as_nanos();
                    components[2] += value.reduction.as_nanos();
                    components[3] += value.total.as_nanos();
                }
            }
            durations.push(started.elapsed().as_nanos() as f64 / iterations as f64);
            if profile {
                profiles.push(components.map(|value| value as f64 / iterations as f64));
            }
        }
        self.pipeline.reduction.set_profiling(false);
        let output = match self.postcheck(stage) {
            Ok(output) => output,
            Err(error) => {
                return Ok(
                    json!({"id":id,"status":"postcheck_failed","error":error,"total_ns":durations,"profile_ns":profiles}),
                );
            }
        };
        Ok(
            json!({"id":id,"status":"measured","stage":stage,"warmup":warmup,"iterations":iterations,"repeats":repeats,"total_ns":durations,"profiled":profile,
            "profile_ns":profiles,"profile_order":["arguments","masters","reduction","total"],"output":output}),
        )
    }
}

enum State {
    F64(Prepared<f64>),
    Dd(Prepared<DoubleFloat>),
    Float(Prepared<Float>),
}

fn prepare(configuration: Configuration) -> Result<(State, Value), String> {
    let samples = match configuration.input {
        InputDocument::Batch { samples } => samples,
        InputDocument::Single(sample) => vec![sample],
    };
    let source = Source {
        id: configuration.id,
        model: configuration.model,
        dot: configuration.dot,
        options: PrepareOptions {
            source: NumeratorSource::FeynmanRules,
            external_wavefunctions: true,
            component: Some(configuration.component),
        },
        samples,
        accuracy_bits: configuration.accuracy_bits,
        algorithm: configuration.algorithm,
    };
    match configuration.precision {
        Precision::F64 => {
            Prepared::new(source, 0f64).map(|(state, value)| (State::F64(state), value))
        }
        Precision::Dd => Prepared::new(source, DoubleFloat::from(0.))
            .map(|(state, value)| (State::Dd(state), value)),
        Precision::Float if configuration.bits >= 2 => {
            Prepared::new(source, Float::new(configuration.bits))
                .map(|(state, value)| (State::Float(state), value))
        }
        Precision::Float => Err("Float precision must be at least two bits".into()),
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut state = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim() == "QUIT" {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let response =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Value, String> {
                match serde_json::from_str::<Request>(&line).map_err(|error| error.to_string())? {
                    Request::Prepare { configuration } => {
                        state = None;
                        let (prepared, response) = prepare(configuration)?;
                        state = Some(prepared);
                        Ok(response)
                    }
                    Request::Measure {
                        id,
                        stage,
                        warmup,
                        iterations,
                        repeats,
                        profile,
                    } => match state.as_mut().ok_or("prepare a physical fixture first")? {
                        State::F64(value) => {
                            value.measure(&id, stage, warmup, iterations, repeats, profile)
                        }
                        State::Dd(value) => {
                            value.measure(&id, stage, warmup, iterations, repeats, profile)
                        }
                        State::Float(value) => {
                            value.measure(&id, stage, warmup, iterations, repeats, profile)
                        }
                    },
                }
            }));
        let value = match response {
            Ok(Ok(value)) => value,
            Ok(Err(error)) => json!({"status":"error","error":error}),
            Err(_) => {
                state = None;
                json!({"status":"panic","error":"physical preparation or evaluation panicked"})
            }
        };
        writeln!(stdout, "RESULT {value}").unwrap();
        stdout.flush().unwrap();
    }
}
