//! Checked supplied-G pentagon dispatcher for the private packed N6 path.
//!
//! Every original p/G jet is certified before selecting an execution actor.
//! Planar samples enter the existing masked planar wrapper directly; only a
//! certified regular primal can prepare the ordinary supplied-G PV workspace.

use std::time::Instant;

use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use symbolica::{
    atom::Atom,
    domains::{dual::HyperDual, float::Complex, rational::Rational},
    evaluate::EvaluationDomain,
};

use crate::{
    BatchResult, CertificateExecutionStats, Convention, ExecutionProfile, ExecutionStats,
    InputLayout, KinematicsBatch, LaneStatus, ReductionError, ReductionPlan, ReductionSettings,
    ResultMetadata, TensorWeights, Workspace,
    basis::{Monomial, tensor_basis},
    certified_planar::{
        CertifiedPlanarReductionPlan, CertifiedPlanarWorkspace, supplied_six_point_child_settings,
    },
    exact_zero_certificate::{
        ExactZeroCertificate, ExactZeroConstraintPlan, ExactZeroConstraintWorkspace,
    },
    execution::{WorkspacePolicy, validate_input_row},
    pv::planar_chart_polynomials,
};

type Shape = Option<Vec<Vec<usize>>>;

#[derive(Clone)]
pub(super) struct CheckedFiveWorkspace<T: NativeFloat> {
    settings: ReductionSettings,
    prototype: T,
    shape: Shape,
    components: usize,
    accuracy_bits: u32,
    basis: Vec<Monomial>,
    metadata: ResultMetadata,
    layout: InputLayout,
    compatibility: ExactZeroConstraintWorkspace,
    rank: ExactZeroConstraintWorkspace,
    // Neither numerical actor nor an ordinary ReductionPlan is constructed by
    // new(). Sample-specific certificates are never stored in these options.
    planar: Option<CertifiedPlanarWorkspace<T>>,
    generic: Option<Workspace<T>>,
    stats: ExecutionStats,
    certificate_stats: CertificateExecutionStats,
    diagnostics: Vec<Vec<String>>,
    profiling: bool,
    profile: ExecutionProfile,
}

impl<T: NativeFloat> CheckedFiveWorkspace<T>
where
    Complex<T>: EvaluationDomain,
{
    pub(super) fn new(
        settings: ReductionSettings,
        prototype: T,
        shape: Shape,
    ) -> Result<Self, ReductionError> {
        if !supplied_six_point_child_settings(&settings)
            || settings.rank > 5
            || settings.convention != Convention::IregiNative
            || settings.tensor_weights != TensorWeights::Components
        {
            return Err(ReductionError::Unsupported(
                "checked packed pentagon children require preserved supplied-G N5, rank<=5, unit powers, h=0, PV and native raw components".into(),
            ));
        }
        let accuracy_bits = prototype.get_precision();
        if accuracy_bits == 0 {
            return Err(preparation("working precision must be positive"));
        }
        if let Some(shape) = &shape {
            iregi_masters::validate_dual_structure(&HyperDual::<u8>::new(shape.clone()))
                .map_err(preparation)?;
        }
        let (mut parameters, compatibility) =
            crate::certified_tensor::compatibility_constraints(5);
        let shifts: Vec<[Atom; 4]> = (0..5)
            .map(|i| std::array::from_fn(|mu| parameters[4 * i + mu].clone()))
            .collect();
        let masses: Vec<_> = (0..5).map(|i| variable(&format!("m2_{i}"))).collect();
        parameters.extend(masses.iter().cloned());
        parameters.push(variable("mu"));
        // Reuse the existing determinant construction. The other chart
        // polynomials are prepared by the masked actor if planar lanes exist.
        let rank = planar_chart_polynomials(&shifts, &masses)
            .map_err(preparation)?
            .into_iter()
            .next()
            .ok_or_else(|| preparation("missing planar determinant predicate"))?;
        let compatibility = proof(&parameters, &compatibility, &shape)?.instantiate();
        let rank = proof(&parameters, &[rank], &shape)?.instantiate();
        Ok(Self {
            components: shape.as_ref().map_or(1, Vec::len),
            basis: tensor_basis(settings.rank),
            metadata: ResultMetadata::new(settings.convention, settings.tensor_weights, shape.clone()),
            settings,
            prototype,
            shape,
            accuracy_bits,
            layout: InputLayout {
                shifts: 0..20,
                scalar_products: Some(20..35),
                masses: 35..40,
                scale: 40,
            },
            compatibility,
            rank,
            planar: None,
            generic: None,
            stats: ExecutionStats::default(),
            certificate_stats: CertificateExecutionStats::default(),
            diagnostics: Vec::new(),
            profiling: false,
            profile: ExecutionProfile::default(),
        })
    }

    pub(super) fn set_accuracy_precision(&mut self, bits: u32) -> Result<(), ReductionError> {
        if bits == 0 || bits > self.prototype.get_precision() {
            return Err(ReductionError::InvalidInput(
                "checked pentagon accuracy precision must be positive and no greater than working precision".into(),
            ));
        }
        if let Some(workspace) = &mut self.planar {
            workspace.set_accuracy_precision(bits)?;
        }
        if let Some(workspace) = &mut self.generic {
            workspace.set_accuracy_precision(bits)?;
        }
        self.accuracy_bits = bits;
        Ok(())
    }

    pub(super) fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
        if let Some(workspace) = &mut self.planar {
            workspace.set_profiling(enabled);
        }
        if let Some(workspace) = &mut self.generic {
            workspace.set_profiling(enabled);
        }
    }

    pub(super) fn last_execution(&self) -> ExecutionStats { self.stats }
    pub(super) fn last_planar_certificate_execution(&self) -> CertificateExecutionStats {
        self.certificate_stats
    }
    pub(super) fn last_diagnostics(&self) -> &[Vec<String>] { &self.diagnostics }
    pub(super) fn last_profile(&self) -> Option<ExecutionProfile> {
        self.profiling.then_some(self.profile)
    }

    pub(super) fn reduce_batch(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<BatchResult<T>, ReductionError> {
        self.stats = ExecutionStats { input_rows: input.rows, ..Default::default() };
        self.certificate_stats = CertificateExecutionStats { input_rows: input.rows, ..Default::default() };
        self.diagnostics.clear();
        self.profile = ExecutionProfile::default();
        let started = self.profiling.then(Instant::now);
        let result = self.reduce_batch_impl(input);
        if let Some(started) = started { self.profile.total = started.elapsed(); }
        result
    }

    fn reduce_batch_impl(&mut self, input: &KinematicsBatch<T>) -> Result<BatchResult<T>, ReductionError> {
        let arguments_started = self.profiling.then(Instant::now);
        let input_width = product(41, self.components)?;
        if self.components == 0 || input.components != self.components
            || input.values.len() != product(input.rows, input_width)?
        {
            return Err(ReductionError::InvalidInput(
                "checked pentagon input dimensions do not match its full supplied-G layout".into(),
            ));
        }
        let output_width = product(product(self.basis.len(), 4)?, self.components)?;
        let output_len = product(input.rows, output_width)?;
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        let mut result = BatchResult {
            basis: self.basis.clone(), components: self.components, metadata: self.metadata.clone(),
            values: vec![zero; output_len],
            statuses: vec![LaneStatus::Unstable("checked pentagon input not certified".into()); input.rows],
            fallback_used: vec![false; input.rows], analytic_used: vec![false; input.rows],
            primal_envelopes: vec![None; input.rows], primal_envelope_errors: vec![None; input.rows],
            estimated_errors: Vec::new(), best_effort: Vec::new(),
        };
        self.diagnostics.resize_with(input.rows, Vec::new);
        let mut lanes = Vec::with_capacity(input.rows);
        let mut exact = Vec::with_capacity(input.values.len());
        for (lane, row) in input.values.chunks_exact(input_width).enumerate() {
            if let Some(status) = validate_input_row(row, &self.layout, self.components) {
                result.statuses[lane] = status;
                continue;
            }
            let previous = exact.len();
            for value in row {
                match (value.re.exact_rational(), value.im.exact_rational()) {
                    (Some(re), Some(im)) => exact.push(Complex::new(re, im)),
                    _ => {
                        result.statuses[lane] = LaneStatus::Unstable(
                            "numeric type cannot certify its exact represented supplied-G child inputs".into(),
                        );
                        break;
                    }
                }
            }
            if exact.len() == previous + input_width { lanes.push(lane); }
            else { exact.truncate(previous); }
        }
        // A full check, with no selected-component shortcut, precedes even the
        // rank predicate. All p/G/mass/mu arguments keep their original order.
        let certificates = self.compatibility.check_batch(&exact, lanes.len());
        add_proof_work(&mut self.certificate_stats, &self.compatibility)?;
        let certificates = certificates?;
        check_rows(certificates.len(), lanes.len())?;
        let mut compatible_lanes = Vec::with_capacity(lanes.len());
        let mut compatible_exact = Vec::with_capacity(exact.len());
        for (compact, (lane, certificate)) in lanes.into_iter().zip(certificates).enumerate() {
            match certificate {
                ExactZeroCertificate::Zero => {
                    compatible_lanes.push(lane);
                    compatible_exact.extend_from_slice(&exact[compact * input_width..(compact + 1) * input_width]);
                    self.diagnostics[lane].push("exact full-Taylor supplied-G/Cartesian compatibility for packed pentagon child".into());
                }
                ExactZeroCertificate::Nonzero { constraint, component } => result.statuses[lane] = LaneStatus::InvalidInput(format!(
                    "packed pentagon supplied-G/Cartesian constraint {constraint} is certified nonzero in Taylor component {component}; input was preserved"
                )),
                ExactZeroCertificate::Unresolved { working_bits } => result.statuses[lane] = LaneStatus::Unstable(format!(
                    "packed pentagon compatibility certificate unresolved at {working_bits} bits; input was preserved"
                )),
            }
        }
        let certificates = self.rank.check_batch(&compatible_exact, compatible_lanes.len());
        add_proof_work(&mut self.certificate_stats, &self.rank)?;
        let certificates = certificates?;
        check_rows(certificates.len(), compatible_lanes.len())?;
        let mut planar_lanes = Vec::new();
        let mut regular_lanes = Vec::new();
        for (lane, certificate) in compatible_lanes.into_iter().zip(certificates) {
            match certificate {
                ExactZeroCertificate::Zero => {
                    planar_lanes.push(lane);
                    self.diagnostics[lane].push("exact full-Taylor planar determinant; dispatching directly to masked planar child".into());
                }
                ExactZeroCertificate::Nonzero { constraint: 0, component: 0 } => {
                    regular_lanes.push(lane);
                    self.diagnostics[lane].push("certified regular Cartesian primal; dispatching supplied-G PV child".into());
                }
                ExactZeroCertificate::Nonzero { component, .. } => result.statuses[lane] = LaneStatus::Unstable(format!(
                    "packed pentagon determinant is nonzero in Taylor component {component} without a certified regular primal; transverse rank-changing jets require another path"
                )),
                ExactZeroCertificate::Unresolved { working_bits } => result.statuses[lane] = LaneStatus::Unstable(format!(
                    "packed pentagon rank certificate unresolved at {working_bits} bits; input was preserved"
                )),
            }
        }
        if let Some(started) = arguments_started { self.profile.arguments = started.elapsed(); }
        // Each branch receives one full-layout batch. A preparation/evaluation
        // failure marks only that branch's lanes and cannot erase its neighbor.
        self.dispatch_planar(input, &planar_lanes, &mut result)?;
        self.dispatch_generic(input, &regular_lanes, &mut result)?;
        Ok(result)
    }

    fn prepare_planar(&mut self) -> Result<(), ReductionError> {
        if self.planar.is_some() { return Ok(()); }
        let plan = CertifiedPlanarReductionPlan::new_for_six_point_child(self.settings.clone())?;
        let mut workspace = match &self.shape {
            None => CertifiedPlanarWorkspace::new_with_policies(
                &plan, self.prototype.clone(), WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica,
            ),
            Some(shape) => CertifiedPlanarWorkspace::new_dual_with_policies(
                &plan, self.prototype.clone(), HyperDual::<Complex<Rational>>::new(shape.clone()),
                WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica,
            ),
        }?;
        workspace.set_accuracy_precision(self.accuracy_bits)?;
        workspace.set_profiling(self.profiling);
        self.planar = Some(workspace);
        Ok(())
    }

    fn prepare_generic(&mut self) -> Result<(), ReductionError> {
        if self.generic.is_some() { return Ok(()); }
        // This is the sole ordinary plan construction site. It is reached
        // only after a batch contains a certified regular primal lane.
        let plan = ReductionPlan::new(self.settings.clone())?;
        let mut workspace = match &self.shape {
            None => Workspace::new_with_policies(
                &plan, self.prototype.clone(), WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica,
            ),
            Some(shape) => Workspace::new_dual_with_policies(
                &plan, self.prototype.clone(), HyperDual::<Complex<Rational>>::new(shape.clone()),
                WorkspacePolicy::RETAIN_ENVELOPES, MasterExecutionPolicy::Symbolica,
            ),
        }?;
        workspace.set_accuracy_precision(self.accuracy_bits)?;
        workspace.set_profiling(self.profiling);
        self.generic = Some(workspace);
        Ok(())
    }

    fn dispatch_planar(
        &mut self, input: &KinematicsBatch<T>, lanes: &[usize], result: &mut BatchResult<T>,
    ) -> Result<(), ReductionError> {
        if lanes.is_empty() { return Ok(()); }
        let started = self.profiling.then(Instant::now);
        let prepared = self.prepare_planar();
        if let Some(started) = started { self.profile.arguments += started.elapsed(); }
        if let Err(error) = prepared {
            fail_group(result, lanes, format!("checked planar child preparation failed: {error}"));
            return Ok(());
        }
        let packed = pack(input, lanes, self.components)?;
        let workspace = self.planar.as_mut().expect("prepared planar child");
        let reduced = workspace.reduce_batch(&packed);
        add_execution(&mut self.stats, workspace.last_execution())?;
        add_certificates(&mut self.certificate_stats, workspace.last_certificate_execution())?;
        add_profile(&mut self.profile, workspace.last_profile());
        append_diagnostics(&mut self.diagnostics, lanes, workspace.last_diagnostics());
        self.scatter_group(result, lanes, reduced, "checked planar child")
    }

    fn dispatch_generic(
        &mut self, input: &KinematicsBatch<T>, lanes: &[usize], result: &mut BatchResult<T>,
    ) -> Result<(), ReductionError> {
        if lanes.is_empty() { return Ok(()); }
        let started = self.profiling.then(Instant::now);
        let prepared = self.prepare_generic();
        if let Some(started) = started { self.profile.arguments += started.elapsed(); }
        if let Err(error) = prepared {
            fail_group(result, lanes, format!("checked regular PV child preparation failed: {error}"));
            return Ok(());
        }
        let packed = pack(input, lanes, self.components)?;
        let workspace = self.generic.as_mut().expect("prepared regular child");
        let reduced = workspace.reduce_batch(&packed);
        add_execution(&mut self.stats, workspace.last_execution())?;
        add_certificates(&mut self.certificate_stats, workspace.last_planar_certificate_execution())?;
        self.certificate_stats.reduction_batches = count(self.certificate_stats.reduction_batches, 1)?;
        add_profile(&mut self.profile, workspace.last_profile());
        append_diagnostics(&mut self.diagnostics, lanes, workspace.last_diagnostics());
        self.scatter_group(result, lanes, reduced, "checked regular PV child")
    }

    fn scatter_group(
        &self, result: &mut BatchResult<T>, lanes: &[usize],
        reduced: Result<BatchResult<T>, ReductionError>, label: &str,
    ) -> Result<(), ReductionError> {
        let reduced = match reduced {
            Ok(reduced) => reduced,
            Err(error) => {
                fail_group(result, lanes, format!("{label} evaluation failed: {error}"));
                return Ok(());
            }
        };
        if let Err(error) = super::validate_result_layout(
            &reduced, &self.basis, self.components, &self.metadata, lanes.len(),
        ) {
            fail_group(result, lanes, format!("{label} result layout failed: {error}"));
            return Ok(());
        }
        let width = product(product(self.basis.len(), 4)?, self.components)?;
        for (compact, &lane) in lanes.iter().enumerate() {
            result.statuses[lane] = reduced.statuses[compact].clone();
            result.fallback_used[lane] = reduced.fallback_used[compact];
            result.analytic_used[lane] = reduced.analytic_used[compact];
            // Preserve the exact actual-route context and its coupled errors;
            // no synthetic envelope is inferred from the returned moments.
            result.primal_envelopes[lane] = reduced.primal_envelopes[compact].clone();
            result.primal_envelope_errors[lane] = reduced.primal_envelope_errors[compact].clone();
            if let Ok(values) = reduced.lane(compact) {
                result.values[lane * width..(lane + 1) * width].clone_from_slice(values);
            }
        }
        Ok(())
    }
}

fn proof(parameters: &[Atom], residuals: &[Atom], shape: &Shape) -> Result<ExactZeroConstraintPlan, ReductionError> {
    match shape {
        None => ExactZeroConstraintPlan::new(parameters, residuals),
        Some(shape) => ExactZeroConstraintPlan::new_dual(parameters, residuals, &HyperDual::<u8>::new(shape.clone())),
    }
}
fn variable(name: &str) -> Atom { Atom::var(symbolica::symbol!(format!("iregi::{name}"))) }
fn preparation(error: impl ToString) -> ReductionError { ReductionError::Preparation(error.to_string()) }
fn product(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b).ok_or_else(|| ReductionError::InvalidInput("checked pentagon batch size overflow".into()))
}
fn count(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_add(b).ok_or_else(|| ReductionError::Evaluation("checked pentagon execution counter overflow".into()))
}
fn check_rows(actual: usize, expected: usize) -> Result<(), ReductionError> {
    if actual == expected { Ok(()) }
    else { Err(ReductionError::Evaluation("checked pentagon certificate row count mismatch".into())) }
}
fn pack<T: NativeFloat>(input: &KinematicsBatch<T>, lanes: &[usize], components: usize) -> Result<KinematicsBatch<T>, ReductionError> {
    let width = product(41, components)?;
    let mut packed = KinematicsBatch { rows: lanes.len(), components, values: Vec::with_capacity(product(lanes.len(), width)?) };
    for &lane in lanes { packed.values.extend_from_slice(&input.values[lane * width..(lane + 1) * width]); }
    Ok(packed)
}
fn fail_group<T: NativeFloat>(result: &mut BatchResult<T>, lanes: &[usize], message: String) {
    for &lane in lanes { result.statuses[lane] = LaneStatus::MasterFailure(message.clone()); }
}
fn append_diagnostics(destination: &mut [Vec<String>], lanes: &[usize], source: &[Vec<String>]) {
    for (compact, &lane) in lanes.iter().enumerate() {
        if let Some(notes) = source.get(compact) { destination[lane].extend_from_slice(notes); }
    }
}
fn add_proof_work(total: &mut CertificateExecutionStats, proof: &ExactZeroConstraintWorkspace) -> Result<(), ReductionError> {
    total.evaluated_rows = count(total.evaluated_rows, proof.evaluated_rows())?;
    total.passes = count(total.passes, proof.passes())?;
    Ok(())
}
fn add_certificates(total: &mut CertificateExecutionStats, work: CertificateExecutionStats) -> Result<(), ReductionError> {
    total.evaluated_rows = count(total.evaluated_rows, work.evaluated_rows)?;
    total.passes = count(total.passes, work.passes)?;
    total.reduction_batches = count(total.reduction_batches, work.reduction_batches)?;
    Ok(())
}
fn add_execution(total: &mut ExecutionStats, work: ExecutionStats) -> Result<(), ReductionError> {
    // Logical input rows belong to this actor, not the sum of child attempts.
    macro_rules! add { ($field:ident) => { total.$field = count(total.$field, work.$field)?; }; }
    add!(argument_rows); add!(master_family_batches); add!(master_rows); add!(pentagon_box_rows);
    add!(reduction_rows); add!(fallback_rows); add!(fallback_successes); add!(analytic_rows);
    Ok(())
}
fn add_profile(total: &mut ExecutionProfile, profile: Option<ExecutionProfile>) {
    if let Some(profile) = profile {
        total.arguments += profile.arguments;
        total.scalar_masters += profile.scalar_masters;
        total.reduction += profile.reduction;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Algorithm, InputPolicy, InvariantInput};

    fn settings() -> ReductionSettings {
        let mut settings = ReductionSettings::new(5, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        settings.invariants = InvariantInput::SuppliedScalarProducts;
        settings
    }
    fn constant_planar_row() -> Vec<Complex<f64>> {
        // p=[0,ex,ey,ez,ex+ey+ez], eta=(+---), with exact packed G input.
        [0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,1,
         0,0,0,0,0, -1,0,0,-1, -1,0,-1, -1,-1,-3,
         4,9,16,25,36,3].into_iter().map(|v| Complex::new(f64::from(v),0.)).collect()
    }
    fn mixed_shape() -> Shape { Some(vec![vec![0,0],vec![0,1],vec![1,0],vec![1,1]]) }
    fn constant_jets() -> Vec<Complex<f64>> {
        constant_planar_row().into_iter().flat_map(|v| [v,Complex::new(0.,0.),Complex::new(0.,0.),Complex::new(0.,0.)]).collect()
    }
    #[test]
    fn checked_actor_constructor_is_lazy_and_domain_is_narrow() {
        for shape in [None, Some(vec![vec![0]]), mixed_shape()] {
            let mut workspace = CheckedFiveWorkspace::new(settings(),0_f64,shape.clone()).unwrap();
            assert!(workspace.generic.is_none() && workspace.planar.is_none());
            assert_eq!(workspace.layout.scalar_products,Some(20..35));
            assert_eq!(workspace.layout.masses,35..40);
            assert_eq!(workspace.layout.scale,40);
            assert_eq!(workspace.metadata.dual_shape(),shape.as_deref());
            workspace.set_accuracy_precision(52).unwrap();
            assert!(workspace.set_accuracy_precision(54).is_err());
            workspace.set_profiling(true);
            let empty = workspace.reduce_batch(&KinematicsBatch { rows:0,components:workspace.components,values:vec![] }).unwrap();
            assert!(empty.statuses.is_empty());
            assert!(workspace.generic.is_none() && workspace.planar.is_none());
            assert_eq!(workspace.last_execution(),ExecutionStats::default());
            assert_eq!(workspace.last_planar_certificate_execution(),CertificateExecutionStats::default());
            assert!(workspace.last_profile().is_some());
        }
        for field in 0..9 {
            let mut invalid=settings();
            match field {
                0=>invalid.rank=6, 1=>invalid.dimension_shift=1, 2=>invalid.powers[0]=2,
                3=>invalid.algorithm=Algorithm::Hybrid, 4=>invalid.invariants=InvariantInput::FromShifts,
                5=>invalid.input_policy=InputPolicy::legacy_onshell(), 6=>invalid.propagators=6,
                7=>invalid.convention=Convention::OneLoop, _=>invalid.tensor_weights=TensorWeights::Multinomial,
            }
            assert!(CheckedFiveWorkspace::new(invalid,0_f64,None).is_err());
        }
    }
    #[test]
    fn bad_supplied_products_reject_before_any_numerical_actor_exists() {
        let mut workspace=CheckedFiveWorkspace::new(settings(),0_f64,mixed_shape()).unwrap();
        let mut mixed=constant_jets(); mixed[20*4+3].re=1.;
        let mut primal=constant_jets(); primal[20*4].re=1.;
        let input=KinematicsBatch { rows:2,components:4,values:[mixed,primal].concat() };
        let original=input.values.clone();
        let result=workspace.reduce_batch(&input).unwrap();
        assert_eq!(input.values,original);
        for (lane,component) in [(0,3),(1,0)] {
            assert!(matches!(&result.statuses[lane],LaneStatus::InvalidInput(message)
                if message.contains("supplied-G/Cartesian") && message.contains(&format!("component {component}"))));
            assert!(result.lane(lane).is_err());
        }
        assert!(workspace.generic.is_none() && workspace.planar.is_none());
        assert_eq!(workspace.last_execution().master_rows,0);
        assert_eq!(workspace.last_planar_certificate_execution().evaluated_rows,2);
        assert_eq!(workspace.last_planar_certificate_execution().reduction_batches,0);
        assert_eq!(workspace.rank.evaluated_rows(),0);
        assert!(workspace.reduce_batch(&KinematicsBatch { rows:1,components:4,values:vec![] }).is_err());
        assert_eq!(workspace.last_planar_certificate_execution().evaluated_rows,0);
        assert!(workspace.last_diagnostics().is_empty());
    }
    #[test]
    fn transverse_mixed_rank_rejects_before_generic_or_planar_preparation() {
        let mut workspace=CheckedFiveWorkspace::new(settings(),0_f64,mixed_shape()).unwrap();
        let mut mixed=constant_jets();
        // p4.time=st leaves the primal planar. Its square is nilpotent zero in
        // this shape, so the unchanged supplied G remains exactly compatible.
        mixed[16*4+3].re=1.;
        let input=KinematicsBatch { rows:1,components:4,values:mixed };
        let original=input.values.clone();
        let result=workspace.reduce_batch(&input).unwrap();
        assert_eq!(input.values,original);
        assert!(matches!(&result.statuses[0],LaneStatus::Unstable(message)
            if message.contains("Taylor component 3") && message.contains("transverse rank-changing")));
        assert!(result.lane(0).is_err());
        assert!(workspace.generic.is_none() && workspace.planar.is_none());
        assert_eq!(workspace.last_execution().argument_rows,0);
        assert_eq!(workspace.last_execution().master_rows,0);
        assert_eq!(workspace.last_planar_certificate_execution().evaluated_rows,2);
        assert_eq!(workspace.last_planar_certificate_execution().passes,2);
        assert_eq!(workspace.last_planar_certificate_execution().reduction_batches,0);
    }
}
