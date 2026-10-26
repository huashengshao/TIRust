//! Low-point h0 raised powers from Symbolica auxiliary mass Taylor coefficients.
//!
//! A unit-power N5 tensor is evaluated in aux×user coefficient space. Selecting
//! aux^(nu-1) introduces no factorial. This module only packs/indexes values;
//! Symbolica owns all derivative products. Selected values, full-component
//! sensitivity amplitudes and Euclidean quadrature diagnostics enter a FRESH
//! identity parent with dotted settings and dimensions. A unit primal envelope
//! is never relabelled as a dotted envelope.
mod layout;
mod checked_regular;
use checked_regular::CheckedRegularUnitGuard;
use layout::{DottedLayout,DottedLayoutBudget};
use std::{sync::Arc,time::Instant};
use iregi_masters::{NativeFloat,MasterExecutionPolicy};
use symbolica::{atom::{Atom,AtomCore},domains::{dual::{DualNumber,DualNumberStructure,HyperDual},float::Complex},evaluate::EvaluationDomain};
use crate::{Algorithm,BatchResult,CertificateExecutionStats,ContractionPlan,ExecutionProfile,ExecutionStats,InputLayout,InputPolicy,InvariantInput,
    KinematicsBatch,LaneStatus,ReductionError,ReductionPlan,ReductionSettings,ResultMetadata,Workspace,
    basis::Monomial,execution::{WorkspacePolicy,quadrature_context::{CoupledQuadratureBatch,QuadratureErrorContext,QuadratureErrorOrigin}},
    stability::quadrature::QuadratureStabilityPlan,types::PrimalEnvelopeRoute};

/// Explicit control-work/count limits, not mathematical power or point caps.
/// These do not bound the evaluator's total allocator/RSS; the caller still
/// controls process time/address limits for expensive qualification work.
#[derive(Clone,Copy,Debug)]
pub struct MassTaylorBudget {
    pub max_joint_components:usize,pub max_shape_entries:usize,pub max_metadata_pair_visits:usize,
    pub max_projection_parameters:usize,pub max_batch_values:usize,
}
impl Default for MassTaylorBudget {
    fn default()->Self {Self{max_joint_components:1024,max_shape_entries:65536,
        max_metadata_pair_visits:2_100_000,max_projection_parameters:1_000_000,max_batch_values:16_000_000}}
}
impl MassTaylorBudget {
    fn layout(self)->DottedLayoutBudget {DottedLayoutBudget{max_joint_components:self.max_joint_components,
        max_shape_entries:self.max_shape_entries,max_metadata_pair_visits:self.max_metadata_pair_visits}}
}
fn invalid(s:impl Into<String>)->ReductionError {ReductionError::InvalidInput(s.into())}
fn prep(s:impl Into<String>)->ReductionError {ReductionError::Preparation(s.into())}
fn mul(a:usize,b:usize)->Result<usize,ReductionError>{a.checked_mul(b).ok_or_else(||invalid("mass-Taylor dimensions overflow"))}
fn add(a:usize,b:usize)->Result<usize,ReductionError>{a.checked_add(b).ok_or_else(||invalid("mass-Taylor dimensions overflow"))}
fn var(name:&str)->Atom {Atom::var(symbolica::symbol!(format!("iregi_mass_taylor::{name}")))}

#[derive(Clone,Debug)]
pub struct MassTaylorReductionPlan {
    settings:ReductionSettings,unit:Arc<ReductionPlan>,budget:MassTaylorBudget,notes:Vec<String>,
    // Selects a private runtime policy, never stored numerical proof evidence.
    checked_regular_unit:bool,
}
impl MassTaylorReductionPlan {
    pub fn new(settings:ReductionSettings)->Result<Self,ReductionError>{Self::with_budget(settings,MassTaylorBudget::default())}
    pub fn with_budget(settings:ReductionSettings,budget:MassTaylorBudget)->Result<Self,ReductionError>{
        validate_settings(&settings)?;
        let _=DottedLayout::new(&settings.powers,None,budget.layout())?;
        let mut unit=settings.clone();unit.powers.fill(1);
        Self::with_shared_unit(settings,Arc::new(ReductionPlan::new(unit)?),budget)
    }
    /// Internal reuse boundary. A proof-selected supplied-G/planar unit recipe
    /// must have been selected by its owning checked executor using the FULL
    /// aux×user shape. This method creates/prunes no charts or cofactor masks;
    /// it cannot turn a primal-only certificate into a full-jet certificate.
    pub(crate) fn with_shared_unit(settings:ReductionSettings,unit:Arc<ReductionPlan>,budget:MassTaylorBudget)
        ->Result<Self,ReductionError>{
        validate_settings(&settings)?;
        let _=DottedLayout::new(&settings.powers,None,budget.layout())?;
        let u=unit.settings();
        if u.propagators!=settings.propagators||u.rank!=settings.rank||u.dimension_shift!=settings.dimension_shift
            ||u.powers.iter().any(|&p|p!=1)||u.powers.len()!=settings.powers.len()
            ||u.algorithm!=settings.algorithm||u.convention!=settings.convention||u.tensor_weights!=settings.tensor_weights
            ||u.invariants!=settings.invariants||u.input_policy!=settings.input_policy||!unit.external_primal_seeds.is_empty()
            ||unit.input_layout().masses.len()!=settings.propagators||unit.input_layout().masses.end>unit.parameter_count()
            ||unit.input_layout().scale>=unit.parameter_count(){
            return Err(prep("mass-Taylor shared unit recipe does not preserve requested settings/layout"));
        }
        let needed=add(unit.parameter_count(),mul(mul(unit.basis().len(),4)?,2)?)?;
        if needed>budget.max_projection_parameters||budget.max_batch_values==0 {
            return Err(prep("mass-Taylor explicit projection/batch resource budget exceeded"));
        }
        let notes=vec![format!("MassTaylor N5 h0 powers {:?}: requested {:?}; actual unit {:?}; Symbolica aux mass coefficient, no extra factorial; selected context rebuilt at dotted dimensions",settings.powers,settings.algorithm,unit.algorithm_used())];
        Ok(Self{settings,unit,budget,notes,checked_regular_unit:false})
    }
    /// Private checked affine fallback. Shared structural unit preparation is
    /// reused, but each lifted aux x user row must independently pass fresh
    /// complete supplied-G compatibility and regular-primal checks at runtime.
    pub(crate) fn with_shared_checked_regular_unit(settings:ReductionSettings,unit:Arc<ReductionPlan>,budget:MassTaylorBudget)
        ->Result<Self,ReductionError>{
        if settings.invariants!=InvariantInput::SuppliedScalarProducts {
            return Err(prep("checked regular MassTaylor requires authoritative supplied-G inputs"));
        }
        let mut plan=Self::with_shared_unit(settings,unit,budget)?;
        let layout=plan.input_layout();
        if layout.shifts!=(0..20)||layout.scalar_products!=Some(20..35)||layout.masses!=(35..40)
            ||layout.scale!=40||plan.parameter_count()!=41 {
            return Err(prep("checked regular MassTaylor requires the full original N5 supplied-G geometry layout"));
        }
        plan.checked_regular_unit=true;
        plan.notes.push("Private checked regular supplied-G unit: every lifted auxiliary-mass x user Taylor row is freshly certified before unit numerical evaluation; only a certified regular primal is admitted".into());
        Ok(plan)
    }
    pub fn settings(&self)->&ReductionSettings{&self.settings}
    pub fn basis(&self)->&[Monomial]{self.unit.basis()}
    pub fn input_layout(&self)->&InputLayout{self.unit.input_layout()}
    pub fn parameter_count(&self)->usize{self.unit.parameter_count()}
    pub fn algorithm_used(&self)->Algorithm{self.unit.algorithm_used()}
    pub fn unit_algorithm_used(&self)->Algorithm{self.unit.algorithm_used()}
    pub fn unit_plan(&self)->&ReductionPlan{&self.unit}
    pub fn preparation_notes(&self)->&[String]{&self.notes}
    pub fn contraction_plan(&self,terms:&[Monomial])->Result<ContractionPlan,ReductionError>{
        ContractionPlan::from_basis(self.basis(),self.settings.convention,self.settings.tensor_weights,terms)
    }
}
pub(crate) fn settings_supported(settings:&ReductionSettings)->bool {
    settings.propagators==5&&settings.rank<=6&&settings.dimension_shift==0&&settings.powers.len()==5
        &&!settings.powers.contains(&0)&&settings.powers.iter().any(|&p|p!=1)&&settings.input_policy==InputPolicy::Preserve
}
fn validate_settings(settings:&ReductionSettings)->Result<(),ReductionError>{
    if !settings_supported(settings) {
        return Err(ReductionError::Unsupported("MassTaylor currently requires N5 h0, rank<=6, preserved routing and positive nonunit powers".into()));
    }
    Ok(())
}

#[derive(Clone,Debug,Default)]
pub struct MassTaylorExecutionStats {
    pub input_rows:usize,pub user_components:usize,pub internal_components:usize,
    pub unit_execution:ExecutionStats,pub projection_execution:ExecutionStats,
    pub unit_algorithm_used:Algorithm,pub requested_algorithm:Algorithm,
    /// Exact lifted-row compatibility/rank proof work, separate from numerical
    /// kernel counts. The summary does not add these rows or passes again.
    pub certificate_execution:CertificateExecutionStats,
}
impl MassTaylorExecutionStats {
    pub fn summary(&self)->Result<ExecutionStats,ReductionError>{
        let a=self.unit_execution;let b=self.projection_execution;
        let count=|a:usize,b:usize|a.checked_add(b).ok_or_else(||ReductionError::Evaluation("mass-Taylor execution summary counter overflow".into()));
        Ok(ExecutionStats{input_rows:self.input_rows,argument_rows:count(a.argument_rows,b.argument_rows)?,
            master_family_batches:count(a.master_family_batches,b.master_family_batches)?,master_rows:count(a.master_rows,b.master_rows)?,
            pentagon_box_rows:count(a.pentagon_box_rows,b.pentagon_box_rows)?,reduction_rows:count(a.reduction_rows,b.reduction_rows)?,
            fallback_rows:count(a.fallback_rows,b.fallback_rows)?,fallback_successes:count(a.fallback_successes,b.fallback_successes)?,
            analytic_rows:count(a.analytic_rows,b.analytic_rows)?})
    }
}
struct ProjectionRecipe {plan:ReductionPlan,checks:QuadratureStabilityPlan,exact_zero_outputs:Vec<bool>}
impl ProjectionRecipe {
    fn prepare(plan:&MassTaylorReductionPlan,shape:Option<&[Vec<usize>]>)->Result<Self,ReductionError>{
        let count=mul(plan.basis().len(),4)?;let mut parameters=plan.unit.parameters.clone();
        let values:Vec<_>=(0..count).map(|i|var(&format!("selected_value_{i}"))).collect();
        let amplitudes:Vec<_>=(0..count).map(|i|var(&format!("selected_amplitude_{i}"))).collect();
        parameters.extend(values.iter().cloned());parameters.extend(amplitudes.iter().cloned());
        if parameters.len()>plan.budget.max_projection_parameters{return Err(prep("mass-Taylor projection parameter budget exceeded"));}
        let power_sum=plan.settings.powers.iter().try_fold(0i32,|sum,&p|sum.checked_add(i32::from(p)))
            .ok_or_else(||prep("mass-Taylor dimension sum overflow"))?;
        let base=4i32.checked_sub(power_sum.checked_mul(2).ok_or_else(||prep("mass-Taylor dimension overflow"))?)
            .ok_or_else(||prep("mass-Taylor dimension overflow"))?;
        let mut external=Vec::with_capacity(count);let mut moments=Vec::with_capacity(plan.basis().len());
        for(moment,basis)in plan.basis().iter().enumerate(){
            let dimension=base.checked_add(i32::try_from(basis.rank()).map_err(|_|prep("mass-Taylor tensor dimension overflow"))?)
                .ok_or_else(||prep("mass-Taylor tensor dimension overflow"))?;
            moments.push(std::array::from_fn(|pole|values[4*moment+pole].clone()));
            for pole in 0..4 {let i=4*moment+pole;external.push((values[i].clone(),amplitudes[i].clone(),dimension));}
        }
        let order=shape.map_or(Ok(0u16),|s|s.iter().try_fold(0u16,|maximum,row|
            row.iter().try_fold(0usize,|sum,&n|sum.checked_add(n)).and_then(|n|u16::try_from(n).ok())
                .map(|n|maximum.max(n)).ok_or_else(||prep("mass-Taylor user analytic order exceeds u16"))))?;
        let imported=ReductionPlan::new_lowpoint_imported_tensor_parent(plan.settings.clone(),order,parameters,external,
            plan.input_layout().clone(),plan.unit.raw_distances.clone(),moments,Vec::new(),
            vec!["MassTaylor imported identity: selected coefficient AND selected sensitivity amplitudes; fresh target dotted dimensions, no copied unit envelope".into()])?;
        let exact_zero_outputs=imported.moments.iter().flatten().map(AtomCore::is_zero).collect();
        let checks=QuadratureStabilityPlan::imported_dense_checks(&imported)?;
        Ok(Self{plan:imported,checks,exact_zero_outputs})
    }
}

#[derive(Clone)]
pub struct MassTaylorWorkspace<T:NativeFloat> {
    plan:MassTaylorReductionPlan,prototype:T,accuracy_bits:u32,layout:Arc<DottedLayout>,
    unit:Workspace<T>,projection:Workspace<T>,checks:QuadratureStabilityPlan,exact_zero_outputs:Vec<bool>,
    stats:MassTaylorExecutionStats,diagnostics:Vec<Vec<String>>,profiling:bool,profile:ExecutionProfile,
    checked_regular:Option<CheckedRegularUnitGuard>,
}
impl<T:NativeFloat> MassTaylorWorkspace<T>where Complex<T>:EvaluationDomain {
    pub fn new(plan:&MassTaylorReductionPlan,prototype:T)->Result<Self,ReductionError>{Self::new_with_master_policy(plan,prototype,MasterExecutionPolicy::Default)}
    pub fn new_with_master_policy(plan:&MassTaylorReductionPlan,prototype:T,masters:MasterExecutionPolicy)->Result<Self,ReductionError>{
        Self::prepare(plan,prototype,DottedLayout::new(&plan.settings.powers,None,plan.budget.layout())?,masters)
    }
    pub fn new_dual<D:DualNumberStructure>(plan:&MassTaylorReductionPlan,prototype:T,user:D)->Result<Self,ReductionError>{
        Self::new_dual_with_master_policy(plan,prototype,user,MasterExecutionPolicy::Default)
    }
    pub fn new_dual_with_master_policy<D:DualNumberStructure>(plan:&MassTaylorReductionPlan,prototype:T,user:D,masters:MasterExecutionPolicy)
        ->Result<Self,ReductionError>{
        let shape:Vec<Vec<usize>>=user.get_shape().into_iter().map(<[usize]>::to_vec).collect();
        let layout=DottedLayout::new(&plan.settings.powers,Some(&shape),plan.budget.layout())?;
        iregi_masters::validate_dual_structure(&user).map_err(|e|invalid(e.to_string()))?;
        Self::prepare(plan,prototype,layout,masters)
    }
    fn prepare(plan:&MassTaylorReductionPlan,prototype:T,layout:DottedLayout,masters:MasterExecutionPolicy)->Result<Self,ReductionError>{
        let recipe=ProjectionRecipe::prepare(plan,layout.user_shape())?;
        let joint=HyperDual::<u8>::new(layout.joint_shape().to_vec());
        let checked_regular=if plan.checked_regular_unit {Some(CheckedRegularUnitGuard::new(layout.joint_shape())?)}else{None};
        let unit=Workspace::new_dual_with_policies(&plan.unit,prototype.clone(),joint,WorkspacePolicy::AFFINE_QUADRATURE_CHILD,masters)?;
        let projection=match layout.user_shape(){
            None=>Workspace::new_with_policies(&recipe.plan,prototype.clone(),WorkspacePolicy::AFFINE_QUADRATURE_PARENT,masters)?,
            Some(s)=>Workspace::new_dual_with_policies(&recipe.plan,prototype.clone(),HyperDual::<u8>::new(s.to_vec()),WorkspacePolicy::AFFINE_QUADRATURE_PARENT,masters)?,
        };
        let mut workspace=Self{plan:plan.clone(),accuracy_bits:prototype.get_precision(),prototype,layout:Arc::new(layout),unit,projection,
            checks:recipe.checks,exact_zero_outputs:recipe.exact_zero_outputs,stats:Default::default(),diagnostics:Vec::new(),profiling:false,profile:Default::default(),checked_regular};
        workspace.reset(0);Ok(workspace)
    }
    pub fn component_count(&self)->usize{self.layout.user_components()}
    pub fn dual_shape(&self)->Option<&[Vec<usize>]>{self.layout.user_shape()}
    pub fn binary_precision(&self)->u32{self.prototype.get_precision()}
    pub fn unit_algorithm_used(&self)->Algorithm{self.plan.unit_algorithm_used()}
    pub fn algorithm_used(&self)->Algorithm{self.unit_algorithm_used()}
    pub fn preparation_notes(&self)->&[String]{self.plan.preparation_notes()}
    pub fn set_accuracy_precision(&mut self,bits:u32)->Result<(),ReductionError>{
        self.unit.set_accuracy_precision(bits)?;self.projection.set_accuracy_precision(bits)?;self.accuracy_bits=bits;Ok(())
    }
    pub fn set_profiling(&mut self,on:bool){self.profiling=on;self.unit.set_profiling(on);self.projection.set_profiling(on);}
    pub fn last_profile(&self)->Option<ExecutionProfile>{self.profiling.then_some(self.profile)}
    pub fn last_execution(&self)->&MassTaylorExecutionStats{&self.stats}
    pub fn execution_summary(&self)->Result<ExecutionStats,ReductionError>{self.stats.summary()}
    pub fn unit_execution(&self)->ExecutionStats{self.stats.unit_execution}
    pub fn projection_execution(&self)->ExecutionStats{self.stats.projection_execution}
    pub fn last_diagnostics(&self)->&[Vec<String>]{&self.diagnostics}
    fn reset(&mut self,rows:usize){self.stats=MassTaylorExecutionStats{input_rows:rows,user_components:self.layout.user_components(),
        internal_components:self.layout.joint_components(),unit_algorithm_used:self.plan.unit_algorithm_used(),requested_algorithm:self.plan.settings.algorithm,..Default::default()};
        self.diagnostics.clear();self.profile=Default::default();}
    pub fn reduce_batch(&mut self,batch:&KinematicsBatch<T>)->Result<BatchResult<T>,ReductionError>{Ok(self.reduce_batch_with_quadrature_context(batch)?.into_parts().0)}
    pub(crate) fn reduce_batch_with_quadrature_context(&mut self,batch:&KinematicsBatch<T>)->Result<CoupledQuadratureBatch<T>,ReductionError>{
        self.reset(batch.rows);let started=self.profiling.then(Instant::now);
        let evaluated=self.reduce_impl(batch);if let Some(started)=started{self.profile.total=started.elapsed();}evaluated
    }
    fn reduce_impl(&mut self,batch:&KinematicsBatch<T>)->Result<CoupledQuadratureBatch<T>,ReductionError>{
        let geometry=self.plan.parameter_count();let user=self.layout.user_components();let joint=self.layout.joint_components();
        if batch.components!=user||batch.values.len()!=mul(mul(batch.rows,geometry)?,user)?{return Err(invalid("mass-Taylor input shape/dimensions differ"));}
        let lifted_count=mul(mul(batch.rows,geometry)?,joint)?;
        let outputs=mul(self.plan.basis().len(),4)?;
        let unit_outputs=mul(mul(batch.rows,outputs)?,joint)?;
        let target_outputs=mul(mul(batch.rows,outputs)?,user)?;
        let projection_inputs=mul(mul(batch.rows,add(geometry,mul(outputs,2)?)?)?,user)?;
        if [lifted_count,unit_outputs,target_outputs,projection_inputs].iter().any(|&n|n>self.plan.budget.max_batch_values){
            return Err(prep("mass-Taylor dense input/output resource budget exceeded"));
        }
        let zero=Complex::new(self.prototype.zero(),self.prototype.zero());let one=Complex::new(self.prototype.one(),self.prototype.zero());
        let physical_width=mul(geometry,user)?;
        let mut admitted=Vec::new();let mut rejected=vec![None;batch.rows];
        for(row,physical)in batch.values.chunks_exact(physical_width).enumerate(){
            if let Some(reason)=auxiliary_mass_boundary(physical,self.plan.input_layout(),&self.layout){rejected[row]=Some(reason);}
            else{admitted.push(row);}
        }
        // Pack admitted physical rows once. A rejected zero-mass direction
        // enters neither scalar-master activation nor any unit analytic route.
        let mut lifted=KinematicsBatch{rows:admitted.len(),components:joint,
            values:vec![zero.clone();mul(mul(admitted.len(),geometry)?,joint)?]};
        for(compact,&row)in admitted.iter().enumerate(){
            for parameter in 0..geometry{for(component,&embedded)in self.layout.embedded_user_indices().iter().enumerate(){
                lifted.values[(compact*geometry+parameter)*joint+embedded]=batch.values[(row*geometry+parameter)*user+component].clone();
            }}
            for &(line,index)in self.layout.mass_seeds(){
                let mass=self.plan.input_layout().masses.start.checked_add(line).ok_or_else(||invalid("mass-Taylor mass index overflow"))?;
                if !self.plan.input_layout().masses.contains(&mass){return Err(prep("mass-Taylor mass seed outside authoritative layout"));}
                lifted.values[(compact*geometry+mass)*joint+index]=one.clone();
            }
        }
        if self.checked_regular.is_some(){
            return self.reduce_checked_regular(batch,lifted,&admitted,&rejected);
        }
        let evaluated=self.unit.reduce_batch_with_quadrature_context(&lifted);
        self.stats.unit_execution=self.unit.last_execution();self.diagnostics.resize_with(batch.rows,Vec::new);
        for(compact,&row)in admitted.iter().enumerate(){if let Some(notes)=self.unit.last_diagnostics().get(compact){self.diagnostics[row].extend_from_slice(notes);}}
        if let Some(p)=self.unit.last_profile(){add_profile(&mut self.profile,p);}
        let evaluated=scatter_admitted_joint(evaluated?,batch.rows,&admitted,&rejected,&self.prototype,self.accuracy_bits)?;
        self.project(batch,evaluated)
    }
    fn reduce_checked_regular(&mut self,batch:&KinematicsBatch<T>,lifted:KinematicsBatch<T>,
        admitted:&[usize],rejected:&[Option<String>])->Result<CoupledQuadratureBatch<T>,ReductionError>{
        if admitted.len()!=lifted.rows||rejected.len()!=batch.rows||admitted.iter().any(|&row|row>=batch.rows)
            ||admitted.windows(2).any(|w|w[0]>=w[1]) {
            return Err(prep("checked regular MassTaylor lifted physical lane map differs"));
        }
        self.diagnostics.resize_with(batch.rows,Vec::new);
        let mut statuses=vec![LaneStatus::Unstable("checked regular MassTaylor row was not admitted".into());batch.rows];
        for(row,reason)in rejected.iter().enumerate(){if let Some(reason)=reason{
            statuses[row]=LaneStatus::Unstable(reason.clone());self.diagnostics[row].push(reason.clone());
        }}
        let started=self.profiling.then(Instant::now);
        let guard=self.checked_regular.as_mut().ok_or_else(||prep("checked regular MassTaylor guard is absent"))?;
        let selected=guard.select(&lifted);
        // Harvest every attempted proof, even if the selector returns an error.
        self.stats.certificate_execution=guard.last_certificate_execution();
        if let Some(started)=started{self.profile.arguments+=started.elapsed();}
        let selected=selected?;
        if selected.statuses.len()!=lifted.rows||selected.accepted.iter().any(|&row|row>=lifted.rows)
            ||selected.accepted.windows(2).any(|w|w[0]>=w[1]) {
            return Err(prep("checked regular MassTaylor selection lane inventory differs"));
        }
        for(compact,&row)in admitted.iter().enumerate(){
            if rejected[row].is_some(){return Err(prep("checked regular MassTaylor row is both admitted and rejected"));}
            let accepted=selected.accepted.binary_search(&compact).is_ok();
            if accepted!=(selected.statuses[compact]==LaneStatus::Success){
                return Err(prep("checked regular MassTaylor provisional status differs from accepted lane map"));
            }
            statuses[row]=if accepted {
                // Only the actual dotted projection may assign final Success.
                self.diagnostics[row].push("fresh complete auxiliary-mass x user supplied-G compatibility and regular-primal certificate accepted before unit numerical evaluation".into());
                LaneStatus::Unstable("checked regular MassTaylor numerical evaluation has not completed".into())
            }else{
                self.diagnostics[row].push(format!("checked regular MassTaylor rejected lifted row: {:?}",selected.statuses[compact]));
                selected.statuses[compact].clone()
            };
        }
        let accepted:Vec<_>=selected.accepted.iter().map(|&row|admitted[row]).collect();
        let metadata=ResultMetadata::new(self.plan.settings.convention,self.plan.settings.tensor_weights,
            self.layout.user_shape().map(<[Vec<usize>]>::to_vec));
        if accepted.is_empty(){
            return scatter_checked_dotted(None,batch.rows,&accepted,&statuses,self.plan.basis(),&metadata,
                self.layout.user_components(),&self.prototype,self.accuracy_bits,&mut self.diagnostics);
        }
        let compact_lifted=gather_mass_rows(&lifted,&selected.accepted,self.plan.parameter_count())?;
        let compact_physical=gather_mass_rows(batch,&accepted,self.plan.parameter_count())?;
        // Existing project() operates on its own physical sample inventory.
        // Give it compact original-user geometry, then restore/scatter notes
        // regardless of whether unit evaluation or projection returned an error.
        let mut full_diagnostics=std::mem::take(&mut self.diagnostics);
        self.diagnostics=accepted.iter().map(|&row|full_diagnostics[row].clone()).collect();
        let evaluated=self.unit.reduce_batch_with_quadrature_context(&compact_lifted);
        self.stats.unit_execution=self.unit.last_execution();
        for(compact,notes)in self.unit.last_diagnostics().iter().enumerate(){
            if let Some(destination)=self.diagnostics.get_mut(compact){destination.extend_from_slice(notes);}
        }
        if let Some(profile)=self.unit.last_profile(){add_profile(&mut self.profile,profile);}
        let projected=match evaluated{Ok(evaluated)=>self.project(&compact_physical,evaluated),Err(error)=>Err(error)};
        let compact_diagnostics=std::mem::take(&mut self.diagnostics);
        for(compact,&row)in accepted.iter().enumerate(){if let Some(notes)=compact_diagnostics.get(compact){full_diagnostics[row]=notes.clone();}}
        self.diagnostics=full_diagnostics;
        scatter_checked_dotted(Some(projected?),batch.rows,&accepted,&statuses,self.plan.basis(),&metadata,
            self.layout.user_components(),&self.prototype,self.accuracy_bits,&mut self.diagnostics)
    }
    fn project(&mut self,batch:&KinematicsBatch<T>,joint:CoupledQuadratureBatch<T>)->Result<CoupledQuadratureBatch<T>,ReductionError>{
        let source=joint.result();let rows=batch.rows;let user=self.layout.user_components();let internal=self.layout.joint_components();
        let outputs=mul(self.plan.basis().len(),4)?;let source_width=mul(outputs,internal)?;let width=mul(outputs,user)?;
        let metadata=ResultMetadata::new(self.plan.settings.convention,self.plan.settings.tensor_weights,self.layout.user_shape().map(<[Vec<usize>]>::to_vec));
        let unit_metadata=ResultMetadata::new(self.plan.settings.convention,self.plan.settings.tensor_weights,Some(self.layout.joint_shape().to_vec()));
        if source.rows()!=rows||source.basis!=self.plan.basis()||source.metadata()!=&unit_metadata||source.components!=internal
            ||source.values.len()!=mul(rows,source_width)?||source.fallback_used.len()!=rows||source.analytic_used.len()!=rows{
            return Err(prep("mass-Taylor complete joint result interpretation differs"));
        }
        let zero=Complex::new(self.prototype.zero(),self.prototype.zero());
        let mut result=BatchResult{basis:source.basis.clone(),metadata,components:user,values:vec![zero.clone();mul(rows,width)?],
            statuses:source.statuses.clone(),fallback_used:source.fallback_used.clone(),analytic_used:source.analytic_used.clone(),
            primal_envelopes:vec![None;rows],primal_envelope_errors:vec![None;rows]};
        let mut context=QuadratureErrorContext::missing(rows);let mut selected_rows=Vec::new();let mut selected_errors=Vec::new();let mut origins=Vec::new();
        let projection_parameters=add(self.plan.parameter_count(),mul(outputs,2)?)?;
        let mut input=KinematicsBatch{rows:0,components:user,values:Vec::new()};
        for row in 0..rows{
            let geometry_width=mul(self.plan.parameter_count(),user)?;
            if let Some(reason)=auxiliary_mass_boundary(&batch.values[row*geometry_width..(row+1)*geometry_width],
                self.plan.input_layout(),&self.layout) {
                // Scalar-master validate_point already rejects seeded zero-mass
                // boundaries. Repeat that exact admission rule at this owning
                // coefficient boundary, including cases whose active master
                // graph happened to use a constant scalar boundary formula.
                result.statuses[row]=LaneStatus::Unstable(reason.clone());
                self.diagnostics[row].push(reason.clone());
                context.reject(row,reason).map_err(ReductionError::Evaluation)?;continue;
            }
            if source.statuses[row]!=LaneStatus::Success{continue;}
            let selected=match select_coefficients(&joint,row,&self.layout,&self.prototype,self.accuracy_bits){
                Ok(v)=>v,Err(reason)=>{result.statuses[row]=LaneStatus::Unstable(reason.clone());self.diagnostics[row].push(reason.clone());context.reject(row,reason).map_err(ReductionError::Evaluation)?;continue;}
            };
            let next=mul(add(input.rows,1)?,mul(projection_parameters,user)?)?;
            if next>self.plan.budget.max_batch_values{return Err(prep("mass-Taylor projection input resource budget exceeded"));}
            let geometry_width=mul(self.plan.parameter_count(),user)?;
            input.values.extend_from_slice(&batch.values[row*geometry_width..(row+1)*geometry_width]);
            input.values.extend(selected.values);input.values.extend(selected.amplitudes);
            selected_rows.push(row);selected_errors.push(selected.errors);origins.push(selected.origin);input.rows+=1;
        }
        if input.rows==0{return CoupledQuadratureBatch::new(result,context);}
        let projected=self.projection.reduce_batch(&input);self.stats.projection_execution=self.projection.last_execution();
        if let Some(p)=self.projection.last_profile(){add_profile(&mut self.profile,p);}
        for(compact,&row)in selected_rows.iter().enumerate(){if let Some(notes)=self.projection.last_diagnostics().get(compact){self.diagnostics[row].extend_from_slice(notes);}}
        let projected=projected?;
        if projected.rows()!=input.rows||projected.basis!=result.basis||projected.metadata()!=result.metadata()||projected.components!=user
            ||projected.values.len()!=mul(input.rows,width)?{return Err(prep("mass-Taylor identity parent result layout differs"));}
        for(compact,&row)in selected_rows.iter().enumerate(){
            result.statuses[row]=projected.statuses[compact].clone();let Ok(values)=projected.lane(compact)else{continue;};
            result.values[row*width..(row+1)*width].clone_from_slice(values);
            result.fallback_used[row]|=projected.fallback_used[compact];result.analytic_used[row]|=projected.analytic_used[compact];
            let errors=&mut selected_errors[compact];
            // The existing final-UV power-counting rule can make target slots
            // exact zero. Only those proven zero expressions erase delta jets.
            for(output,&is_zero)in self.exact_zero_outputs.iter().enumerate(){if is_zero{errors[output*user..(output+1)*user].fill(zero.clone());}}
            let input_width=mul(projection_parameters,user)?;
            let checked=self.projection.inspect_imported_quadrature_output(&self.checks,&projected,compact,
                &input.values[compact*input_width..(compact+1)*input_width],errors);
            let reason=match checked{Ok(None)=>None,Ok(Some(s))|Err(s)=>Some(s)};
            if let Some(reason)=reason{result.statuses[row]=LaneStatus::Unstable(reason.clone());self.diagnostics[row].push(reason.clone());context.reject(row,reason).map_err(ReductionError::Evaluation)?;continue;}
            // Inherit ONLY the fresh dotted identity parent's envelope. Its
            // amplitude inputs came from validated selected component amplitudes.
            result.inherit_primal_envelope(row,&projected,compact,PrimalEnvelopeRoute::Ordinary);
            let installed=result.validated_primal_envelope(row,self.prototype.get_precision(),self.accuracy_bits)
                .map(|_|()).and_then(|_|if user>1 {result.validated_component_amplitudes(row,self.prototype.get_precision(),self.accuracy_bits).map(|_|())}else{Ok(())})
                .and_then(|_|context.install_dense(&result,row,self.prototype.get_precision(),self.accuracy_bits,errors,origins[compact]));
            if let Err(reason)=installed{result.statuses[row]=LaneStatus::Unstable(reason.clone());result.primal_envelopes[row]=None;result.primal_envelope_errors[row]=Some(reason.clone());self.diagnostics[row].push(reason.clone());context.reject(row,reason).map_err(ReductionError::Evaluation)?;}
        }
        CoupledQuadratureBatch::new(result,context)
    }
    pub fn reduce_typed_dual_batch<D:DualNumber<Complex<T>>>(&mut self,values:&[D],rows:usize)->Result<BatchResult<T>,ReductionError>{
        self.reset(rows);let shape=self.layout.user_shape().ok_or_else(||invalid("MassTaylor::new_dual required for typed input"))?;
        let expected:Vec<_>=shape.iter().map(Vec::as_slice).collect();
        if values.len()!=mul(rows,self.plan.parameter_count())?{return Err(invalid("mass-Taylor typed input length differs"));}
        let packed_count=mul(values.len(),self.layout.user_components())?;
        if packed_count>self.plan.budget.max_batch_values{return Err(prep("mass-Taylor typed input resource budget exceeded"));}
        let mut packed=Vec::with_capacity(packed_count);
        for value in values{if value.get_shape()!=expected||value.values().len()!=self.layout.user_components(){return Err(invalid("mass-Taylor typed user shape/order differs"));}packed.extend_from_slice(value.values());}
        self.reduce_batch(&KinematicsBatch{rows,components:self.layout.user_components(),values:packed})
    }
}
/// Gather complete physical rows without changing any parameter or Taylor slot.
fn gather_mass_rows<T:NativeFloat>(input:&KinematicsBatch<T>,rows:&[usize],parameters:usize)
    ->Result<KinematicsBatch<T>,ReductionError>{
    let width=mul(parameters,input.components)?;
    if input.components==0||input.values.len()!=mul(input.rows,width)?||rows.iter().any(|&row|row>=input.rows)
        ||rows.windows(2).any(|w|w[0]>=w[1]){
        return Err(prep("checked regular MassTaylor compact physical layout differs"));
    }
    let mut result=KinematicsBatch{rows:rows.len(),components:input.components,values:Vec::with_capacity(mul(rows.len(),width)?)};
    for &row in rows{result.values.extend_from_slice(&input.values[row*width..(row+1)*width]);}
    Ok(result)
}

/// Scatter FINAL dotted outputs. A unit envelope is never reinterpreted here:
/// the source is the existing projected identity actor in the user algebra.
fn scatter_checked_dotted<T:NativeFloat>(source:Option<CoupledQuadratureBatch<T>>,rows:usize,accepted:&[usize],
    statuses:&[LaneStatus],basis:&[Monomial],metadata:&ResultMetadata,components:usize,prototype:&T,accuracy:u32,
    diagnostics:&mut [Vec<String>])->Result<CoupledQuadratureBatch<T>,ReductionError>{
    let width=mul(mul(basis.len(),4)?,components)?;
    if components==0||metadata.dual_shape().map_or(1,|shape|shape.len())!=components
        ||statuses.len()!=rows||diagnostics.len()!=rows||statuses.iter().any(|status|*status==LaneStatus::Success)
        ||accepted.iter().any(|&row|row>=rows)||accepted.windows(2).any(|w|w[0]>=w[1]){
        return Err(prep("checked regular MassTaylor final scatter lane inventory differs"));
    }
    let zero=Complex::new(prototype.zero(),prototype.zero());
    let mut result=BatchResult{basis:basis.to_vec(),metadata:metadata.clone(),components,values:vec![zero;mul(rows,width)?],
        statuses:statuses.to_vec(),fallback_used:vec![false;rows],analytic_used:vec![false;rows],
        primal_envelopes:vec![None;rows],primal_envelope_errors:vec![None;rows]};
    let mut context=QuadratureErrorContext::missing(rows);
    for row in 0..rows{if accepted.binary_search(&row).is_err(){
        let reason=format!("checked regular MassTaylor row rejected: {:?}",result.statuses[row]);
        result.primal_envelope_errors[row]=Some(reason.clone());
        context.reject(row,reason).map_err(ReductionError::Evaluation)?;
    }}
    let Some(source)=source else{
        if !accepted.is_empty(){return Err(prep("checked regular MassTaylor accepted rows have no evaluated dotted source"));}
        return CoupledQuadratureBatch::new(result,context);
    };
    let(source_result,source_context)=source.into_parts();
    if source_result.rows()!=accepted.len()||source_result.basis!=basis||source_result.metadata()!=metadata
        ||source_result.components!=components||source_result.values.len()!=mul(accepted.len(),width)?
        ||source_result.fallback_used.len()!=accepted.len()||source_result.analytic_used.len()!=accepted.len()
        ||source_result.primal_envelopes.len()!=accepted.len()||source_result.primal_envelope_errors.len()!=accepted.len(){
        return Err(prep("checked regular MassTaylor final dotted source interpretation differs"));
    }
    for(compact,&row)in accepted.iter().enumerate(){
        result.statuses[row]=source_result.statuses[compact].clone();
        result.fallback_used[row]=source_result.fallback_used[compact];result.analytic_used[row]=source_result.analytic_used[compact];
        result.primal_envelope_errors[row]=source_result.primal_envelope_errors[compact].clone();
        let Ok(values)=source_result.lane(compact)else{
            context.reject(row,format!("checked regular MassTaylor evaluated source failed: {:?}",result.statuses[row]))
                .map_err(ReductionError::Evaluation)?;continue;
        };
        result.values[row*width..(row+1)*width].clone_from_slice(values);
        let source_valid=source_result.validated_primal_envelope(compact,prototype.get_precision(),accuracy)
            .map(|_|()).and_then(|_|if components>1{
                source_result.validated_component_amplitudes(compact,prototype.get_precision(),accuracy).map(|_|())
            }else{Ok(())});
        let installed=source_valid.and_then(|_|{
            result.inherit_primal_envelope(row,&source_result,compact,PrimalEnvelopeRoute::Ordinary);
            result.validated_primal_envelope(row,prototype.get_precision(),accuracy).map(|_|())
                .and_then(|_|if components>1{result.validated_component_amplitudes(row,prototype.get_precision(),accuracy).map(|_|())}else{Ok(())})
                .and_then(|_|context.inherit_lane(&result,row,&source_result,&source_context,compact,prototype.get_precision(),accuracy))
        });
        if let Err(reason)=installed{
            result.statuses[row]=LaneStatus::Unstable(reason.clone());result.primal_envelopes[row]=None;
            result.primal_envelope_errors[row]=Some(reason.clone());diagnostics[row].push(reason.clone());
            context.reject(row,reason).map_err(ReductionError::Evaluation)?;
        }
    }
    CoupledQuadratureBatch::new(result,context)
}

/// Scatter a same-algebra joint child batch back to original sample rows.
/// This is ordinary lane transport, not a projection of a unit envelope to dots.
fn scatter_admitted_joint<T:NativeFloat>(source:CoupledQuadratureBatch<T>,rows:usize,admitted:&[usize],
    rejected:&[Option<String>],prototype:&T,accuracy:u32)->Result<CoupledQuadratureBatch<T>,ReductionError>{
    let(source_result,source_context)=source.into_parts();let width=mul(mul(source_result.basis.len(),4)?,source_result.components)?;
    if source_result.rows()!=admitted.len()||rejected.len()!=rows||source_result.values.len()!=mul(admitted.len(),width)?
        ||source_result.fallback_used.len()!=admitted.len()||source_result.analytic_used.len()!=admitted.len()
        ||admitted.iter().any(|&row|row>=rows)||admitted.windows(2).any(|w|w[0]>=w[1]) {
        return Err(prep("mass-Taylor admitted joint lane map differs"));
    }
    let zero=Complex::new(prototype.zero(),prototype.zero());
    let mut result=BatchResult{basis:source_result.basis.clone(),metadata:source_result.metadata().clone(),components:source_result.components,
        values:vec![zero;mul(rows,width)?],statuses:vec![LaneStatus::Unstable("mass-Taylor joint row not admitted".into());rows],
        fallback_used:vec![false;rows],analytic_used:vec![false;rows],primal_envelopes:vec![None;rows],primal_envelope_errors:vec![None;rows]};
    let mut context=QuadratureErrorContext::missing(rows);
    for(row,reason)in rejected.iter().enumerate(){if let Some(reason)=reason{
        result.statuses[row]=LaneStatus::Unstable(reason.clone());result.primal_envelope_errors[row]=Some(reason.clone());
        context.reject(row,reason.clone()).map_err(ReductionError::Evaluation)?;
    }}
    for(compact,&row)in admitted.iter().enumerate(){
        if rejected[row].is_some(){return Err(prep("mass-Taylor lane is both admitted and rejected"));}
        result.statuses[row]=source_result.statuses[compact].clone();
        result.fallback_used[row]=source_result.fallback_used[compact];result.analytic_used[row]=source_result.analytic_used[compact];
        result.values[row*width..(row+1)*width].clone_from_slice(&source_result.values[compact*width..(compact+1)*width]);
        if result.statuses[row]==LaneStatus::Success {
            result.inherit_primal_envelope(row,&source_result,compact,PrimalEnvelopeRoute::Ordinary);
            let _=context.inherit_lane(&result,row,&source_result,&source_context,compact,prototype.get_precision(),accuracy);
        }
    }
    CoupledQuadratureBatch::new(result,context)
}

/// Existing master-domain rule (iregi_masters::MasterEvaluator::validate_point):
/// a primal zero-mass scalar branch does not define transverse mass derivatives.
/// Each auxiliary seed is +1 by construction, so a dotted zero mass is transverse.
/// Use represented exact zero only; no f64 conversion, threshold, or snapping.
fn auxiliary_mass_boundary<T:NativeFloat>(physical:&[Complex<T>],input:&InputLayout,layout:&DottedLayout)->Option<String>{
    let user=layout.user_components();
    for &(line,_)in layout.mass_seeds(){
        let Some(parameter)=input.masses.start.checked_add(line)else{return Some("MassTaylor auxiliary mass index overflow".into());};
        if !input.masses.contains(&parameter){return Some("MassTaylor auxiliary mass is outside the authoritative layout".into());}
        let Some(index)=parameter.checked_mul(user)else{return Some("MassTaylor auxiliary component index overflow".into());};
        let Some(mass)=physical.get(index)else{return Some("MassTaylor auxiliary mass primal is missing".into());};
        if mass.re.is_zero()&&mass.im.is_zero(){
            return Some(format!("MassTaylor auxiliary mass direction on propagator {line} crosses an exact zero-mass branch: {}",
                iregi_masters::LaneError::UnsupportedDerivativeDomain));
        }
    }
    None
}

#[derive(Clone,Debug)]
struct SelectedCoefficients<T:NativeFloat> {
    values:Vec<Complex<T>>,amplitudes:Vec<Complex<T>>,errors:Vec<Complex<T>>,origin:QuadratureErrorOrigin,
}
/// Pure index projection after BOTH owning source contexts validate all source
/// outputs/components and their precision/signed-zero snapshots. This helper
/// performs no Taylor arithmetic and copies no unit PrimalEnvelope.
fn select_coefficients<T:NativeFloat>(joint:&CoupledQuadratureBatch<T>,row:usize,layout:&DottedLayout,
    prototype:&T,accuracy:u32)->Result<SelectedCoefficients<T>,String>{
    let source=joint.result();let internal=layout.joint_components();let user=layout.user_components();
    if source.components!=internal||source.metadata().dual_shape()!=Some(layout.joint_shape()) {
        return Err("mass-Taylor source joint shape/order differs".into());
    }
    let amplitudes=source.validated_component_amplitudes(row,prototype.get_precision(),accuracy)?;
    let errors=joint.errors(row,prototype.get_precision(),accuracy)?;
    let origin=joint.origin(row,prototype.get_precision(),accuracy)?;
    if user>1&&origin==QuadratureErrorOrigin::PropagatedEuclidean {
        return Err("mass-Taylor user higher-component combined quadrature/roundoff policy is not qualified".into());
    }
    let source_values=source.lane(row).map_err(|status|format!("mass-Taylor requires successful source: {status:?}"))?;
    let outputs=source.basis.len().checked_mul(4).ok_or("mass-Taylor projection dimensions overflow")?;
    let width=outputs.checked_mul(user).ok_or("mass-Taylor projection dimensions overflow")?;
    let mut selected=SelectedCoefficients{values:Vec::with_capacity(width),amplitudes:Vec::with_capacity(width),errors:Vec::with_capacity(width),origin};
    for output in 0..outputs {for &index in layout.selected_indices(){
        selected.values.push(source_values[output*internal+index].clone());
        selected.amplitudes.push(amplitudes[output*internal+index].clone());
        selected.errors.push(errors[output*internal+index].clone());
    }}
    Ok(selected)
}
fn add_profile(target:&mut ExecutionProfile,source:ExecutionProfile){target.arguments+=source.arguments;target.scalar_masters+=source.scalar_masters;target.reduction+=source.reduction;}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path="mass_taylor/checked_actor_tests.rs"]
mod checked_actor_tests;
