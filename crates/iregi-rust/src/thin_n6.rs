//! Private implementation behind the restricted six-point API: one packed N5
//! rank-lowered child batch and a guarded Symbolica parent. Internal generic
//! rank and Dual support exceed the public, numerically qualified domain.
use iregi_masters::NativeFloat;
use symbolica::{domains::{dual::DualNumberStructure, float::Complex}, evaluate::EvaluationDomain};
use crate::{Algorithm, BatchResult, CertificateExecutionStats, Convention, ExecutionStats, InputLayout, KinematicsBatch,
    LaneStatus, ReductionError, ReductionPlan, ReductionSettings, ResultMetadata, TensorWeights,
    Workspace, basis::{Monomial,tensor_basis}, execution::{WorkspacePolicy, validate_input_row}};

#[derive(Clone, Debug)]
pub(crate) struct ThinSixReductionPlan { parent:ReductionPlan, child:ReductionPlan }
impl ThinSixReductionPlan {
    pub fn new(settings:ReductionSettings)->Result<Self,ReductionError> {
        let child_rank=settings.rank.saturating_sub(1);
        let parent=ReductionPlan::new_thin_six(settings,0)?;
        let mut settings=ReductionSettings::new(5,child_rank); settings.algorithm=Algorithm::PassarinoVeltman;
        let child=ReductionPlan::new(settings)?;
        Ok(Self{parent,child})
    }
    pub fn basis(&self)->&[Monomial] { self.parent.basis() }
    pub fn settings(&self)->&ReductionSettings { self.parent.settings() }
    pub fn alias_count(&self)->usize { self.parent.aliases().len() }
    pub(crate) fn input_layout(&self)->&InputLayout { self.parent.input_layout() }
    pub(crate) fn contraction_plan(&self,monomials:&[Monomial])->Result<crate::ContractionPlan,ReductionError> { crate::ContractionPlan::new(&self.parent,monomials) }
}
#[derive(Clone,Debug,Default)]
pub struct SixPointExecutionStats {
    pub parent_input_rows:usize, pub child_batch_calls:usize, pub packed_child_rows:usize,
    pub parent_batch_calls:usize, pub packed_parent_rows:usize,
    pub child_execution:ExecutionStats, pub parent_execution:ExecutionStats,
    pub child_certificate_execution:CertificateExecutionStats,
    pub parent_certificate_execution:CertificateExecutionStats,
}
#[derive(Clone)]
pub(crate) struct ThinSixWorkspace<T:NativeFloat> {
    prototype:T, accuracy_bits:u32, components:usize, metadata:ResultMetadata, basis:Vec<Monomial>,
    child:Workspace<T>, parent:Workspace<T>, child_basis:Vec<Monomial>, stats:SixPointExecutionStats, diagnostics:Vec<Vec<String>>,
}
impl<T:NativeFloat> ThinSixWorkspace<T> where Complex<T>:EvaluationDomain {
    pub fn new(plan:&ThinSixReductionPlan,prototype:T)->Result<Self,ReductionError> {
        let child=Workspace::new_with_policy(&plan.child,prototype.clone(),WorkspacePolicy::SIX_POINT_CHILD)?;
        let parent=Workspace::new_with_policy(&plan.parent,prototype.clone(),WorkspacePolicy::RETAIN_ENVELOPES)?;
        Ok(Self{accuracy_bits:prototype.get_precision(),prototype,components:1,
            metadata:ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,None),
            basis:plan.basis().to_vec(),child_basis:plan.child.basis().to_vec(),child,parent,stats:Default::default(),diagnostics:Vec::new()})
    }
    pub fn new_dual<D:DualNumberStructure>(plan:&ThinSixReductionPlan,prototype:T,dual:D)->Result<Self,ReductionError> {
        iregi_masters::validate_dual_structure(&dual).map_err(|error|ReductionError::Preparation(error.to_string()))?;
        let shape:Vec<Vec<usize>>=dual.get_shape().into_iter().map(<[usize]>::to_vec).collect();
        let order=shape.iter().try_fold(0u16,|max,row|row.iter().try_fold(0usize,|sum,&n|sum.checked_add(n))
            .and_then(|n|u16::try_from(n).ok()).map(|n|max.max(n))
            .ok_or_else(||ReductionError::Preparation("Dual order exceeds u16".into())))?;
        let parent_plan=ReductionPlan::new_thin_six(plan.settings().clone(),order)?;
        let child=Workspace::new_dual_with_policy(&plan.child,prototype.clone(),BorrowedDual(&dual),WorkspacePolicy::SIX_POINT_CHILD)?;
        let parent=Workspace::new_dual_with_policy(&parent_plan,prototype.clone(),BorrowedDual(&dual),WorkspacePolicy::RETAIN_ENVELOPES)?;
        Ok(Self{accuracy_bits:prototype.get_precision(),prototype,components:shape.len(),
            metadata:ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,Some(shape)),
            basis:plan.basis().to_vec(),child_basis:plan.child.basis().to_vec(),child,parent,stats:Default::default(),diagnostics:Vec::new()})
    }
    pub fn set_accuracy_precision(&mut self,bits:u32)->Result<(),ReductionError> {
        if bits==0 || bits>self.prototype.get_precision() { return Err(ReductionError::InvalidInput("invalid thin N6 accuracy precision".into())); }
        self.child.set_accuracy_precision(bits)?; self.parent.set_accuracy_precision(bits)?;
        self.accuracy_bits=bits; Ok(())
    }
    pub fn last_execution(&self)->&SixPointExecutionStats { &self.stats }
    pub fn last_diagnostics(&self)->&[Vec<String>] { &self.diagnostics }
    pub(crate) fn component_count(&self)->usize { self.components }
    pub(crate) fn binary_precision(&self)->u32 { self.prototype.get_precision() }
    pub fn reduce_batch(&mut self,batch:&KinematicsBatch<T>)->Result<BatchResult<T>,ReductionError> {
        self.stats=SixPointExecutionStats{parent_input_rows:batch.rows,..Default::default()};
        self.diagnostics.clear();
        let input_width=31usize.checked_mul(self.components).ok_or_else(||ReductionError::InvalidInput("thin N6 dimensions overflow".into()))?;
        if self.components==0 || batch.components!=self.components || batch.rows.checked_mul(input_width)!=Some(batch.values.len()) {
            return Err(ReductionError::InvalidInput("thin N6 input dimensions do not match plan".into()));
        }
        let width=self.basis.len().checked_mul(4).and_then(|n|n.checked_mul(self.components))
            .ok_or_else(||ReductionError::InvalidInput("thin N6 result dimensions overflow".into()))?;
        let output_len=batch.rows.checked_mul(width).ok_or_else(||ReductionError::InvalidInput("thin N6 output dimensions overflow".into()))?;
        // Validate every checked dimension before allocating row-sized scratch.
        self.diagnostics.resize_with(batch.rows,Vec::new);
        let zero=Complex::new(self.prototype.zero(),self.prototype.zero());
        let mut result=BatchResult{basis:self.basis.clone(),components:self.components,metadata:self.metadata.clone(),
            values:vec![zero.clone();output_len],
            statuses:vec![LaneStatus::Success;batch.rows],fallback_used:vec![false;batch.rows],analytic_used:vec![false;batch.rows],
            primal_envelopes:vec![None;batch.rows],primal_envelope_errors:vec![None;batch.rows]};
        let layout=InputLayout{shifts:0..24,scalar_products:None,masses:24..30,scale:30};
        let mut active=Vec::new();
        for (lane,row) in batch.values.chunks_exact(input_width).enumerate() {
            if let Some(status)=validate_input_row(row,&layout,self.components) {result.statuses[lane]=status;} else {active.push(lane);}
        }
        if active.is_empty() {return Ok(result);}
        let child_rows=active.len().checked_mul(6).ok_or_else(||ReductionError::InvalidInput("thin N6 child batch dimensions overflow".into()))?;
        let mut child_input=KinematicsBatch{rows:child_rows,components:self.components,values:Vec::new()};
        // Subset-major lane packing. Absolute shifts preserve the integration
        // variable. Every Taylor component is copied, even for primal-zero weights.
        for omitted in 0..6 {for &lane in &active {
            let row=&batch.values[lane*input_width..(lane+1)*input_width];
            for i in (0..6).filter(|&i|i!=omitted) {child_input.values.extend_from_slice(&row[4*i*self.components..4*(i+1)*self.components]);}
            for i in (0..6).filter(|&i|i!=omitted) {child_input.values.extend_from_slice(&row[(24+i)*self.components..(25+i)*self.components]);}
            child_input.values.extend_from_slice(&row[30*self.components..31*self.components]);
        }}
        self.stats.child_batch_calls=1; self.stats.packed_child_rows=child_input.rows;
        // The child entry starts fresh guard contexts; no proof state is
        // transferred across the parent and packed-child parameter layouts.
        let children=self.child.reduce_batch(&child_input);
        self.stats.child_execution=self.child.last_execution();
        self.stats.child_certificate_execution=self.child.last_planar_certificate_execution();
        for (compact,&lane) in active.iter().enumerate() { for subset in 0..6 {
            if let Some(notes)=self.child.last_diagnostics().get(subset*active.len()+compact) {
                self.diagnostics[lane].extend(notes.iter().map(|note|format!("N5 subset {subset}: {note}")));
            }
        }}
        let children=children?;
        validate_result_layout(&children, &self.child_basis, self.components, &self.metadata, child_input.rows)?;
        let mut parent_input=KinematicsBatch{rows:0,components:self.components,values:Vec::new()};
        let mut parent_lanes=Vec::new();
        for (compact,&lane) in active.iter().enumerate() {
            let mut failure=None;
            for subset in 0..6 {
                let child_row=subset*active.len()+compact;
                result.fallback_used[lane]|=children.fallback_used[child_row];
                result.analytic_used[lane]|=children.analytic_used[child_row];
                if let Err(status)=children.lane(child_row) {
                    if failure.is_none() {failure=Some(format!("N5 subset {subset}: {status:?}"));}
                    continue;
                }
                if let Err(error)=children.validated_primal_envelope(child_row,self.prototype.get_precision(),self.accuracy_bits) {
                    if failure.is_none() {failure=Some(format!("N5 subset {subset}: unavailable actual-route primal context: {error}"));}
                }
            }
            if let Some(error)=failure {self.diagnostics[lane].push(error.clone());result.statuses[lane]=LaneStatus::Unstable(error);continue;}
            parent_lanes.push(lane);parent_input.rows+=1;
            parent_input.values.extend_from_slice(&batch.values[lane*input_width..(lane+1)*input_width]);
            for subset in 0..6 {parent_input.values.extend_from_slice(children.lane(subset*active.len()+compact).expect("verified child success"));}
            for subset in 0..6 {
                let envelope=children.validated_primal_envelope(subset*active.len()+compact,self.prototype.get_precision(),self.accuracy_bits).expect("verified child envelope");
                for operand in &envelope.operands {
                    parent_input.values.push(operand.amplitude.clone());
                    parent_input.values.extend(std::iter::repeat_n(zero.clone(),self.components-1));
                }
            }
        }
        if parent_input.rows==0 {return Ok(result);}
        self.stats.parent_batch_calls=1; self.stats.packed_parent_rows=parent_input.rows;
        let parent=self.parent.reduce_batch(&parent_input);
        self.stats.parent_execution=self.parent.last_execution();
        self.stats.parent_certificate_execution=self.parent.last_planar_certificate_execution();
        for (compact,&lane) in parent_lanes.iter().enumerate() {
            if let Some(notes)=self.parent.last_diagnostics().get(compact) {self.diagnostics[lane].extend(notes.iter().cloned());}
        }
        let parent=parent?;
        validate_result_layout(&parent, &self.basis, self.components, &self.metadata, parent_input.rows)?;
        for (compact,&lane) in parent_lanes.iter().enumerate() {
            result.statuses[lane]=parent.statuses[compact].clone();
            if let Ok(values)=parent.lane(compact) {
                result.values[lane*width..(lane+1)*width].clone_from_slice(values);
                result.primal_envelopes[lane]=parent.primal_envelopes[compact].clone();
                result.primal_envelope_errors[lane]=parent.primal_envelope_errors[compact].clone();
            }
        }
        Ok(result)
    }
}
/// Validate complete dynamic layouts before any lane indexing or scatter.
/// Per-lane envelope content and precision remain separately checked below.
fn validate_result_layout<T:NativeFloat>(result:&BatchResult<T>, basis:&[Monomial], components:usize, metadata:&ResultMetadata, rows:usize)->Result<(),ReductionError> {
    let values=rows.checked_mul(basis.len()).and_then(|n|n.checked_mul(4)).and_then(|n|n.checked_mul(components))
        .ok_or_else(||ReductionError::Evaluation("thin N6 result dimensions overflow".into()))?;
    if result.basis!=basis || result.components!=components || result.metadata()!=metadata || result.rows()!=rows
        || result.values.len()!=values || result.fallback_used.len()!=rows || result.analytic_used.len()!=rows
        || result.primal_envelopes.len()!=rows || result.primal_envelope_errors.len()!=rows {
        return Err(ReductionError::Evaluation("thin N6 complete result layout mismatch".into()));
    }
    Ok(())
}

struct BorrowedDual<'a,D>(&'a D);
impl<D:DualNumberStructure> DualNumberStructure for BorrowedDual<'_,D> {
    fn get_len(&self)->usize{self.0.get_len()}
    fn get_shape(&self)->Vec<&[usize]>{self.0.get_shape()}
    fn get_multiplication_table(&self)->&[(usize,usize,usize)]{self.0.get_multiplication_table()}
    fn get_max_depth(&self)->usize{self.0.get_max_depth()}
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::{atom::{Atom,AtomCore}, domains::{dual::HyperDual,rational::{Rational,Q}}, evaluate::{Dualizer,ExpressionEvaluator}};
    use crate::plan::assemble_thin_six_moments;
    #[test]
    fn malformed_dimensions_reset_before_any_row_sized_allocation() {
        // This private synthetic wrapper supplies cheap ordinary N1 scratch;
        // no child or parent is evaluated, and it makes no N6 numerical claim.
        let plan=ReductionPlan::new(ReductionSettings::new(1,0)).unwrap();
        let scratch=Workspace::new(&plan,0f64).unwrap();
        let mut workspace=ThinSixWorkspace {
            prototype:0f64,accuracy_bits:53,components:1,
            metadata:ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,None),
            basis:vec![Monomial([0;4])],child_basis:vec![Monomial([0;4])],
            child:scratch.clone(),parent:scratch,stats:Default::default(),diagnostics:Vec::new(),
        };
        for (rows,input_components,stored_components) in [
            (1,0,1),(1,usize::MAX,1),(usize::MAX,1,1),(usize::MAX/31,1,1),
            (0,usize::MAX,usize::MAX),(0,0,0),
        ] {
            workspace.components=stored_components;
            workspace.stats=SixPointExecutionStats { child_batch_calls:7,parent_batch_calls:3,..Default::default() };
            workspace.diagnostics=vec![vec!["previous call".into()]];
            let capacity=workspace.diagnostics.capacity();
            let result=workspace.reduce_batch(&KinematicsBatch {rows,components:input_components,values:Vec::new()});
            assert!(matches!(result,Err(ReductionError::InvalidInput(_))));
            assert!(workspace.diagnostics.is_empty());
            assert_eq!(workspace.diagnostics.capacity(),capacity);
            assert_eq!(workspace.stats.parent_input_rows,rows);
            assert_eq!(workspace.stats.child_batch_calls,0);
            assert_eq!(workspace.stats.parent_batch_calls,0);
            assert_eq!(workspace.stats.child_execution,ExecutionStats::default());
            assert_eq!(workspace.stats.parent_execution,ExecutionStats::default());
            assert_eq!(workspace.stats.child_certificate_execution,CertificateExecutionStats::default());
            assert_eq!(workspace.stats.parent_certificate_execution,CertificateExecutionStats::default());
        }
        workspace.components=1;
        let empty=workspace.reduce_batch(&KinematicsBatch {rows:0,components:1,values:Vec::new()}).unwrap();
        assert_eq!(empty.rows(),0);
        assert_eq!(workspace.stats.parent_input_rows,0);
        assert_eq!(workspace.stats.child_batch_calls,0);
        assert!(workspace.diagnostics.is_empty());
    }

    #[test]
    fn actual_assembly_all_slots_and_mixed_dual_sentinels() {
        let x=Atom::var(symbolica::symbol!("thin_test::x"));
        let y=Atom::var(symbolica::symbol!("thin_test::y"));
        let weights:[Vec<Atom>;5]=std::array::from_fn(|column|(0..6).map(|subset|
            &x*Atom::num((column*6+subset+1) as i64)+&y*Atom::num((column+2*subset+1) as i64)).collect());
        let values:Vec<_>=(0..120).map(|index|Atom::num((index+1) as i64)+&x*Atom::num((index+3) as i64)+&y).collect();
        let actual=assemble_thin_six_moments(&tensor_basis(2),&tensor_basis(1),&weights,&values);
        // Independent fixed IREGI rank-two ordering, with distinct Laurent and
        // subset sentinels. The scalar entry uses the constant identity column.
        let map=[(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(1,2),(1,3),(1,4),(2,2),(2,3),(2,4),(3,3),(3,4),(4,4)];
        let expected:Vec<Atom>=map.iter().flat_map(|&(column,child)| (0..4).map(move |pole|(column,child,pole)))
            .map(|(column,child,pole)|Atom::add_many(&(0..6).map(|subset|&weights[column][subset]*&values[20*subset+4*child+pole]).collect::<Vec<_>>())).collect();
        for (actual,expected) in actual.iter().flatten().zip(&expected) {assert!((actual-expected).expand().is_zero());}
        let expressions:Vec<_>=actual.iter().flatten().cloned().chain(expected).collect();
        let exact:ExpressionEvaluator<Complex<Rational>>=Atom::evaluator_multiple(&expressions,&[x,y]).direct_translation(true).cores(1).horner_iterations(0).build().unwrap();
        let shape=vec![vec![0,0],vec![1,0],vec![0,1],vec![2,0],vec![1,1],vec![0,2]];
        let mut eval=exact.vectorize(&Dualizer::new(HyperDual::<Complex<Rational>>::new(shape),vec![])).unwrap().map_to_ring(&Q).unwrap();
        let mut inputs=vec![Rational::from(0);12];inputs[1]=Rational::from(1);inputs[8]=Rational::from(1);
        let mut outputs=vec![Rational::from(0);120*6];eval.evaluate_in_ring(&inputs,&mut outputs,&Q);
        assert_eq!(&outputs[..360],&outputs[360..]);
        for slot in 0..60 {assert_eq!(outputs[slot*6],Rational::from(0));assert_ne!(outputs[slot*6+4],Rational::from(0));}
    }
    #[test]
    fn five_rhs_integrand_identities_keep_dimensionally_regulated_denominators() {
        let points=[[0,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2],[1,0,0,0],[0,2,2,0]];
        let q:[Atom;4]=std::array::from_fn(|mu|Atom::var(symbolica::symbol!(format!("thin_test::q{mu}"))));
        let transverse=Atom::var(symbolica::symbol!("thin_test::q_perp_squared"));
        fn dot(p:&[Atom;4])->Atom{&p[0]*&p[0]-&p[1]*&p[1]-&p[2]*&p[2]-&p[3]*&p[3]}
        for variant in 0..5 {
            let mut p:Vec<[Atom;4]>=points.iter().map(|p|p.map(|n|Atom::num(n)/Atom::num(2))).collect();
            let mut masses:Vec<_>=[1,2,3,4,2,3].into_iter().map(Atom::num).collect();
            match variant {
                1=>for row in &mut p {for (mu,x) in row.iter_mut().enumerate(){*x=&*x+Atom::num((mu+1) as i64)/Atom::num(3);}},
                2=>for row in &mut p {for x in row {*x=-&*x;}},
                3=>{p.reverse();masses.reverse();},
                4=>for (i,m) in masses.iter_mut().enumerate(){*m=&*m-Atom::i()*Atom::num((i+1) as i64)/Atom::num(16);},
                _=>(),
            }
            let mut compiler=crate::pv::PvCompiler::new(p.clone(),masses.clone()).unwrap();
            let weights=compiler.thin_six_weights().unwrap();
            let denominators:Vec<_>=p.iter().zip(&masses).map(|(p,m)|dot(&std::array::from_fn(|mu|&q[mu]+&p[mu]))+&transverse-m).collect();
            for column in 0..5 {
                let rhs=Atom::add_many(&(0..6).map(|i|&weights[column][i]*&denominators[i]).collect::<Vec<_>>());
                let expected=if column==0{Atom::num(1)}else{q[column-1].clone()};
                assert!((rhs-expected).expand().together().is_zero(),"variant={variant},column={column}");
            }
        }
    }
}

#[cfg(test)]
#[path="thin_n6/generic_tests.rs"]
mod generic_tests;
