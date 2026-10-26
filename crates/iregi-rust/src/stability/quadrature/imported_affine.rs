//! PRIVATE SOURCE-ONLY PROPOSAL. Intended child module of stability::quadrature.
//! One shared signed affine coefficient kernel, one positive Dual contraction.
//! Output slots are internal work rows; they never become physical sample lanes
//! or Taylor directions. No integral graph is inlined into this program.
use std::collections::{BTreeSet,HashMap};
use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom,AtomCore},
    domains::{dual::DualNumberStructure,float::Complex,rational::Rational},
    evaluate::{BatchExpressionEvaluator,EvaluationDomain,ExpressionEvaluator},
};
use crate::{KinematicsBatch,ReductionError,ResultMetadata,basis::Monomial,
    execution::{compile_exact,map_exact}};
// Proposed sibling execution module, separately owned from BatchResult.
use crate::execution::quadrature_context::CoupledQuadratureBatch;
use super::{affine_projection,depends_on_projection,variable};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) enum ErrorInputNorm {
    RawComplexInfinity,
    ImportedEuclidean,
}
impl ErrorInputNorm {
    fn euclidean_factor(self)->Atom {
        match self {Self::RawComplexInfinity=>Atom::num(2),Self::ImportedEuclidean=>Atom::num(1)}
    }
}

/// Structural recipe for sum_i b_i * child_i[output], independent of which
/// moment/Laurent output is selected. The factory must use this same exact
/// coefficient vector for every output and retain its existing chart witness.
pub(crate) struct PreparedImportedAffineQuadrature {
    coefficients:ExpressionEvaluator<Complex<Rational>>,
    positive:ExpressionEvaluator<Complex<Rational>>,
    coefficient_parameters:usize,
    sources:usize,
    outputs:usize,
    shape:Option<Vec<Vec<usize>>>,
}

impl PreparedImportedAffineQuadrature {
    pub(crate) fn prepare<D:DualNumberStructure>(
        parameters:&[Atom],aliases:&[(Atom,Atom)],representative:&Atom,
        child_slots:&[Atom],outputs:usize,dual:Option<&D>,
    )->Result<Self,ReductionError> {
        if child_slots.is_empty() || outputs==0 || child_slots.iter().collect::<BTreeSet<_>>().len()!=child_slots.len() {
            return Err(ReductionError::Preparation("invalid shared affine quadrature source/output layout".into()));
        }
        if let Some(dual)=dual {
            iregi_masters::validate_dual_structure(dual)
                .map_err(|error|ReductionError::Preparation(error.to_string()))?;
        }
        let shape:Option<Vec<Vec<usize>>>=dual.map(|dual|dual.get_shape().into_iter().map(<[usize]>::to_vec).collect());
        let count=child_slots.len();
        let mut projections=HashMap::new();
        for (index,slot)in child_slots.iter().enumerate() {
            let mut unit=vec![Atom::new();count];unit[index]=Atom::num(1);projections.insert(slot.clone(),unit);
        }
        // Same alias recipe as QuadratureStabilityPlan::prepare. Symbolica's
        // existing affine_projection performs all polynomial coefficient work.
        let mut projected_aliases=Vec::new();
        for(alias,body)in aliases {
            if !depends_on_projection(body,&projections) {continue;}
            let mut projected=affine_projection(body,&projections,count)?;
            for(source,expression)in projected.iter_mut().enumerate() {
                if expression.is_zero(){continue;}
                let handle=variable(&format!("imported_coefficient_{source}_{}",projected_aliases.len()));
                projected_aliases.push((handle.clone(),expression.clone()));*expression=handle;
            }
            projections.insert(alias.clone(),projected);
        }
        let coefficients=affine_projection(representative,&projections,count)?;
        let all_aliases:Vec<_>=aliases.iter().cloned().chain(projected_aliases).collect();
        let definitions:HashMap<_,_>=all_aliases.iter().cloned().collect();
        let mut reachable=BTreeSet::new();let mut pending=coefficients.clone();
        while let Some(body)=pending.pop() {
            for symbol in body.get_all_symbols(false) {
                let atom=Atom::var(symbol);
                if let Some(value)=definitions.get(&atom) {
                    if reachable.insert(atom){pending.push(value.clone());}
                } else if projections.contains_key(&atom) {
                    return Err(ReductionError::Preparation("imported quadrature coefficient still depends on child outputs".into()));
                }
            }
        }
        let coefficient_aliases:Vec<_>=all_aliases.into_iter().filter(|(a,_)|reachable.contains(a)).collect();
        let coefficients=compile_exact(&coefficients,parameters,&coefficient_aliases,dual)?;
        let magnitudes:Vec<_>=(0..count).map(|i|variable(&format!("imported_magnitude_{i}"))).collect();
        let errors:Vec<_>=(0..count).map(|i|variable(&format!("imported_error_{i}"))).collect();
        let mut positive=Atom::new();
        for(magnitude,error)in magnitudes.iter().zip(&errors) {
            // This sidecar already contains Euclidean OUTPUT estimates. Its
            // raw E6 infinity-norm conversion has happened in the child.
            positive+=ErrorInputNorm::ImportedEuclidean.euclidean_factor()*magnitude*error;
        }
        let positive_parameters:Vec<_>=magnitudes.into_iter().chain(errors).collect();
        let positive=compile_exact(&[positive],&positive_parameters,&[],dual)?;
        Ok(Self{coefficients,positive,coefficient_parameters:parameters.len(),sources:count,outputs,shape})
    }
    pub(crate) fn instantiate<T:NativeFloat>(&self,prototype:&T)->ImportedAffineQuadrature<T>
    where Complex<T>:EvaluationDomain {
        ImportedAffineQuadrature{coefficients:map_exact(&self.coefficients,prototype),positive:map_exact(&self.positive,prototype),
            coefficient_parameters:self.coefficient_parameters,sources:self.sources,outputs:self.outputs,
            shape:self.shape.clone(),working_precision:prototype.get_precision(),
            coefficient_values:Vec::new(),positive_inputs:Vec::new(),dense_errors:Vec::new(),last_work_rows:0,last_components:0,last_failures:Vec::new()}
    }
}

#[derive(Clone)]
pub(crate) struct ImportedAffineQuadrature<T:NativeFloat> {
    coefficients:BatchExpressionEvaluator<Complex<T>>,
    positive:BatchExpressionEvaluator<Complex<T>>,
    coefficient_parameters:usize,sources:usize,outputs:usize,
    shape:Option<Vec<Vec<usize>>>,working_precision:u32,
    coefficient_values:Vec<Complex<T>>,positive_inputs:Vec<Complex<T>>,
    dense_errors:Vec<Complex<T>>,last_work_rows:usize,last_components:usize,
    last_failures:Vec<Option<String>>,
}
impl<T:NativeFloat> ImportedAffineQuadrature<T>where Complex<T>:EvaluationDomain {
    /// `source_rows[sample][source]` is the frontstage's owned subset-major map
    /// into the ONE packed N5 child batch. All child contexts are validated
    /// against their actual returned values before any positive inputs are made.
    pub(crate) fn evaluate(
        &mut self,geometry:&KinematicsBatch<T>,children:&CoupledQuadratureBatch<T>,
        source_rows:&[Vec<usize>],expected_basis:&[Monomial],expected_metadata:&ResultMetadata,
        prototype:&T,accuracy_bits:u32,
    )->Result<(),ReductionError> {
        self.last_work_rows=0;self.last_components=0;self.last_failures.clear();self.coefficient_values.clear();self.positive_inputs.clear();self.dense_errors.clear();
        let fail=|message:&str|ReductionError::Evaluation(message.into());
        let components=geometry.components;
        let input_width=self.coefficient_parameters.checked_mul(components).ok_or_else(||fail("imported quadrature input size overflow"))?;
        let dense_width=self.outputs.checked_mul(components).ok_or_else(||fail("imported quadrature output size overflow"))?;
        let coefficient_width=self.sources.checked_mul(components).ok_or_else(||fail("imported quadrature coefficient size overflow"))?;
        let work_rows=geometry.rows.checked_mul(self.outputs).ok_or_else(||fail("imported quadrature work-row size overflow"))?;
        let positive_width=coefficient_width.checked_mul(2).ok_or_else(||fail("imported quadrature positive-input size overflow"))?;
        let positive_sample_width=self.outputs.checked_mul(positive_width).ok_or_else(||fail("imported quadrature positive-input size overflow"))?;
        geometry.rows.checked_mul(positive_sample_width).ok_or_else(||fail("imported quadrature positive-input size overflow"))?;
        if components==0 || components!=self.shape.as_ref().map_or(1,Vec::len)
            || expected_metadata.dual_shape()!=self.shape.as_deref()
            || prototype.get_precision()!=self.working_precision || accuracy_bits==0 || accuracy_bits>self.working_precision
            || geometry.rows.checked_mul(input_width)!=Some(geometry.values.len())
            || source_rows.len()!=geometry.rows || source_rows.iter().any(|row|row.len()!=self.sources)
            || source_rows.iter().flatten().any(|&row|row>=children.result().rows())
            || expected_basis.len().checked_mul(4)!=Some(self.outputs)
            || children.result().basis!=expected_basis || children.result().metadata()!=expected_metadata
            || children.result().components!=components {
            return Err(fail("imported quadrature layout or complete child interpretation differs"));
        }
        self.last_failures.resize(geometry.rows,None);
        self.coefficient_values.resize(geometry.rows.checked_mul(coefficient_width).ok_or_else(||fail("imported quadrature coefficient size overflow"))?,
            Complex::new(prototype.zero(),prototype.zero()));
        self.coefficients.evaluate_batch(&geometry.values,&mut self.coefficient_values,geometry.rows)
            .map_err(|error|ReductionError::Evaluation(error.to_string()))?;
        // Coefficient jets are evaluated only ONCE per physical sample.
        for(sample,source_rows)in source_rows.iter().enumerate() {
            let coefficients=&self.coefficient_values[sample*coefficient_width..(sample+1)*coefficient_width];
            let magnitudes:Vec<_>=coefficients.iter().map(|z|Complex::new(NativeFloat::hypot(&z.re,&z.im),prototype.zero())).collect();
            if magnitudes.iter().any(|z|!z.re.is_finite()||!z.im.is_finite()) {
                self.last_failures[sample]=Some("nonfinite imported quadrature coefficient jet".into());
            }
            let errors=source_rows.iter().map(|&row|children.errors(row,prototype.get_precision(),accuracy_bits))
                .collect::<Result<Vec<_>,_>>();
            let errors=match errors {
                Ok(errors) if errors.iter().all(|row|row.len()==dense_width)=>Some(errors),
                Ok(_)=>{
                    self.last_failures[sample].get_or_insert_with(||"dense child quadrature size differs".into());None
                }
                Err(reason)=>{
                    self.last_failures[sample].get_or_insert_with(||format!("child quadrature context unavailable: {reason}"));None
                }
            };
            if self.last_failures[sample].is_some() {
                // Failed physical samples still occupy their fixed instruction-
                // major work rows. Deterministic zeros avoid contaminating a
                // successful sibling; no lane accessor can publish these zeros.
                self.positive_inputs.extend(std::iter::repeat_with(||
                    Complex::new(prototype.zero(),prototype.zero())).take(positive_sample_width));
                continue;
            }
            let errors=errors.expect("validated successful child contexts");
            for output in 0..self.outputs {
                self.positive_inputs.extend_from_slice(&magnitudes);
                for error in &errors {self.positive_inputs.extend_from_slice(&error[output*components..(output+1)*components]);}
            }
        }
        self.dense_errors.resize(work_rows.checked_mul(components).ok_or_else(||fail("imported quadrature output size overflow"))?,
            Complex::new(prototype.zero(),prototype.zero()));
        // This tiny shared Symbolica-Dualized expression owns all Taylor
        // products/multiplicities. The extra work-row axis is tensor output,
        // while every original ordered Taylor component stays inside its row.
        self.positive.evaluate_batch(&self.positive_inputs,&mut self.dense_errors,work_rows)
            .map_err(|error|ReductionError::Evaluation(error.to_string()))?;
        for (sample,errors) in self.dense_errors.chunks_exact(dense_width).enumerate() {
            if errors.iter().any(|z|!z.re.is_finite()||z.re<prototype.zero()||!z.im.is_finite()||!z.im.is_zero()) {
                self.last_failures[sample].get_or_insert_with(||"invalid dense imported quadrature estimate".into());
            }
        }
        self.last_work_rows=work_rows;self.last_components=components;Ok(())
    }
    pub(crate) fn lane(&self,row:usize,components:usize)->Option<&[Complex<T>]> {
        if components!=self.last_components || self.last_work_rows==0 || self.failure(row).is_some()
            || row.checked_mul(self.outputs)? >= self.last_work_rows {return None;}
        let width=self.outputs.checked_mul(components)?;let start=row.checked_mul(width)?;
        self.dense_errors.get(start..start.checked_add(width)?)
    }
    pub(crate) fn failure(&self,row:usize)->Option<&str> {
        self.last_failures.get(row).and_then(Option::as_deref)
    }
    pub(crate) fn last_work_rows(&self)->usize{self.last_work_rows}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_kernel_keeps_all_four_slots_second_order_jets_and_reversed_source_rows() {
        use symbolica::domains::dual::HyperDual;
        use crate::{Convention,LaneStatus,TensorWeights};
        use crate::execution::quadrature_context::{QuadratureErrorContext,QuadratureErrorOrigin};
        let x=Atom::var(symbolica::symbol!("affine_quadrature_test::x"));
        let child=Atom::var(symbolica::symbol!("affine_quadrature_test::child"));
        let shape=vec![vec![0],vec![1],vec![2]];
        let dual=HyperDual::<Complex<Rational>>::new(shape.clone());
        let prepared=PreparedImportedAffineQuadrature::prepare(&[x.clone()],&[],
            &(&x*&x*&child),&[child],4,Some(&dual)).unwrap();
        let c=|x|Complex::new(x,0.0);
        let metadata=ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,Some(shape));
        let basis=vec![Monomial([0;4])];
        let children=crate::BatchResult {
            basis:basis.clone(),components:3,metadata:metadata.clone(),values:vec![c(1.0);24],
            statuses:vec![LaneStatus::Success;2],fallback_used:vec![false;2],analytic_used:vec![false;2],
            primal_envelopes:vec![],primal_envelope_errors:vec![],
        };
        let mut context=QuadratureErrorContext::missing(2);
        let first=[1.,2.,3.,2.,4.,6.,3.,6.,9.,4.,8.,12.].map(c);
        let second=[4.,5.,6.,8.,10.,12.,12.,15.,18.,16.,20.,24.].map(c);
        context.install_dense(&children,0,53,40,&first,QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        context.install_dense(&children,1,53,40,&second,QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        let owned=CoupledQuadratureBatch::new(children,context).unwrap();
        let geometry=KinematicsBatch{rows:2,components:3,values:[2.,1.,0.,1.,0.,1.].map(c).to_vec()};
        let mut kernel=prepared.instantiate(&0.0);
        kernel.evaluate(&geometry,&owned,&[vec![1],vec![0]],&basis,&metadata,&0.0,40).unwrap();
        // Independent ordinary Taylor products: (4+4t+t²)*(4+5t+6t²)
        // and (1+2t²)*(1+2t+3t²), with each Laurent slot scaled distinctly.
        assert_eq!(kernel.lane(0,3).unwrap(),[16.,36.,48.,32.,72.,96.,48.,108.,144.,64.,144.,192.].map(c));
        assert_eq!(kernel.lane(1,3).unwrap(),[1.,2.,5.,2.,4.,10.,3.,6.,15.,4.,8.,20.].map(c));
        assert_eq!(kernel.last_work_rows(),8); // 2 samples * 4 outputs, not 24 Taylor samples.
        assert!(kernel.lane(0,1).is_none()); // A wrong component view cannot truncate a successful mixed/jet result.
        let zero_primal_weight=KinematicsBatch{rows:1,components:3,values:[0.,1.,0.].map(c).to_vec()};
        kernel.evaluate(&zero_primal_weight,&owned,&[vec![0]],&basis,&metadata,&0.0,40).unwrap();
        assert_eq!(kernel.lane(0,3).unwrap(),[0.,0.,1.,0.,0.,2.,0.,0.,3.,0.,0.,4.].map(c));
        // x² has a zero primal but a nonzero normalized second-order jet.
        // Keeping this source is essential; primal-only cofactor masking fails.
        let (mut wrong_result,_)=owned.clone().into_parts();
        let wrong_metadata=ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,
            Some(vec![vec![0,0],vec![1,0],vec![0,1]]));
        wrong_result.metadata=wrong_metadata.clone();
        let mut wrong_context=QuadratureErrorContext::missing(2);
        wrong_context.install_dense(&wrong_result,0,53,40,&first,QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        wrong_context.install_dense(&wrong_result,1,53,40,&second,QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        let wrong_owner=CoupledQuadratureBatch::new(wrong_result,wrong_context).unwrap();
        assert!(kernel.evaluate(&geometry,&wrong_owner,&[vec![1],vec![0]],&basis,&wrong_metadata,&0.0,40).is_err());
        let bad=KinematicsBatch{rows:2,components:1,values:vec![]};
        assert!(kernel.evaluate(&bad,&owned,&[vec![1],vec![0]],&basis,&metadata,&0.0,40).is_err());
        assert!(kernel.lane(0,3).is_none());assert_eq!(kernel.last_work_rows(),0);
    }

    // Additional integration skeleton, intentionally not claimed executed:
    // 1. Reuse quadrature::tests::signed_coefficient_jets_feed_positive_symbolica_convolution
    //    with x*x*external_child, shape [[0],[1],[2]], x=(2,1,0), δ=(1,2,3).
    //    Raw E6 existing expected [8,24,42] must remain unchanged.
    //    Imported output expected [4,12,21], exactly, without another factor2.
    // 2. Use four distinct child-slot markers, one for each Laurent output,
    //    and two physical samples with reversed source_rows mapping [2,0].
    //    Include mixed shape [[0,0],[1,0],[0,1],[1,1]] and a primal-zero,
    //    mixed-nonzero weight so no source is discarded by primal-only tests.
    // 3. Compare all outputs to Symbolica's explicit derivative oracle after
    //    exact Rational evaluation; test static and dynamic identical shapes.
    // 4. Preserve lazy IF inactive singular coefficient behavior and assert
    //    no child-dependent predicate/non-affine expression is accepted.

    #[test]
    fn coefficient_and_positive_overflow_failures_do_not_corrupt_good_sample_rows() {
        use symbolica::domains::dual::HyperDual;
        use crate::{Convention,LaneStatus,TensorWeights};
        use crate::execution::quadrature_context::{QuadratureErrorContext,QuadratureErrorOrigin};
        let x=Atom::var(symbolica::symbol!("affine_quadrature_isolation::x"));
        let slot=Atom::var(symbolica::symbol!("affine_quadrature_isolation::child"));
        let prepared=PreparedImportedAffineQuadrature::prepare::<HyperDual<Complex<Rational>>>(
            &[x.clone()],&[],&(&x*&x*&slot),&[slot],4,None).unwrap();
        let c=|x|Complex::new(x,0.0);
        let basis=vec![Monomial([0;4])];
        let metadata=ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,None);
        let result=crate::BatchResult{basis:basis.clone(),components:1,metadata:metadata.clone(),
            values:vec![c(1.);12],statuses:vec![LaneStatus::Success;3],fallback_used:vec![false;3],analytic_used:vec![false;3],
            primal_envelopes:vec![],primal_envelope_errors:vec![]};
        let mut context=QuadratureErrorContext::missing(3);
        context.install_dense(&result,0,53,40,&[c(1.);4],QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        context.install_dense(&result,1,53,40,&[c(2.);4],QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        context.install_dense(&result,2,53,40,&[c(f64::MAX);4],QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        let owned=CoupledQuadratureBatch::new(result,context).unwrap();
        // Row1 overflows x² in the coefficient pass; row2 has finite x²=4
        // but overflows the positive contraction against a finite child error.
        let geometry=KinematicsBatch{rows:3,components:1,values:[2.,f64::MAX,2.].map(c).to_vec()};
        let mut kernel=prepared.instantiate(&0.0);
        kernel.evaluate(&geometry,&owned,&[vec![0],vec![1],vec![2]],&basis,&metadata,&0.,40).unwrap();
        assert_eq!(kernel.lane(0,1).unwrap(),[c(4.);4]);
        assert!(kernel.failure(0).is_none());
        assert!(kernel.failure(1).unwrap().contains("coefficient"));assert!(kernel.lane(1,1).is_none());
        assert!(kernel.failure(2).unwrap().contains("estimate"));assert!(kernel.lane(2,1).is_none());
        assert_eq!(kernel.last_work_rows(),12);
        let good=KinematicsBatch{rows:1,components:1,values:vec![c(3.)]};
        kernel.evaluate(&good,&owned,&[vec![1]],&basis,&metadata,&0.,40).unwrap();
        assert!(kernel.failure(0).is_none());assert!(kernel.failure(1).is_none());
        assert_eq!(kernel.lane(0,1).unwrap(),[c(18.);4]);
    }

}
