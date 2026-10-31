//! Coupled dense output quadrature estimates for explicit affine composition.
//!
//! Successive-rule Euclidean estimates are numerical diagnostics, not rigorous
//! error bounds. This owner is separate from heuristic roundoff amplitudes.
//! It does not implement any Taylor arithmetic or silently infer zero error.
use iregi_masters::NativeFloat;
use symbolica::domains::float::Complex;
use crate::{BatchResult, ReductionError, ResultMetadata, basis::Monomial};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuadratureErrorOrigin {
    /// Output estimates already include raw-master infinity->Euclidean factor2.
    PropagatedEuclidean,
    /// The actual successful execution route had no quadrature dependency.
    /// This must be an executor assertion, never inferred from a missing sidecar.
    KnownNoQuadrature,
}

#[derive(Clone, Debug)]
struct CoupledQuadratureEnvelope<T: NativeFloat> {
    basis: Vec<Monomial>,
    metadata: ResultMetadata,
    components: usize,
    working_precision: u32,
    accuracy_bits: u32,
    origin: QuadratureErrorOrigin,
    output_snapshot: Vec<Complex<T>>,
    /// Dense [moment][UV1,IR1,IR2,finite][ordered Taylor component].
    /// Every entry is finite, nonnegative real with exactly zero imaginary part.
    errors: Vec<Complex<T>>,
    error_formats: Vec<(u32, bool, u32, bool)>,
}

/// One-call lane map, owned separately from BatchResult's roundoff envelopes.
#[derive(Clone, Debug)]
pub(crate) struct QuadratureErrorContext<T: NativeFloat> {
    lanes: Vec<Result<CoupledQuadratureEnvelope<T>, String>>,
}

/// The result and its actual-route estimates travel together across the packed
/// child boundary. Keeping result private prevents unvalidated direct mutation.
#[derive(Clone, Debug)]
pub(crate) struct CoupledQuadratureBatch<T: NativeFloat> {
    result: BatchResult<T>,
    context: QuadratureErrorContext<T>,
}

impl<T: NativeFloat> QuadratureErrorContext<T> {
    pub(crate) fn missing(rows: usize) -> Self {
        Self { lanes: (0..rows).map(|_| Err("actual-route quadrature context is missing".into())).collect() }
    }

    /// Reset is required at the public entry before typed/dimension preflight.
    /// Empty or failed calls cannot expose an earlier successful execution.
    pub(crate) fn clear(&mut self) { self.lanes.clear(); }

    pub(crate) fn is_empty(&self) -> bool { self.lanes.is_empty() }

    pub(crate) fn origin(&self, result: &BatchResult<T>, row: usize,
        working_precision: u32, accuracy_bits: u32,
    ) -> Result<QuadratureErrorOrigin, String> {
        self.validated(result, row, working_precision, accuracy_bits)?;
        Ok(self.lanes[row].as_ref().map_err(Clone::clone)?.origin)
    }

    pub(crate) fn reject(&mut self, row: usize, reason: String) -> Result<(), String> {
        let lane=self.lanes.get_mut(row).ok_or("quadrature context row is out of bounds")?;
        *lane=Err(reason); Ok(())
    }

    pub(crate) fn install_dense(
        &mut self, result:&BatchResult<T>, row:usize, working_precision:u32,
        accuracy_bits:u32, errors:&[Complex<T>], origin:QuadratureErrorOrigin,
    )->Result<(),String> {
        if self.lanes.len()!=result.rows() {return Err("quadrature context lane inventory differs from result".into());}
        // Clear a previous envelope before every fallible replacement.
        self.reject(row,"actual-route quadrature context replacement did not complete".into())?;
        let values=checked_successful_lane(result,row)?;
        validate_target_precision(working_precision,accuracy_bits)?;
        if errors.len()!=values.len() {return Err("dense quadrature-error dimensions differ from output".into());}
        for error in errors {
            validate_error(error)?;
            if origin==QuadratureErrorOrigin::KnownNoQuadrature && !error.re.is_zero() {
                return Err("known-no-quadrature route has a nonzero estimate".into());
            }
        }
        self.lanes[row]=Ok(CoupledQuadratureEnvelope {
            basis:result.basis.clone(),metadata:result.metadata().clone(),components:result.components,
            working_precision,accuracy_bits,origin,output_snapshot:values.to_vec(),
            errors:errors.to_vec(),error_formats:errors.iter().map(format).collect(),
        });
        self.validated(result,row,working_precision,accuracy_bits).map(|_|())
    }

    pub(crate) fn validated<'a>(
        &'a self,result:&BatchResult<T>,row:usize,working_precision:u32,accuracy_bits:u32,
    )->Result<&'a [Complex<T>],String> {
        if self.lanes.len()!=result.rows() {return Err("quadrature context lane inventory differs from result".into());}
        let values=checked_successful_lane(result,row)?;
        let envelope=self.lanes.get(row).ok_or("quadrature context row is out of bounds")?
            .as_ref().map_err(Clone::clone)?;
        validate_target_precision(working_precision,accuracy_bits)?;
        if envelope.basis!=result.basis || envelope.metadata!=*result.metadata()
            || envelope.components!=result.components || envelope.working_precision!=working_precision
            || envelope.accuracy_bits!=accuracy_bits || envelope.errors.len()!=values.len()
            || envelope.output_snapshot.len()!=values.len() || envelope.error_formats.len()!=values.len() {
            return Err("quadrature context basis, ordered metadata, precision or dimensions differ".into());
        }
        for (actual,saved) in values.iter().zip(&envelope.output_snapshot) {
            if !same_component(actual,saved) {return Err("quadrature context does not describe a returned Taylor coefficient".into());}
        }
        for (error,saved_format) in envelope.errors.iter().zip(&envelope.error_formats) {
            validate_error(error)?;
            if format(error)!=*saved_format {return Err("quadrature estimate precision or signed zero changed".into());}
            if envelope.origin==QuadratureErrorOrigin::KnownNoQuadrature && !error.re.is_zero() {
                return Err("known-no-quadrature route has a nonzero estimate".into());
            }
        }
        Ok(&envelope.errors)
    }

    /// Scatter only the actual successful route; source/destination outputs and
    /// their complete interpretation must agree after the caller copies values.
    pub(crate) fn inherit_lane(
        &mut self,result:&BatchResult<T>,row:usize,source:&BatchResult<T>,
        source_context:&Self,source_row:usize,working_precision:u32,accuracy_bits:u32,
    )->Result<(),String> {
        self.reject(row,"replacement route has no validated quadrature context".into())?;
        source_context.validated(source,source_row,working_precision,accuracy_bits)?;
        let source_envelope=source_context.lanes[source_row].as_ref().map_err(Clone::clone)?.clone();
        self.lanes[row]=Ok(source_envelope);
        if let Err(reason)=self.validated(result,row,working_precision,accuracy_bits) {
            self.reject(row,reason.clone())?;return Err(reason);
        }
        Ok(())
    }
}

impl<T: NativeFloat> CoupledQuadratureBatch<T> {
    pub(crate) fn new(result:BatchResult<T>,context:QuadratureErrorContext<T>)->Result<Self,ReductionError> {
        if context.lanes.len()!=result.rows() {return Err(ReductionError::Evaluation("quadrature owner lane inventory differs from result".into()));}
        Ok(Self{result,context})
    }
    pub(crate) fn result(&self)->&BatchResult<T>{&self.result}
    pub(crate) fn errors(&self,row:usize,working_precision:u32,accuracy_bits:u32)->Result<&[Complex<T>],String> {
        self.context.validated(&self.result,row,working_precision,accuracy_bits)
    }
    pub(crate) fn origin(&self,row:usize,working_precision:u32,accuracy_bits:u32)->Result<QuadratureErrorOrigin,String> {
        self.context.origin(&self.result,row,working_precision,accuracy_bits)
    }
    pub(crate) fn into_parts(self)->(BatchResult<T>,QuadratureErrorContext<T>){(self.result,self.context)}
}

fn checked_successful_lane<T:NativeFloat>(result:&BatchResult<T>,row:usize)->Result<&[Complex<T>],String> {
    if result.statuses.get(row)!=Some(&crate::LaneStatus::Success) {return Err("quadrature context requires a successful lane".into());}
    let width=result.basis.len().checked_mul(4).and_then(|n|n.checked_mul(result.components))
        .ok_or("quadrature output dimensions overflow")?;
    if result.basis.is_empty() || result.components==0
        || result.metadata().dual_shape().map_or(1, |shape| shape.len())!=result.components
        || result.rows().checked_mul(width)!=Some(result.values.len())
        || result.fallback_used.len()!=result.rows() || result.analytic_used.len()!=result.rows() {
        return Err("quadrature result layout is inconsistent".into());
    }
    let values=&result.values[row*width..(row+1)*width];
    if values.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()) {return Err("quadrature context has a nonfinite output".into());}
    Ok(values)
}
fn validate_target_precision(working:u32,accuracy:u32)->Result<(),String> {
    if working==0||accuracy==0||accuracy>working {return Err("invalid quadrature context precision or accuracy target".into());}
    Ok(())
}
fn validate_error<T:NativeFloat>(value:&Complex<T>)->Result<(),String> {
    if !value.re.is_finite()||!value.im.is_zero()||!value.im.is_finite()||value.re<value.re.zero() {
        return Err("quadrature estimate must be finite nonnegative real".into());
    }
    Ok(())
}
fn format<T:NativeFloat>(value:&Complex<T>)->(u32,bool,u32,bool) {
    (value.re.get_precision(),value.re.negative_sign(),value.im.get_precision(),value.im.negative_sign())
}
fn same_component<T:NativeFloat>(a:&Complex<T>,b:&Complex<T>)->bool {
    a.re==b.re&&a.im==b.im&&format(a)==format(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convention,LaneStatus,TensorWeights};
    use symbolica::domains::float::{Float,FloatLike};
    fn fixture(rows:usize)->BatchResult<Float> {
        let shape=Some(vec![vec![0,0],vec![1,0],vec![0,1],vec![1,1]]);
        BatchResult {basis:vec![Monomial([0;4]),Monomial([1,0,0,0])],components:4,
            metadata:ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,shape),
            values:(0..rows*8*4).map(|n|Complex::new(Float::with_val(192,(n+1)as i64),Float::with_val(160,0))).collect(),
            statuses:vec![LaneStatus::Success;rows],fallback_used:vec![false;rows],analytic_used:vec![false;rows],
            primal_envelopes:vec![],primal_envelope_errors:vec![],
            estimated_errors: Vec::new(), best_effort: Vec::new(),}
    }
    fn errors()->Vec<Complex<Float>> {(0..32).map(|n|Complex::new(Float::with_val(256,n as i64),Float::with_val(256,0))).collect()}

    #[test]
    fn dense_context_covers_every_laurent_and_mixed_component_without_padding_precision() {
        let result=fixture(3);let mut context=QuadratureErrorContext::missing(3);
        context.install_dense(&result,2,256,53,&errors(),QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        let owner=CoupledQuadratureBatch::new(result,context).unwrap();
        assert_eq!(owner.errors(2,256,53).unwrap().len(),32);
        assert!(owner.errors(0,256,53).unwrap_err().contains("missing"));
        for slot in owner.result().lane(2).unwrap() {assert_eq!(slot.re.get_precision(),192);assert_eq!(slot.im.get_precision(),160);}
        assert!(owner.errors(2,128,53).is_err());assert!(owner.errors(2,256,54).is_err());
    }

    #[test]
    fn stale_primal_higher_mixed_precision_and_signed_zero_mutations_are_rejected() {
        let mut result=fixture(1);result.values[3].im=Float::with_val(160,-0.0);
        let mut context=QuadratureErrorContext::missing(1);
        context.install_dense(&result,0,256,53,&errors(),QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        for index in 0..32 {
            let mut bad=result.clone();bad.values[index].re=Float::with_val(192,-1);assert!(context.validated(&bad,0,256,53).is_err());
        }
        let mut rounded=result.clone();rounded.values[1].re=rounded.values[1].re.at_precision(64);
        assert!(context.validated(&rounded,0,256,53).is_err());
        let mut sign=result.clone();sign.values[3].im=Float::with_val(160,0.0);
        assert!(context.validated(&sign,0,256,53).is_err());
        let mut reordered=result.clone();reordered.metadata=ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,
            Some(vec![vec![0,0],vec![0,1],vec![1,0],vec![1,1]]));
        assert!(context.validated(&reordered,0,256,53).is_err());
        let mut wrongbasis=result.clone();wrongbasis.basis[1]=Monomial([0,1,0,0]);assert!(context.validated(&wrongbasis,0,256,53).is_err());
    }

    #[test]
    fn invalid_estimates_failed_replacement_and_explicit_no_quadrature_are_distinct() {
        let result=fixture(1);let mut context=QuadratureErrorContext::missing(1);
        context.install_dense(&result,0,256,53,&errors(),QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        let mut negative=errors();negative[31].re=Float::with_val(256,-1);
        assert!(context.install_dense(&result,0,256,53,&negative,QuadratureErrorOrigin::PropagatedEuclidean).is_err());
        assert!(context.validated(&result,0,256,53).is_err());
        let mut nonfinite=errors();nonfinite[2].re=Float::with_val(256,f64::INFINITY);
        assert!(context.install_dense(&result,0,256,53,&nonfinite,QuadratureErrorOrigin::PropagatedEuclidean).is_err());
        let mut imaginary=errors();imaginary[1].im=Float::with_val(256,1);
        assert!(context.install_dense(&result,0,256,53,&imaginary,QuadratureErrorOrigin::PropagatedEuclidean).is_err());
        let zero=vec![Complex::new(Float::with_val(256,0),Float::with_val(256,0));32];
        context.install_dense(&result,0,256,53,&zero,QuadratureErrorOrigin::KnownNoQuadrature).unwrap();
        assert!(context.install_dense(&result,0,256,53,&errors(),QuadratureErrorOrigin::KnownNoQuadrature).is_err());
        let mut overflow=result.clone();overflow.components=usize::MAX;
        assert!(context.validated(&overflow,0,256,53).is_err());
        context.clear();assert!(context.validated(&result,0,256,53).is_err());
        let empty=CoupledQuadratureBatch::new(fixture(0),QuadratureErrorContext::missing(0)).unwrap();
        assert!(empty.errors(0,256,53).is_err());
    }

    #[test]
    fn reversed_selected_lane_scatter_keeps_context_failure_isolation_and_all_slots() {
        let source=fixture(3);let mut source_context=QuadratureErrorContext::missing(3);
        source_context.install_dense(&source,2,256,53,&errors(),QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        source_context.install_dense(&source,0,256,53,&errors(),QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        let mut target=fixture(2);let mut target_context=QuadratureErrorContext::missing(2);
        for (target_row,source_row)in [2,0].into_iter().enumerate() {
            target.values[target_row*32..(target_row+1)*32].clone_from_slice(source.lane(source_row).unwrap());
            target_context.inherit_lane(&target,target_row,&source,&source_context,source_row,256,53).unwrap();
            assert_eq!(target_context.validated(&target,target_row,256,53).unwrap(),errors());
        }
        let mut failed=source.clone();failed.statuses[2]=LaneStatus::MasterFailure("E6 endpoint domain".into());
        assert!(target_context.inherit_lane(&target,0,&failed,&source_context,2,256,53).is_err());
        assert!(target_context.validated(&target,0,256,53).is_err());
        assert!(target_context.validated(&target,1,256,53).is_ok());
    }

    #[test]
    fn zero_estimate_origin_is_explicit_and_equal_values_do_not_select_a_route() {
        let source=fixture(1);let zero=vec![Complex::new(Float::with_val(256,0),Float::with_val(256,0));32];
        let mut local=QuadratureErrorContext::missing(1);
        local.install_dense(&source,0,256,53,&zero,QuadratureErrorOrigin::KnownNoQuadrature).unwrap();
        let mut imported=QuadratureErrorContext::missing(1);
        imported.install_dense(&source,0,256,53,&zero,QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        assert_eq!(local.validated(&source,0,256,53).unwrap(),imported.validated(&source,0,256,53).unwrap());
        assert_eq!(local.origin(&source,0,256,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
        // Even numerically zero E6 estimates remain propagated estimates, so
        // the caller cannot use their values as proof of a quadrature-free jet.
        assert_eq!(imported.origin(&source,0,256,53).unwrap(),QuadratureErrorOrigin::PropagatedEuclidean);
        let mut replacement=errors();replacement[31].re=Float::with_val(256,91);
        imported.install_dense(&source,0,256,53,&replacement,QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
        local.inherit_lane(&source,0,&source,&imported,0,256,53).unwrap();
        assert_eq!(local.validated(&source,0,256,53).unwrap(),replacement);
        assert_eq!(local.origin(&source,0,256,53).unwrap(),QuadratureErrorOrigin::PropagatedEuclidean);
    }

}
