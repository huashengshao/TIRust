//! Transport/rejection fixtures only. No dotted integral value is evaluated.
use super::*;
use crate::{Convention,TensorWeights,types::{PrimalEnvelope,PrimalOperand,PrimalOperandKind}};
fn c(value:f64)->Complex<f64>{Complex::new(value,0.)}
fn metadata()->ResultMetadata{
    ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,Some(vec![vec![0],vec![1]]))
}
fn dotted_source()->CoupledQuadratureBatch<f64>{
    let metadata=metadata();let basis=vec![Monomial([0;4])];let components=2;let width=8;
    let values:Vec<_>=(0..16).map(|i|c((i+1)as f64)).collect();
    let amplitudes:Vec<_>=(0..16).map(|i|c((i+100)as f64)).collect();
    let envelopes=(0..2).map(|row|Some(PrimalEnvelope{basis:basis.clone(),metadata:metadata.clone(),components,
        working_precision:53,accuracy_bits:53,operands:(0..4).map(|output_index|PrimalOperand{output_index,
            amplitude:amplitudes[row*width+output_index*components],kind:PrimalOperandKind::UncancelledMasters}).collect(),
        output_primals:(0..4).map(|i|values[row*width+i*components]).collect(),
        component_amplitudes:Some(amplitudes[row*width..(row+1)*width].to_vec()),
        output_components:Some(values[row*width..(row+1)*width].to_vec()),routes:vec![PrimalEnvelopeRoute::Ordinary]})).collect();
    let result=BatchResult{basis,metadata,components,values,statuses:vec![LaneStatus::Success;2],
        fallback_used:vec![true,false],analytic_used:vec![false,true],primal_envelopes:envelopes,primal_envelope_errors:vec![None;2],
        estimated_errors: Vec::new(), best_effort: Vec::new(),};
    let mut context=QuadratureErrorContext::missing(2);
    for row in 0..2{context.install_dense(&result,row,53,53,&[c(0.);8],QuadratureErrorOrigin::KnownNoQuadrature).unwrap();}
    CoupledQuadratureBatch::new(result,context).unwrap()
}
fn pending()->Vec<LaneStatus>{vec![LaneStatus::Unstable("pending actual dotted output".into()),
    LaneStatus::InvalidInput("independent bad supplied G".into()),LaneStatus::Unstable("pending actual dotted output".into())]}

#[test]
fn checked_compaction_copies_every_original_and_joint_component_without_reconstructing_geometry(){
    for components in [1,3,6]{
        let source=KinematicsBatch{rows:3,components,values:(0..3*41*components).map(|i|Complex::new((i+1)as f64,-(i as f64)-2.)).collect()};
        let before=source.values.clone();let packed=gather_mass_rows(&source,&[0,2],41).unwrap();
        assert_eq!(source.values,before);assert_eq!(packed.rows,2);assert_eq!(packed.components,components);
        assert_eq!(packed.values.len(),2*41*components);
        assert_eq!(&packed.values[..41*components],&source.values[..41*components]);
        assert_eq!(&packed.values[41*components..],&source.values[82*components..]);
        for invalid in [&[0,0][..],&[2,0][..],&[3][..]]{assert!(gather_mass_rows(&source,invalid,41).is_err());}
        assert!(gather_mass_rows(&source,&[0],40).is_err());
        assert!(gather_mass_rows(&source,&[],41).unwrap().values.is_empty());
    }
}

#[test]
fn final_dotted_scatter_preserves_actual_sidecars_and_rejects_stale_context_per_lane(){
    let source=dotted_source();let expected=source.clone();let basis=source.result().basis.clone();
    let mut diagnostics=vec![vec![];3];
    let scattered=scatter_checked_dotted(Some(source.clone()),3,&[0,2],&pending(),&basis,&metadata(),2,&0.,53,&mut diagnostics).unwrap();
    assert_eq!(scattered.result().statuses[0],LaneStatus::Success);
    assert!(matches!(scattered.result().statuses[1],LaneStatus::InvalidInput(_)));
    assert_eq!(scattered.result().statuses[2],LaneStatus::Success);
    assert_eq!(scattered.result().fallback_used,[true,false,false]);
    assert_eq!(scattered.result().analytic_used,[false,false,true]);
    for(destination,original)in [(0,0),(2,1)]{
        assert_eq!(scattered.result().lane(destination).unwrap(),expected.result().lane(original).unwrap());
        assert_eq!(scattered.result().validated_component_amplitudes(destination,53,53).unwrap(),
            expected.result().validated_component_amplitudes(original,53,53).unwrap());
        assert_eq!(scattered.errors(destination,53,53).unwrap(),expected.errors(original,53,53).unwrap());
    }
    assert!(scattered.result().lane(1).is_err());assert!(scattered.errors(1,53,53).is_err());
    let(mut changed,context)=source.into_parts();changed.values[1].re+=1.;
    let changed=CoupledQuadratureBatch::new(changed,context).unwrap();
    let failed=scatter_checked_dotted(Some(changed),3,&[0,2],&pending(),&basis,&metadata(),2,&0.,53,&mut diagnostics).unwrap();
    assert!(matches!(failed.result().statuses[0],LaneStatus::Unstable(_)));
    assert!(failed.result().primal_envelopes[0].is_none());assert!(failed.errors(0,53,53).is_err());
    assert_eq!(failed.result().statuses[2],LaneStatus::Success);assert!(failed.errors(2,53,53).is_ok());
}

#[test]
fn checked_empty_scatter_never_leaks_provisional_success_and_proofs_stay_out_of_numeric_summary(){
    let statuses=[LaneStatus::InvalidInput("bad G".into()),LaneStatus::Unstable("transverse rank jet".into())];
    let basis=[Monomial([0;4])];let mut diagnostics=vec![vec![];2];
    let rejected=scatter_checked_dotted::<f64>(None,2,&[],&statuses,&basis,&metadata(),2,&0.,53,&mut diagnostics).unwrap();
    assert_eq!(rejected.result().statuses,statuses);assert!(rejected.result().lane(0).is_err());assert!(rejected.result().lane(1).is_err());
    assert!(scatter_checked_dotted::<f64>(None,2,&[0],&statuses,&basis,&metadata(),2,&0.,53,&mut diagnostics).is_err());
    let provisional=[LaneStatus::Success,LaneStatus::Unstable("pending".into())];
    assert!(scatter_checked_dotted::<f64>(None,2,&[],&provisional,&basis,&metadata(),2,&0.,53,&mut diagnostics).is_err());
    let stats=MassTaylorExecutionStats{input_rows:3,user_components:2,internal_components:4,
        unit_execution:ExecutionStats{input_rows:2,master_rows:7,reduction_rows:2,..Default::default()},
        projection_execution:ExecutionStats{input_rows:2,reduction_rows:2,..Default::default()},
        certificate_execution:CertificateExecutionStats{input_rows:3,evaluated_rows:usize::MAX,passes:usize::MAX,reduction_batches:0},
        ..Default::default()};
    let summary=stats.summary().unwrap();assert_eq!(summary.input_rows,3);assert_eq!(summary.master_rows,7);assert_eq!(summary.reduction_rows,4);
    assert_eq!(summary.argument_rows,0);
}

#[test]
fn actual_checked_rejection_precedes_unit_and_projection_calls_and_resets_empty_error_stats(){
    // Cheap N1 numerical scratch is an intentional sentinel: this test admits
    // no unit lane, evaluates no integral, and makes no N5 numerical claim.
    // The plan metadata below supplies the real full N5 input/certificate map.
    let scratch_plan=ReductionPlan::new(ReductionSettings::new(1,0)).unwrap();
    let scratch=Workspace::new_with_policies(&scratch_plan,0.,WorkspacePolicy::AFFINE_QUADRATURE_CHILD,MasterExecutionPolicy::Symbolica).unwrap();
    let mut settings=ReductionSettings::new(5,0);settings.algorithm=Algorithm::Ibp;
    settings.invariants=InvariantInput::SuppliedScalarProducts;settings.powers[0]=2;
    let mut unit=scratch_plan.clone();unit.settings=settings.clone();unit.settings.powers.fill(1);
    let(parameters,_)=crate::certified_tensor::compatibility_constraints(5);unit.parameters=parameters;
    unit.parameters.extend((0..5).map(|i|var(&format!("checked_rejection_m2_{i}"))));unit.parameters.push(var("checked_rejection_mu"));
    unit.layout=InputLayout{shifts:0..20,scalar_products:Some(20..35),masses:35..40,scale:40};
    let unit=Arc::new(unit);
    let ordinary=MassTaylorReductionPlan::with_shared_unit(settings.clone(),unit.clone(),MassTaylorBudget::default()).unwrap();
    assert!(!ordinary.checked_regular_unit);
    let plan=MassTaylorReductionPlan::with_shared_checked_regular_unit(settings.clone(),unit,MassTaylorBudget::default()).unwrap();
    assert!(plan.checked_regular_unit);assert_eq!(plan.parameter_count(),41);
    let layout=Arc::new(DottedLayout::new(&settings.powers,None,MassTaylorBudget::default().layout()).unwrap());
    let checked_regular=Some(CheckedRegularUnitGuard::new(layout.joint_shape()).unwrap());
    let mut workspace=MassTaylorWorkspace{plan,prototype:0.,accuracy_bits:53,layout,
        unit:scratch.clone(),projection:scratch,checks:QuadratureStabilityPlan::imported_dense_checks(&scratch_plan).unwrap(),
        exact_zero_outputs:vec![false;4],stats:Default::default(),diagnostics:vec![],profiling:false,profile:Default::default(),checked_regular};
    let mut bad=vec![c(0.);41];for mass in 35..40{bad[mass]=c(1.);}bad[40]=c(3.);bad[20]=c(1.);
    let input=KinematicsBatch{rows:2,components:1,values:[bad.clone(),bad].concat()};let before=input.values.clone();
    let result=workspace.reduce_batch(&input).unwrap();assert_eq!(input.values,before);
    assert!(result.statuses.iter().all(|status|matches!(status,LaneStatus::InvalidInput(_))));
    assert_eq!(workspace.last_execution().unit_execution,ExecutionStats::default());
    assert_eq!(workspace.last_execution().projection_execution,ExecutionStats::default());
    assert_eq!(workspace.last_execution().certificate_execution.input_rows,2);
    assert_eq!(workspace.last_execution().certificate_execution.evaluated_rows,2);
    assert_eq!(workspace.execution_summary().unwrap().input_rows,2);
    assert!(workspace.reduce_batch(&KinematicsBatch{rows:0,components:1,values:vec![]}).unwrap().statuses.is_empty());
    assert_eq!(workspace.last_execution().certificate_execution,CertificateExecutionStats::default());
    assert!(workspace.reduce_batch(&KinematicsBatch{rows:1,components:2,values:vec![]}).is_err());
    assert_eq!(workspace.last_execution().certificate_execution,CertificateExecutionStats::default());
    assert!(workspace.last_diagnostics().is_empty());
}
