//! Source-only tests: exact layout oracles live in layout.rs. These fixtures
//! exercise actual context selection and the existing imported identity guard;
//! they are not native integral values or a numerical qualification claim.
use super::*;
use crate::{Convention,TensorWeights,types::{PrimalEnvelope,PrimalOperand,PrimalOperandKind}};
fn c(x:f64)->Complex<f64>{Complex::new(x,0.)}
fn budget()->DottedLayoutBudget {MassTaylorBudget::default().layout()}
fn layout(users:Option<&[Vec<usize>]>)->DottedLayout{DottedLayout::new(&[2,1,1,1,1],users,budget()).unwrap()}
fn joint_fixture(layout:&DottedLayout,rows:usize,origin:QuadratureErrorOrigin)->CoupledQuadratureBatch<f64>{
    let components=layout.joint_components();let width=components*4;let basis=vec![Monomial([0;4])];
    let metadata=ResultMetadata::new(Convention::IregiNative,TensorWeights::Components,Some(layout.joint_shape().to_vec()));
    let values:Vec<_>=(0..rows*width).map(|i|c((i+1)as f64)).collect();
    let amplitudes:Vec<_>=(0..rows*width).map(|i|c((i+1000)as f64)).collect();
    let envelopes=(0..rows).map(|row|Some(PrimalEnvelope{basis:basis.clone(),metadata:metadata.clone(),components,
        working_precision:53,accuracy_bits:53,operands:(0..4).map(|output_index|PrimalOperand{output_index,
            amplitude:amplitudes[row*width+output_index*components],kind:PrimalOperandKind::UncancelledMasters}).collect(),
        output_primals:(0..4).map(|i|values[row*width+i*components]).collect(),
        component_amplitudes:Some(amplitudes[row*width..(row+1)*width].to_vec()),
        output_components:Some(values[row*width..(row+1)*width].to_vec()),routes:vec![PrimalEnvelopeRoute::Ordinary]})).collect();
    let result=BatchResult{basis,metadata,components,values,statuses:vec![LaneStatus::Success;rows],fallback_used:vec![false;rows],
        analytic_used:vec![false;rows],primal_envelopes:envelopes,primal_envelope_errors:vec![None;rows],
        estimated_errors: Vec::new(), best_effort: Vec::new(),};
    let mut context=QuadratureErrorContext::missing(rows);
    let errors:Vec<_>=(0..width).map(|i|c(if origin==QuadratureErrorOrigin::KnownNoQuadrature{0.}else{(i+1)as f64*1e-16})).collect();
    for row in 0..rows{context.install_dense(&result,row,53,53,&errors,origin).unwrap();}
    CoupledQuadratureBatch::new(result,context).unwrap()
}

#[test]
fn selector_keeps_physical_rows_all_four_slots_and_ordered_user_components(){
    let users=vec![vec![0,0],vec![0,1],vec![1,0]];let layout=layout(Some(&users));
    let source=joint_fixture(&layout,3,QuadratureErrorOrigin::KnownNoQuadrature);
    for row in [2,0] {
        let selected=select_coefficients(&source,row,&layout,&0.,53).unwrap();
        assert_eq!(selected.values.len(),12);assert_eq!(selected.amplitudes.len(),12);assert_eq!(selected.errors,[c(0.);12]);
        for output in 0..4{for(user,&index)in layout.selected_indices().iter().enumerate(){
            assert_eq!(selected.values[output*3+user],source.result().lane(row).unwrap()[output*layout.joint_components()+index]);
            assert_eq!(selected.amplitudes[output*3+user],source.result().validated_component_amplitudes(row,53,53).unwrap()[output*layout.joint_components()+index]);
        }}
        assert_ne!(selected.values[0],source.result().lane(row).unwrap()[0]);
        assert_ne!(selected.amplitudes[0],source.result().primal_envelope(row).unwrap().operands[0].amplitude);
    }
}

#[test]
fn selector_rejects_stale_selected_or_unselected_context_and_preserves_good_neighbor(){
    let layout=layout(None);let original=joint_fixture(&layout,3,QuadratureErrorOrigin::KnownNoQuadrature);
    for index in [0,layout.selected_indices()[0]]{
        let(mut result,context)=original.clone().into_parts();result.values[index].re+=1.;
        let changed=CoupledQuadratureBatch::new(result,context).unwrap();
        assert!(select_coefficients(&changed,0,&layout,&0.,53).is_err());
        assert!(select_coefficients(&changed,2,&layout,&0.,53).is_ok());
    }
    // Refresh only the component-amplitude snapshot: the independent dense
    // quadrature owner must still reject a changed selected coefficient.
    let(mut result,context)=original.clone().into_parts();let selected=layout.selected_indices()[0];
    result.values[selected].re+=1.;
    result.primal_envelopes[0].as_mut().unwrap().output_components.as_mut().unwrap()[selected]=result.values[selected];
    assert!(result.validated_component_amplitudes(0,53,53).is_ok());
    let changed=CoupledQuadratureBatch::new(result,context).unwrap();
    assert!(select_coefficients(&changed,0,&layout,&0.,53).is_err());
    let(mut result,context)=original.clone().into_parts();result.statuses[1]=LaneStatus::Unstable("independent child failure".into());
    let changed=CoupledQuadratureBatch::new(result,context).unwrap();
    assert!(select_coefficients(&changed,1,&layout,&0.,53).is_err());
    assert!(select_coefficients(&changed,0,&layout,&0.,53).is_ok());
    assert!(select_coefficients(&changed,2,&layout,&0.,53).is_ok());
    let(mut result,context)=original.into_parts();
    let index=layout.selected_indices()[0];result.values[index].im=-0.;
    let changed=CoupledQuadratureBatch::new(result,context).unwrap();
    assert!(select_coefficients(&changed,0,&layout,&0.,53).is_err());
}

#[test]
fn propagated_zero_retains_origin_for_ordinary_and_higher_user_selection(){
    let ordinary=layout(None);let source=joint_fixture(&ordinary,1,QuadratureErrorOrigin::PropagatedEuclidean);
    let selected=select_coefficients(&source,0,&ordinary,&0.,53).unwrap();
    assert_eq!(selected.origin,QuadratureErrorOrigin::PropagatedEuclidean);
    for output in 0..4{assert_eq!(selected.errors[output],source.errors(0,53,53).unwrap()[output*ordinary.joint_components()+ordinary.selected_indices()[0]]);}
    let(result,mut context)=source.into_parts();
    context.install_dense(&result,0,53,53,&vec![c(0.);result.values.len()],QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
    let ordinary_zero=CoupledQuadratureBatch::new(result,context).unwrap();
    let selected=select_coefficients(&ordinary_zero,0,&ordinary,&0.,53).unwrap();
    assert_eq!(selected.errors,[c(0.);4]);assert_eq!(selected.origin,QuadratureErrorOrigin::PropagatedEuclidean);
    let users=vec![vec![0],vec![1]];let higher=layout(Some(&users));
    let source=joint_fixture(&higher,1,QuadratureErrorOrigin::PropagatedEuclidean);
    let(result,mut context)=source.into_parts();let errors=vec![c(0.);result.values.len()];
    context.install_dense(&result,0,53,53,&errors,QuadratureErrorOrigin::PropagatedEuclidean).unwrap();
    let zero_estimate=CoupledQuadratureBatch::new(result,context).unwrap();
    let selected=select_coefficients(&zero_estimate,0,&higher,&0.,53).unwrap();
    assert_eq!(selected.origin,QuadratureErrorOrigin::PropagatedEuclidean);
    assert_eq!(selected.errors,[c(0.);8]);
    assert_eq!(selected.amplitudes.len(),8);
    // Selection is not acceptance: the owning dotted identity's imported
    // all-component inspector is still required before values are returned.
}

#[test]
fn selected_coefficient_gets_fresh_dotted_dimension_primal_and_combined_error_guard(){
    let layout=layout(None);let source=joint_fixture(&layout,1,QuadratureErrorOrigin::PropagatedEuclidean);
    let selected=select_coefficients(&source,0,&layout,&0.,53).unwrap();
    let mut settings=ReductionSettings::new(5,0);settings.powers[0]=2;settings.algorithm=Algorithm::Ibp;
    let geometry:Vec<_>=(0..26).map(|i|var(&format!("identity_test_geometry_{i}"))).collect();
    let values:Vec<_>=(0..4).map(|i|var(&format!("identity_test_value_{i}"))).collect();
    let amplitudes:Vec<_>=(0..4).map(|i|var(&format!("identity_test_amplitude_{i}"))).collect();
    let shifts:Vec<[Atom;4]>=geometry[..20].chunks_exact(4).map(|p|std::array::from_fn(|i|p[i].clone())).collect();
    let distances=shifts.iter().map(|p|shifts.iter().map(|q|crate::plan::squared_difference(p,q)).collect()).collect();
    let mut parameters=geometry;parameters.extend(values.iter().cloned());parameters.extend(amplitudes.iter().cloned());
    let seeds=(0..4).map(|i|(values[i].clone(),amplitudes[i].clone(),-8)).collect();
    let parent=ReductionPlan::new_lowpoint_imported_tensor_parent(settings,0,parameters,seeds,
        InputLayout{shifts:0..20,scalar_products:None,masses:20..25,scale:25},distances,
        vec![std::array::from_fn(|i|values[i].clone())],vec![],vec!["Synthetic identity transport fixture; no integral evaluation".into()]).unwrap();
    let checks=QuadratureStabilityPlan::imported_dense_checks(&parent).unwrap();
    let mut workspace=Workspace::new_with_policies(&parent,0.,WorkspacePolicy::AFFINE_QUADRATURE_PARENT,MasterExecutionPolicy::Symbolica).unwrap();
    let mut input=vec![c(0.);26];for i in 20..26{input[i]=c(1.);}input.extend_from_slice(&selected.values);input.extend_from_slice(&selected.amplitudes);
    let result=workspace.reduce_batch(&KinematicsBatch{rows:1,components:1,values:input.clone()}).unwrap();
    assert_eq!(result.statuses,[LaneStatus::Success]);assert_eq!(result.metadata().dual_shape(),None);
    assert_eq!(result.lane(0).unwrap()[0],c(0.)); // exact target convergent UV rule
    assert_eq!(result.lane(0).unwrap()[3],selected.values[3]);
    assert_eq!(result.primal_envelope(0).unwrap().operands[3].amplitude,selected.amplitudes[3]);
    let mut errors=selected.errors;errors[0]=c(0.); // zero only because parent moment is exactly zero
    assert!(workspace.inspect_imported_quadrature_output(&checks,&result,0,&input,&errors).unwrap().is_none());
    errors[3]=c(1.);assert!(workspace.inspect_imported_quadrature_output(&checks,&result,0,&input,&errors).unwrap().is_some());
}

#[test]
fn metadata_domain_and_execution_summary_have_no_extra_sample_axis(){
    let mut settings=ReductionSettings::new(5,6);settings.powers[4]=3;
    assert!(settings_supported(&settings));settings.invariants=crate::InvariantInput::SuppliedScalarProducts;assert!(settings_supported(&settings));
    // Pure validation controls; no high-point plan is built.
    let mut bad=settings.clone();bad.dimension_shift=1;assert!(!settings_supported(&bad));
    let mut bad=settings.clone();bad.propagators=6;assert!(!settings_supported(&bad));
    let mut bad=settings.clone();bad.input_policy=InputPolicy::legacy_onshell();assert!(!settings_supported(&bad));
    let mut bad=settings.clone();bad.powers.pop();assert!(!settings_supported(&bad));
    let mut bad=settings.clone();bad.rank=7;assert!(!settings_supported(&bad));
    let mut bad=settings.clone();bad.powers.fill(1);assert!(!settings_supported(&bad));
    let mut bad=settings.clone();bad.powers[0]=0;assert!(!settings_supported(&bad));
    let stats=MassTaylorExecutionStats{input_rows:3,user_components:6,internal_components:18,
        unit_execution:ExecutionStats{input_rows:3,master_rows:12,reduction_rows:3,..Default::default()},
        projection_execution:ExecutionStats{input_rows:2,reduction_rows:2,..Default::default()},
        requested_algorithm:Algorithm::Ibp,unit_algorithm_used:Algorithm::PassarinoVeltman,..Default::default()};
    let summary=stats.summary().unwrap();assert_eq!(summary.input_rows,3);assert_eq!(summary.master_rows,12);assert_eq!(summary.reduction_rows,5);
    let mut overflow=stats;overflow.unit_execution.reduction_rows=usize::MAX;
    assert!(matches!(overflow.summary(),Err(ReductionError::Evaluation(_))));
}


#[test]
fn auxiliary_mass_branch_admission_uses_exact_primal_zero_and_only_differentiated_lines(){
    let layout=layout(Some(&[vec![0],vec![1]]));
    let input=InputLayout{shifts:0..20,scalar_products:None,masses:20..25,scale:25};
    let mut row=vec![c(0.);52];for i in 20..26{row[2*i]=c(1.);}
    let saved=row.clone();assert!(auxiliary_mass_boundary(&row,&input,&layout).is_none());assert_eq!(row,saved);
    row[2*21]=c(0.);assert!(auxiliary_mass_boundary(&row,&input,&layout).is_none()); // undifferentiated mass
    row[2*20]=c(-0.);row[2*20+1]=c(7.);
    assert!(auxiliary_mass_boundary(&row,&input,&layout).unwrap().contains("propagator 0"));
    row[2*20]=c(f64::MIN_POSITIVE);assert!(auxiliary_mass_boundary(&row,&input,&layout).is_none());
    row[2*20]=Complex::new(0.,-1.);assert!(auxiliary_mass_boundary(&row,&input,&layout).is_none());
}


#[test]
fn admitted_joint_scatter_excludes_boundary_rows_without_losing_good_neighbor_context(){
    let layout=layout(None);let source=joint_fixture(&layout,2,QuadratureErrorOrigin::PropagatedEuclidean);
    let expected_first=select_coefficients(&source,0,&layout,&0.,53).unwrap();
    let expected_last=select_coefficients(&source,1,&layout,&0.,53).unwrap();
    let rejected=vec![None,Some("exact zero differentiated mass; unsupported derivative domain".into()),None];
    let expanded=scatter_admitted_joint(source.clone(),3,&[0,2],&rejected,&0.,53).unwrap();
    assert!(matches!(expanded.result().statuses[1],LaneStatus::Unstable(_)));
    assert!(expanded.errors(1,53,53).is_err());
    assert_eq!(select_coefficients(&expanded,0,&layout,&0.,53).unwrap().values,expected_first.values);
    let selected=select_coefficients(&expanded,2,&layout,&0.,53).unwrap();
    assert_eq!(selected.values,expected_last.values);assert_eq!(selected.errors,expected_last.errors);
    assert_eq!(selected.origin,QuadratureErrorOrigin::PropagatedEuclidean);
    assert!(scatter_admitted_joint(source,3,&[0,0],&rejected,&0.,53).is_err());
}
