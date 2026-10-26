//! Small execution/ownership tests. The injected recurrence below is a routing
//! fixture, not an independent integral or D6 oracle. All tests are unexecuted
//! until the integrated candidate is built under its separately owned lease.
use super::*;
use crate::{Algorithm, Convention, ReductionSettings};

fn c(x:f64)->Complex<f64>{Complex::new(x,0.)}
fn plan()->ReductionPlan {
    let mut settings=ReductionSettings::new(1,0);
    settings.algorithm=Algorithm::PassarinoVeltman;
    settings.convention=Convention::OneLoop;
    ReductionPlan::new(settings).unwrap()
}
fn workspace(plan:&ReductionPlan,policy:WorkspacePolicy)->Workspace<f64> {
    Workspace::new_with_policies(plan,0.,policy,MasterExecutionPolicy::Symbolica).unwrap()
}
fn rows(points:&[[f64;6]])->KinematicsBatch<f64> {
    KinematicsBatch{rows:points.len(),components:1,
        values:points.iter().flatten().copied().map(c).collect()}
}

#[test]
fn actual_base_child_all_slots_match_ordinary_and_failure_neighbor_is_isolated() {
    let plan=plan();
    let mut ordinary=workspace(&plan,WorkspacePolicy::ORDINARY);
    let mut child=workspace(&plan,WorkspacePolicy::AFFINE_QUADRATURE_CHILD);
    // Retention uses the exact same structural recipe/cache, not a new graph.
    assert_eq!(plan.execution_cache.lock().unwrap().len(),1);
    let batch=rows(&[[0.,0.,0.,0.,2.,1.],[f64::NAN,0.,0.,0.,2.,1.],[0.,0.,0.,0.,3.,1.]]);
    let expected=ordinary.reduce_batch(&batch).unwrap();
    let coupled=child.reduce_batch_with_quadrature_context(&batch).unwrap();
    assert_eq!(coupled.result().statuses,expected.statuses);
    assert_eq!(coupled.result().values,expected.values);
    assert_eq!(coupled.result().metadata(),expected.metadata());
    assert_eq!(child.last_execution(),ordinary.last_execution());
    assert!(child.last_quadrature_context.is_empty()); // ownership transferred
    for row in [2,0] {
        assert_eq!(coupled.errors(row,53,53).unwrap(),[c(0.);4]);
        assert_eq!(coupled.origin(row,53,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
        coupled.result().validated_primal_envelope(row,53,53).unwrap();
    }
    assert!(coupled.errors(1,53,53).is_err());
    assert!(coupled.origin(1,53,53).is_err());
    assert!(ordinary.last_quadrature_context.is_empty());
    assert_eq!(expected.primal_envelopes.capacity(),0);
}

#[test]
fn context_resets_before_raw_and_typed_preflight_and_empty_reuse() {
    let plan=plan();let mut child=workspace(&plan,WorkspacePolicy::AFFINE_QUADRATURE_CHILD);
    let batch=rows(&[[0.,0.,0.,0.,2.,1.]]);
    let previous=child.reduce_batch(&batch).unwrap();
    assert!(child.last_quadrature_context.validated(&previous,0,53,53).is_ok());
    let clone=child.clone();
    assert!(child.reduce_batch(&KinematicsBatch{rows:1,components:1,values:vec![]}).is_err());
    assert!(child.last_quadrature_context.is_empty());
    assert!(clone.last_quadrature_context.validated(&previous,0,53,53).is_ok());
    child.reduce_batch(&batch).unwrap();
    assert!(!child.last_quadrature_context.is_empty());
    assert!(child.reduce_batch(&KinematicsBatch{rows:1,components:2,values:vec![c(0.);12]}).is_err());
    assert!(child.last_quadrature_context.is_empty());
    child.reduce_batch(&batch).unwrap();
    assert!(!child.last_quadrature_context.is_empty());
    let typed:Vec<HyperDual<Complex<f64>>>=Vec::new();
    assert!(child.reduce_hyperdual_batch(&typed,1).is_err());
    assert!(child.last_quadrature_context.is_empty());
    child.reduce_batch(&batch).unwrap();
    let empty=child.reduce_batch_with_quadrature_context(&rows(&[])).unwrap();
    assert_eq!(empty.result().rows(),0);
    assert!(empty.errors(0,53,53).is_err());
    assert_eq!(child.last_execution(),ExecutionStats::default());
    let mut ordinary=workspace(&plan,WorkspacePolicy::ORDINARY);
    assert!(ordinary.reduce_batch_with_quadrature_context(&batch).is_err());
    assert!(ordinary.last_quadrature_context.is_empty());
}

#[test]
fn parent_without_local_quadrature_stays_pending_and_base_child_rejects_imports_early() {
    let plan=plan();
    let mut parent=workspace(&plan,WorkspacePolicy::AFFINE_QUADRATURE_PARENT);
    let result=parent.reduce_batch_with_quadrature_context(&rows(&[[0.,0.,0.,0.,2.,1.]])).unwrap();
    assert_eq!(result.result().statuses,[LaneStatus::Success]);
    result.result().validated_primal_envelope(0,53,53).unwrap();
    assert!(result.errors(0,53,53).unwrap_err().contains("pending"));
    assert!(result.origin(0,53,53).is_err());
    let mut imported=self::plan();
    imported.external_primal_seeds.push((Atom::num(1),Atom::num(1),0));
    assert_eq!(imported.execution_cache.lock().unwrap().len(),0);
    assert!(matches!(Workspace::new_with_policy(&imported,0.,WorkspacePolicy::AFFINE_QUADRATURE_CHILD),
        Err(ReductionError::Preparation(reason)) if reason.contains("imported")));
    assert_eq!(imported.execution_cache.lock().unwrap().len(),0);
    // The recursive instantiate path applies the same invariant even when no
    // ReductionPlan reference is present (e.g. an exact analytic child).
    let mut prepared=PreparedExecution::build::<HyperDual<Complex<Rational>>>(
        &self::plan(),None,&mut PreparedMasterCache::default(),MasterExecutionPolicy::Symbolica).unwrap();
    prepared.zero_profile_offset=None; // isolate instantiate's recipe invariant
    assert!(Workspace::instantiate(None,0.,&prepared,WorkspacePolicy::AFFINE_QUADRATURE_CHILD).is_ok());
    prepared.has_external_primal_seeds=true;
    assert!(matches!(Workspace::instantiate(None,0.,&prepared,WorkspacePolicy::AFFINE_QUADRATURE_CHILD),
        Err(ReductionError::Preparation(reason)) if reason.contains("imported")));
    assert!(WorkspacePolicy::AFFINE_QUADRATURE_PARENT.validate_quadrature_recipe(true).is_ok());
}

#[test]
fn actual_generic_fallback_replaces_context_and_preserves_successful_neighbor() {
    let baseline=plan();
    let mut injected=plan();
    // Symbolica owns the injected pole. It forces exactly the zero-origin lane
    // to fail the primary output guard and take the real executor fallback.
    injected.moments[0][3]=Atom::num(1)/&injected.parameters[0];
    let mut parent=workspace(&injected,WorkspacePolicy::AFFINE_QUADRATURE_CHILD);
    parent.fallback=Some(Box::new(workspace(&baseline,WorkspacePolicy::AFFINE_QUADRATURE_CHILD)));
    let batch=rows(&[[0.,0.,0.,0.,2.,1.],[1.,0.,0.,0.,2.,1.]]);
    let expected=workspace(&baseline,WorkspacePolicy::AFFINE_QUADRATURE_CHILD).reduce_batch(&batch).unwrap();
    let actual=parent.reduce_batch_with_quadrature_context(&batch).unwrap();
    assert_eq!(actual.result().statuses,[LaneStatus::Success,LaneStatus::Success]);
    assert_eq!(actual.result().fallback_used,[true,false]);
    assert_eq!(actual.result().lane(0).unwrap(),expected.lane(0).unwrap());
    assert_eq!(actual.result().lane(1).unwrap()[3],c(1.));
    for row in 0..2 {
        assert_eq!(actual.origin(row,53,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
        assert_eq!(actual.errors(row,53,53).unwrap(),[c(0.);4]);
        actual.result().validated_primal_envelope(row,53,53).unwrap();
    }
    assert_eq!(parent.last_execution().fallback_rows,1);
    assert_eq!(parent.last_execution().fallback_successes,1);
}

#[test]
fn actual_analytic_profile_and_failed_optional_profile_preserve_context_route() {
    let plan=plan();let batch=rows(&[[0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,2.,1.]]);
    let mut healthy=workspace(&plan,WorkspacePolicy::AFFINE_QUADRATURE_CHILD);
    let expected=healthy.reduce_batch_with_quadrature_context(&batch).unwrap();
    assert_eq!(expected.result().analytic_used,[true,false]);
    for row in 0..2 {assert_eq!(expected.origin(row,53,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);}

    let mut injected=self::plan();injected.moments[0][3]=Atom::num(1)/&injected.parameters[0];
    let mut failed=workspace(&injected,WorkspacePolicy::AFFINE_QUADRATURE_CHILD);
    failed.zero_profile_offset=None; // test fixture cannot recursively reroute itself
    let mut parent=workspace(&plan,WorkspacePolicy::AFFINE_QUADRATURE_CHILD);
    parent.zero_profiles.insert(ZeroInvariantProfile{masses:vec![true],distances:vec![]},Ok(Box::new(failed)));
    let actual=parent.reduce_batch_with_quadrature_context(&batch).unwrap();
    assert_eq!(actual.result().statuses,expected.result().statuses);
    assert_eq!(actual.result().values,expected.result().values);
    assert_eq!(actual.result().analytic_used,[false,false]);
    assert_eq!(actual.result().fallback_used,[false,false]);
    assert!(parent.last_diagnostics()[0].iter().any(|note|note.contains("ordinary reduction retained")));
    for row in 0..2 {
        assert_eq!(actual.errors(row,53,53).unwrap(),[c(0.);4]);
        assert_eq!(actual.origin(row,53,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
    }
}

#[test]
fn imported_inspector_uses_returned_route_amplitudes_and_rejects_stale_values() {
    let plan=plan();let mut child=workspace(&plan,WorkspacePolicy::AFFINE_QUADRATURE_CHILD);
    let batch=rows(&[[0.,0.,0.,0.,2.,1.]]);
    let actual=child.reduce_batch(&batch).unwrap();
    let checks=QuadratureStabilityPlan::imported_dense_checks(&plan).unwrap();
    let mut errors=vec![c(0.);4];
    assert!(child.inspect_imported_quadrature_output(&checks,&actual,0,&batch.values,&errors).unwrap().is_none());
    for slot in 0..4 {
        errors[slot]=c(1.);
        assert!(child.inspect_imported_quadrature_output(&checks,&actual,0,&batch.values,&errors)
            .unwrap().unwrap().contains(&format!("Laurent slot {slot}")));
        errors[slot]=c(0.);
    }
    let mut changed=actual.clone();changed.values[0]=c(123.);
    assert!(child.inspect_imported_quadrature_output(&checks,&changed,0,&batch.values,&errors).is_err());
    assert!(child.inspect_imported_quadrature_output(&checks,&actual,9,&batch.values,&errors).is_err());
    child.set_accuracy_precision(32).unwrap();
    assert!(child.inspect_imported_quadrature_output(&checks,&actual,0,&batch.values,&errors).is_err());
}


#[test]
fn actual_sparse_builder_maps_e6_dependencies_in_every_laurent_slot() {
    // Synthetic dependency fixture: UV/IR placement here tests the generic
    // affine graph adapter, not a claim of physical E6 UV/IR poles.
    let mut settings=ReductionSettings::new(1,0);settings.algorithm=Algorithm::PassarinoVeltman;
    let mut graph=ReductionPlan::new(settings).unwrap();
    graph.leaves[0].kind=MasterLeafKind::SixDimensionalPentagon;
    graph.leaves[0].active=vec![0,1,2,3,4];graph.aliases.clear();
    for pole in 0..4 {graph.moments[0][pole]=Atom::num((pole+1)as i64)*&graph.leaves[0].slots[3];}
    let stability=StabilityPlan::prepare(&graph).unwrap();
    let dual=HyperDual::<Complex<Rational>>::new(vec![vec![0],vec![2],vec![1]]);
    let exact=PreparedQuadrature::prepare(&graph,&stability,Some(&dual)).unwrap().unwrap();
    let mut workspace=QuadratureWorkspace::instantiate(&exact,&0.,1).unwrap();
    workspace.reset(3,3,&c(0.)).unwrap();
    workspace.store_errors(0,2,3,&[2.,5.,7.],&0.).unwrap();
    workspace.select_row(2,3); // survivor index differs from source row
    let input_width=graph.parameters.len()*3;let width=input_width+4*3;
    workspace.evaluate(&vec![c(0.);width],1,input_width,width,3,&c(0.)).unwrap();
    let dense=workspace.plan.dense_output_errors(workspace.lane(0,3),3,&c(0.)).unwrap();
    assert_eq!(dense,[c(4.),c(10.),c(14.),c(8.),c(20.),c(28.),c(12.),c(30.),c(42.),c(16.),c(40.),c(56.)]);
}


#[test]
fn actual_typed_shape_rejection_clears_previous_dual_context_and_profiling_is_observational() {
    let plan=plan();let shape=vec![vec![0],vec![1]];
    let dual=HyperDual::<Complex<Rational>>::new(shape.clone());
    let mut child=Workspace::new_dual_with_policies(&plan,0.,dual,
        WorkspacePolicy::AFFINE_QUADRATURE_CHILD,MasterExecutionPolicy::Symbolica).unwrap();
    let batch=KinematicsBatch{rows:1,components:2,values:[0.,0.,0.,0.,2.,1.].into_iter()
        .flat_map(|x|[c(x),c(0.)]).collect()};
    let baseline=child.reduce_batch(&batch).unwrap();
    assert_eq!(baseline.statuses,[LaneStatus::Success]);
    child.last_quadrature_context.validated(&baseline,0,53,53).unwrap();
    let wrong=HyperDual::<Complex<f64>>::new(vec![vec![0],vec![1],vec![2]]);
    let rejected=child.reduce_hyperdual_batch(&vec![wrong;6],1).unwrap_err();
    assert!(rejected.to_string().contains("shape"));
    assert!(child.last_quadrature_context.is_empty());
    child.set_profiling(true);
    let actual=child.reduce_batch_with_quadrature_context(&batch).unwrap();
    assert_eq!(actual.result().values,baseline.values);
    assert_eq!(actual.result().statuses,baseline.statuses);
    assert_eq!(actual.origin(0,53,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
    assert_eq!(actual.errors(0,53,53).unwrap(),[c(0.);8]);
    assert!(child.last_profile().is_some());
}


#[test]
fn imported_dual_joint_inspector_requires_actual_full_component_owner_after_fallback() {
    // new_dual otherwise rebuilds a default-order plan and discards this
    // synthetic primary pole. Match the requested second-order shape first.
    let ordinary=plan().with_analytic_jet_order(2).unwrap();
    let mut primary=(*plan().with_analytic_jet_order(2).unwrap()).clone();
    primary.moments[0][3]=Atom::num(1)/&primary.parameters[0];
    let shape=vec![vec![0,0],vec![1,0],vec![0,1],vec![2,0],vec![1,1],vec![0,2]];
    let make=|plan:&ReductionPlan|Workspace::new_dual_with_policies(plan,0.,
        HyperDual::<Complex<Rational>>::new(shape.clone()),WorkspacePolicy::AFFINE_QUADRATURE_CHILD,
        MasterExecutionPolicy::Symbolica).unwrap();
    let mut child=make(&primary);child.zero_profile_offset=None;child.fallback=Some(Box::new(make(&ordinary)));
    let mut input=vec![c(0.);36];input[24]=c(2.);input[25]=c(1.);input[30]=c(1.);
    let batch=KinematicsBatch{rows:1,components:6,values:input.clone()};
    let actual=child.reduce_batch_with_quadrature_context(&batch).unwrap();
    assert_eq!(actual.result().statuses,[LaneStatus::Success]);
    assert_eq!(actual.result().fallback_used,[true]);assert_eq!(child.last_execution().fallback_rows,1);
    let result=actual.result();let checks=QuadratureStabilityPlan::imported_dense_checks(&ordinary).unwrap();
    let errors=vec![c(0.);24];
    assert!(child.inspect_imported_quadrature_output(&checks,result,0,&input,&errors).unwrap().is_none());
    for mutation in 0..5 {
        let mut changed=result.clone();
        match mutation {
            0=>changed.primal_envelopes[0].as_mut().unwrap().component_amplitudes=None,
            1=>changed.values[19].re+=1.,
            2=>{ assert!(changed.values[1].im == 0.); changed.values[1].im=if changed.values[1].im.is_sign_negative(){0.}else{-0.}; },
            3=>changed.primal_envelopes[0].as_mut().unwrap().accuracy_bits=32,
            _=>changed.primal_envelopes[0].as_mut().unwrap().output_components.as_mut().unwrap().swap(19,20),
        }
        assert!(child.inspect_imported_quadrature_output(&checks,&changed,0,&input,&errors).is_err(),"mutation {mutation}");
    }
    let mut errors=errors;errors[22]=c(1.);
    assert!(child.inspect_imported_quadrature_output(&checks,result,0,&input,&errors).unwrap().unwrap().contains("Taylor component 4"));
    assert_eq!(actual.origin(0,53,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
}

#[test]
fn ordinary_e6_dual_constructor_instantiates_joint_sensitivity_without_export_policy() {
    let mut settings=ReductionSettings::new(5,0);
    settings.algorithm=Algorithm::Ibp;settings.dimension_shift=1;
    let plan=ReductionPlan::new(settings).unwrap();
    let dual=HyperDual::<Complex<Rational>>::new(vec![vec![0],vec![1],vec![2]]);
    let workspace=Workspace::new_dual_with_policies(&plan,0.,dual,
        WorkspacePolicy::ORDINARY,MasterExecutionPolicy::Symbolica).unwrap();
    assert!(workspace.quadrature.is_some());assert!(workspace.component_sensitivity.is_some());
    assert!(!workspace.policy.retain_primal_envelopes);
    assert_eq!(workspace.last_execution(),ExecutionStats::default());
    assert!(workspace.last_quadrature_context.is_empty());
}

#[test]
fn e6_dual_zero_profiles_are_optional_and_generic_contexts_stay_coupled() {
    let mut settings = ReductionSettings::new(5, 0);
    settings.algorithm = Algorithm::Ibp;
    settings.dimension_shift = 1;
    settings.convention = Convention::OneLoop;
    let plan = ReductionPlan::new(settings).unwrap();
    let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
    let mut child = Workspace::new_dual_with_policies(
        &plan, 0., shape, WorkspacePolicy::AFFINE_QUADRATURE_CHILD,
        MasterExecutionPolicy::Symbolica,
    ).unwrap();
    assert!(child.quadrature.is_some());
    assert!(child.component_sensitivity.is_some());
    assert!(child.zero_profile_offset.is_some());
    let mut ordinary = child.clone();
    ordinary.zero_profile_offset = None;
    ordinary.zero_profile_plan = None;

    // The first lane is the first five shifts/masses of the full-rank SAMPLE
    // witness. Its time-to-spatial separations are exactly null, with every
    // Taylor coefficient zero. Changing only the time shift in the neighbor
    // removes those optional profiles without changing the positive-F domain.
    let batch = KinematicsBatch {
        rows: 2,
        components: 2,
        values: [1., 2.].into_iter().flat_map(|time| {
            [0., 0., 0., 0., time, 0., 0., 0., 0., 1., 0., 0.,
                0., 0., 1., 0., 0., 0., 0., 1., 10., 12., 15., 18., 22., 1.]
                .into_iter().enumerate().flat_map(|(argument, value)| {
                    [c(value), c(if (20..25).contains(&argument) { 1. } else { 0. })]
                })
        }).collect(),
    };
    let expected = ordinary.reduce_batch_with_quadrature_context(&batch).unwrap();
    let actual = child.reduce_batch_with_quadrature_context(&batch).unwrap();
    assert_eq!(actual.result().statuses, [LaneStatus::Success, LaneStatus::Success]);
    assert_eq!(actual.result().values, expected.result().values);
    assert_eq!(actual.result().metadata(), expected.result().metadata());
    assert_eq!(child.last_execution(), ordinary.last_execution());
    assert!(child.last_execution().pentagon_box_rows > 0);
    assert_eq!(child.last_execution().analytic_rows, 0);
    assert!(child.zero_profiles.is_empty());
    let skipped = "exact-zero profile specialization skipped for full-component quadrature; ordinary reduction retained";
    assert!(child.last_diagnostics()[0].iter().any(|note| note == skipped));
    assert!(!child.last_diagnostics()[1].iter().any(|note| note == skipped));
    for lane in 0..2 {
        assert_eq!(actual.origin(lane, 53, 53).unwrap(), QuadratureErrorOrigin::PropagatedEuclidean);
        assert_eq!(actual.errors(lane, 53, 53).unwrap(), expected.errors(lane, 53, 53).unwrap());
        actual.result().validated_primal_envelope(lane, 53, 53).unwrap();
    }
    // Independent D6 parameter integral and common-mass derivative (minus
    // the D4 scalar) already establish this normalization and derivative sign.
    let output = actual.result().lane(0).unwrap();
    assert!(output[..6].iter().all(|value| *value == c(0.)));
    assert!((output[6].re + 0.0001771401382640081806).abs() < 2e-14);
    assert!((output[7].re - 0.0000236059994160983338257).abs() < 2e-14);
    assert!(output[6].im.abs() < 2e-14 && output[7].im.abs() < 2e-14);
}
