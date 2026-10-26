use super::*;
use symbolica::domains::float::{Float, FloatLike, SingleFloat};

fn checked(n:usize)->ReductionSettings {
    let mut settings=ReductionSettings::new(n,0);
    settings.algorithm=Algorithm::Ibp;
    settings.invariants=InvariantInput::SuppliedScalarProducts;
    settings
}
fn bubble()->Vec<Complex<Float>> {
    // p0=0, p1=(0,1,0,0), packed G=(0,0,-1), m²=(1,2), mu=3.
    [0,0,0,0,0,1,0,0,0,0,-1,1,2,3].into_iter()
        .map(|v|Complex::new(Float::with_val(128,v),Float::with_val(128,0)))
        .collect()
}
#[test]
fn checked_outer_admission_keeps_public_cartesian_domain_and_private_layouts_distinct() {
    let settings=checked(6);
    assert!(AffinePowerReductionPlan::new(settings.clone()).is_err());
    let plan=AffinePowerReductionPlan::new_checked_supplied(settings).unwrap();
    assert!(plan.checked_supplied);assert_eq!(plan.parameter_count(),52);
    assert_eq!(plan.input_layout().scalar_products,Some(24..45));
    assert_eq!(plan.input_layout().masses,45..51);
    assert!(AffinePowerReductionPlan::new_checked_supplied(checked(5)).is_err());
    let recursive=AffinePowerReductionPlan::new_checked_internal(checked(5),AffinePowerBudget::default()).unwrap();
    assert_eq!(recursive.parameter_count(),41);
    assert!(recursive.children.lock().unwrap().is_empty());
    assert!(recursive.guards.lock().unwrap().is_empty());
    assert!(recursive.profiles.lock().unwrap().is_empty());
}
#[test]
fn checked_incompatible_gram_is_rejected_before_any_direct_or_recursive_child_prepare() {
    let plan=AffinePowerReductionPlan::new_checked_internal(checked(2),AffinePowerBudget::default()).unwrap();
    let mut workspace=AffinePowerWorkspace::new(&plan,Float::new(128)).unwrap();
    let mut row=bubble();row[10]=Complex::new(Float::with_val(128,0),Float::with_val(128,0));
    let result=workspace.reduce_coupled_batch(&KinematicsBatch{values:row,rows:1,components:1}).unwrap();
    assert!(matches!(result.result().statuses[0],LaneStatus::InvalidInput(_)));
    assert!(workspace.children.is_empty());assert!(workspace.profiles.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls,0);
    assert!(result.quadrature_estimates(0).is_err());
}
#[test]
fn checked_regular_rows_use_one_batched_lowpoint_route_and_scatter_its_owned_context() {
    let settings=checked(2);
    let plan=AffinePowerReductionPlan::new_checked_internal(settings.clone(),AffinePowerBudget::default()).unwrap();
    let mut workspace=AffinePowerWorkspace::new(&plan,Float::new(128)).unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    let row=bubble();let mut bad=row.clone();bad[10]=Complex::new(Float::with_val(128,0),Float::with_val(128,0));
    let values=row.iter().cloned().chain(bad).chain(row.iter().cloned()).collect();
    let result=workspace.reduce_coupled_batch(&KinematicsBatch{values,rows:3,components:1}).unwrap();
    assert_eq!(result.result().statuses[0],LaneStatus::Success);
    assert!(matches!(result.result().statuses[1],LaneStatus::InvalidInput(_)));
    assert_eq!(result.result().statuses[2],LaneStatus::Success);
    let raw_plan=ReductionPlan::new(settings).unwrap();
    let mut raw=Workspace::new(&raw_plan,Float::new(128)).unwrap();raw.set_accuracy_precision(53).unwrap();
    let expected=raw.reduce_batch(&KinematicsBatch{values:row,rows:1,components:1}).unwrap();
    assert_eq!(result.result().lane(0).unwrap(),expected.lane(0).unwrap());
    assert_eq!(result.result().lane(2).unwrap(),expected.lane(0).unwrap());
    for row in [0,2] {
        assert_eq!(result.owned.origin(row,128,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
        assert_eq!(result.quadrature_estimates(row).unwrap().len(),4);
        assert!(result.quadrature_estimates(row).unwrap().iter().all(|z|z.re.is_zero()&&z.im.is_zero()));
    }
    assert!(result.quadrature_estimates(1).is_err());
    assert!(workspace.profiles.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls,1);
    assert_eq!(workspace.last_execution().packed_child_rows,2);
    assert_eq!(workspace.last_execution().child_methods[0].powers,vec![1,1]);
    let empty=workspace.reduce_coupled_batch(&KinematicsBatch{values:vec![],rows:0,components:1}).unwrap();
    assert_eq!(empty.result().rows(),0);assert_eq!(workspace.last_execution().child_batch_calls,0);
}
#[test]
fn checked_nilpotent_only_coordinate_rank_change_never_prepares_regular_lowpoint() {
    let plan=AffinePowerReductionPlan::new_checked_internal(checked(2),AffinePowerBudget::default()).unwrap();
    let shape=HyperDual::<u8>::new(vec![vec![0],vec![1]]);
    let mut workspace=AffinePowerWorkspace::new_dual(&plan,Float::new(128),shape).unwrap();
    // p1 = t*e_x and G11=-t² has zero represented orders0/1. Compatibility
    // holds at every represented order, but the nonzero coordinate direction
    // is nilpotent only and cannot certify a regular primal direct route.
    let mut values=Vec::new();
    for parameter in 0..14 {
        let primal=match parameter {11=>1,12=>2,13=>3,_=>0};
        values.push(Complex::new(Float::with_val(128,primal),Float::with_val(128,0)));
        values.push(Complex::new(Float::with_val(128,i64::from(parameter==5)),Float::with_val(128,0)));
    }
    let result=workspace.reduce_coupled_batch(&KinematicsBatch{values,rows:1,components:2}).unwrap();
    assert!(matches!(result.result().statuses[0],LaneStatus::Unstable(_)));
    assert!(workspace.children.is_empty());assert!(workspace.profiles.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls,0);
    assert!(result.quadrature_estimates(0).is_err());
}
