//! Public typed-input failures must describe the failed call, not earlier work.
use std::time::Duration;

use iregi_rust::{
    Algorithm, Convention, ExecutionStats, InputPolicy,
    LaneStatus, ReductionError, ReductionPlan, ReductionSettings, Workspace,
};
use symbolica::domains::{dual::{DualNumber, HyperDual}, float::Complex};

symbolica::create_hyperdual_from_components!(ConstantJet, [[0]]);
symbolica::create_hyperdual_from_components!(MassJet, [[0], [1]]);

fn c(value: f64) -> Complex<f64> { Complex::new(value,0.) }

fn reset_observations(workspace: &Workspace<f64>, rows: usize) {
    assert_eq!(workspace.last_execution(),ExecutionStats {input_rows:rows,..Default::default()});
    assert!(workspace.last_diagnostics().is_empty());
    let profile=workspace.last_profile().expect("profiling remains enabled");
    assert_eq!(profile.total,Duration::ZERO);
    assert_eq!(profile.arguments,Duration::ZERO);
    assert_eq!(profile.scalar_masters,Duration::ZERO);
    assert_eq!(profile.reduction,Duration::ZERO);
}

#[test]
fn typed_errors_clear_real_bubble_work_and_legacy_boundary_diagnostics() {
    let mut settings=ReductionSettings::new(2,0);
    settings.algorithm=Algorithm::PassarinoVeltman;
    settings.convention=Convention::OneLoop;
    // This existing legacy boundary fixture produces real equality-branch
    // diagnostic messages while retaining successful B0 work. The
    // primal-only Dual layout is a supported public typed carrier, not mocked
    // state. The second lane's invalid mu checks a real lane rejection too.
    settings.input_policy=InputPolicy::LegacyOnShell {threshold:"1/4".into()};
    let plan=ReductionPlan::new(settings).unwrap();
    let mut workspace=Workspace::new_dual(&plan,0.,ConstantJet {values:[c(0.)]}).unwrap();
    workspace.set_profiling(true);
    let input:Vec<_>=[3.,0.].into_iter().flat_map(|mu|
        [0.,0.,0.,0.,1.,0.,0.,0.,4.,9.,mu].map(|v|ConstantJet {values:[c(v)]})).collect();
    let original:Vec<_>=input.iter().map(|v|v.values).collect();
    for failure in ["length","shape","component_buffer","overflow"] {
        let prior=workspace.reduce_typed_dual_batch(&input,2).unwrap();
        assert_eq!(prior.statuses[0],LaneStatus::Success);
        assert!(matches!(prior.statuses[1],LaneStatus::InvalidInput(_)));
        assert_eq!(prior.lane(0).unwrap().len(),4);
        assert!(workspace.last_execution().master_rows>0,"real B0 work must precede the invalid typed call");
        assert!(workspace.last_execution().reduction_rows>0);
        assert!(workspace.last_profile().unwrap().total>Duration::ZERO);
        assert!(workspace.last_diagnostics().iter().flatten()
            .any(|note|note.contains("strict-comparison boundary")),"a real recorded diagnostic must be cleared");
        let (error,rows)=match failure {
            "length"=>(workspace.reduce_typed_dual_batch(&input[..input.len()-1],2).unwrap_err(),2),
            "shape"=>{
                let wrong:Vec<_>=input.iter().map(|v|HyperDual::from_values(
                    vec![vec![0],vec![1]],vec![v.values[0],c(0.)])).collect();
                (workspace.reduce_hyperdual_batch(&wrong,2).unwrap_err(),2)
            }
            "component_buffer"=>(workspace.reduce_dual_batch(&input,2,|v|&v.values[..0]).unwrap_err(),2),
            "overflow"=>(workspace.reduce_typed_dual_batch(&[] as &[ConstantJet<Complex<f64>>],usize::MAX).unwrap_err(),usize::MAX),
            _=>unreachable!(),
        };
        assert!(matches!(error,ReductionError::InvalidInput(_)));
        reset_observations(&workspace,rows);
        assert_eq!(input.iter().map(|v|v.values).collect::<Vec<_>>(),original,"typed validation changed inputs");
        // The previous owned result remains readable despite workspace reset.
        assert!(prior.lane(0).unwrap()[3].re.is_finite());
        assert!(prior.lane(1).is_err());
    }
    let recovered=workspace.reduce_typed_dual_batch(&input,2).unwrap();
    assert_eq!(recovered.statuses[0],LaneStatus::Success);
    assert!(workspace.last_execution().master_rows>0);
}

#[test]
fn genuine_mass_dual_recovers_after_malformed_typed_length_and_shape() {
    let mut settings=ReductionSettings::new(1,0);
    settings.algorithm=Algorithm::PassarinoVeltman;
    settings.convention=Convention::OneLoop;
    let plan=ReductionPlan::new(settings).unwrap();
    let mut workspace=Workspace::new_dual(&plan,0.,MassJet {values:[c(0.);2]}).unwrap();
    workspace.set_profiling(true);
    let input:Vec<_>=[0.,0.,0.,0.,2.,1.].into_iter().enumerate().map(|(i,value)|
        MassJet {values:[c(value),c(if i==4 {1.} else {0.})]}).collect();
    for wrong_shape in [false,true] {
        let before=workspace.reduce_typed_dual_batch(&input,1).unwrap();
        assert_eq!(before.statuses,[LaneStatus::Success]);
        assert!(before.lane(0).unwrap()[7].re!=0.,"genuine nonzero A0 mass derivative must be evaluated");
        assert!(workspace.last_execution().master_rows>0);
        assert!(workspace.last_profile().unwrap().total>Duration::ZERO);
        let error=if wrong_shape {
            let dynamic:Vec<_>=input.iter().map(|v|HyperDual::from_values(
                vec![vec![0],vec![1],vec![2]],vec![v.values()[0],v.values()[1],c(0.)])).collect();
            workspace.reduce_typed_dual_batch(&dynamic,1).unwrap_err()
        } else {workspace.reduce_typed_dual_batch(&input[..5],1).unwrap_err()};
        assert!(matches!(error,ReductionError::InvalidInput(_)));
        reset_observations(&workspace,1);
        let recovered=workspace.reduce_typed_dual_batch(&input,1).unwrap();
        assert_eq!(recovered.lane(0).unwrap(),before.lane(0).unwrap());
    }
}
