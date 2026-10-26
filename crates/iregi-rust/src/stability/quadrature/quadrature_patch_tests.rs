//! SOURCE-ONLY candidate tests for quadrature.private.patch; intended as a
//! cfg(test) child module of stability::quadrature. No tests were executed.
use super::*;
use crate::{Algorithm,ReductionSettings};
fn c(x:f64)->Complex<f64>{Complex::new(x,0.0)}

#[test]
fn dense_export_preserves_all_laurent_component_positions_and_structural_zeros() {
    let plan=QuadratureStabilityPlan {
        leaf_indices:vec![0],coefficient_parameters:vec![],coefficient_expressions:vec![],
        coefficient_aliases:vec![],contraction_parameters:vec![],contraction_expressions:vec![],
        checks:vec![OutputCheck{output:3,dimension:0,primal_amplitude:0,label:"finite".into()}],
        moment_outputs:4,primal_amplitudes_are_outputs:false,
    };
    let dense=plan.dense_output_errors(&[c(1.),c(2.),c(3.)],3,&c(0.)).unwrap();
    assert_eq!(dense,[c(0.),c(0.),c(0.),c(0.),c(0.),c(0.),c(0.),c(0.),c(0.),c(1.),c(2.),c(3.)]);
    assert!(plan.dense_output_errors(&[c(1.),c(2.)],3,&c(0.)).is_err());
    assert!(plan.dense_output_errors(&[c(1.),c(-1.),c(3.)],3,&c(0.)).is_err());
    assert!(plan.dense_output_errors(&[c(1.),c(f64::INFINITY),c(3.)],3,&c(0.)).is_err());
}

#[test]
fn imported_check_plan_accepts_dense_actual_route_amplitudes_and_checks_higher_components() {
    // A cheap structural graph fixture, not a fake D6 numerical oracle.
    let mut settings=ReductionSettings::new(1,0);settings.algorithm=Algorithm::PassarinoVeltman;
    let plan=ReductionPlan::new(settings).unwrap();
    let checks=QuadratureStabilityPlan::imported_dense_checks(&plan).unwrap();
    let stability=StabilityPlan::prepare(&plan).unwrap();
    let mut input=vec![c(0.);6*3];input[4*3]=c(1.);input[5*3]=c(1.);
    let output=vec![c(0.);4*3];let mut errors=vec![c(0.);4*3];
    let actual_route_amplitudes=vec![c(0.);4];
    assert!(stability.inspect_quadrature_lane(&checks,&errors,&output,&input,3,&0.,53,Some(&actual_route_amplitudes)).is_none());
    for slot in 0..4 {
        for component in 0..3 {
            errors[slot*3+component]=c(1.);
            let reason=stability.inspect_quadrature_lane(&checks,&errors,&output,&input,3,&0.,53,Some(&actual_route_amplitudes)).unwrap();
            assert!(reason.contains(&format!("Laurent slot {slot}")));
            assert!(reason.contains(&format!("Taylor component {component}")));
            errors[slot*3+component]=c(0.);
        }
    }
    // A missing actual-route amplitude inventory remains a failure even with
    // numerically zero errors; it is never interpreted as zero roundoff.
    assert!(stability.inspect_quadrature_lane(&checks,&errors,&output,&input,3,&0.,53,None).is_some());
}
