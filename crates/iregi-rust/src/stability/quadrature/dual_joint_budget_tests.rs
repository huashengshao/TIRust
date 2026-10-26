//! Joint-budget regressions, not synthetic numerical integral oracles.
use super::*;
use crate::{Algorithm, ReductionSettings};
fn c(x: f64) -> Complex<f64> { Complex::new(x, 0.) }
fn fixture() -> (StabilityPlan, QuadratureStabilityPlan) {
    let mut settings = ReductionSettings::new(1, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    let plan = ReductionPlan::new(settings).unwrap();
    (StabilityPlan::prepare(&plan).unwrap(),
     QuadratureStabilityPlan::imported_dense_checks(&plan).unwrap())
}
fn input(components: usize) -> Vec<Complex<f64>> {
    let mut input = vec![c(0.); 6 * components];
    input[4 * components] = c(1.); input[5 * components] = c(1.); input
}
#[test]
fn full_component_joint_budget_rejects_two_individually_passing_contributions() {
    let (stability, checks) = fixture(); let components = 6; let input = input(components);
    let output = vec![c(0.); 24]; let primal = vec![c(0.); 4];
    // target=2^-34; each contribution is exactly3/4 target and separately passes.
    let error = dyadic(&0f64, -36) * 3.; let amplitude = dyadic(&0f64, 16) * 3.;
    for slot in 0..4 { for component in 0..components {
        let index = slot * components + component;
        let mut errors = output.clone(); errors[index] = c(error);
        let mut dense = output.clone();
        assert!(stability.inspect_quadrature_lane_components(&checks,&errors,&output,
            &input,components,&0.,53,Some(&primal),&dense).is_none());
        dense[index] = c(amplitude);
        assert!(stability.inspect_quadrature_lane_components(&checks,&output,&output,
            &input,components,&0.,53,Some(&primal),&dense).is_none());
        let failure = stability.inspect_quadrature_lane_components(&checks,&errors,&output,
            &input,components,&0.,53,Some(&primal),&dense).unwrap();
        assert!(failure.contains(&format!("Laurent slot {slot}")));
        assert!(failure.contains(&format!("Taylor component {component}")));
    }}
}
#[test]
fn component_zero_uses_worse_roundoff_once_without_weakening_legacy_budget() {
    let (stability,checks)=fixture();let components=6;let input=input(components);
    let output=vec![c(0.);24];let mut errors=output.clone();
    let mut primal=vec![c(0.);4];let mut dense=output.clone();
    let half_amplitude=dyadic(&0f64,17);errors[18]=c(dyadic(&0f64,-35));
    primal[3]=c(half_amplitude);dense[18]=c(half_amplitude);
    let inspect=|primal:&[Complex<f64>],dense:&[Complex<f64>]|stability
        .inspect_quadrature_lane_components(&checks,&errors,&output,&input,components,&0.,53,Some(primal),dense);
    assert!(inspect(&primal,&dense).is_none()); // equal budget, not double-counted.
    primal[3]=c(half_amplitude*1.5);dense[18]=c(0.);
    assert!(inspect(&primal,&dense).unwrap().contains("Taylor component 0"));
    primal[3]=c(0.);dense[18]=c(half_amplitude*1.5);
    assert!(inspect(&primal,&dense).unwrap().contains("Taylor component 0"));
}
#[test]
fn sparse_checks_use_output_index_and_cached_normalization_matches_uncached() {
    let(stability,mut checks)=fixture();checks.checks.retain(|check|check.output==3);
    let components=6;let input=input(components);let output=vec![c(0.);24];
    let primal=vec![c(0.);4];let mut dense=output.clone();let errors=vec![c(0.);6];
    dense[4]=c(f64::MAX); // unchecked UV slot cannot stand in for finite slot.
    assert!(stability.inspect_quadrature_lane_components(&checks,&errors,&output,
        &input,components,&0.,53,Some(&primal),&dense).is_none());
    dense[22]=c(dyadic(&0f64,19));
    let uncached=stability.inspect_quadrature_lane_components(&checks,&errors,&output,
        &input,components,&0.,53,Some(&primal),&dense).unwrap();
    assert!(uncached.contains("Laurent slot 3")&&uncached.contains("Taylor component 4"));
    let mut context=GuardContext::new(&stability,&0.,53);context.begin_lane();
    assert_eq!(Some(uncached),stability.inspect_quadrature_lane_components_context(&checks,&errors,&output,
        &input,components,&0.,53,Some(&primal),&dense,&mut context));
}
#[test]
fn dense_amplitudes_require_complete_finite_nonnegative_output_layout() {
    let(stability,checks)=fixture();let components=6;let input=input(components);
    let output=vec![c(0.);24];let primal=vec![c(0.);4];
    let inspect=|dense:&[Complex<f64>]|stability.inspect_quadrature_lane_components(
        &checks,&output,&output,&input,components,&0.,53,Some(&primal),dense);
    assert!(inspect(&output[..23]).unwrap().contains("full-component"));
    for value in [c(-1.),c(f64::NAN),c(f64::INFINITY),Complex::new(0.,1.)]{
        let mut dense=output.clone();dense[22]=value;
        assert!(inspect(&dense).unwrap().contains("full-component"));
    }
}
#[test]
fn higher_component_roundoff_retains_operand_precision_and_zero_metadata() {
    use symbolica::domains::float::Float;
    let(stability,checks)=fixture();let prototype=Float::new(256);let components=6;
    let number=|bits,value|Complex::new(Float::with_val(bits,value),Float::new(bits));
    let mut input=vec![number(256,0);36];input[24]=number(256,1);input[30]=number(256,1);
    let output=vec![number(256,0);24];let primal=vec![number(256,0);4];
    let inspect=|bits,value|{
        let mut dense=output.clone();dense[22]=number(bits,value);
        stability.inspect_quadrature_lane_components(&checks,&output,&output,&input,
            components,&prototype,128,Some(&primal),&dense)
    };
    assert!(inspect(80,1).unwrap().contains("Taylor component 4"));
    assert!(inspect(256,1).is_none());assert!(inspect(1,0).is_none());
}

#[test]
fn higher_component_combined_budget_keeps_d6_output_dimensions() {
    let(stability,mut checks)=fixture();
    // A dimensioned guard fixture: scale2, natural2^(-2)=1/4. This directly
    // distinguishes using the stored output dimension from a universal1.
    for check in &mut checks.checks { check.dimension=-2; }
    let components=3;let mut input=input(components);input[4*components]=c(4.);
    let output=vec![c(0.);12];let primal=vec![c(0.);4];let mut errors=output.clone();
    let mut dense=output.clone();
    errors[10]=c(dyadic(&0f64,-38)*3.);dense[10]=c(dyadic(&0f64,14)*3.);
    let reason=stability.inspect_quadrature_lane_components(&checks,&errors,&output,&input,
        components,&0.,53,Some(&primal),&dense).unwrap();
    assert!(reason.contains("Taylor component 1"));
    for check in &mut checks.checks { check.dimension=0; }
    assert!(stability.inspect_quadrature_lane_components(&checks,&errors,&output,&input,
        components,&0.,53,Some(&primal),&dense).is_none());
}
