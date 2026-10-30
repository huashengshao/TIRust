//! Constructor/metadata checks only. These add no native integral capture or
//! high-point evaluator construction to the default focused test set.
use super::*;
use crate::{Algorithm,Convention,InputPolicy,InvariantInput,TensorWeights};
use symbolica::domains::dual::HyperDual;

#[test]
fn affine_constructor_has_dynamic_point_count_preserves_semantics_and_avoids_generic_graph() {
    let mut settings=ReductionSettings::new(17,6);
    settings.powers[0]=4;settings.powers[16]=5;
    settings.dimension_shift=1;settings.algorithm=Algorithm::Ibp;
    settings.convention=Convention::MadLoop;settings.tensor_weights=TensorWeights::Multinomial;
    let before=serde_json::to_value(&settings).unwrap();
    let plan=TensorReductionPlan::new(settings).unwrap();
    assert_eq!(plan.engine_kind(),TensorEngineKind::AffinePower);
    assert_eq!(serde_json::to_value(plan.settings()).unwrap(),before);
    assert_eq!(plan.parameter_count(),86);
    assert_eq!(plan.input_layout().shifts,0..68);
    assert_eq!(plan.input_layout().masses,68..85);
    assert_eq!(plan.input_layout().scale,85);
    assert_eq!(plan.basis().len(),210);
    assert!(plan.generic.get().is_none());
    let support=[Monomial([0;4]),Monomial([1,1,0,0])];
    let contraction=plan.contraction_plan(&support).unwrap();
    assert_eq!(contraction.monomials(),support);
    assert!(plan.contraction_plan(&[Monomial([7,0,0,0])]).is_err());
    assert!(plan.generic.get().is_none());
    let selected=plan.affine_plan().unwrap();
    assert!(Arc::ptr_eq(&selected,&plan.affine_plan().unwrap()));
    assert!(plan.generic.get().is_none());
    let cloned=plan.clone();assert_eq!(cloned.engine_kind(),TensorEngineKind::AffinePower);
    assert!(Arc::ptr_eq(&plan.generic,&cloned.generic));
    assert!(Arc::ptr_eq(&plan.affine,&cloned.affine));
}

#[test]
fn affine_selection_keeps_existing_fastpaths_and_has_explicit_input_domain() {
    let mut six=ReductionSettings::new(6,6);six.algorithm=Algorithm::PassarinoVeltman;
    assert_eq!(ordinary_kind(&six),TensorEngineKind::SixPoint);
    six.algorithm=Algorithm::Hybrid;
    assert_eq!(ordinary_kind(&six),TensorEngineKind::SixPointHybrid);
    six.powers[5]=3;
    assert_eq!(ordinary_kind(&six),TensorEngineKind::AffinePower);
    let mut seven=ReductionSettings::new(7,0);seven.algorithm=Algorithm::PassarinoVeltman;
    assert_eq!(ordinary_kind(&seven),TensorEngineKind::SevenPoint);
    seven.dimension_shift=-2;
    assert_eq!(ordinary_kind(&seven),TensorEngineKind::AffinePower);
    for n in [8,17] {
        let mut s=ReductionSettings::new(n,6);s.powers[n-1]=4;
        assert_eq!(ordinary_kind(&s),TensorEngineKind::AffinePower);
        let mut bad=s.clone();bad.powers[n-1]=0;assert_eq!(ordinary_kind(&bad),TensorEngineKind::Generic);
        let mut bad=s.clone();bad.powers.pop();assert_eq!(ordinary_kind(&bad),TensorEngineKind::Generic);
        let mut bad=s.clone();bad.rank=7;assert_eq!(ordinary_kind(&bad),TensorEngineKind::Generic);
        let mut bad=s.clone();bad.invariants=InvariantInput::SuppliedScalarProducts;
        assert_eq!(ordinary_kind(&bad),TensorEngineKind::Generic);
        let mut bad=s.clone();bad.input_policy=InputPolicy::legacy_onshell();
        assert_eq!(ordinary_kind(&bad),TensorEngineKind::Generic);
    }
    let mut low=ReductionSettings::new(5,0);low.powers[0]=2;
    assert_eq!(ordinary_kind(&low),TensorEngineKind::Generic);
}

#[test]
fn affine_explicit_dual_rescue_is_rejected_before_unrelated_generic_preparation() {
    let mut settings=ReductionSettings::new(6,6);settings.powers[0]=2;
    let plan=TensorReductionPlan::new(settings).unwrap();
    assert!(plan.generic.get().is_none());
    let dual=HyperDual::<f64>::new(vec![vec![0],vec![1]]);
    assert!(matches!(plan.rescue_dual_workspace(0.,RescueSettings::default(),dual),
        Err(ReductionError::Unsupported(reason)) if reason.contains("joint-context")));
    assert!(plan.generic.get().is_none());
}


#[test]
fn thin_ordinary_plan_caches_an_affine_capability_recipe_without_generic_preparation() {
    // Plan construction only: no N6 numerical workspace or evaluator is built.
    let mut settings=ReductionSettings::new(6,0);
    settings.algorithm=Algorithm::PassarinoVeltman;
    let plan=TensorReductionPlan::new_without_direct(settings).unwrap();
    assert_eq!(plan.engine_kind(),TensorEngineKind::SixPoint);
    assert!(plan.generic.get().is_none());assert!(plan.affine.get().is_none());
    let affine=plan.affine_plan().unwrap();
    assert_eq!(affine.settings().powers,[1;6]);
    assert_eq!(affine.settings().algorithm,Algorithm::PassarinoVeltman);
    assert!(Arc::ptr_eq(&affine,&plan.affine_plan().unwrap()));
    assert_eq!(plan.engine_kind(),TensorEngineKind::SixPoint); // compatible ordinary path unchanged
    let dual=HyperDual::<f64>::new(vec![vec![0],vec![1]]);
    assert!(matches!(plan.rescue_dual_workspace(0.,RescueSettings::default(),dual),
        Err(ReductionError::Unsupported(reason)) if reason.contains("joint-context")));
    assert!(plan.generic.get().is_none());
}
