//! Both explicit SixPoint constructor routes pin their children to Symbolica.
use super::*;
use symbolica::domains::{dual::HyperDual, rational::Rational};

#[test]
fn composition_six_point_constructors_pin_symbolica_with_no_backend_option() {
    let mut settings = ReductionSettings::new(6, 0);
    settings.algorithm = Algorithm::PassarinoVeltman;
    let plan = ThinSixReductionPlan::new(settings).unwrap();
    let ordinary = ThinSixWorkspace::new(&plan, 0.).unwrap();
    let dual = ThinSixWorkspace::new_dual(
        &plan, 0., HyperDual::<Complex<Rational>>::new(vec![vec![0]]),
    ).unwrap();
    for workspace in [&ordinary, &dual] {
        let ChildWorkspace::Ordinary(child)=&workspace.child else {panic!("ordinary constructor changed child engine")};
        assert!(crate::execution::policy_composition_tests::assert_context(
            child, WorkspacePolicy::SIX_POINT_CHILD,
            MasterExecutionPolicy::Symbolica, true,
        ) > 0);
        crate::execution::policy_composition_tests::assert_context(
            &workspace.parent, WorkspacePolicy::RETAIN_ENVELOPES,
            MasterExecutionPolicy::Symbolica, true,
        );
    }
    let ChildWorkspace::Ordinary(ordinary_child)=&ordinary.child else {panic!("ordinary child")};
    let ChildWorkspace::Ordinary(dual_child)=&dual.child else {panic!("dual child")};
    assert_eq!(ordinary_child.dual_shape(), None);
    assert_eq!(ordinary.parent.dual_shape(), None);
    assert_eq!(dual_child.dual_shape(), Some([vec![0]].as_slice()));
    assert_eq!(dual.parent.dual_shape(), Some([vec![0]].as_slice()));
}
