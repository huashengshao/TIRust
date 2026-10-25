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
        assert!(crate::execution::policy_composition_tests::assert_context(
            &workspace.child, WorkspacePolicy::SIX_POINT_CHILD,
            MasterExecutionPolicy::Symbolica, true,
        ) > 0);
        crate::execution::policy_composition_tests::assert_context(
            &workspace.parent, WorkspacePolicy::RETAIN_ENVELOPES,
            MasterExecutionPolicy::Symbolica, true,
        );
    }
    assert_eq!(ordinary.child.dual_shape(), None);
    assert_eq!(ordinary.parent.dual_shape(), None);
    assert_eq!(dual.child.dual_shape(), Some([vec![0]].as_slice()));
    assert_eq!(dual.parent.dual_shape(), Some([vec![0]].as_slice()));
}
