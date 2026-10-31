//! COLLIER-style regulators against COLLIER 1.2.10 itself: the massless
//! on-shell box D0 (s = 10, t = -3, mu^2 = 2), whose double and single IR
//! poles test the scale shift and the c(eps) normalization. COLLIER values
//! from `D0_cll` with `SetDeltaIR_cll` varied (finite part, then the
//! coefficients of Delta_IR1 and Delta_IR2), `SetMuUV2_cll(2)`.
use iregi_rust::{
    Convention, KinematicsBatch, LaneStatus, ReductionSettings, RegulatorNormalization, Regulators,
    TensorReductionPlan, TensorWorkspace,
};
use symbolica::domains::float::Complex;

/// q_0 = 0, q_1 = k1, q_2 = k1 + k2, q_3 = k1 + k2 + k3 with lightlike
/// k_i, s = (k1 + k2)^2 = 10 and t = (k2 + k3)^2 = -3.
fn massless_box() -> Vec<Complex<f64>> {
    let e = 10f64.sqrt() / 2.0;
    let cos = -0.4f64;
    let sin = (1.0 - cos * cos).sqrt();
    let k1 = [e, 0.0, 0.0, e];
    let k2 = [e, 0.0, 0.0, -e];
    let k3 = [-e, -e * sin, 0.0, -e * cos];
    let add = |a: [f64; 4], b: [f64; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let shifts = [[0.0; 4], k1, add(k1, k2), add(add(k1, k2), k3)];
    let mut values: Vec<Complex<f64>> = shifts.iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
    values.extend([Complex::new(0.0, 0.0); 4]);
    values.push(Complex::new(2f64.sqrt(), 0.0));
    values
}

fn close(a: Complex<f64>, b: (f64, f64), what: &str) {
    let error = (a.re - b.0).hypot(a.im - b.1) / b.0.hypot(b.1);
    assert!(error < 1e-11, "{what}: {a:?} against COLLIER {b:?} (relative {error:e})");
}

#[test]
fn collier_regulators_reproduce_collier_on_an_ir_divergent_box() {
    let mut settings = ReductionSettings::new(4, 0);
    settings.convention = Convention::OneLoop;
    let plan = TensorReductionPlan::new(settings).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    let batch = KinematicsBatch { rows: 1, components: 1, values: massless_box() };
    // (mu_IR^2, COLLIER finite part, coefficient of Delta_IR1).
    let cases = [
        (2.0, (0.50480662780551067, 0.084920413661306690), (0.13432686803615101, -0.20943951023931953)),
        (5.0, (0.57191651167415347, -0.10698706845924401), (0.012154770452930341, -0.20943951023931956)),
    ];
    for (mu_ir2, finite, single) in cases {
        workspace.set_regulators(Regulators {
            mu_ir_squared: Some(mu_ir2),
            normalization: RegulatorNormalization::Collier,
            ..Regulators::default()
        });
        let result = workspace.reduce_batch(&batch).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        let slots = result.lane(0).unwrap();
        close(slots[3], finite, "finite part");
        close(slots[1], single, "IR single pole");
        close(slots[2], (-0.13333333333333336, 0.0), "IR double pole");
        assert_eq!(slots[0], Complex::new(0.0, 0.0), "no UV pole");
        // Numerical poles as COLLIER's SetDeltaIR_cll(1, 2): the finite slot
        // gains dIR1 + 2 dIR2.
        workspace.set_regulators(Regulators {
            mu_ir_squared: Some(mu_ir2),
            delta_ir1: 1.0,
            delta_ir2: 2.0,
            normalization: RegulatorNormalization::Collier,
            ..Regulators::default()
        });
        let result = workspace.reduce_batch(&batch).unwrap();
        let total = (finite.0 + single.0 + 2.0 * -0.13333333333333336, finite.1 + single.1);
        close(result.lane(0).unwrap()[3], total, "finite part with Delta_IR1 = 1, Delta_IR2 = 2");
    }
    // The default regulators change nothing; COLLIER normalization needs
    // Convention::OneLoop.
    workspace.set_regulators(Regulators::default());
    let plain = workspace.reduce_batch(&batch).unwrap();
    let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    close(plain.lane(0).unwrap()[3], (0.50480662780551067 + pi2_6 * -0.13333333333333336, 0.084920413661306690),
          "OneLoop finite part (COLLIER's plus pi^2/6 times the double pole)");
    let native = TensorReductionPlan::new(ReductionSettings::new(4, 0)).unwrap();
    let mut workspace = TensorWorkspace::new(&native, 0.0f64).unwrap();
    workspace.set_regulators(Regulators { normalization: RegulatorNormalization::Collier, ..Regulators::default() });
    assert!(workspace.reduce_batch(&batch).is_err());
}
