//! Embedder mode of the facade (`TensorWorkspace::new_embedded`): the direct
//! engine at the workspace's precision with conditioning gates, then only the
//! configured method choice, partial rescue and precision tiers. No
//! loop-rotation test, no rescue-first routing of refused lanes (the standalone
//! facade's `PrecisionRescue::Automatic` sends a refused box straight to its
//! double-double tier) and never the symbolic engine.
use iregi_rust::{
    Algorithm, Convention, DirectReductionPlan, EmbedderOptions, InvariantInput, KinematicsBatch, LaneRoute,
    LaneStatus, PrecisionRescue, ReductionError, ReductionSettings, ScaleInput, TensorEngineExecution,
    TensorReductionPlan, TensorWeights, TensorWorkspace, TensorWorkspaceOptions,
};
use symbolica::domains::float::{Complex, DoubleFloat, Float, Real, RealLike, SingleFloat};

fn row(shifts: &[[f64; 4]], masses: &[(f64, f64)], mu: f64) -> Vec<Complex<f64>> {
    let mut values: Vec<Complex<f64>> = shifts.iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
    values.extend(masses.iter().map(|&(re, im)| Complex::new(re, im)));
    values.push(Complex::new(mu, 0.0));
    values
}

fn batch<T: symbolica::domains::float::SingleFloat>(values: Vec<Complex<T>>) -> KinematicsBatch<T> {
    KinematicsBatch { rows: 1, components: 1, values }
}

fn bits(values: &[Complex<f64>]) -> Vec<(u64, u64)> {
    values.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
}

/// gg -> HH top-loop box at sqrt(s) = 1 TeV whose Gram systems the binary64
/// gate refuses (the facade's `GATE_REFUSED_BOX`, `tests/direct_engine.rs`).
const GATE_REFUSED_BOX: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [500.0, 0.0, 0.0, 500.0],
    [1000.0, 0.0, 0.0, 0.0],
    [499.99999999999994, 21.6322892694395, -42.18555103069406, 481.7960391547444],
];

/// Small Higgs pT gg -> HH box (box-r4 #5046): PV keeps only E/A 1e-3 in
/// binary64, the small-Gram expansion recovers it.
const SMALL_GRAM_BOX: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [500.0, 0.0, 0.0, 500.0],
    [1000.0, 0.0, 0.0, 0.0],
    [499.9999999999999, 8.840408903567376, -6.981798561273453, -483.99184048831603],
];

fn box_row(shifts: &[[f64; 4]; 4]) -> Vec<Complex<f64>> {
    let mt2 = 173.0f64 * 173.0;
    row(shifts, &[(mt2, 0.0); 4], 1000.0)
}

/// Array at 256 bits (gate 2^-216) of the same exactly lifted inputs, rounded.
fn reference_256(settings: &ReductionSettings, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let lifted: Vec<Complex<Float>> =
        values.iter().map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im))).collect();
    let mut workspace = DirectReductionPlan::new(settings.clone()).unwrap().workspace(Float::new(256)).unwrap();
    workspace.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    let result = workspace.reduce_batch(&batch(lifted)).unwrap();
    result.lane(0).unwrap().iter().map(|z| Complex::new(z.re.to_f64(), z.im.to_f64())).collect()
}

/// E/A with moments weighted by mu^-(rank + 4 - 2N).
fn array_error(basis: &[iregi_rust::basis::Monomial], n: usize, mu: f64, a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    let (mut error, mut amplitude) = (0.0f64, 0.0f64);
    for (m, monomial) in basis.iter().enumerate() {
        let weight = mu.powi(monomial.rank() as i32 + 4 - 2 * n as i32);
        for slot in 0..4 {
            let (x, y) = (a[4 * m + slot], b[4 * m + slot]);
            error = error.max((x.re - y.re).hypot(x.im - y.im) / weight);
            amplitude = amplitude.max(y.re.hypot(y.im) / weight);
        }
    }
    error / amplitude
}

fn stats(workspace: &TensorWorkspace<f64>) -> iregi_rust::DirectExecutionStats {
    let TensorEngineExecution::Direct(stats) = workspace.engine_execution() else {
        panic!("expected the direct engine");
    };
    stats
}

#[test]
fn a_refused_box_gets_the_method_choice_not_the_rescue_first_route() {
    let settings = ReductionSettings::new(4, 6);
    let values = box_row(&GATE_REFUSED_BOX);
    let plan = TensorReductionPlan::new(settings.clone()).unwrap();
    // The standalone facade (precision rescue on, huashengshao/TIRust#4):
    // the refused box goes to the double-double tier directly.
    let mut standalone = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    standalone.reduce_batch(&batch(values.clone())).unwrap();
    let standalone_stats = stats(&standalone);
    assert_eq!(standalone_stats.precision_rescue.first().map(|tier| tier.bits), Some(106), "{standalone_stats:?}");
    assert_eq!(standalone_stats.expansion_rows, 0, "{standalone_stats:?}");
    // Embedder mode: gates, then the a priori method choice (unchecked,
    // flagged), no tier, no rotation test, no symbolic engine.
    let mut embedded = TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()).unwrap();
    assert_eq!(embedded.engine_kind(), iregi_rust::TensorEngineKind::Direct);
    let result = embedded.reduce_batch(&batch(values.clone())).unwrap();
    let embedded_stats = stats(&embedded);
    assert_eq!(embedded_stats.expansion_rows, 1, "{embedded_stats:?}");
    assert!(embedded_stats.precision_rescue.is_empty(), "{embedded_stats:?}");
    assert_eq!((embedded_stats.rotation_rows, embedded_stats.symbolic_rows), (0, 0), "{embedded_stats:?}");
    assert_eq!(result.statuses[0], LaneStatus::Success, "{:?}", embedded.last_diagnostics());
    assert_eq!(embedded.last_lane_routes(), Some(&[LaneRoute::MethodChoice][..]));
    assert!(result.fallback_used[0] && result.estimated_errors[0].is_none());
    let reference = reference_256(&settings, &values);
    let error = array_error(&result.basis, 4, 1000.0, result.lane(0).unwrap(), &reference);
    println!("gate-refused box, embedder method choice: E/A {error:e}");
    assert!(error <= 1e-6, "E/A = {error:e}");
    // Without the method choice the lane stays refused: nothing escalates.
    let options = EmbedderOptions { method_choice: false, ..Default::default() };
    let mut embedded = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
    let result = embedded.reduce_batch(&batch(values.clone())).unwrap();
    assert!(matches!(result.statuses[0], LaneStatus::Unstable(_)), "{:?}", result.statuses[0]);
    assert_eq!(embedded.last_lane_routes(), Some(&[LaneRoute::Refused][..]));
    let refused_stats = stats(&embedded);
    assert_eq!((refused_stats.expansion_rows, refused_stats.fallback_rows, refused_stats.symbolic_rows), (0, 0, 0));
    assert!(!embedded.last_diagnostics()[0].is_empty());
    // Precision tiers only when asked for (no rotation check in this mode).
    let options = EmbedderOptions { method_choice: false, precision_tiers: true, ..Default::default() };
    let mut embedded = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
    let result = embedded.reduce_batch(&batch(values.clone())).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success, "{:?}", embedded.last_diagnostics());
    let Some(&[LaneRoute::Tier(tier_bits)]) = embedded.last_lane_routes() else { panic!() };
    let error = array_error(&result.basis, 4, 1000.0, result.lane(0).unwrap(), &reference);
    println!("gate-refused box, embedder tier {tier_bits} bits: E/A {error:e}");
    assert!(error <= 1e-12, "E/A = {error:e}");
    assert_eq!(stats(&embedded).symbolic_rows, 0);
}

/// A massless rank-4 pentagon of MadLoop's QCD u u~ > z g g g at 1 TeV
/// (shifts and the supplied invariants s_ij = (p_i - p_j)^2, column-major, as
/// MadLoop's TIRust bridge passes them; mu^2 = 10^6). The binary64 gates and
/// the method choice refuse it, with a conditioning estimate inside the
/// default 8-bit margin above the double-double tier's limit 2^-66.
const ZGGG_PENTAGON_SHIFTS: [u64; 20] = [
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x407f40000000000a,
    0xbd00000000000000, 0x3d36000000000000, 0x407f40000000000a, 0x4055945286e7a19c, 0xc06bd4785c87a13c,
    0xc07594ea4e9ead65, 0x408121f7eecf071e, 0xc0432f7033be507e, 0xc0692d49abfa7a3e, 0xc077fcc682af1d94,
    0x4083685e2c7546a3, 0xc0671604ba4c69f8, 0xc058eedca3c58e8a, 0xc0721f6791e15459, 0x4081ebb254e902cc,
];
const ZGGG_PENTAGON_INVARIANTS: [u64; 25] = [
    0x0000000000000000, 0x0000000000000000, 0xc11c31a2187cccfa, 0xc121757e578777e7, 0xc117ba97fedfc9b2,
    0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x40f55196c85c661f, 0x41168b809dee9f90,
    0xc11c31a2187cccfa, 0x0000000000000000, 0x0000000000000000, 0x40c03da02c0a4a08, 0x40eaadde2f25ae46,
    0xc121757e578777e7, 0x40f55196c85c661f, 0x40c03da02c0a4a08, 0x0000000000000000, 0x0000000000000000,
    0xc117ba97fedfc9b2, 0x41168b809dee9f90, 0x40eaadde2f25ae46, 0x0000000000000000, 0x0000000000000000,
];

#[test]
fn a_zero_tier_margin_starts_near_limit_lanes_in_double_double() {
    let n = 5;
    let mut settings = ReductionSettings::new(n, 4);
    settings.convention = Convention::MadLoop;
    settings.tensor_weights = TensorWeights::Components;
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    let invariants = ZGGG_PENTAGON_INVARIANTS.map(f64::from_bits);
    let mut values: Vec<Complex<f64>> =
        ZGGG_PENTAGON_SHIFTS.iter().map(|&x| Complex::new(f64::from_bits(x), 0.0)).collect();
    for i in 0..n {
        for j in i + 1..n {
            values.push(Complex::new(invariants[i + n * j], 0.0));
        }
    }
    values.extend((0..n).map(|_| Complex::new(0.0, 0.0)));
    values.push(Complex::new(1e6, 0.0));
    let plan = TensorReductionPlan::new(settings).unwrap();
    assert_eq!(EmbedderOptions::default().tier_margin_bits, 8);
    let run = |tier_margin_bits| {
        let options = EmbedderOptions { precision_tiers: true, tier_margin_bits, ..Default::default() };
        let mut embedded = TensorWorkspace::new_embedded(&plan, 0.0f64, options).unwrap();
        let result = embedded.reduce_batch(&batch(values.clone())).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success, "{:?}", embedded.last_diagnostics());
        let route = embedded.last_lane_routes().unwrap()[0];
        (result.lane(0).unwrap().to_vec(), route, result.basis.clone())
    };
    // The default margin skips the double-double tier for this estimate;
    // without it, the double-double tier's own gate accepts the lane.
    let (skipped, skipped_route, basis) = run(8);
    let (direct, direct_route, _) = run(0);
    assert_eq!(skipped_route, LaneRoute::Tier(256));
    assert_eq!(direct_route, LaneRoute::Tier(106));
    // Both tiers bound the amplification of their rounding by 2^-40.
    let error = array_error(&basis, n, 1000.0, &direct, &skipped);
    println!("near-limit pentagon: double-double against 256-bit tier E/A {error:e}");
    assert!(error <= 2f64.powi(-40), "E/A = {error:e}");
}

#[test]
fn the_method_choice_keeps_a_small_gram_box() {
    let settings = ReductionSettings::new(4, 4);
    let values = box_row(&SMALL_GRAM_BOX);
    let plan = TensorReductionPlan::new(settings.clone()).unwrap();
    let mut embedded = TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()).unwrap();
    let result = embedded.reduce_batch(&batch(values.clone())).unwrap();
    let routes = embedded.last_lane_routes().unwrap().to_vec();
    let reference = reference_256(&settings, &values);
    let error = array_error(&result.basis, 4, 1000.0, result.lane(0).unwrap(), &reference);
    println!("small-Gram box, embedder {routes:?}: E/A {error:e}");
    assert_eq!(result.statuses[0], LaneStatus::Success);
    assert!(error <= 1e-10, "E/A = {error:e}");
}

#[test]
fn gated_lanes_equal_the_direct_workspace_bitwise() {
    // A batch of boxes and a pentagon the gates accept: the embedder's lanes
    // are the plain direct workspace's, bit for bit (no rotation test).
    let mt2 = 173.0f64 * 173.0;
    let rows = [
        [[0.0, 0.0, 0.0, 0.0], [500.0, 0.0, 0.0, 500.0], [1000.0, 0.0, 0.0, 0.0], [600.0, 100.0, -50.0, 300.0]],
        [[0.0, 0.0, 0.0, 0.0], [300.0, 10.0, 20.0, 250.0], [700.0, -40.0, 60.0, 100.0], [900.0, 30.0, -80.0, -90.0]],
    ];
    for rank in [0u16, 3, 4] {
        let settings = ReductionSettings::new(4, rank);
        let mut values = Vec::new();
        for shifts in &rows {
            values.extend(row(shifts, &[(mt2, 0.0); 4], 1000.0));
        }
        let batch = KinematicsBatch { rows: rows.len(), components: 1, values };
        let expected = DirectReductionPlan::new(settings.clone()).unwrap().workspace(0.0f64).unwrap().reduce_batch(&batch).unwrap();
        let plan = TensorReductionPlan::new(settings).unwrap();
        let mut embedded = TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()).unwrap();
        let got = embedded.reduce_batch(&batch).unwrap();
        assert_eq!(got.statuses, expected.statuses);
        for lane in 0..rows.len() {
            assert_eq!(expected.statuses[lane], LaneStatus::Success, "rank {rank} lane {lane}");
            assert_eq!(bits(got.lane(lane).unwrap()), bits(expected.lane(lane).unwrap()));
        }
        assert!(embedded.last_lane_routes().unwrap().iter().all(|&route| route == LaneRoute::Direct));
        assert_eq!(stats(&embedded).rotation_rows, 0);
    }
}

#[test]
fn double_double_and_mpfr_run_on_the_direct_engine() {
    // A massive box in double-double and at 256 bits through the embedder
    // facade (supplied invariants, mu^2 input): the direct engine's values
    // bit for bit, double-double within 1e-24 of MPFR. The standalone facade
    // runs these precisions on the symbolic engine.
    let mt2 = 173.0f64 * 173.0;
    let shifts = [[0.0, 0.0, 0.0, 0.0], [500.0, 0.0, 0.0, 500.0], [1000.0, 0.0, 0.0, 0.0], [600.0, 100.0, -50.0, 300.0]];
    let mut settings = ReductionSettings::new(4, 3);
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    fn lifted<T: iregi_masters::NativeFloat>(prototype: &T, shifts: &[[f64; 4]; 4], mt2: f64) -> Vec<Complex<T>> {
        let lift = |x: f64| prototype.from_rational(&Float::with_val(53, x).try_to_rational().unwrap());
        let real = |x: T| Complex::new(x, prototype.zero());
        let q: Vec<[T; 4]> = shifts.iter().map(|s| s.map(lift)).collect();
        let mut values: Vec<Complex<T>> = q.iter().flatten().cloned().map(real).collect();
        for i in 0..4 {
            for j in i + 1..4 {
                let d: Vec<T> = (0..4).map(|a| q[i][a].clone() - &q[j][a]).collect();
                let s = d[0].clone() * &d[0] - d[1].clone() * &d[1] - d[2].clone() * &d[2] - d[3].clone() * &d[3];
                values.push(real(s));
            }
        }
        values.extend((0..4).map(|_| real(lift(mt2))));
        values.push(real(lift(1.0e6)));
        values
    }
    let plan = TensorReductionPlan::new(settings.clone()).unwrap();
    let dd_values = lifted(&DoubleFloat::from(0.0), &shifts, mt2);
    let mut embedded = TensorWorkspace::new_embedded(&plan, DoubleFloat::from(0.0), EmbedderOptions::default()).unwrap();
    assert_eq!(embedded.binary_precision(), 106);
    let dd = embedded.reduce_batch(&batch(dd_values.clone())).unwrap();
    let direct = DirectReductionPlan::new(settings.clone())
        .unwrap()
        .workspace(DoubleFloat::from(0.0))
        .unwrap()
        .reduce_batch(&batch(dd_values))
        .unwrap();
    assert_eq!(dd.statuses, vec![LaneStatus::Success]);
    for (a, b) in dd.lane(0).unwrap().iter().zip(direct.lane(0).unwrap()) {
        assert!(a.re == b.re && a.im == b.im);
    }
    let TensorEngineExecution::Direct(dd_stats) = embedded.engine_execution() else { panic!() };
    assert_eq!(dd_stats.symbolic_rows, 0);
    let mut embedded = TensorWorkspace::new_embedded(&plan, Float::new(256), EmbedderOptions::default()).unwrap();
    let mpfr = embedded.reduce_batch(&batch(lifted(&Float::new(256), &shifts, mt2))).unwrap();
    assert_eq!(mpfr.statuses, vec![LaneStatus::Success]);
    let (mut error, mut scale) = (Float::with_val(256, 0), Float::with_val(256, 0));
    for (a, b) in dd.lane(0).unwrap().iter().zip(mpfr.lane(0).unwrap()) {
        let to = |x: &DoubleFloat| Float::with_val(256, 0).from_rational(&iregi_masters::NativeFloat::exact_rational(x).unwrap());
        let (dr, di) = (to(&a.re) - &b.re, to(&a.im) - &b.im);
        let e = (dr.clone() * &dr + di.clone() * &di).sqrt();
        let s = (b.re.clone() * &b.re + b.im.clone() * &b.im).sqrt();
        if e > error {
            error = e;
        }
        if s > scale {
            scale = s;
        }
    }
    let relative = (error / &scale).to_f64();
    println!("embedder double-double vs MPFR(256): {relative:e}");
    assert!(relative <= 1e-24, "{relative:e}");
}

#[test]
fn embedder_mode_refuses_what_the_direct_engine_does_not_serve() {
    let mut settings = ReductionSettings::new(3, 2);
    settings.algorithm = Algorithm::Ibp;
    let plan = TensorReductionPlan::new(settings).unwrap();
    assert!(matches!(
        TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()),
        Err(ReductionError::Unsupported(_))
    ));
    let plan = TensorReductionPlan::new(ReductionSettings::new(3, 2)).unwrap();
    let options = EmbedderOptions { partial_rescue: true, ..Default::default() };
    assert!(matches!(
        TensorWorkspace::new_embedded(&plan, DoubleFloat::from(0.0), options),
        Err(ReductionError::Unsupported(_))
    ));
    // The standalone facade keeps its own behaviour (rotation test, rescue).
    let workspace = TensorWorkspace::new_with_options(
        &plan,
        0.0f64,
        TensorWorkspaceOptions { precision_rescue: PrecisionRescue::Disabled, ..Default::default() },
    )
    .unwrap();
    assert!(workspace.last_lane_routes().is_none() && workspace.embedder_options().is_none());
}
