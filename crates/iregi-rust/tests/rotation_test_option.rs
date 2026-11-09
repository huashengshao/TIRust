//! `RotationTest` (huashengshao/TIRust#24): the default is the explicit
//! `Enabled` bit for bit; `Disabled` skips only the binary64 first pass's
//! loop-rotation comparison (its gate-accepted lanes come back unchecked,
//! without an estimate, marked and counted), every lane the gates refuse is
//! served exactly as with the test (also in multi-lane batches mixing them
//! with lanes the test refuses), and the option changes nothing where the
//! test never runs, where `rotation_test()` reports `Disabled`.
use iregi_rust::{
    AmplitudeOptions, AmplitudeWorkspace, BatchResult, DirectExecutionStats, EmbedderOptions, KinematicsBatch,
    LaneStatus, ReductionSettings, RotationTest, TensorEngineExecution, TensorReductionPlan, TensorWorkspace,
    TensorWorkspaceOptions,
};
use symbolica::domains::float::{Complex, DoubleFloat};

mod common;
use common::hexagon::{MT2, MU, POINTS};

/// The diagnostic line of an unchecked lane (`RotationTest::Disabled`).
const UNCHECKED: &str = "accepted by the conditioning gates alone (loop-rotation test disabled";

/// The hexagon and its pinches with at least four propagators (6 pentagons,
/// 15 boxes), each at rank = number of propagators.
fn integrals() -> Vec<Vec<usize>> {
    (1u32..1 << 6)
        .filter(|m| m.count_ones() >= 4)
        .map(|m| (0..6).filter(|i| m & (1 << i) != 0).collect())
        .collect()
}

fn batch(members: &[usize], mass2: f64) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    for point in &POINTS {
        for &i in members {
            values.extend(point[i].iter().map(|&x| Complex::new(x, 0.0)));
        }
        values.extend(members.iter().map(|_| Complex::new(mass2, 0.0)));
        values.push(Complex::new(MU, 0.0));
    }
    KinematicsBatch { rows: POINTS.len(), components: 1, values }
}

fn options(rotation_test: RotationTest) -> TensorWorkspaceOptions {
    TensorWorkspaceOptions { rotation_test, ..Default::default() }
}

/// A lane's array (best effort included), status and estimate, bit for bit.
fn lane_bits(result: &BatchResult<f64>, lane: usize) -> (Vec<u64>, String, Option<u64>) {
    let values = match result.lane_with_estimate(lane) {
        Some(best) => best.values.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect(),
        None => vec![u64::MAX],
    };
    (values, format!("{:?}", result.statuses[lane]), result.estimated_errors[lane].map(f64::to_bits))
}

fn stats(workspace: &TensorWorkspace<f64>) -> DirectExecutionStats {
    match workspace.engine_execution() {
        TensorEngineExecution::Direct(stats) => stats,
        other => panic!("not the direct engine: {other:?}"),
    }
}

struct Run {
    result: BatchResult<f64>,
    diagnostics: Vec<Vec<String>>,
    stats: DirectExecutionStats,
}

fn run(members: &[usize], mass2: f64, workspace_options: Option<TensorWorkspaceOptions>) -> Run {
    let plan = TensorReductionPlan::new(ReductionSettings::new(members.len(), members.len() as u16)).unwrap();
    run_plan(&plan, &batch(members, mass2), workspace_options)
}

fn run_plan(plan: &TensorReductionPlan, batch: &KinematicsBatch<f64>, workspace_options: Option<TensorWorkspaceOptions>) -> Run {
    let mut workspace = match workspace_options {
        Some(options) => TensorWorkspace::new_with_options(plan, 0.0f64, options).unwrap(),
        None => TensorWorkspace::new(plan, 0.0f64).unwrap(),
    };
    let result = workspace.reduce_batch(batch).unwrap();
    Run { result, diagnostics: workspace.last_diagnostics().to_vec(), stats: stats(&workspace) }
}

/// One multi-lane batch of `n` propagators at rank `n` (pyAmpliCol's
/// settings, invariants from the shifts or supplied): the method-choice
/// fixture's lanes of that size (lightlike legs; the first pass's gates
/// refuse some of its boxes) interleaved with the hexagon's pinches of that
/// size at both masses, among which are lanes the rotation test refuses
/// today.
fn mixed_batch(n: usize, supplied: bool) -> (TensorReductionPlan, KinematicsBatch<f64>) {
    let fixture: Vec<_> =
        common::lanes(include_str!("fixtures/method_choice_lanes.json")).into_iter().filter(|lane| lane.n == n).collect();
    assert!(fixture.iter().all(|lane| lane.rank == n as u16));
    let mut pinches = Vec::new();
    for mass2 in [MT2, 0.0] {
        for members in integrals().into_iter().filter(|m| m.len() == n) {
            for point in &POINTS {
                let mut row: Vec<Complex<f64>> =
                    members.iter().flat_map(|&i| point[i].map(|x| Complex::new(x, 0.0))).collect();
                if supplied {
                    // (V_i - V_j)^2, i < j, row-major, in binary64.
                    for (a, &i) in members.iter().enumerate() {
                        for &j in &members[a + 1..] {
                            let d: [f64; 4] = std::array::from_fn(|c| point[i][c] - point[j][c]);
                            row.push(Complex::new(d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3], 0.0));
                        }
                    }
                }
                row.extend(members.iter().map(|_| Complex::new(mass2, 0.0)));
                row.push(Complex::new(MU, 0.0));
                pinches.push(row);
            }
        }
    }
    let mut values = Vec::new();
    for k in 0..pinches.len().max(fixture.len()) {
        if let Some(row) = pinches.get(k) {
            values.extend_from_slice(row);
        }
        if let Some(lane) = fixture.get(k) {
            values.extend(common::row(lane, supplied));
        }
    }
    let rows = pinches.len() + fixture.len();
    assert_eq!(values.len(), rows * common::row(&fixture[0], supplied).len());
    let plan = TensorReductionPlan::new(common::settings(&fixture[0], supplied)).unwrap();
    (plan, KinematicsBatch { rows, components: 1, values })
}

/// Counts of [`compare_disabled`].
#[derive(Default, Debug)]
struct Lanes {
    unchecked: usize,
    refused_today: usize,
    gate_refused: usize,
}

/// `Disabled` against `Enabled` on one batch, lane by lane: lanes marked
/// unchecked were accepted by the gates alone (kept as the first pass
/// computed them, without an estimate); every other lane takes the same
/// route, bit for bit.
fn compare_disabled(label: &str, plan: &TensorReductionPlan, batch: &KinematicsBatch<f64>) -> Lanes {
    let enabled = run_plan(plan, batch, Some(options(RotationTest::Enabled)));
    let disabled = run_plan(plan, batch, Some(options(RotationTest::Disabled)));
    assert_eq!((disabled.stats.rotation_rows, disabled.stats.rotation_failures), (0, 0), "{label}");
    let mut lanes = Lanes::default();
    for lane in 0..batch.rows {
        let notes = &disabled.diagnostics[lane];
        if notes.iter().any(|note| note.contains(UNCHECKED)) {
            lanes.unchecked += 1;
            assert_eq!(notes.len(), 1, "{label} lane {lane}: {notes:?}");
            assert_eq!(disabled.result.statuses[lane], LaneStatus::Success);
            assert_eq!(disabled.result.estimated_errors[lane], None);
            // With the test, a lane it accepts has no diagnostics; one it
            // refuses is served by a later stage (cyclic orders, the
            // precision rescue), which says so.
            if enabled.diagnostics[lane].is_empty() {
                // The rotation test accepted it: the same array.
                assert_eq!(lane_bits(&enabled.result, lane).0, lane_bits(&disabled.result, lane).0, "{label} lane {lane}");
            } else {
                lanes.refused_today += 1;
            }
        } else {
            // Refused by the gates: the same route, bit for bit.
            lanes.gate_refused += 1;
            assert_eq!(lane_bits(&enabled.result, lane), lane_bits(&disabled.result, lane), "{label} lane {lane}");
            assert_eq!(enabled.diagnostics[lane], disabled.diagnostics[lane], "{label} lane {lane}");
        }
    }
    assert_eq!(disabled.stats.rotation_unchecked, lanes.unchecked, "{label}");
    lanes
}

#[test]
fn the_default_is_the_explicit_enabled_setting_bit_for_bit() {
    for mass2 in [MT2, 0.0] {
        for members in integrals() {
            let default = run(&members, mass2, None);
            let enabled = run(&members, mass2, Some(options(RotationTest::Enabled)));
            for lane in 0..POINTS.len() {
                assert_eq!(lane_bits(&default.result, lane), lane_bits(&enabled.result, lane), "{members:?} lane {lane}");
            }
            assert_eq!(default.diagnostics, enabled.diagnostics, "{members:?}");
            assert_eq!(format!("{:?}", default.stats), format!("{:?}", enabled.stats), "{members:?}");
            assert_eq!(enabled.stats.rotation_unchecked, 0);
        }
    }
}

#[test]
fn disabled_skips_only_the_first_pass_rotation_comparison() {
    let mut total = Lanes::default();
    let mut add = |lanes: &Lanes| {
        total.unchecked += lanes.unchecked;
        total.refused_today += lanes.refused_today;
        total.gate_refused += lanes.gate_refused;
    };
    // The hexagon's pinches, one integral per batch: no lane the gates refuse.
    for mass2 in [MT2, 0.0] {
        for members in integrals() {
            let plan = TensorReductionPlan::new(ReductionSettings::new(members.len(), members.len() as u16)).unwrap();
            add(&compare_disabled(&format!("{members:?} m2 {mass2}"), &plan, &batch(&members, mass2)));
        }
    }
    // Multi-lane batches mixing lanes the gates refuse with lanes the
    // rotation test refuses today and lanes it accepts.
    // Boxes, invariants from the shifts and supplied: today 3 gate-refused, 11
    // and 8 refused by the test, of 126 lanes. Pentagons: the gates accept
    // every fixture pentagon, but the batch holds 20 lanes the test refuses.
    for (n, supplied, mixes) in [(4, false, true), (4, true, true), (5, false, false)] {
        let (plan, batch) = mixed_batch(n, supplied);
        let label = format!("mixed N={n}, invariants supplied {supplied}");
        let lanes = compare_disabled(&label, &plan, &batch);
        eprintln!("{label}: {} lanes, {lanes:?}", batch.rows);
        assert!(lanes.refused_today > 0 && lanes.unchecked > lanes.refused_today, "{label}: {lanes:?}");
        assert!(!mixes || lanes.gate_refused > 0, "{label} has no gate-refused lane: {lanes:?}");
        add(&lanes);
    }
    eprintln!("{total:?}");
    assert!(total.unchecked > 0 && total.refused_today > 0 && total.gate_refused > 0, "{total:?}");
}

/// Lanes the first pass's gates refuse take the same route with the test
/// disabled, bit for bit (pyAmpliCol's method-choice lanes: sub-triangle and
/// pentagon sub-systems of Gram conditioning 6e-5 .. 4e-3 with lightlike
/// legs, and top-loop pentagons and hexagons; invariants from the shifts or
/// supplied).
#[test]
fn gate_refused_lanes_are_served_alike_with_the_test_disabled() {
    let (mut refused, mut marked) = (0, 0);
    for lane in common::lanes(include_str!("fixtures/method_choice_lanes.json")) {
        for supplied in [false, true] {
            let plan = TensorReductionPlan::new(common::settings(&lane, supplied)).unwrap();
            let mut outputs = Vec::new();
            for test in [RotationTest::Enabled, RotationTest::Disabled] {
                let mut workspace = TensorWorkspace::new_with_options(&plan, 0.0f64, options(test)).unwrap();
                let result = workspace.reduce_batch(&common::batch(&lane, supplied)).unwrap();
                outputs.push((lane_bits(&result, 0), workspace.last_diagnostics()[0].clone()));
            }
            if outputs[1].1.iter().any(|note| note.contains(UNCHECKED)) {
                marked += 1;
            } else {
                refused += 1;
                assert_eq!(outputs[0], outputs[1], "{} (supplied {supplied})", lane.label);
            }
        }
    }
    eprintln!("gate-refused lanes {refused}, unchecked {marked}");
    assert!(refused > 0, "no gate-refused lane");
}

#[test]
fn the_setting_is_reported_and_can_be_changed() {
    let plan = TensorReductionPlan::new(ReductionSettings::new(4, 4)).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    assert_eq!(workspace.rotation_test(), RotationTest::Enabled);
    workspace.set_rotation_test(RotationTest::Disabled);
    assert_eq!(workspace.rotation_test(), RotationTest::Disabled);
    let members = [0, 1, 2, 3];
    workspace.reduce_batch(&batch(&members, MT2)).unwrap();
    let after = stats(&workspace);
    assert_eq!(after.rotation_rows, 0);
    assert_eq!(after.rotation_unchecked, after.direct_successes);
    workspace.set_rotation_test(RotationTest::Enabled);
    workspace.reduce_batch(&batch(&members, MT2)).unwrap();
    assert_eq!(stats(&workspace).rotation_unchecked, 0);
    assert!(stats(&workspace).rotation_rows >= POINTS.len());
}

/// Like `precision_rescue()`, `rotation_test()` reports the effective
/// setting: `Disabled` wherever later calls run no rotation test, whatever
/// was set.
#[test]
fn the_getter_reports_the_effective_setting() {
    let facade = |settings: ReductionSettings| {
        TensorWorkspace::new(&TensorReductionPlan::new(settings).unwrap(), 0.0f64).unwrap()
    };
    let mut dotted = ReductionSettings::new(4, 3);
    dotted.powers = vec![2, 1, 1, 1];
    for (label, mut workspace) in [("triangle", facade(ReductionSettings::new(3, 3))), ("dotted box", facade(dotted))] {
        for test in [RotationTest::Enabled, RotationTest::Disabled] {
            workspace.set_rotation_test(test);
            assert_eq!(workspace.rotation_test(), RotationTest::Disabled, "{label} set to {test:?}");
        }
    }
    // Workspaces of other engines, as `precision_rescue()` reports them.
    let plan = TensorReductionPlan::new(ReductionSettings::new(4, 1)).unwrap();
    let mut double_double = TensorWorkspace::new(&plan, DoubleFloat::from(0.0)).unwrap();
    double_double.set_rotation_test(RotationTest::Enabled);
    assert_eq!(double_double.rotation_test(), RotationTest::Disabled);
    assert_eq!(double_double.precision_rescue(), iregi_rust::PrecisionRescue::Disabled);
    let mut embedded = TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()).unwrap();
    embedded.set_rotation_test(RotationTest::Enabled);
    assert_eq!(embedded.rotation_test(), RotationTest::Disabled);
    assert_eq!(embedded.precision_rescue(), iregi_rust::PrecisionRescue::Disabled);
}

#[test]
fn disabled_changes_nothing_where_the_test_never_runs() {
    // Fewer than four propagators: no rotation test either way.
    for members in [vec![0, 1, 2], vec![2, 4]] {
        let enabled = run(&members, MT2, Some(options(RotationTest::Enabled)));
        let disabled = run(&members, MT2, Some(options(RotationTest::Disabled)));
        for lane in 0..POINTS.len() {
            assert_eq!(lane_bits(&enabled.result, lane), lane_bits(&disabled.result, lane), "{members:?}");
        }
        assert_eq!(enabled.diagnostics, disabled.diagnostics);
        assert_eq!(format!("{:?}", enabled.stats), format!("{:?}", disabled.stats));
        assert_eq!(disabled.stats.rotation_unchecked, 0);
    }
    // Unequal powers (a dotted propagator).
    let members = [0, 1, 2, 3];
    let mut settings = ReductionSettings::new(4, 3);
    settings.powers = vec![2, 1, 1, 1];
    let plan = TensorReductionPlan::new(settings).unwrap();
    let mut outputs = Vec::new();
    for test in [RotationTest::Enabled, RotationTest::Disabled] {
        let mut workspace = TensorWorkspace::new_with_options(&plan, 0.0f64, options(test)).unwrap();
        let result = workspace.reduce_batch(&batch(&members, MT2)).unwrap();
        let lanes: Vec<_> = (0..POINTS.len()).map(|lane| lane_bits(&result, lane)).collect();
        outputs.push((lanes, workspace.last_diagnostics().to_vec()));
    }
    assert_eq!(outputs[0], outputs[1], "dotted box");
    // Another number type: the double-double facade runs no rotation test
    // (a low rank keeps its symbolic plan small).
    let plan = TensorReductionPlan::new(ReductionSettings::new(4, 1)).unwrap();
    let lift = |b: KinematicsBatch<f64>| KinematicsBatch {
        rows: b.rows,
        components: b.components,
        values: b.values.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect(),
    };
    let mut outputs = Vec::new();
    for test in [RotationTest::Enabled, RotationTest::Disabled] {
        let mut workspace = TensorWorkspace::new_with_options(&plan, DoubleFloat::from(0.0), options(test)).unwrap();
        let result = workspace.reduce_batch(&lift(batch(&members, MT2))).unwrap();
        let arrays: Vec<String> =
            (0..result.rows()).map(|lane| format!("{:?}", result.lane_with_estimate(lane).map(|best| best.values.to_vec()))).collect();
        outputs.push((arrays, format!("{:?}", result.statuses), workspace.last_diagnostics().to_vec()));
    }
    assert_eq!(outputs[0], outputs[1], "double-double facade");
    // Embedder mode: no rotation test; the setter is accepted and ignored.
    let plan = TensorReductionPlan::new(ReductionSettings::new(4, 4)).unwrap();
    let mut outputs = Vec::new();
    for test in [RotationTest::Enabled, RotationTest::Disabled] {
        let mut workspace = TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()).unwrap();
        workspace.set_rotation_test(test);
        assert_eq!(workspace.rotation_test(), RotationTest::Disabled);
        let result = workspace.reduce_batch(&batch(&members, MT2)).unwrap();
        let lanes: Vec<_> = (0..POINTS.len()).map(|lane| lane_bits(&result, lane)).collect();
        outputs.push((lanes, workspace.last_diagnostics().to_vec()));
    }
    assert_eq!(outputs[0], outputs[1], "embedder mode");
}

#[test]
fn amplitude_sharing_stays_bit_identical_with_the_test_disabled() {
    let members = integrals();
    let plans: Vec<TensorReductionPlan> = members
        .iter()
        .map(|m| TensorReductionPlan::new(ReductionSettings::new(m.len(), m.len() as u16)).unwrap())
        .collect();
    let batches: Vec<KinematicsBatch<f64>> = members.iter().map(|m| batch(m, MT2)).collect();
    let amplitude_options = AmplitudeOptions { workspace: options(RotationTest::Disabled), ..Default::default() };
    let mut amplitude = AmplitudeWorkspace::new_with_options(&plans, 0.0f64, amplitude_options).unwrap();
    let shared = amplitude.reduce_batches(&batches).unwrap();
    assert!(amplitude.last_sharing().reused > 0);
    amplitude.set_subset_sharing(false);
    let separate = amplitude.reduce_batches(&batches).unwrap();
    for (k, (a, b)) in shared.iter().zip(&separate).enumerate() {
        for lane in 0..POINTS.len() {
            assert_eq!(lane_bits(a, lane), lane_bits(b, lane), "integral {:?} lane {lane}", members[k]);
        }
    }
    // Every integral's workspace carries the setting and marks its lanes.
    let unchecked: usize = (0..amplitude.len())
        .map(|k| match amplitude.workspace(k).engine_execution() {
            TensorEngineExecution::Direct(stats) => stats.rotation_unchecked,
            _ => 0,
        })
        .sum();
    assert!(unchecked > 0);
    assert!((0..amplitude.len()).all(|k| amplitude.workspace(k).rotation_test() == RotationTest::Disabled));
}
