//! The persistent tensor store (`TensorStore`): a lifetime the caller owns,
//! sharing between workspaces of different plans across separate calls,
//! bit-identical results with and without it (any call order, sharing on or
//! off, any byte budget), translation-invariant master keys, and the
//! optional masters provider.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use iregi_rust::{
    BatchResult, DirectReductionPlan, DirectWorkspace, EmbedderOptions, InvariantInput, KinematicsBatch, LaneStatus,
    MasterFamily, MasterProvider, ReductionSettings, TensorReductionPlan, TensorStore, TensorStoreOptions,
    TensorStoreStatistics, TensorWorkspace, direct::EXPANSION_TOLERANCE,
};
use symbolica::domains::float::{Complex, DoubleFloat};

/// Four top-loop gg -> gggg hexagon points (tests/amplitude_sharing.rs):
/// shifts of the six propagators in the hexagon's loop-momentum frame.
const POINTS: [[[f64; 4]; 6]; 4] = [
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [792.7874729156998, -82.84167294384719, -92.33145751079824, -165.97948836665938],
        [496.1459371514851, -309.53730217508144, -274.5389438821032, -224.33807246752178],
        [365.75699134826516, -254.15872221430251, -237.93327663297802, -112.1127847893476],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [628.3485871137328, -126.38562954754646, -64.66171687618484, 343.46805923528484],
        [493.6035005043037, -75.99778103019892, 59.08875070855528, 360.8785405535477],
        [342.2594345535183, -151.0438342019983, 151.3291327069494, 267.2578797427977],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [720.5411460864439, 41.03231623557652, 269.0508926250419, -63.44459971258532],
        [580.5195723069281, 86.55172419413077, 347.3583949823081, 43.335359537087676],
        [328.9555327352336, -146.2746278878062, 292.60284001366847, -34.62735579481515],
    ],
    [
        [0.0, 0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0, 500.0],
        [1000.0, 0.0, 0.0, 0.0],
        [684.7727360619465, -266.1383293706448, -32.469872857286525, 165.7839706555041],
        [491.59350180547904, -99.14409525080174, -121.72713563298282, 127.51801502347416],
        [243.28769056900498, 107.6510206808816, -17.784570933495488, 217.44853912012346],
    ],
];
const MT2: f64 = 29929.0;
const MU: f64 = 1000.0;

/// The hexagon's sub-topologies with `size` propagators (members in the
/// hexagon's order: the aligned frame).
fn subsets(size: u32) -> Vec<Vec<usize>> {
    (1u32..1 << 6).filter(|m| m.count_ones() == size).map(|m| (0..6).filter(|i| m & (1 << i) != 0).collect()).collect()
}

/// One batch per propagator count: every integral of that size on every
/// point (an embedder's (N, R) bucket), in the aligned frame.
fn bucket(size: u32, mass2: f64) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    let mut rows = 0;
    for members in subsets(size) {
        for point in &POINTS {
            for &i in &members {
                values.extend(point[i].iter().map(|&x| Complex::new(x, 0.0)));
            }
            values.extend(members.iter().map(|_| Complex::new(mass2, 0.0)));
            values.push(Complex::new(MU, 0.0));
            rows += 1;
        }
    }
    KinematicsBatch { rows, components: 1, values }
}

fn plan(size: usize) -> DirectReductionPlan {
    DirectReductionPlan::new(ReductionSettings::new(size, size as u16)).unwrap()
}

/// Every lane's array (best-effort ones included) and status, exactly (the
/// shortest round-trip forms of every limb).
fn fingerprint<T: iregi_masters::NativeFloat>(result: &BatchResult<T>) -> (Vec<String>, String) {
    let values = (0..result.rows())
        .map(|lane| result.lane_with_estimate(lane).map_or_else(|| "refused".to_string(), |best| format!("{:?}", best.values)))
        .collect();
    (values, format!("{:?}", result.statuses))
}

/// The binary64 workspaces of an embedder's two passes: the conditioning
/// gates, and the a priori method choice for the lanes the gates refuse.
fn workspace(size: usize, choice: bool) -> DirectWorkspace<f64> {
    let mut workspace = plan(size).workspace(0.0f64).unwrap();
    if choice {
        workspace.set_stability_threshold(1e-15);
        workspace.enable_estimated_method_choice(12, 1e-15);
        workspace.set_alternative_first(Some(EXPANSION_TOLERANCE));
    }
    workspace
}

/// The buckets in `order` (sizes), each through its own workspace, sharing
/// `store` (if any) across the calls.
fn reduce_buckets(order: &[u32], choice: bool, mass2: f64, store: Option<&TensorStore>) -> Vec<(u32, (Vec<String>, String))> {
    let mut results = Vec::new();
    for &size in order {
        let mut workspace = workspace(size as usize, choice);
        workspace.set_tensor_store(store);
        let result = workspace.reduce_batch(&bucket(size, mass2)).unwrap();
        results.push((size, fingerprint(&result)));
    }
    results.sort_by_key(|(size, _)| *size);
    results
}

#[test]
fn parent_and_pinches_meet_across_calls_bit_identically() {
    for choice in [false, true] {
        let alone = reduce_buckets(&[6, 5, 4], choice, MT2, None);
        let lanes: usize = alone.iter().map(|(_, (_, statuses))| statuses.matches("Success").count()).sum();
        assert!(lanes > 0, "no lane served (choice {choice})");
        for order in [[6u32, 5, 4], [4, 5, 6]] {
            let store = TensorStore::new();
            let shared = reduce_buckets(&order, choice, MT2, Some(&store));
            assert_eq!(shared, alone, "results differ with the store (choice {choice}, order {order:?})");
            let stats = store.statistics();
            eprintln!("choice {choice}, order {order:?}: {stats:?}");
            // Scalar masters of the common boxes and triangles meet in every order.
            assert!(stats.masters_reused > 0, "no master met across calls: {stats:?}");
            // Sub-integrals (and, with the method choice, expansions) meet in
            // every order.
            assert!(stats.subsets_reused > 0, "no sub-integral met across calls: {stats:?}");
            assert!(stats.entries > 0 && stats.bytes > 0 && stats.not_stored == 0, "{stats:?}");
        }
    }
}

#[test]
fn the_store_lives_until_the_caller_clears_it() {
    let store = TensorStore::new();
    let mut workspace = workspace(5, false);
    workspace.set_tensor_store(Some(&store));
    let batch = bucket(5, MT2);
    let first = workspace.reduce_batch(&batch).unwrap();
    let computed = store.statistics();
    assert!(computed.masters_computed > 0 && computed.entries > 0, "{computed:?}");
    // A second call on the same points: nothing new is computed.
    let second = workspace.reduce_batch(&batch).unwrap();
    let again = store.statistics();
    assert_eq!(fingerprint(&first), fingerprint(&second));
    assert_eq!(again.masters_computed, computed.masters_computed, "{again:?}");
    assert_eq!(again.entries, computed.entries, "{again:?}");
    assert!(again.masters_reused > computed.masters_reused, "{again:?}");
    // Cleared by the caller: empty, and computed again.
    store.clear();
    assert_eq!(store.statistics(), TensorStoreStatistics::default());
    let third = workspace.reduce_batch(&batch).unwrap();
    assert_eq!(fingerprint(&first), fingerprint(&third));
    assert_eq!(store.statistics().masters_computed, computed.masters_computed);
    // Detached: nothing more is recorded.
    workspace.set_tensor_store(None);
    let before = store.statistics();
    workspace.reduce_batch(&batch).unwrap();
    assert_eq!(store.statistics(), before);
}

#[test]
fn a_byte_budget_bounds_the_store_and_never_changes_values() {
    let alone = reduce_buckets(&[6, 5, 4], true, MT2, None);
    let full = TensorStore::new();
    reduce_buckets(&[6, 5, 4], true, MT2, Some(&full));
    let full = full.statistics();
    for budget in [0u64, 4096, full.bytes / 2] {
        let store = TensorStore::with_options(TensorStoreOptions { byte_budget: Some(budget), ..Default::default() });
        assert_eq!(store.byte_budget(), Some(budget));
        assert_eq!(reduce_buckets(&[6, 5, 4], true, MT2, Some(&store)), alone, "budget {budget}");
        let stats = store.statistics();
        eprintln!("budget {budget}: {stats:?}");
        assert!(stats.bytes <= budget, "{stats:?}");
        assert!(stats.not_stored > 0, "{stats:?}");
        assert!(stats.entries < full.entries, "{stats:?} vs {full:?}");
    }
}

#[test]
fn sharing_off_shares_nothing_and_changes_nothing() {
    let alone = reduce_buckets(&[6, 5, 4], true, 0.0, None);
    let store = TensorStore::new();
    store.set_sharing(false);
    assert!(!store.sharing());
    assert_eq!(reduce_buckets(&[6, 5, 4], true, 0.0, Some(&store)), alone);
    assert_eq!(store.statistics(), TensorStoreStatistics::default());
    // Massless propagators (IR poles in every slot) with sharing on.
    store.set_sharing(true);
    assert_eq!(reduce_buckets(&[6, 5, 4], true, 0.0, Some(&store)), alone);
    let stats = store.statistics();
    assert!(stats.masters_reused > 0 && stats.moments_reused > 0, "{stats:?}");
}

/// A generic box at lane `k`: shifts in the box's own frame.
fn box_shifts(k: usize) -> [[f64; 4]; 4] {
    let x = k as f64;
    [
        [0.0, 0.0, 0.0, 0.0],
        [210.0 + x, 13.0, -27.0 + x, 190.0],
        [520.0, 61.0 - x, -40.0, 110.0 + 2.0 * x],
        [330.0 - x, -75.0, 32.0, -95.0 + x],
    ]
}

/// The supplied invariants (p_i - p_j)^2, i < j, of `shifts`, formed once.
fn invariants(shifts: &[[f64; 4]; 4]) -> Vec<f64> {
    let mut s = Vec::new();
    for i in 0..4 {
        for j in i + 1..4 {
            let d: Vec<f64> = (0..4).map(|a| shifts[i][a] - shifts[j][a]).collect();
            s.push(d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]);
        }
    }
    s
}

/// The boxes translated by `offset` (another loop-momentum frame), with
/// the invariants of the untranslated shifts supplied (bit for bit equal).
fn supplied_boxes(offset: [f64; 4]) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    let lanes = 6;
    for k in 0..lanes {
        let shifts = box_shifts(k);
        for shift in &shifts {
            values.extend((0..4).map(|a| Complex::new(shift[a] + offset[a], 0.0)));
        }
        values.extend(invariants(&shifts).into_iter().map(|s| Complex::new(s, 0.0)));
        values.extend([MT2, MT2, 4.0 * MT2, 0.25 * MT2].map(|m| Complex::new(m, 0.0)));
        values.push(Complex::new(MU, 0.0));
    }
    KinematicsBatch { rows: lanes, components: 1, values }
}

#[test]
fn master_keys_are_translation_invariant_and_nothing_false_is_shared() {
    let mut settings = ReductionSettings::new(4, 3);
    settings.invariants = InvariantInput::SuppliedInvariants;
    let plan = DirectReductionPlan::new(settings).unwrap();
    let frame = supplied_boxes([0.0; 4]);
    let moved = supplied_boxes([137.25, -41.5, 18.75, 3.0625]);
    let alone = |batch: &KinematicsBatch<f64>| fingerprint(&plan.workspace(0.0f64).unwrap().reduce_batch(batch).unwrap());
    let store = TensorStore::new();
    let mut workspace = plan.workspace(0.0f64).unwrap();
    workspace.set_tensor_store(Some(&store));
    let first = fingerprint(&workspace.reduce_batch(&frame).unwrap());
    let before = store.statistics();
    let second = fingerprint(&workspace.reduce_batch(&moved).unwrap());
    let after = store.statistics();
    assert_eq!(first, alone(&frame));
    assert_eq!(second, alone(&moved));
    assert!(first.1.matches("Success").count() == 6, "{}", first.1);
    // Every scalar master of the moved boxes is met (their arguments are the
    // supplied invariants, the masses and mu^2), and nothing is computed
    // again; sub-integrals and moments key on the shifts and do not meet.
    assert_eq!(after.masters_computed, before.masters_computed, "{after:?}");
    assert!(after.masters_reused >= before.masters_reused + before.masters_computed, "{before:?} -> {after:?}");
    assert_eq!(after.subsets_reused, before.subsets_reused, "{before:?} -> {after:?}");
    assert_eq!(after.moments_reused, before.moments_reused, "{before:?} -> {after:?}");
    // The moments themselves move with the frame (rank 3): they differ.
    assert_ne!(first.0, second.0);
}

#[test]
fn double_double_workspaces_share_their_masters() {
    let plan = plan(5);
    let batch = bucket(5, MT2);
    let lifted = KinematicsBatch {
        rows: batch.rows,
        components: 1,
        values: batch.values.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect(),
    };
    // Binary64 inputs lifted exactly, as a rescue tier evaluates them (the
    // lightlike rule of their precision: the massless gluons' bubbles).
    let make = || {
        let mut workspace = plan.workspace(DoubleFloat::from(0.0)).unwrap();
        workspace.set_input_precision(53);
        workspace
    };
    let alone = make().reduce_batch(&lifted).unwrap();
    assert!(alone.statuses.contains(&LaneStatus::Success), "{:?}", alone.statuses);
    let alone = fingerprint(&alone);
    let store = TensorStore::new();
    let mut workspace = make();
    workspace.set_tensor_store(Some(&store));
    assert_eq!(fingerprint(&workspace.reduce_batch(&lifted).unwrap()), alone);
    let stats = store.statistics();
    // The pentagons' common boxes and triangles share their masters.
    assert!(stats.dd_masters_reused > 0 && stats.dd_masters_computed > 0, "{stats:?}");
    assert_eq!((stats.masters_computed, stats.subsets_computed, stats.moments_computed), (0, 0, 0), "{stats:?}");
}

#[test]
fn facade_and_embedder_mode_workspaces_take_the_store() {
    for embedded in [false, true] {
        let make = |size: usize| {
            let plan = TensorReductionPlan::new(ReductionSettings::new(size, size as u16)).unwrap();
            if embedded {
                TensorWorkspace::new_embedded(&plan, 0.0f64, EmbedderOptions::default()).unwrap()
            } else {
                TensorWorkspace::new(&plan, 0.0f64).unwrap()
            }
        };
        let run = |store: Option<&TensorStore>| {
            [6u32, 5, 4]
                .iter()
                .map(|&size| {
                    let mut workspace = make(size as usize);
                    workspace.set_tensor_store(store);
                    fingerprint(&workspace.reduce_batch(&bucket(size, MT2)).unwrap())
                })
                .collect::<Vec<_>>()
        };
        let alone = run(None);
        let store = TensorStore::new();
        assert_eq!(run(Some(&store)), alone, "embedded {embedded}");
        let stats = store.statistics();
        eprintln!("embedded {embedded}: {stats:?}");
        assert!(stats.masters_reused > 0 && stats.subsets_reused > 0, "{stats:?}");
        // The hexagon gathers its Gram-free pentagons; the pentagon call
        // finds them gathered.
        assert!(stats.moments_reused > 0, "{stats:?}");
    }
}

/// TIRust's own native kernels behind the provider interface, counting.
struct KernelProvider {
    kernels: Mutex<Vec<oneloop::NativeEvaluator<f64>>>,
    calls: AtomicU64,
    scale_boxes: f64,
    decline: bool,
}

impl KernelProvider {
    fn new(scale_boxes: f64, decline: bool) -> Self {
        let kernels = [
            iregi_masters::ScalarIntegral::A0,
            iregi_masters::ScalarIntegral::B0,
            iregi_masters::ScalarIntegral::C0,
            iregi_masters::ScalarIntegral::D0,
        ]
        .into_iter()
        .map(|family| oneloop::NativeEvaluator::with_binary_precision(family, 53).unwrap())
        .collect();
        Self { kernels: Mutex::new(kernels), calls: AtomicU64::new(0), scale_boxes, decline }
    }
}

impl MasterProvider for KernelProvider {
    fn master(&self, family: MasterFamily, arguments: &[Complex<f64>]) -> Option<[Complex<f64>; 3]> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(arguments.len(), family.arguments());
        if self.decline {
            return None;
        }
        let index = match family {
            MasterFamily::Tadpole => 0,
            MasterFamily::Bubble => 1,
            MasterFamily::Triangle => 2,
            MasterFamily::Box => 3,
        };
        let mut output = vec![Complex::new(0.0, 0.0); 3];
        self.kernels.lock().unwrap()[index].evaluate(arguments, &mut output).ok()?;
        let scale = if family == MasterFamily::Box { self.scale_boxes } else { 1.0 };
        Some([0, 1, 2].map(|k| Complex::new(scale * output[k].re, scale * output[k].im)))
    }
}

#[test]
fn a_masters_provider_serves_every_master_and_changes_values_only_by_its_own() {
    let alone = reduce_buckets(&[6, 5, 4], false, MT2, None);
    // TIRust's own kernels behind the provider: bit-identical.
    let provider = Arc::new(KernelProvider::new(1.0, false));
    let store = TensorStore::with_options(TensorStoreOptions { masters: Some(provider.clone()), ..Default::default() });
    assert!(store.has_master_provider());
    assert_eq!(reduce_buckets(&[6, 5, 4], false, MT2, Some(&store)), alone);
    let stats = store.statistics();
    eprintln!("kernel provider: {stats:?}");
    assert!(stats.masters_provided > 0, "{stats:?}");
    assert_eq!(stats.masters_provided, stats.masters_computed, "{stats:?}");
    assert_eq!(provider.calls.load(Ordering::Relaxed), stats.masters_provided);
    // With sharing off the provider still serves every master (no lookup).
    let store = TensorStore::with_options(TensorStoreOptions { masters: Some(provider.clone()), ..Default::default() });
    store.set_sharing(false);
    let calls = provider.calls.load(Ordering::Relaxed);
    assert_eq!(reduce_buckets(&[6, 5, 4], false, MT2, Some(&store)), alone);
    assert!(provider.calls.load(Ordering::Relaxed) > calls + stats.masters_provided);
    assert_eq!(store.statistics().masters_provided, provider.calls.load(Ordering::Relaxed) - calls);
    // A declining provider leaves every master to TIRust.
    let declining = Arc::new(KernelProvider::new(1.0, true));
    let store = TensorStore::with_options(TensorStoreOptions { masters: Some(declining.clone()), ..Default::default() });
    assert_eq!(reduce_buckets(&[6, 5, 4], false, MT2, Some(&store)), alone);
    assert_eq!(store.statistics().masters_provided, 0);
    assert!(declining.calls.load(Ordering::Relaxed) > 0);
    // The provider's values are the ones used: scalar boxes doubled double
    // a scalar box integral.
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, 0)).unwrap();
    let batch = bucket(4, MT2);
    let reference = plan.workspace(0.0f64).unwrap().reduce_batch(&batch).unwrap();
    let store = TensorStore::with_options(TensorStoreOptions {
        masters: Some(Arc::new(KernelProvider::new(2.0, false))),
        ..Default::default()
    });
    let mut workspace = plan.workspace(0.0f64).unwrap();
    workspace.set_tensor_store(Some(&store));
    let doubled = workspace.reduce_batch(&batch).unwrap();
    for lane in 0..batch.rows {
        assert_eq!(reference.statuses[lane], LaneStatus::Success);
        let (a, b) = (reference.lane(lane).unwrap(), doubled.lane(lane).unwrap());
        for (x, y) in a.iter().zip(b) {
            assert!((y.re - 2.0 * x.re).abs() <= 1e-13 * x.re.abs().max(x.im.abs()), "{x:?} -> {y:?}");
            assert!((y.im - 2.0 * x.im).abs() <= 1e-13 * x.re.abs().max(x.im.abs()), "{x:?} -> {y:?}");
        }
    }
}
