//! Direct-engine concurrency (G11): independent workspaces on many threads,
//! one per thread, give every lane bit for bit what one workspace gives on
//! one thread, whatever the number of threads and the lanes' grouping into
//! batches; a tensor store shared by all threads changes nothing either.
//! `DirectWorkspace`s in binary64 (gates, method choice, partial rescue),
//! double-double and MPFR; embedder-mode and facade `TensorWorkspace`s are
//! not covered here.
use std::thread;

use iregi_rust::{
    BatchResult, DirectReductionPlan, DirectWorkspace, KinematicsBatch, ReductionSettings, TensorStore,
    direct::EXPANSION_TOLERANCE,
};
use iregi_masters::NativeFloat;
use symbolica::domains::float::{Complex, DoubleFloat};

mod common;
use common::hexagon::{Lcg, POINTS};

fn send_sync<T: Send + Sync>() {}
fn send<T: Send>() {}

#[test]
fn workspaces_plans_and_stores_cross_threads() {
    send::<DirectWorkspace<f64>>();
    send::<DirectWorkspace<DoubleFloat>>();
    send::<DirectWorkspace<symbolica::domains::float::Float>>();
    send_sync::<DirectReductionPlan>();
    send_sync::<TensorStore>();
}

/// Lanes of an `n`-point plan: the physical hexagon points' sub-topologies
/// (massive top loop and massless; gate-refused lanes among them), then
/// generic massive lanes, interleaved.
fn lanes(n: usize, generic: usize) -> Vec<Vec<Complex<f64>>> {
    let mut physical = Vec::new();
    for mass2 in [29929.0, 0.0] {
        for members in (1u32..1 << 6).filter(|m| m.count_ones() as usize == n) {
            let members: Vec<usize> = (0..6).filter(|i| members & (1 << i) != 0).collect();
            for point in &POINTS {
                let mut row: Vec<Complex<f64>> =
                    members.iter().flat_map(|&i| point[i].map(|x| Complex::new(x, 0.0))).collect();
                row.extend(members.iter().map(|_| Complex::new(mass2, 0.0)));
                row.push(Complex::new(1000.0, 0.0));
                physical.push(row);
            }
        }
    }
    let mut rng = Lcg(97 + n as u64);
    let mut random = Vec::new();
    for _ in 0..generic {
        let mut row = Vec::new();
        for i in 0..n {
            let shift = if i == 0 { [0.0; 4] } else { [0.0; 4].map(|_| 600.0 * rng.next() - 300.0) };
            row.extend(shift.map(|x| Complex::new(x, 0.0)));
        }
        row.extend((0..n).map(|_| Complex::new(1000.0 + 40000.0 * rng.next(), 0.0)));
        row.push(Complex::new(500.0, 0.0));
        random.push(row);
    }
    let mut all = Vec::new();
    let (mut a, mut b) = (physical.into_iter(), random.into_iter());
    loop {
        match (a.next(), b.next()) {
            (None, None) => break,
            (x, y) => all.extend(x.into_iter().chain(y)),
        }
    }
    all
}

fn batch<T: NativeFloat>(rows: &[&Vec<Complex<f64>>], lift: &impl Fn(f64) -> T) -> KinematicsBatch<T> {
    KinematicsBatch {
        rows: rows.len(),
        components: 1,
        values: rows.iter().flat_map(|row| row.iter().map(|z| Complex::new(lift(z.re), lift(z.im)))).collect(),
    }
}

/// One lane's array, status and conditioning estimate, exactly.
type Lane = (String, String, Option<u64>);

fn lanes_of<T: NativeFloat>(result: &BatchResult<T>, conditions: &[Option<f64>]) -> Vec<Lane> {
    (0..result.rows())
        .map(|lane| {
            let values = result.lane_with_estimate(lane).map_or_else(|| "refused".into(), |best| format!("{:?}", best.values));
            (values, format!("{:?}", result.statuses[lane]), conditions[lane].map(f64::to_bits))
        })
        .collect()
}

/// Every lane through `threads` threads, each with its own workspace (made
/// by `make`) and its share of the lanes, contiguous or round-robin, in
/// batches of at most `chunk` lanes; optionally one store for all threads.
fn run<T: NativeFloat + Send>(
    rows: &[Vec<Complex<f64>>],
    lift: &(impl Fn(f64) -> T + Sync),
    make: &(impl Fn() -> DirectWorkspace<T> + Sync),
    threads: usize,
    round_robin: bool,
    chunk: usize,
    store: Option<&TensorStore>,
) -> Vec<Lane>
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let owner = |lane: usize| if round_robin { lane % threads } else { lane * threads / rows.len() };
    let mut out: Vec<Option<Lane>> = vec![None; rows.len()];
    let parts: Vec<Vec<(usize, Lane)>> = thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let mine: Vec<usize> = (0..rows.len()).filter(|&lane| owner(lane) == t).collect();
                    let mut workspace = make();
                    workspace.set_tensor_store(store);
                    let mut done = Vec::new();
                    for part in mine.chunks(chunk.max(1)) {
                        let rows: Vec<&Vec<Complex<f64>>> = part.iter().map(|&lane| &rows[lane]).collect();
                        let result = workspace.reduce_batch(&batch(&rows, lift)).unwrap();
                        let lanes = lanes_of(&result, workspace.last_conditions());
                        done.extend(part.iter().copied().zip(lanes));
                    }
                    done
                })
            })
            .collect();
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });
    for (lane, value) in parts.into_iter().flatten() {
        out[lane] = Some(value);
    }
    out.into_iter().map(Option::unwrap).collect()
}

fn check<T: NativeFloat + Send>(
    name: &str,
    rows: &[Vec<Complex<f64>>],
    lift: &(impl Fn(f64) -> T + Sync),
    make: &(impl Fn() -> DirectWorkspace<T> + Sync),
    thread_counts: &[usize],
    pinches: bool,
) where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    // One workspace, one thread, one batch.
    let reference = run(rows, lift, make, 1, false, rows.len(), None);
    let served = reference.iter().filter(|lane| lane.1 == "Success").count();
    let refused = reference.len() - served;
    eprintln!("{name}: {} lanes, {served} served, {refused} refused", reference.len());
    assert!(served > 0, "{name}: no lane served");
    for &threads in thread_counts {
        for round_robin in [false, true] {
            for chunk in [rows.len(), 3] {
                let parallel = run(rows, lift, make, threads, round_robin, chunk, None);
                assert_eq!(parallel, reference, "{name}: {threads} threads, round robin {round_robin}, chunks of {chunk}");
            }
        }
        let store = TensorStore::new();
        let shared = run(rows, lift, make, threads, true, 3, Some(&store));
        assert_eq!(shared, reference, "{name}: {threads} threads sharing one store");
        // The sub-topologies of one point share masters (a hexagon point has
        // one hexagon: nothing to meet).
        let stats = store.statistics();
        assert!(!pinches || stats.masters_reused + stats.dd_masters_reused > 0, "{name}: {stats:?}");
    }
}

#[test]
fn binary64_lanes_are_bitwise_independent_of_the_thread_count() {
    for (n, rank) in [(4usize, 4u16), (5, 5), (6, 6)] {
        let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank)).unwrap();
        let rows = lanes(n, 24);
        let gates = || plan.workspace(0.0f64).unwrap();
        check(&format!("N{n} R{rank} gates"), &rows, &|x| x, &gates, &[1, 2, 3, 4, 8], n < 6);
        let choice = || {
            let mut workspace = plan.workspace(0.0f64).unwrap();
            workspace.set_stability_threshold(1e-15);
            workspace.enable_estimated_method_choice(12, 1e-15);
            workspace.set_alternative_first(Some(EXPANSION_TOLERANCE));
            workspace
        };
        check(&format!("N{n} R{rank} method choice"), &rows, &|x| x, &choice, &[1, 2, 4, 8], n < 6);
        let rescue = || {
            let mut workspace = plan.workspace(0.0f64).unwrap();
            workspace.enable_partial_rescue(1e-20);
            workspace
        };
        check(&format!("N{n} R{rank} partial rescue"), &rows, &|x| x, &rescue, &[1, 4], n < 6);
    }
}

#[test]
fn double_double_lanes_are_bitwise_independent_of_the_thread_count() {
    let plan = DirectReductionPlan::new(ReductionSettings::new(5, 5)).unwrap();
    let rows = lanes(5, 8);
    let double = || plan.workspace(DoubleFloat::from(0.0)).unwrap();
    check("N5 R5 double-double", &rows, &DoubleFloat::from, &double, &[1, 3, 4], true);
}

#[test]
fn mpfr_lanes_are_bitwise_independent_of_the_thread_count() {
    use symbolica::domains::float::Float;
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, 4)).unwrap();
    // Every sixth lane (physical and generic boxes): MPFR is slow.
    let rows: Vec<_> = lanes(4, 6).into_iter().step_by(6).collect();
    let mpfr = || plan.workspace(Float::new(128)).unwrap();
    check("N4 R4 MPFR 128 bits", &rows, &|x| Float::with_val(128, x), &mpfr, &[1, 3], false);
}
