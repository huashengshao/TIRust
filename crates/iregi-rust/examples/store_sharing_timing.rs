//! Sharing of a parent family and its pinches ACROSS calls (the persistent
//! `TensorStore`), and the cost of the rotation copy: warm binary64 timings.
//!
//! The integrals of a top-loop gg -> gggg hexagon amplitude (the hexagon and
//! its 56 pinches, rank = number of propagators, m_t^2 = 29929, mu = 1000) on
//! `POINTS` physical points (RAMBO at sqrt(s) = 1 TeV, pT > 30 GeV,
//! |eta| < 2.5, Delta R > 0.4; fixed seed), every integral in the hexagon's
//! loop-momentum frame (the aligned frame). As an embedder reduces them: one
//! call per propagator count (an (N, R) bucket: every integral of that size
//! on every point), one workspace per bucket, so the parent and its pinches
//! never meet inside a call. Modes:
//!
//! - `separate`: no store;
//! - `store-down` / `store-up`: one `TensorStore` attached to the five
//!   workspaces, cleared once per chunk of points; calls from the hexagon down
//!   to the bubbles, or up;
//! - `own-separate` / `own-store`: every integral in its own frame (its last
//!   propagator at the origin) with the invariants supplied, without and with
//!   the store (hexagon first). A partially aligned frame: integrals with the
//!   same last propagator share their shifts bit for bit (the 30 pinches
//!   containing the hexagon's last propagator are in the hexagon's own
//!   frame), so their sub-integrals and moments meet as in the aligned frame;
//!   between integrals with different last propagators only the
//!   translation-invariant keys (the scalar masters) meet;
//! - `amplitude-off` / `amplitude-on`: one `AmplitudeWorkspace` call over all
//!   57 integrals (sharing inside one call; the facade only);
//! - `rotation-two` / `rotation-copy`: per bucket, the primary and its copy
//!   rotated by one as two calls, or as one `reduce_batch_with_rotation`.
//!
//! Workspaces (`--engine`): `gates` (DirectWorkspace, conditioning gates
//! only: an embedder's first pass), `choice` (DirectWorkspace with the a
//! priori method choice, alternative PV first: an embedder's second pass, here
//! on every lane), `embedded` (TensorWorkspace in embedder mode: gates, then
//! the method choice for refused lanes) and `facade` (TensorWorkspace
//! default: rotation test and precision rescue).
//!
//! usage: store_sharing_timing ENGINE POINTS ROUNDS MODE...
//! Prints one JSON line per mode and round (ns per chunk) and a FINAL line
//! with medians, sharing statistics and every mode's checksum (bitwise
//! identity of the store modes with `separate`).
use std::{error::Error, hint::black_box, time::Instant};

use iregi_rust::{
    AmplitudeOptions, AmplitudeWorkspace, BatchResult, Convention, DirectReductionPlan, DirectWorkspace, EmbedderOptions,
    InvariantInput, KinematicsBatch, ReductionSettings, TensorReductionPlan, TensorStore, TensorWeights, TensorWorkspace,
    direct::EXPANSION_TOLERANCE,
};
use symbolica::domains::float::Complex;

type Fallible<T> = Result<T, Box<dyn Error>>;

const MT2: f64 = 29929.0;
const MU: f64 = 1000.0;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }
}

/// RAMBO: `n` massless momenta of total energy `e` in the centre-of-mass frame.
fn rambo(rng: &mut Lcg, n: usize, e: f64) -> Vec<[f64; 4]> {
    let q: Vec<[f64; 4]> = (0..n)
        .map(|_| {
            let c = 2.0 * rng.next() - 1.0;
            let s = (1.0 - c * c).sqrt();
            let f = 2.0 * std::f64::consts::PI * rng.next();
            let q0 = -(rng.next() * rng.next()).ln();
            [q0, q0 * s * f.cos(), q0 * s * f.sin(), q0 * c]
        })
        .collect();
    let mut r = [0.0; 4];
    for k in &q {
        for a in 0..4 {
            r[a] += k[a];
        }
    }
    let m = (r[0] * r[0] - r[1] * r[1] - r[2] * r[2] - r[3] * r[3]).sqrt();
    let b = [-r[1] / m, -r[2] / m, -r[3] / m];
    let g = r[0] / m;
    let a = 1.0 / (1.0 + g);
    let x = e / m;
    q.iter()
        .map(|k| {
            let bq = b[0] * k[1] + b[1] * k[2] + b[2] * k[3];
            let t = a * bq + k[0];
            [x * (g * k[0] + bq), x * (k[1] + b[0] * t), x * (k[2] + b[1] * t), x * (k[3] + b[2] * t)]
        })
        .collect()
}

fn passes(k: &[[f64; 4]]) -> bool {
    let pt = |p: &[f64; 4]| p[1].hypot(p[2]);
    let eta = |p: &[f64; 4]| {
        let modulus = (p[1] * p[1] + p[2] * p[2] + p[3] * p[3]).sqrt();
        0.5 * ((modulus + p[3]) / (modulus - p[3])).ln()
    };
    let phi = |p: &[f64; 4]| p[2].atan2(p[1]);
    for (i, p) in k.iter().enumerate() {
        if pt(p) < 30.0 || eta(p).abs() > 2.5 {
            return false;
        }
        for q in &k[i + 1..] {
            let mut dphi = (phi(p) - phi(q)).abs();
            if dphi > std::f64::consts::PI {
                dphi = 2.0 * std::f64::consts::PI - dphi;
            }
            if (eta(p) - eta(q)).hypot(dphi) < 0.4 {
                return false;
            }
        }
    }
    true
}

/// The hexagon's six shifts at each point: 0, p1, p1 + p2, then minus the
/// outgoing gluons one by one.
fn points(count: usize) -> Vec<[[f64; 4]; 6]> {
    let mut rng = Lcg(20261107);
    let mut out = Vec::new();
    while out.len() < count {
        let k = rambo(&mut rng, 4, 1000.0);
        if !passes(&k) {
            continue;
        }
        let mut shifts = [[0.0; 4]; 6];
        shifts[1] = [500.0, 0.0, 0.0, 500.0];
        shifts[2] = [1000.0, 0.0, 0.0, 0.0];
        for i in 0..3 {
            for a in 0..4 {
                shifts[3 + i][a] = shifts[2 + i][a] - k[i][a];
            }
        }
        out.push(shifts);
    }
    out
}

/// The integrals of one size: subsets of the six propagators (the hexagon's order).
fn subsets(size: usize) -> Vec<Vec<usize>> {
    (1u32..1 << 6).filter(|m| m.count_ones() as usize == size).map(|m| (0..6).filter(|i| m & (1 << i) != 0).collect()).collect()
}

fn settings(size: usize) -> ReductionSettings {
    let mut settings = ReductionSettings::new(size, size as u16);
    settings.convention = Convention::OneLoop;
    settings.tensor_weights = TensorWeights::Components;
    settings
}

/// The settings of the own-frame buckets: the invariants supplied.
fn supplied_settings(size: usize) -> ReductionSettings {
    let mut settings = settings(size);
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings
}

/// Every integral of `size` on every point in its OWN frame (its last
/// propagator at the origin, as an embedder's families each have their own),
/// with the invariants supplied: formed once per point and pair of the
/// hexagon's propagators, so equal for every integral bit for bit.
fn own_bucket(points: &[[[f64; 4]; 6]], size: usize) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    let mut rows = 0;
    for members in subsets(size) {
        for point in points {
            let last = point[*members.last().unwrap()];
            for &i in &members {
                values.extend((0..4).map(|a| Complex::new(point[i][a] - last[a], 0.0)));
            }
            for (k, &i) in members.iter().enumerate() {
                for &j in &members[k + 1..] {
                    let d: Vec<f64> = (0..4).map(|a| point[i][a] - point[j][a]).collect();
                    values.push(Complex::new(d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3], 0.0));
                }
            }
            values.extend(members.iter().map(|_| Complex::new(MT2, 0.0)));
            values.push(Complex::new(MU, 0.0));
            rows += 1;
        }
    }
    KinematicsBatch { rows, components: 1, values }
}

/// Every integral of `size` on every point (integral-major).
fn bucket(points: &[[[f64; 4]; 6]], size: usize) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    let mut rows = 0;
    for members in subsets(size) {
        for point in points {
            for &i in &members {
                values.extend(point[i].iter().map(|&x| Complex::new(x, 0.0)));
            }
            values.extend(members.iter().map(|_| Complex::new(MT2, 0.0)));
            values.push(Complex::new(MU, 0.0));
            rows += 1;
        }
    }
    KinematicsBatch { rows, components: 1, values }
}

/// One integral on every point.
fn integral(points: &[[[f64; 4]; 6]], members: &[usize]) -> KinematicsBatch<f64> {
    let mut values = Vec::new();
    for point in points {
        for &i in members {
            values.extend(point[i].iter().map(|&x| Complex::new(x, 0.0)));
        }
        values.extend(members.iter().map(|_| Complex::new(MT2, 0.0)));
        values.push(Complex::new(MU, 0.0));
    }
    KinematicsBatch { rows: points.len(), components: 1, values }
}

/// FNV-1a over every lane's array (best-effort ones included) and status.
fn eat(hash: &mut u64, result: &BatchResult<f64>) {
    let mut word = |x: u64| {
        for byte in x.to_le_bytes() {
            *hash ^= u64::from(byte);
            *hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for lane in 0..result.rows() {
        match result.lane_with_estimate(lane) {
            Some(best) => best.values.iter().for_each(|z| {
                word(z.re.to_bits());
                word(z.im.to_bits());
            }),
            None => word(u64::MAX),
        }
        for byte in format!("{:?}", result.statuses[lane]).bytes() {
            word(u64::from(byte));
        }
    }
}

enum Bucket {
    Direct(Box<DirectWorkspace<f64>>),
    Facade(Box<TensorWorkspace<f64>>),
}

impl Bucket {
    fn new(engine: &str, settings: ReductionSettings) -> Fallible<Self> {
        Ok(match engine {
            "gates" | "choice" => {
                let mut workspace = DirectReductionPlan::new(settings)?.workspace(0.0f64)?;
                if engine == "choice" {
                    workspace.set_stability_threshold(1e-15);
                    workspace.enable_estimated_method_choice(12, 1e-15);
                    workspace.set_alternative_first(Some(EXPANSION_TOLERANCE));
                }
                Self::Direct(Box::new(workspace))
            }
            "embedded" => Self::Facade(Box::new(TensorWorkspace::new_embedded(
                &TensorReductionPlan::new(settings)?,
                0.0f64,
                EmbedderOptions::default(),
            )?)),
            "facade" => Self::Facade(Box::new(TensorWorkspace::new(&TensorReductionPlan::new(settings)?, 0.0f64)?)),
            _ => return Err(format!("unknown engine {engine}").into()),
        })
    }

    fn set_store(&mut self, store: Option<&TensorStore>) {
        match self {
            Self::Direct(workspace) => workspace.set_tensor_store(store),
            Self::Facade(workspace) => workspace.set_tensor_store(store),
        }
    }

    fn reduce(&mut self, batch: &KinematicsBatch<f64>) -> Fallible<BatchResult<f64>> {
        Ok(match self {
            Self::Direct(workspace) => workspace.reduce_batch(batch)?,
            Self::Facade(workspace) => workspace.reduce_batch(batch)?,
        })
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { 0.5 * (values[n / 2 - 1] + values[n / 2]) }
}

fn main() -> Fallible<()> {
    let args: Vec<String> = std::env::args().collect();
    let engine = args.get(1).ok_or("ENGINE")?.clone();
    let count: usize = args.get(2).ok_or("POINTS")?.parse()?;
    let rounds: usize = args.get(3).ok_or("ROUNDS")?.parse()?;
    let modes: Vec<String> = args[4..].to_vec();
    let points = points(count);
    let sizes = [6usize, 5, 4, 3, 2];
    let batches: Vec<(usize, KinematicsBatch<f64>)> = sizes.iter().map(|&s| (s, bucket(&points, s))).collect();
    let mut buckets: Vec<Bucket> = sizes.iter().map(|&s| Bucket::new(&engine, settings(s))).collect::<Fallible<_>>()?;
    let own_frames = modes.iter().any(|m| m.starts_with("own"));
    let own_batches: Vec<KinematicsBatch<f64>> =
        if own_frames { sizes.iter().map(|&s| own_bucket(&points, s)).collect() } else { Vec::new() };
    let mut own_buckets: Vec<Bucket> = if own_frames {
        sizes.iter().map(|&s| Bucket::new(&engine, supplied_settings(s))).collect::<Fallible<_>>()?
    } else {
        Vec::new()
    };
    let rotation_plans: Vec<DirectReductionPlan> =
        sizes.iter().map(|&s| DirectReductionPlan::new(settings(s))).collect::<Result<_, _>>()?;
    let mut rotation_workspaces: Vec<DirectWorkspace<f64>> = rotation_plans
        .iter()
        .map(|plan| {
            let mut workspace = plan.workspace(0.0f64)?;
            if engine == "choice" {
                workspace.set_stability_threshold(1e-15);
                workspace.enable_estimated_method_choice(12, 1e-15);
                workspace.set_alternative_first(Some(EXPANSION_TOLERANCE));
            }
            Ok(workspace)
        })
        .collect::<Fallible<_>>()?;
    let rotated: Vec<KinematicsBatch<f64>> =
        rotation_plans.iter().zip(&batches).map(|(plan, (_, batch))| plan.rotated_batch(batch, 1)).collect::<Result<_, _>>()?;
    // The amplitude workspace over all 57 integrals (facade workspaces).
    let members: Vec<Vec<usize>> = sizes.iter().flat_map(|&s| subsets(s)).collect();
    let integrals: Vec<KinematicsBatch<f64>> = members.iter().map(|m| integral(&points, m)).collect();
    let amplitude_plans: Vec<TensorReductionPlan> =
        members.iter().map(|m| TensorReductionPlan::new(settings(m.len()))).collect::<Result<_, _>>()?;
    let mut amplitude = if modes.iter().any(|m| m.starts_with("amplitude")) {
        Some(AmplitudeWorkspace::new_with_options(&amplitude_plans, 0.0f64, AmplitudeOptions::default())?)
    } else {
        None
    };
    let store = TensorStore::new();
    let mut checksums = vec![0u64; modes.len()];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
    let mut statistics = vec![String::new(); modes.len()];
    let mut lanes = vec![(0usize, 0usize); modes.len()];
    // Round 0 warms every workspace (plans, variants, scratch); not timed.
    for round in 0..=rounds {
        for (index, mode) in modes.iter().enumerate() {
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            let (mut served, mut total) = (0usize, 0usize);
            let mut count = |result: &BatchResult<f64>, hash: &mut u64| {
                eat(hash, result);
                total += result.rows();
                served += result.statuses.iter().filter(|s| **s == iregi_rust::LaneStatus::Success).count();
            };
            let started;
            match mode.as_str() {
                "separate" | "store-down" | "store-up" | "own-separate" | "own-store" => {
                    let shared = mode.contains("store");
                    let own = mode.starts_with("own");
                    let (buckets, inputs): (&mut Vec<Bucket>, Vec<&KinematicsBatch<f64>>) = if own {
                        (&mut own_buckets, own_batches.iter().collect())
                    } else {
                        (&mut buckets, batches.iter().map(|(_, batch)| batch).collect())
                    };
                    for bucket in buckets.iter_mut() {
                        bucket.set_store(shared.then_some(&store));
                    }
                    let mut order: Vec<usize> = (0..sizes.len()).collect();
                    if mode == "store-up" {
                        order.reverse();
                    }
                    let mut results = Vec::with_capacity(order.len());
                    started = Instant::now();
                    if shared {
                        store.clear();
                    }
                    for &b in &order {
                        results.push((b, buckets[b].reduce(inputs[b])?));
                    }
                    let elapsed = started.elapsed();
                    if shared {
                        statistics[index] = format!("{:?}", store.statistics());
                    }
                    results.sort_by_key(|(b, _)| *b);
                    for (_, result) in &results {
                        count(black_box(result), &mut hash);
                    }
                    if round > 0 {
                        times[index].push(elapsed.as_nanos() as f64);
                    }
                }
                "amplitude-off" | "amplitude-on" => {
                    let amplitude = amplitude.as_mut().ok_or("amplitude workspace")?;
                    amplitude.set_subset_sharing(mode == "amplitude-on");
                    started = Instant::now();
                    let results = amplitude.reduce_batches(&integrals)?;
                    let elapsed = started.elapsed();
                    statistics[index] = format!("{:?}", amplitude.last_sharing());
                    for result in &results {
                        count(black_box(result), &mut hash);
                    }
                    if round > 0 {
                        times[index].push(elapsed.as_nanos() as f64);
                    }
                }
                "rotation-two" | "rotation-copy" => {
                    let mut results = Vec::new();
                    started = Instant::now();
                    for (b, workspace) in rotation_workspaces.iter_mut().enumerate() {
                        if mode == "rotation-two" {
                            let primary = workspace.reduce_batch(&batches[b].1)?;
                            let copy = workspace.reduce_batch(&rotated[b])?;
                            results.push((primary, copy));
                        } else {
                            results.push(workspace.reduce_batch_with_rotation(&batches[b].1, 1)?);
                        }
                    }
                    let elapsed = started.elapsed();
                    for (primary, copy) in &results {
                        count(black_box(primary), &mut hash);
                        count(black_box(copy), &mut hash);
                    }
                    if round > 0 {
                        times[index].push(elapsed.as_nanos() as f64);
                    }
                }
                _ => return Err(format!("unknown mode {mode}").into()),
            }
            if round == 0 {
                checksums[index] = hash;
                lanes[index] = (served, total);
            } else if checksums[index] != hash {
                return Err(format!("{mode}: checksum changed between rounds").into());
            }
            if round > 0 {
                println!(
                    "{{\"engine\":\"{engine}\",\"mode\":\"{mode}\",\"round\":{round},\"ns\":{}}}",
                    times[index].last().unwrap()
                );
            }
        }
    }
    let mut summary = Vec::new();
    for (index, mode) in modes.iter().enumerate() {
        let min = times[index].iter().copied().fold(f64::INFINITY, f64::min);
        let med = median(&mut times[index].clone());
        summary.push(format!(
            "{{\"mode\":\"{mode}\",\"median_ms\":{:.3},\"min_ms\":{:.3},\"served\":{},\"lanes\":{},\"checksum\":\"{:016x}\",\"statistics\":\"{}\"}}",
            med / 1e6,
            min / 1e6,
            lanes[index].0,
            lanes[index].1,
            checksums[index],
            statistics[index]
        ));
    }
    println!(
        "FINAL {{\"engine\":\"{engine}\",\"points\":{count},\"rounds\":{rounds},\"modes\":[{}]}}",
        summary.join(",")
    );
    Ok(())
}
