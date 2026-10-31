//! Phase timings of the direct engine at the precision-rescue tiers on a
//! near-coplanar box (development aid for the rescue cost).
//!
//! usage: direct_tier_profile [RANK] [DELTA] [REPEATS]
use std::time::Instant;

use iregi_rust::{DirectReductionPlan, DirectWorkspace, KinematicsBatch, ReductionSettings};
use iregi_masters::NativeFloat;
use symbolica::{
    domains::float::{Complex, DoubleFloat, Float, RealLike},
    evaluate::EvaluationDomain,
};

fn batch<R: NativeFloat>(lift: impl Fn(f64) -> R, delta: f64) -> KinematicsBatch<R> {
    let shifts = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, delta]];
    let mut values = Vec::new();
    for s in shifts {
        values.extend(s.map(|x| Complex::new(lift(x), lift(0.0))));
    }
    values.extend([1.0, 2.0, 3.0, 3.0].map(|m| Complex::new(lift(m), lift(0.0))));
    values.push(Complex::new(lift(3.0), lift(0.0)));
    KinematicsBatch { rows: 1, components: 1, values }
}

fn profile<R: NativeFloat>(name: &str, mut workspace: DirectWorkspace<R>, input: KinematicsBatch<R>, threshold: R, repeats: usize)
where
    Complex<R>: EvaluationDomain,
{
    workspace.set_stability_threshold(threshold);
    workspace.set_profiling(true);
    let first = Instant::now();
    let result = workspace.reduce_batch(&input).unwrap();
    let first = first.elapsed();
    let started = Instant::now();
    for _ in 0..repeats {
        workspace.reduce_batch(&input).unwrap();
    }
    let total = started.elapsed() / repeats as u32;
    let p = workspace.last_profile();
    println!(
        "{name:>8}: status {:?} first {first:?} steady {total:?} | geometry {:?} masters {:?} recurrence {:?} gather {:?} output {:?}",
        result.statuses[0], p.geometry, p.masters, p.recurrence, p.gather, p.output
    );
}

/// Phase timings of one stability-study point (`POINTS=file INTEGRAL=id INDEX=k`).
fn profile_point(path: &str) {
    let points: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let id = std::env::var("INTEGRAL").unwrap();
    let index: usize = std::env::var("INDEX").unwrap().parse().unwrap();
    let integral = points["integrals"].as_array().unwrap().iter().find(|i| i["id"] == id.as_str()).unwrap();
    let n = integral["n"].as_u64().unwrap() as usize;
    let rank = integral["rank"].as_u64().unwrap() as u16;
    let mu = integral["mu"].as_f64().unwrap();
    let masses: Vec<f64> = integral["masses_squared"].as_array().unwrap().iter().map(|m| m.as_f64().unwrap()).collect();
    let point: Vec<Vec<f64>> = integral["points"][index].as_array().unwrap().iter()
        .map(|s| s.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect();
    fn lane<R: NativeFloat>(lift: impl Fn(f64) -> R, point: &[Vec<f64>], masses: &[f64], mu: f64) -> KinematicsBatch<R> {
        let mut values = Vec::new();
        for shift in point {
            values.extend(shift.iter().map(|&x| Complex::new(lift(x), lift(0.0))));
        }
        values.extend(masses.iter().map(|&m| Complex::new(lift(m), lift(0.0))));
        values.push(Complex::new(lift(mu), lift(0.0)));
        KinematicsBatch { rows: 1, components: 1, values }
    }
    let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank)).unwrap();
    profile("f64", plan.workspace(0.0f64).unwrap(), lane(|x| x, &point, &masses, mu), 1e-12, 5);
    profile("dd", plan.workspace(DoubleFloat::from(0.0)).unwrap(), lane(DoubleFloat::from, &point, &masses, mu),
        DoubleFloat::from(2f64.powi(-66)), 5);
    profile("256", plan.workspace(Float::new(256)).unwrap(), lane(|x| Float::with_val(256, x), &point, &masses, mu),
        Float::with_val(256, 2f64.powi(-216)), 3);
}

fn main() {
    if std::env::var_os("DILOG_COST").is_some() {
        dilog_cost();
        return;
    }
    if let Some(path) = std::env::var_os("POINTS") {
        profile_point(path.to_str().unwrap());
        return;
    }
    let args: Vec<String> = std::env::args().collect();
    let rank: u16 = args.get(1).map_or(4, |a| a.parse().unwrap());
    let delta: f64 = args.get(2).map_or(1.0 / 1024.0, |a| a.parse().unwrap());
    let repeats: usize = args.get(3).map_or(5, |a| a.parse().unwrap());
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, rank)).unwrap();
    let started = Instant::now();
    let workspace = plan.workspace(0.0f64).unwrap();
    println!("f64 workspace {:?}", started.elapsed());
    profile("f64", workspace, batch(|x| x, delta), 0.0, repeats);
    let started = Instant::now();
    let workspace = plan.workspace(DoubleFloat::from(0.0)).unwrap();
    println!("dd workspace {:?}", started.elapsed());
    profile("dd", workspace, batch(DoubleFloat::from, delta), DoubleFloat::from(2f64.powi(-66)), repeats);
    for bits in [256u32, 512] {
        let started = Instant::now();
        let workspace = plan.workspace(Float::new(bits)).unwrap();
        println!("{bits} workspace {:?}", started.elapsed());
        profile(
            &format!("{bits}"),
            workspace,
            batch(|x| Float::with_val(bits, x), delta),
            Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))),
            repeats,
        );
    }
}

/// The oneloopmaster dilogarithm against Symbolica's generic polylog(2, z)
/// at each tier: worst relative difference over a grid that covers every
/// argument map, the real axis and the neighbourhoods of 0 and 1; then the
/// cost of one call each.
pub fn dilog_cost() {
    let mut points = vec![(1.0, 0.0), (-1.0, 0.0), (0.5, 0.0), (4.0, 0.0), (1.5, 0.0), (-7.0, 0.0),
        (1e-30, 0.0), (1.0 + 1e-20, 1e-25), (1e-18, -1e-19), (2.0, 1e-30), (1.0, -1e-15)];
    for i in -12..=12 {
        for j in -12..=12 {
            points.push((0.37 * i as f64, 0.29 * j as f64 + 1e-3));
        }
    }
    for bits in [106u32, 160, 256, 512] {
        let mut worst = (0.0f64, (0.0, 0.0));
        for &(re, im) in &points {
            let z = Complex::new(Float::with_val(bits, re), Float::with_val(bits, im));
            let ours = <Float as NativeFloat>::dilog(&z);
            let reference = oneloop::symbolica_dilog_float(&z);
            let difference = (ours.re.clone() - &reference.re).to_f64().hypot((ours.im.clone() - &reference.im).to_f64());
            let scale = reference.re.to_f64().hypot(reference.im.to_f64()).max(1e-300);
            let relative = difference / scale;
            if relative > worst.0 || !relative.is_finite() {
                worst = (relative, (re, im));
            }
            if im == 0.0 && ours.im.is_sign_negative() != reference.im.is_sign_negative() {
                println!("  sign of zero differs at {re}: {:?} vs {:?}", ours.im, reference.im);
            }
        }
        println!("dilog {bits} bits: worst relative difference {:.2e} at {:?} ({} points)", worst.0, worst.1, points.len());
        let z = Complex::new(Float::with_val(bits, 0.3), Float::with_val(bits, 0.2));
        let started = Instant::now();
        for _ in 0..200 {
            std::hint::black_box(<Float as NativeFloat>::dilog(std::hint::black_box(&z)));
        }
        let ours = started.elapsed() / 200;
        let started = Instant::now();
        for _ in 0..20 {
            std::hint::black_box(oneloop::symbolica_dilog_float(std::hint::black_box(&z)));
        }
        println!("  cost: oneloopmaster {ours:?}, Symbolica polylog {:?}", started.elapsed() / 20);
    }
    // Native double-double kernel against the 160-bit route on the grid.
    let mut worst = (0.0f64, (0.0, 0.0));
    for &(re, im) in &points {
        let z = Complex::new(DoubleFloat::from(re), DoubleFloat::from(im));
        let ours = <DoubleFloat as NativeFloat>::dilog(&z);
        let reference = oneloop::double_dilog_via_float(&z);
        let difference = (ours.re - reference.re).to_f64().hypot((ours.im - reference.im).to_f64());
        let scale = reference.re.to_f64().hypot(reference.im.to_f64()).max(1e-300);
        if difference / scale > worst.0 || !(difference / scale).is_finite() {
            worst = (difference / scale, (re, im));
        }
        if im == 0.0 && ours.im.to_f64().is_sign_negative() != reference.im.to_f64().is_sign_negative() {
            println!("  double-double sign of zero differs at {re}");
        }
    }
    println!("double-double native vs 160-bit route: worst relative difference {:.2e} at {:?}", worst.0, worst.1);
    let z = Complex::new(DoubleFloat::from(0.3), DoubleFloat::from(0.2));
    let started = Instant::now();
    for _ in 0..200 {
        std::hint::black_box(oneloop::double_dilog_via_float(std::hint::black_box(&z)));
    }
    println!("double-double via 160 bits: {:?}", started.elapsed() / 200);
    let started = Instant::now();
    for _ in 0..200 {
        std::hint::black_box(<DoubleFloat as NativeFloat>::dilog(std::hint::black_box(&z)));
    }
    println!("double-double dilog: {:?}", started.elapsed() / 200);
}
