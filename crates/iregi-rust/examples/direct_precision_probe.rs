//! Direct engine at extended precision on near-coplanar boxes (rank 4):
//! self-consistency 128 vs 256 bits and agreement with binary64 at delta=1/8.
use iregi_rust::{DirectReductionPlan, KinematicsBatch, ReductionSettings};
use symbolica::domains::float::{Complex, Float, RealLike};

fn lane(bits: u32, delta: f64) -> Option<Vec<(f64, f64)>> {
    let shifts = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, delta]];
    let masses = [1.0, 2.0, 3.0, 3.0];
    let f = |x: f64| Float::with_val(bits, x);
    let mut values = Vec::new();
    for s in shifts { values.extend(s.map(|x| Complex::new(f(x), f(0.0)))); }
    values.extend(masses.map(|m| Complex::new(f(m), f(0.0))));
    values.push(Complex::new(f(3.0), f(0.0)));
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, 4)).unwrap();
    let mut workspace = plan.workspace(Float::new(bits)).unwrap();
    // Accept conditioning down to 2^-(bits-60) for this probe.
    workspace.set_stability_threshold(Float::with_val(bits, 2.0f64.powi(-(bits as i32 - 60))));
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    match result.lane(0) {
        Ok(v) => Some(v.iter().map(|z| (z.re.to_f64(), z.im.to_f64())).collect()),
        Err(status) => { println!("  {bits} bits delta {delta:e}: {status:?}"); None }
    }
}

fn lane64(delta: f64) -> Option<Vec<(f64, f64)>> {
    let shifts = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, delta]];
    let mut values = Vec::new();
    for s in shifts { values.extend(s.map(|x| Complex::new(x, 0.0))); }
    values.extend([1.0, 2.0, 3.0, 3.0].map(|m| Complex::new(m, 0.0)));
    values.push(Complex::new(3.0, 0.0));
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, 4)).unwrap();
    let r = plan.workspace(0.0f64).unwrap().reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    r.lane(0).ok().map(|v| v.iter().map(|z| (z.re, z.im)).collect())
}

fn error(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let basis = iregi_rust::basis::tensor_basis(4);
    let (mut e, mut m) = (0.0f64, 0.0f64);
    for (k, mono) in basis.iter().enumerate() {
        let w = 3f64.powi(mono.rank() as i32 - 4);
        for s in 0..4 {
            let (x, y) = (a[4 * k + s], b[4 * k + s]);
            e = e.max((x.0 - y.0).hypot(x.1 - y.1) / w);
            m = m.max(y.0.hypot(y.1) / w);
        }
    }
    e / m
}

fn main() {
    for delta in [0.125, 1.0 / 64.0, 1.0 / 1024.0, 1.0 / 65536.0] {
        let (a, b) = (lane(128, delta), lane(256, delta));
        let c = lane64(delta);
        print!("delta {delta:e}:");
        if let (Some(a), Some(b)) = (&a, &b) { print!(" 128 vs 256 E/A {:.2e}", error(a, b)); }
        if let (Some(c), Some(b)) = (&c, &b) { print!("; f64 vs 256 E/A {:.2e}", error(c, b)); }
        println!();
    }
}
