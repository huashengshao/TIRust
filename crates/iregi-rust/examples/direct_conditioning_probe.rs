//! Robustness probe: boxes approaching a coplanar (Gram-singular) configuration.
//! Compares the public binary64 facade (direct engine + fallback) with a
//! 128-bit symbolic reference and reports the error and the path taken.
use iregi_rust::{KinematicsBatch, ReductionSettings, TensorEngineExecution, TensorReductionPlan, TensorWorkspace};
use symbolica::domains::float::{Complex, Float, RealLike};

fn main() {
    let rank = 4;
    for delta in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 0.0] {
        // p3 = e_x + e_y + delta e_z: coplanar with e_x, e_y at delta = 0.
        let shifts = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, delta]];
        let masses = [1.0, 2.0, 3.0, 3.0];
        let mut row = Vec::new();
        for s in shifts { row.extend(s.map(|x| Complex::new(x, 0.0))); }
        row.extend(masses.map(|m| Complex::new(m, 0.0)));
        row.push(Complex::new(3.0, 0.0));
        let batch = KinematicsBatch { rows: 1, components: 1, values: row.clone() };
        let settings = ReductionSettings::new(4, rank);
        let plan = TensorReductionPlan::new(settings.clone()).unwrap();
        let mut facade = TensorWorkspace::new(&plan, 0.0f64).unwrap();
        let result = facade.reduce_batch(&batch).unwrap();
        let fallback = match facade.engine_execution() { TensorEngineExecution::Direct(s) => s.fallback_rows, _ => 99 };
        let symbolic = TensorReductionPlan::new_without_direct(settings).unwrap();
        let mut wide = TensorWorkspace::new(&symbolic, Float::new(128)).unwrap();
        let wide_batch = KinematicsBatch { rows: 1, components: 1, values: row.iter().map(|z| Complex::new(Float::with_val(128, z.re), Float::with_val(128, z.im))).collect() };
        let reference = wide.reduce_batch(&wide_batch).unwrap();
        let (Ok(a), Ok(b)) = (result.lane(0), reference.lane(0)) else {
            println!("delta {delta:e}: facade {:?} reference {:?}", result.statuses[0], reference.statuses[0]);
            continue;
        };
        let (mut e, mut amp) = (0.0f64, 0.0f64);
        for (m, mono) in result.basis.iter().enumerate() {
            let w = 3f64.powi(mono.rank() as i32 - 4);
            for slot in 0..4 {
                let x = a[4 * m + slot];
                let r = &b[4 * m + slot];
                let (rr, ri) = (r.re.to_f64(), r.im.to_f64());
                e = e.max((x.re - rr).hypot(x.im - ri) / w);
                amp = amp.max(rr.hypot(ri) / w);
            }
        }
        println!("delta {delta:e}: E/A {:.2e} fallback_rows {fallback}", e / amp);
    }
}
