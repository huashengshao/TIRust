//! Accuracy of the binary64 direct engine on the box and triangle subsets of
//! stability-study points (development aid for the small-Gram work).
//!
//! Every subset of `SIZE` propagators of each point becomes a standalone
//! integral of rank `RANK`; it is evaluated in binary64 with the default gate,
//! in binary64 with the gate disabled, and at 256 bits. One JSON line per
//! subset: E/A of both binary64 arrays against 256 bits (moments weighted by
//! mu^rank) and the gate's refusal message.
//!
//! usage: subset_accuracy POINTS_JSON INTEGRAL SIZE RANK [FIRST] [COUNT]
use iregi_rust::{BatchResult, DirectReductionPlan, KinematicsBatch, ReductionSettings};
use serde_json::json;
use symbolica::domains::float::{Complex, Float, RealLike};

fn number(value: &serde_json::Value) -> f64 {
    match value.as_str() {
        Some(text) => text.parse().unwrap(),
        None => value.as_f64().unwrap(),
    }
}

fn subsets(n: usize, size: usize) -> Vec<Vec<usize>> {
    (0u32..1 << n)
        .filter(|m| m.count_ones() as usize == size)
        .map(|m| (0..n).filter(|i| m & (1 << i) != 0).collect())
        .collect()
}

fn error<T: RealLike, U: RealLike>(a: &BatchResult<T>, b: &BatchResult<U>, mu: f64) -> Option<f64> {
    let (Ok(x), Ok(y)) = (a.lane(0), b.lane(0)) else { return None };
    let (mut e, mut s) = (0.0f64, 0.0f64);
    for (m, monomial) in a.basis.iter().enumerate() {
        let rank: u16 = monomial.0.iter().sum();
        let w = mu.powi(i32::from(rank));
        for slot in 0..4 {
            let (u, v) = (&x[4 * m + slot], &y[4 * m + slot]);
            let (vr, vi) = (v.re.to_f64(), v.im.to_f64());
            e = e.max(w * (u.re.to_f64() - vr).hypot(u.im.to_f64() - vi));
            s = s.max(w * vr.hypot(vi));
        }
    }
    if std::env::var("DETAIL").is_ok() {
        // Per moment rank and Laurent slot: max |difference| and max |reference|.
        for r in 0..=a.basis.iter().map(|m| m.0.iter().sum::<u16>()).max().unwrap_or(0) {
            for slot in 0..4 {
                let (mut de, mut ds) = (0.0f64, 0.0f64);
                for (m, monomial) in a.basis.iter().enumerate() {
                    if monomial.0.iter().sum::<u16>() != r {
                        continue;
                    }
                    let w = mu.powi(i32::from(r));
                    let (u, v) = (&x[4 * m + slot], &y[4 * m + slot]);
                    let (vr, vi) = (v.re.to_f64(), v.im.to_f64());
                    de = de.max(w * (u.re.to_f64() - vr).hypot(u.im.to_f64() - vi));
                    ds = ds.max(w * vr.hypot(vi));
                }
                eprintln!("rank {r} slot {slot}: |diff| {de:.3e} |ref| {ds:.3e}");
            }
        }
    }
    Some(if s > 0.0 { e / s } else { e })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let points: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let id = &args[2];
    let size: usize = args[3].parse().unwrap();
    let rank: u16 = args[4].parse().unwrap();
    let first: usize = args.get(5).map_or(0, |s| s.parse().unwrap());
    let integral = points["integrals"].as_array().unwrap().iter().find(|i| i["id"] == id.as_str()).unwrap();
    let n = integral["n"].as_u64().unwrap() as usize;
    let mu = number(&integral["mu"]);
    let masses: Vec<f64> = integral["masses_squared"].as_array().unwrap().iter().map(number).collect();
    let lanes = integral["points"].as_array().unwrap();
    let count: usize = args.get(6).map_or(lanes.len() - first, |s| s.parse().unwrap());
    let plan = DirectReductionPlan::new(ReductionSettings::new(size, rank)).unwrap();
    let mut gated = plan.workspace(0.0f64).unwrap();
    if let Ok(t) = std::env::var("THRESH") {
        gated.set_stability_threshold(t.parse().unwrap());
    }
    let mut open = plan.workspace(0.0f64).unwrap();
    open.set_stability_threshold(0.0);
    let bits = 256;
    let mut wide = plan.workspace(Float::new(bits)).unwrap();
    wide.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))));
    for index in first..(first + count).min(lanes.len()) {
        let point: Vec<Vec<f64>> = lanes[index].as_array().unwrap().iter()
            .map(|s| s.as_array().unwrap().iter().map(number).collect()).collect();
        for members in subsets(n, size) {
            let mut values = Vec::new();
            for &i in &members {
                values.extend(point[i].iter().map(|&x| Complex::new(x, 0.0)));
            }
            values.extend(members.iter().map(|&i| Complex::new(masses[i], 0.0)));
            values.push(Complex::new(mu, 0.0));
            let batch = KinematicsBatch { rows: 1, components: 1, values };
            let lifted = KinematicsBatch {
                rows: 1,
                components: 1,
                values: batch.values.iter().map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im))).collect(),
            };
            let reference = wide.reduce_batch(&lifted).unwrap();
            let a = gated.reduce_batch(&batch).unwrap();
            let b = open.reduce_batch(&batch).unwrap();
            // Optional independent reference: the symbolic engine at 128 bits.
            let double = {
                use symbolica::domains::float::DoubleFloat;
                let input = KinematicsBatch { rows: 1, components: 1, values: batch.values.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect() };
                let mut ws = plan.workspace(DoubleFloat::from(0.0)).unwrap();
                ws.set_stability_threshold(DoubleFloat::from(0.0));
                let result = ws.reduce_batch(&input).unwrap();
                (error(&result, &reference, mu), error(&b, &result, mu))
            };
            let symbolic = std::env::var("SYMBOLIC").ok().map(|_| {
                let plan = iregi_rust::TensorReductionPlan::new_without_direct(ReductionSettings::new(size, rank)).unwrap();
                let mut ws = iregi_rust::TensorWorkspace::new(&plan, Float::new(128)).unwrap();
                let input = KinematicsBatch { rows: 1, components: 1, values: batch.values.iter().map(|z| Complex::new(Float::with_val(128, z.re), Float::with_val(128, z.im))).collect() };
                let result = ws.reduce_batch(&input).unwrap();
                eprintln!("symbolic status {:?}", result.statuses[0]);
                (error(&result, &reference, mu), error(&b, &result, mu))
            });
            println!("{}", json!({
                "index": index, "members": members,
                "gated": error(&a, &reference, mu), "open": error(&b, &reference, mu),
                "gate": a.statuses[0].clone().err_message(),
                "dd_vs_256": double.0, "open_vs_dd": double.1, "symbolic_vs_256": symbolic.map(|s| s.0), "open_vs_symbolic": symbolic.map(|s| s.1),
                "reference": format!("{:?}", reference.statuses[0]).chars().take(80).collect::<String>(),
            }));
        }
    }
}

trait Message {
    fn err_message(self) -> Option<String>;
}

impl Message for iregi_rust::LaneStatus {
    fn err_message(self) -> Option<String> {
        match self {
            iregi_rust::LaneStatus::Success => None,
            other => Some(format!("{other:?}").chars().take(160).collect()),
        }
    }
}
