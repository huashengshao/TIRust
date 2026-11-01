//! Loop-rotation agreement over every cyclic propagator order.
//!
//! usage: rotation_probe POINTS_JSON INTEGRAL_ID INDEX...
//! (POINTS_JSON: a stability points file, or benchmark inputs with `cases`,
//! whose lanes carry their own masses.)
//! For each selected lane: the binary64 direct engine (rotation gate 1e-15)
//! in every cyclic order, its E/A against 256 bits per order and the E/A of
//! each order against order 0.
use iregi_rust::{BatchResult, DirectReductionPlan, KinematicsBatch, ReductionSettings};
use serde_json::{Value, json};
use symbolica::domains::float::{Complex, Float, RealLike};

fn number(value: &Value) -> f64 {
    match value.as_str() {
        Some(text) => text.parse().expect("decimal binary64 input"),
        None => value.as_f64().expect("numeric input"),
    }
}

fn error<T: RealLike, U: RealLike>(a: &BatchResult<T>, b: &BatchResult<U>, mu: f64) -> Option<f64> {
    let (Ok(x), Ok(y)) = (a.lane(0), b.lane(0)) else { return None };
    let (mut e, mut s) = (0.0f64, 0.0f64);
    for (m, monomial) in a.basis.iter().enumerate() {
        let w = mu.powi(i32::from(monomial.0.iter().sum::<u16>()));
        for slot in 0..4 {
            let (u, v) = (&x[4 * m + slot], &y[4 * m + slot]);
            let (vr, vi) = (v.re.to_f64(), v.im.to_f64());
            e = e.max((u.re.to_f64() - vr).hypot(u.im.to_f64() - vi) / w);
            s = s.max(vr.hypot(vi) / w);
        }
    }
    Some(if s > 0.0 { e / s } else { e })
}

fn rational(token: &Value) -> f64 {
    let text = token.as_str().unwrap();
    match text.split_once('/') {
        Some((a, b)) => a.parse::<f64>().unwrap() / b.parse::<f64>().unwrap(),
        None => text.parse().unwrap(),
    }
}

/// A benchmark-inputs case in the stability points layout, with per-lane masses.
fn from_case(case: &Value) -> Value {
    let lanes = case["lanes"].as_array().unwrap();
    let points: Vec<Value> = lanes.iter().map(|lane| json!(lane["shifts"].as_array().unwrap().iter()
        .map(|s| s.as_array().unwrap().iter().map(rational).collect::<Vec<f64>>()).collect::<Vec<_>>())).collect();
    let masses: Vec<Value> = lanes.iter().map(|lane| json!(lane["masses_squared"].as_array().unwrap().iter()
        .map(|m| rational(&m[0])).collect::<Vec<f64>>())).collect();
    json!({"n": case["n"], "rank": case["rank"], "mu": rational(&lanes[0]["mu"]), "points": points,
        "masses_squared": masses[0].clone(), "point_masses": masses})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let points: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let integral = match points.get("cases") {
        Some(cases) => from_case(cases.as_array().unwrap().iter().find(|c| c["id"].as_str() == Some(&args[2])).unwrap()),
        None => points["integrals"].as_array().unwrap().iter().find(|i| i["id"].as_str() == Some(&args[2])).unwrap().clone(),
    };
    let integral = &integral;
    let n = integral["n"].as_u64().unwrap() as usize;
    let rank = integral["rank"].as_u64().unwrap() as u16;
    let mu = number(&integral["mu"]);
    let masses: Vec<f64> = integral["masses_squared"].as_array().unwrap().iter().map(number).collect();
    let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank))?;
    let mut workspace = plan.workspace(0.0f64)?;
    workspace.set_stability_threshold(1e-15);
    let bits = 256;
    let mut reference = plan.workspace(Float::new(bits))?;
    reference.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))));
    for index in args[3..].iter().map(|s| s.parse::<usize>().unwrap()) {
        let masses: Vec<f64> = match integral.get("point_masses") {
            Some(all) => all[index].as_array().unwrap().iter().map(number).collect(),
            None => masses.clone(),
        };
        let shifts: Vec<Vec<f64>> = integral["points"][index].as_array().unwrap().iter()
            .map(|s| s.as_array().unwrap().iter().map(number).collect()).collect();
        let lane = |order: usize| {
            let mut values = Vec::new();
            for i in 0..n {
                values.extend(shifts[(i + order) % n].iter().map(|&x| Complex::new(x, 0.0)));
            }
            for i in 0..n {
                values.push(Complex::new(masses[(i + order) % n], 0.0));
            }
            values.push(Complex::new(mu, 0.0));
            KinematicsBatch { rows: 1, components: 1, values }
        };
        let base = lane(0);
        let lifted = KinematicsBatch { rows: 1, components: 1,
            values: base.values.iter().map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im))).collect() };
        let exact = reference.reduce_batch(&lifted)?;
        let orders: Vec<BatchResult<f64>> = (0..n).map(|order| workspace.reduce_batch(&lane(order))).collect::<Result<_, _>>()?;
        let truth: Vec<Option<f64>> = orders.iter().map(|r| error(r, &exact, mu)).collect();
        let against: Vec<Option<f64>> = orders.iter().map(|r| error(r, &orders[0], mu)).collect();
        let pairwise: Vec<Vec<Option<f64>>> = orders.iter().map(|a| orders.iter().map(|b| error(a, b, mu)).collect()).collect();
        println!("{}", json!({"index": index, "error": truth, "difference_to_order0": against, "pairwise": pairwise}));
    }
    Ok(())
}
