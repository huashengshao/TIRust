//! Small-Gram expansion of boxes on stability-study points.
//!
//! For the selected lanes of one box integral: the binary64 direct engine
//! with the small-Gram expansion for boxes its Gram gate refuses (boosts
//! `B`), the same with every box expanded (gate 1), and the 256-bit
//! reference. One JSON line per lane with each status and E/A.
//!
//! usage: gram_probe POINTS_JSON INTEGRAL_ID STRIDE OFFSET LIMIT BOOST...
use iregi_rust::{BatchResult, DirectReductionPlan, KinematicsBatch, LaneStatus, ReductionSettings};
use serde_json::{Value, json};
use symbolica::domains::float::{Complex, Float, RealLike};

fn number(value: &Value) -> f64 {
    match value.as_str() {
        Some(text) => text.parse().expect("decimal binary64 input"),
        None => value.as_f64().expect("numeric input"),
    }
}

fn error<T: RealLike, U: RealLike>(a: &BatchResult<T>, lane: usize, b: &BatchResult<U>, mu: f64) -> Option<f64> {
    let (Ok(x), Ok(y)) = (a.lane(lane), b.lane(lane)) else { return None };
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let points: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let integral = points["integrals"].as_array().unwrap().iter().find(|i| i["id"].as_str() == Some(&args[2])).unwrap();
    let (stride, offset, limit): (usize, usize, usize) = (args[3].parse()?, args[4].parse()?, args[5].parse()?);
    let boosts: Vec<u16> = args[6..].iter().map(|s| s.parse().unwrap()).collect();
    let n = integral["n"].as_u64().unwrap() as usize;
    let rank = integral["rank"].as_u64().unwrap() as u16;
    let mu = number(&integral["mu"]);
    let masses: Vec<f64> = integral["masses_squared"].as_array().unwrap().iter().map(number).collect();
    let lanes: Vec<(usize, &Value)> = integral["points"].as_array().unwrap().iter().enumerate().skip(offset).step_by(stride).take(limit).collect();
    let mut values = Vec::new();
    for (_, point) in &lanes {
        for shift in point.as_array().unwrap() {
            values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(number(x), 0.0)));
        }
        values.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
        values.push(Complex::new(mu, 0.0));
    }
    let rows = lanes.len();
    let batch = KinematicsBatch { rows, components: 1, values: values.clone() };
    let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank))?;
    let gated = plan.workspace(0.0f64)?.reduce_batch(&batch)?;
    let mut expanded = Vec::new();
    for &boost in &boosts {
        let mut workspace = plan.workspace(0.0f64)?;
        workspace.enable_gram_expansion(boost);
        let refused_only = workspace.reduce_batch(&batch)?;
        let mut every = plan.workspace(0.0f64)?;
        every.enable_gram_expansion(boost);
        every.set_stability_threshold(1.0);
        let all = every.reduce_batch(&batch)?;
        expanded.push((boost, refused_only, all));
    }
    let bits = 256;
    let mut reference = plan.workspace(Float::new(bits))?;
    reference.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))));
    let lifted = KinematicsBatch {
        rows,
        components: 1,
        values: values.iter().map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im))).collect(),
    };
    let r = reference.reduce_batch(&lifted)?;
    let status = |res: &BatchResult<f64>, lane: usize| match &res.statuses[lane] {
        LaneStatus::Success => None,
        s => Some(format!("{s:?}")),
    };
    for (lane, (index, _)) in lanes.iter().enumerate() {
        let entries: Vec<Value> = expanded
            .iter()
            .map(|(boost, refused, all)| json!({
                "boost": boost,
                "refused_only": {"error": error(refused, lane, &r, mu), "status": status(refused, lane)},
                "every_box": {"error": error(all, lane, &r, mu), "status": status(all, lane)},
            }))
            .collect();
        println!("{}", json!({
            "index": index,
            "gated": {"error": error(&gated, lane, &r, mu), "status": status(&gated, lane)},
            "expansion": entries,
        }));
    }
    Ok(())
}
