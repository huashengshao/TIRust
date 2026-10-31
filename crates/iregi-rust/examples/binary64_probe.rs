//! Binary64 refusals of the direct engine on stability-study points.
//!
//! For the selected lanes of one integral: the binary64 direct engine with
//! the public gate (1e-15 with the rotation test for N >= 5, the default
//! otherwise), the same with the gate open, its loop-rotated copy, the
//! partial (double-double) rescue, and the 256-bit reference. One JSON line
//! per lane with the refusal message, the conditioning estimate and every
//! E/A (moments divided by mu^rank) against the reference.
//!
//! usage: binary64_probe POINTS_JSON INTEGRAL_ID STRIDE [LIMIT] [OFFSET]
use iregi_rust::{
    BatchResult, DirectReductionPlan, KinematicsBatch, LaneStatus, PrecisionRescue, ReductionSettings,
    TensorReductionPlan, TensorWorkspace, TensorWorkspaceOptions,
};
use serde_json::{Value, json};
use symbolica::domains::float::{Complex, Float, RealLike};

fn number(value: &Value) -> f64 {
    match value.as_str() {
        Some(text) => text.parse().expect("decimal binary64 input"),
        None => value.as_f64().expect("numeric input"),
    }
}

fn error<T: RealLike, U: RealLike>(
    a: &BatchResult<T>,
    lane: usize,
    b: &BatchResult<U>,
    reference_lane: usize,
    mu: f64,
) -> Option<f64> {
    let (Ok(x), Ok(y)) = (a.lane(lane), b.lane(reference_lane)) else { return None };
    let (mut e, mut s) = (0.0f64, 0.0f64);
    for (m, monomial) in a.basis.iter().enumerate() {
        let rank: u16 = monomial.0.iter().sum();
        let w = mu.powi(i32::from(rank));
        for slot in 0..4 {
            let (u, v) = (&x[4 * m + slot], &y[4 * m + slot]);
            let (vr, vi) = (v.re.to_f64(), v.im.to_f64());
            e = e.max((u.re.to_f64() - vr).hypot(u.im.to_f64() - vi) / w);
            s = s.max(vr.hypot(vi) / w);
        }
    }
    Some(if s > 0.0 { e / s } else { e })
}

fn rotated(values: &[Complex<f64>], rows: usize, n: usize) -> Vec<Complex<f64>> {
    let width = 5 * n + 1;
    let mut out = Vec::with_capacity(values.len());
    for lane in 0..rows {
        let row = &values[lane * width..(lane + 1) * width];
        for k in 0..n {
            let from = (k + 1) % n;
            out.extend_from_slice(&row[4 * from..4 * from + 4]);
        }
        for k in 0..n {
            out.push(row[4 * n + (k + 1) % n]);
        }
        out.push(row[5 * n]);
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let points: Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let id = &args[2];
    let stride: usize = args[3].parse()?;
    let limit: usize = args.get(4).map_or(usize::MAX, |s| s.parse().unwrap());
    let offset: usize = args.get(5).map_or(0, |s| s.parse().unwrap());
    let integral = points["integrals"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["id"].as_str() == Some(id))
        .expect("integral id");
    let n = integral["n"].as_u64().unwrap() as usize;
    let rank = integral["rank"].as_u64().unwrap() as u16;
    let mu = number(&integral["mu"]);
    let masses: Vec<f64> = integral["masses_squared"].as_array().unwrap().iter().map(number).collect();
    let lanes: Vec<(usize, &Value)> = integral["points"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .skip(offset)
        .step_by(stride)
        .take(limit)
        .collect();
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
    let rotated_batch = KinematicsBatch { rows, components: 1, values: rotated(&values, rows, n) };
    let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank))?;
    let gate = if n >= 5 { Some(1e-15) } else { None };

    let mut gated = plan.workspace(0.0f64)?;
    if let Some(g) = gate {
        gated.set_stability_threshold(g);
    }
    let a = gated.reduce_batch(&batch)?;
    let conditions = gated.last_conditions().to_vec();
    let mut open = plan.workspace(0.0f64)?;
    open.set_stability_threshold(0.0);
    let b = open.reduce_batch(&batch)?;
    let c = open.reduce_batch(&rotated_batch)?;
    let mut partial = plan.workspace(0.0f64)?;
    if let Some(g) = gate {
        partial.set_stability_threshold(g);
    }
    partial.enable_partial_rescue(2f64.powi(-86));
    let p = partial.reduce_batch(&batch)?;
    let mut gated_rot = plan.workspace(0.0f64)?;
    if let Some(g) = gate {
        gated_rot.set_stability_threshold(g);
    }
    let ar = gated_rot.reduce_batch(&rotated_batch)?;

    // The public facade: binary64 only (rescue disabled) and the default.
    let facade_plan = TensorReductionPlan::new(ReductionSettings::new(n, rank))?;
    let options = TensorWorkspaceOptions { precision_rescue: PrecisionRescue::Disabled, ..Default::default() };
    let mut only64 = TensorWorkspace::new_with_options(&facade_plan, 0.0f64, options)?;
    let f64only = only64.reduce_batch(&batch)?;
    let only64_notes = only64.last_diagnostics().to_vec();
    let mut default = TensorWorkspace::new(&facade_plan, 0.0f64)?;
    let fdefault = default.reduce_batch(&batch)?;

    let bits = 256;
    let mut reference = plan.workspace(Float::new(bits))?;
    reference.set_stability_threshold(Float::with_val(bits, 2f64.powi(-(bits as i32 - 40))));
    let lifted = KinematicsBatch {
        rows,
        components: 1,
        values: values
            .iter()
            .map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im)))
            .collect(),
    };
    let r = reference.reduce_batch(&lifted)?;

    for (lane, (index, _)) in lanes.iter().enumerate() {
        let message = |res: &BatchResult<f64>| match &res.statuses[lane] { LaneStatus::Success => None, s => Some(format!("{s:?}")) };
        println!(
            "{}",
            json!({
                "index": index,
                "reference": r.statuses[lane] == LaneStatus::Success,
                "gated": {"error": error(&a, lane, &r, lane, mu), "refusal": message(&a), "condition": conditions[lane]},
                "gated_rotation": error(&a, lane, &ar, lane, mu),
                "open": {"error": error(&b, lane, &r, lane, mu), "refusal": message(&b)},
                "open_rotated": {"error": error(&c, lane, &r, lane, mu), "difference": error(&b, lane, &c, lane, mu)},
                "partial": {"error": error(&p, lane, &r, lane, mu), "refusal": message(&p)},
                "binary64": {"error": error(&f64only, lane, &r, lane, mu), "refusal": message(&f64only),
                             "fallback_used": f64only.fallback_used[lane], "diagnostics": only64_notes.get(lane)},
                "default": {"error": error(&fdefault, lane, &r, lane, mu), "refusal": message(&fdefault),
                            "fallback_used": fdefault.fallback_used[lane]},
            })
        );
    }
    Ok(())
}
