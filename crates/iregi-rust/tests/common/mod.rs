//! pyAmpliCol's TIR lanes from a fixture (`fixtures/*_lanes.json`; m-lit
//! lit-tir): binary64 inputs, numerator coefficients on NinjaRS's monomials
//! `q0^a q1^b q2^c q3^d` (contracted with the Cartesian moments `M_alpha`)
//! and a truth `[finite, 1/eps, 1/eps^2]` = `[M_fin, M_UV1 + M_IR1, M_IR2]`
//! contracted. Settings as pyAmpliCol's TIR backend runs them
//! (`Convention::OneLoop`, `TensorWeights::Components`), the invariants from
//! the shifts or supplied (`InvariantInput::SuppliedInvariants`).
#![allow(dead_code)]

pub mod hexagon;

use iregi_rust::{
    Convention, DirectReductionPlan, InvariantInput, KinematicsBatch, LaneStatus,
    ReductionSettings, TensorWeights,
};
use serde_json::Value;
use symbolica::domains::float::{Complex, Float, RealLike};

#[derive(Clone)]
pub struct Lane {
    pub label: String,
    pub n: usize,
    pub rank: u16,
    pub shifts: Vec<[f64; 4]>,
    pub masses: Vec<Complex<f64>>,
    pub mu2: f64,
    /// (V_i - V_j)^2, i < j, row-major (the supplied S-matrix).
    pub invariants: Vec<Complex<f64>>,
    pub coefficients: Vec<([u16; 4], Complex<f64>)>,
    pub truth: [Complex<f64>; 3],
}

fn complex(value: &Value) -> Complex<f64> {
    Complex::new(value[0].as_f64().unwrap(), value[1].as_f64().unwrap())
}

pub fn lanes(fixture: &str) -> Vec<Lane> {
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    fixture["lanes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lane| Lane {
            label: lane["label"].as_str().unwrap().to_string(),
            n: lane["n"].as_u64().unwrap() as usize,
            rank: lane["rank"].as_u64().unwrap() as u16,
            shifts: lane["shifts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| std::array::from_fn(|a| v[a].as_f64().unwrap()))
                .collect(),
            masses: lane["masses"]
                .as_array()
                .unwrap()
                .iter()
                .map(complex)
                .collect(),
            mu2: lane["mu2"].as_f64().unwrap(),
            invariants: lane["invariants"]
                .as_array()
                .unwrap()
                .iter()
                .map(complex)
                .collect(),
            coefficients: lane["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        std::array::from_fn(|a| c[0][a].as_u64().unwrap() as u16),
                        complex(&c[1]),
                    )
                })
                .collect(),
            truth: std::array::from_fn(|slot| complex(&lane["truth"][slot])),
        })
        .collect()
}

pub fn settings(lane: &Lane, supplied: bool) -> ReductionSettings {
    let mut settings = ReductionSettings::new(lane.n, lane.rank);
    settings.convention = Convention::OneLoop;
    settings.tensor_weights = TensorWeights::Components;
    if supplied {
        settings.invariants = InvariantInput::SuppliedInvariants;
    }
    settings
}

/// The lane's kinematic row (shifts, supplied invariants, masses, mu).
pub fn row(lane: &Lane, supplied: bool) -> Vec<Complex<f64>> {
    let mut values = Vec::new();
    for shift in &lane.shifts {
        values.extend(shift.map(|x| Complex::new(x, 0.0)));
    }
    if supplied {
        values.extend_from_slice(&lane.invariants);
    }
    values.extend_from_slice(&lane.masses);
    values.push(Complex::new(lane.mu2.sqrt(), 0.0));
    values
}

pub fn batch(lane: &Lane, supplied: bool) -> KinematicsBatch<f64> {
    KinematicsBatch {
        rows: 1,
        components: 1,
        values: row(lane, supplied),
    }
}

/// pyAmpliCol's lane `[finite, 1/eps, 1/eps^2]` from a lane array.
pub fn contract(
    basis: &[iregi_rust::basis::Monomial],
    array: &[Complex<f64>],
    lane: &Lane,
) -> [Complex<f64>; 3] {
    let mut out = [Complex::new(0.0, 0.0); 3];
    for (alpha, coefficient) in &lane.coefficients {
        let b = basis
            .iter()
            .position(|m| m.0 == *alpha)
            .expect("monomial in the basis");
        let s = &array[4 * b..4 * b + 4];
        for (slot, value) in [s[3], s[0] + s[1], s[2]].into_iter().enumerate() {
            out[slot] += *coefficient * value;
        }
    }
    out
}

/// Largest deviation over the three slots, relative to the truth's largest.
pub fn error(value: &[Complex<f64>; 3], truth: &[Complex<f64>; 3]) -> f64 {
    let size = |z: &Complex<f64>| z.re.hypot(z.im);
    let scale = truth.iter().map(size).fold(0.0, f64::max);
    value
        .iter()
        .zip(truth)
        .map(|(a, b)| size(&(*a - *b)))
        .fold(0.0, f64::max)
        / scale
}

/// The lane by the 256-bit direct engine (accepting conditioning down to
/// 2^-216) on the same binary64 inputs, or its status.
pub fn reference(lane: &Lane, supplied: bool) -> Result<[Complex<f64>; 3], LaneStatus> {
    let lifted: Vec<Complex<Float>> = row(lane, supplied)
        .iter()
        .map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
        .collect();
    let mut workspace = DirectReductionPlan::new(settings(lane, supplied))
        .unwrap()
        .workspace(Float::new(256))
        .unwrap();
    workspace.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: lifted,
        })
        .unwrap();
    if result.statuses[0] != LaneStatus::Success {
        return Err(result.statuses[0].clone());
    }
    let array: Vec<Complex<f64>> = result
        .lane(0)
        .unwrap()
        .iter()
        .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
        .collect();
    Ok(contract(&result.basis, &array, lane))
}
