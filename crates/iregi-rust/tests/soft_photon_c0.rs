//! Soft-photon triangles in binary64: oneloop's small-momentum C0 series
//! (alphal00p/oneloopmaster#21, pinned since huashengshao/TIRust#46).
//!
//! A photon k of energy 1e-4 to 1e-2 GeV leaves the lightlike leg
//! q1 = E (1, 0, 0, 1) at an angle of about 1e-3. The triangle has shifts
//! (0, q1, q1 - k), masses (0, mW^2, mW^2) and invariants
//! (q1^2, k^2, (q1 - k)^2) = (0, 0, -2 q1.k), with 2 q1.k = t mW^2 for t
//! from 3.1e-12 to 2.0e-10. Every input is exact in binary64, and the
//! supplied invariants are the exact invariants of the shifts.
//!
//! Up to oneloop 07fc754, binary64 C0 lost about 1e-16 / t of its value
//! here (1.1e-5 at t = 3.1e-12). The rank-1 lanes amplified that to O(1)
//! to O(10) of the lane (5.1, 3.6 and 0.066 at the three points below),
//! and the 1e-12 Gram gate still served them. Since oneloop cdbc912, C0 is
//! evaluated by a cancellation-free series when every external invariant
//! is small against the squared masses (at most one massless propagator,
//! real invariants). `DirectMasters::Native`, the direct engine's default,
//! reaches that series, and C0 now keeps binary64 accuracy (within 2e-16
//! here). The rank-1 lanes keep what PV's own rounding
//! through the Gram determinant, of order (t mW^2)^2, leaves: 0.6 to 2.3
//! times 2^-53 / t.
//!
//! The reference is the direct engine at 256 bits on the same inputs
//! (512 bits gives the same binary64 values). Every lane must be served,
//! the scalar C0 within `C0_LIMIT` and each rank-1 lane within
//! `RANK_ONE_MARGIN * 2^-53 / t`, both relative to the lane's largest
//! entry. At oneloop 07fc754 both fail by more than a factor of 1000.

use iregi_rust::{
    Convention, DirectReductionPlan, InvariantInput, KinematicsBatch, LaneStatus,
    ReductionSettings, ScaleInput,
};
use symbolica::domains::float::{Complex, Float, RealLike};

/// Energy of the lightlike leg q1 (GeV).
const E: f64 = 173.25;
/// W mass squared (GeV^2).
const MW2: f64 = 6467.215561;
/// The photons k = 2^-e (2048^2 + 1, 2 * 2048, 0, 2048^2 - 1): exactly
/// lightlike, at an angle of about 2 / 2048 to q1, so 2 q1.k = 4 E 2^-e.
/// Their energies are 1.2e-4, 9.8e-4 and 7.8e-3 GeV, and t = 3.1e-12,
/// 2.5e-11 and 2.0e-10.
const PHOTON_EXPONENTS: [i32; 3] = [35, 32, 29];
/// Largest accepted error of the binary64 scalar triangle (07fc754: 1.5e-7
/// to 1.1e-5; cdbc912: within 2e-16).
const C0_LIMIT: f64 = 1e-13;
/// Largest accepted error of a binary64 rank-1 lane, in units of 2^-53 / t
/// (07fc754: 1.2e5 to 8.2e5; cdbc912: 0.6 to 2.3).
const RANK_ONE_MARGIN: f64 = 100.0;

/// The MadLoop bridge's settings: supplied invariants and mu^2.
fn settings(rank: u16) -> ReductionSettings {
    let mut settings = ReductionSettings::new(3, rank);
    settings.convention = Convention::MadLoop;
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    settings
}

/// The triangle's row for the photon 2^-`exponent` (shifts, invariants
/// (s01, s02, s12), squared masses, mu^2 = mW^2) and its t = |s02| / mW^2.
fn row(exponent: i32) -> (Vec<Complex<f64>>, f64) {
    let c = 2f64.powi(-exponent);
    let m = 2048.0;
    let q1 = [E, 0.0, 0.0, E];
    let k = [c * (m * m + 1.0), c * 2.0 * m, 0.0, c * (m * m - 1.0)];
    let q2: [f64; 4] = std::array::from_fn(|a| q1[a] - k[a]);
    assert!((0..4).all(|a| q1[a] - q2[a] == k[a]), "q1 - k is not exact");
    let s02 = -4.0 * E * c;
    let real = |x: f64| Complex::new(x, 0.0);
    let mut values: Vec<_> = [[0.0; 4], q1, q2].into_iter().flatten().map(real).collect();
    values.extend([0.0, s02, 0.0].map(real));
    values.extend([0.0, MW2, MW2].map(real));
    values.push(real(MW2));
    (values, s02.abs() / MW2)
}

/// The binary64 lane (or its refusal) from the default direct workspace.
fn binary64(rank: u16, values: &[Complex<f64>]) -> Result<Vec<Complex<f64>>, LaneStatus> {
    let result = DirectReductionPlan::new(settings(rank))
        .unwrap()
        .workspace(0.0f64)
        .unwrap()
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values: values.to_vec(),
        })
        .unwrap();
    result.lane(0).map(<[_]>::to_vec).map_err(Clone::clone)
}

/// The lane at 256 bits (gate 2^-216, binary64 inputs), rounded to binary64.
fn reference(rank: u16, values: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut workspace = DirectReductionPlan::new(settings(rank))
        .unwrap()
        .workspace(Float::new(256))
        .unwrap();
    workspace.set_stability_threshold(Float::with_val(256, 2f64.powi(-216)));
    workspace.set_input_precision(53);
    let values = values
        .iter()
        .map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im)))
        .collect();
    let result = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 1,
            components: 1,
            values,
        })
        .unwrap();
    result
        .lane(0)
        .unwrap()
        .iter()
        .map(|z| Complex::new(z.re.to_f64(), z.im.to_f64()))
        .collect()
}

/// Largest |a - b| over the lane, relative to the lane's largest |b|.
fn error(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    let scale = b.iter().fold(0f64, |s, z| s.max(z.re.hypot(z.im)));
    let difference = a
        .iter()
        .zip(b)
        .fold(0f64, |e, (x, y)| e.max((x.re - y.re).hypot(x.im - y.im)));
    difference / scale
}

/// Each photon's (t, binary64 error, limit) at `rank`; panics on a refusal.
fn errors(rank: u16, limit: impl Fn(f64) -> f64) -> Vec<(f64, f64, f64)> {
    PHOTON_EXPONENTS
        .iter()
        .map(|&exponent| {
            let (values, t) = row(exponent);
            let lane = binary64(rank, &values)
                .unwrap_or_else(|status| panic!("rank {rank}, t = {t:.1e}: {status:?}"));
            let e = error(&lane, &reference(rank, &values));
            println!(
                "rank {rank}, t = {t:.2e}: error {e:.2e}, limit {:.2e}",
                limit(t)
            );
            (t, e, limit(t))
        })
        .collect()
}

#[test]
fn soft_photon_scalar_triangle_keeps_binary64_accuracy() {
    let failures: Vec<_> = errors(0, |_| C0_LIMIT)
        .into_iter()
        .filter(|&(_, e, l)| e > l)
        .collect();
    assert!(
        failures.is_empty(),
        "binary64 C0 beyond {C0_LIMIT:e} (t, error, limit): {failures:?}"
    );
}

#[test]
fn soft_photon_rank_one_triangle_matches_256_bits() {
    let failures: Vec<_> = errors(1, |t| RANK_ONE_MARGIN * 2f64.powi(-53) / t)
        .into_iter()
        .filter(|&(_, e, l)| e > l)
        .collect();
    assert!(
        failures.is_empty(),
        "binary64 rank-1 lanes beyond {RANK_ONE_MARGIN} * 2^-53 / t (t, error, limit): {failures:?}"
    );
}
