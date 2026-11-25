//! Pentagons with one doubled massless line whose (D + 2)-dimensional
//! integrals are IR divergent: the line in a collinear pair (next to a
//! lightlike leg) or soft (between two on-shell legs). Melrose's scalar
//! reduction J5 = sum_i b_i J4^(i) and the Gram-free tensor coefficients
//! (hep-ph/0509141 Eq. (6.10)) drop terms of order (D - 4) times such
//! integrals, so at the doubled line's jet order they are finite. The direct
//! engine keeps them; these tests check every tensor coefficient against the
//! exact contraction identities
//!   2 q.(r_k - r_j) = D_k - D_j - (r_k^2 - m_k^2 - r_j^2 + m_j^2)
//! in every Laurent slot, at 256 bits, against the integrals with line k or
//! j lowered (the doubled line at unit power, another line pinched). The
//! scalar's value is checked against original IREGI in `tests/onia.rs`
//! (`collinear_pairs_reduce_dotted_triangles_boxes_and_pentagons`).
use iregi_rust::{DirectReductionPlan, KinematicsBatch, LaneStatus, ReductionSettings, basis::tensor_basis};
use symbolica::domains::float::{Complex, Float, RealLike};

const BITS: u32 = 256;
const MU: f64 = 3.0;

fn norm(z: &Complex<Float>) -> f64 {
    z.re.to_f64().hypot(z.im.to_f64())
}

/// The lane of `shifts` and squared `masses` (invariants from the shifts).
fn run(shifts: &[[f64; 4]], masses: &[f64], powers: &[u16], rank: u16) -> (LaneStatus, Vec<Complex<Float>>) {
    let mut settings = ReductionSettings::new(shifts.len(), rank);
    settings.powers = powers.to_vec();
    let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(BITS)).unwrap();
    workspace.set_stability_threshold(Float::with_val(BITS, 2f64.powi(-(BITS as i32 - 40))));
    let mut values: Vec<f64> = shifts.iter().flatten().copied().collect();
    values.extend(masses);
    values.push(MU);
    let values = values.iter().map(|&x| Complex::new(Float::with_val(BITS, x), Float::new(BITS))).collect();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
}

/// Binary64 lane, for the size of its error.
fn run64(shifts: &[[f64; 4]], masses: &[f64], powers: &[u16], rank: u16) -> (LaneStatus, Vec<Complex<f64>>) {
    let mut settings = ReductionSettings::new(shifts.len(), rank);
    settings.powers = powers.to_vec();
    let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(0.0f64).unwrap();
    let mut values: Vec<f64> = shifts.iter().flatten().copied().collect();
    values.extend(masses);
    values.push(MU);
    let values = values.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    (result.statuses[0].clone(), result.lane(0).map(<[_]>::to_vec).unwrap_or_default())
}

/// Worst residual of the contraction identities (j = line 0, every k),
/// relative to the largest entry they involve, and binary64's error.
fn residuals(shifts: &[[f64; 4]], masses: &[f64], powers: &[u16], rank: u16, label: &str) -> (f64, f64) {
    let n = shifts.len();
    let (status, t) = run(shifts, masses, powers, rank);
    assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}");
    let lowered = |k: usize| -> Vec<Complex<Float>> {
        let (status, values) = if powers[k] > 1 {
            let mut lower = powers.to_vec();
            lower[k] -= 1;
            run(shifts, masses, &lower, rank - 1)
        } else {
            let keep: Vec<usize> = (0..n).filter(|&i| i != k).collect();
            let s: Vec<[f64; 4]> = keep.iter().map(|&i| shifts[i]).collect();
            let m: Vec<f64> = keep.iter().map(|&i| masses[i]).collect();
            let p: Vec<u16> = keep.iter().map(|&i| powers[i]).collect();
            run(&s, &m, &p, rank - 1)
        };
        assert_eq!(status, LaneStatus::Success, "{label}, rank {rank}, line {k} lowered");
        values
    };
    let lows: Vec<Vec<Complex<Float>>> = (0..n).map(lowered).collect();
    let basis = tensor_basis(rank);
    let position = |m: [u16; 4]| basis.iter().position(|b| b.0 == m).unwrap();
    let big = |x: f64| Complex::new(Float::with_val(BITS, x), Float::new(BITS));
    let square = |i: usize| {
        let [t, x, y, z] = shifts[i];
        t * t - x * x - y * y - z * z
    };
    let metric = [1.0, -1.0, -1.0, -1.0];
    let (mut worst, mut size) = (0.0f64, 0.0f64);
    for k in 1..n {
        // Exact in binary64 for these dyadic shifts and masses.
        let c = big((square(k) - masses[k]) - (square(0) - masses[0]));
        for (row, monomial) in tensor_basis(rank - 1).iter().enumerate() {
            for slot in 0..4 {
                let mut lhs = big(0.0);
                for mu in 0..4 {
                    let mut up = monomial.0;
                    up[mu] += 1;
                    lhs = lhs + t[4 * position(up) + slot].clone() * big(2.0 * metric[mu] * (shifts[k][mu] - shifts[0][mu]));
                }
                let rhs = lows[k][4 * row + slot].clone()
                    - lows[0][4 * row + slot].clone()
                    - c.clone() * t[4 * position(monomial.0) + slot].clone();
                worst = worst.max(norm(&(lhs.clone() - rhs)));
                size = size.max(norm(&lhs)).max(norm(&lows[k][4 * row + slot]));
            }
        }
    }
    let (status, t64) = run64(shifts, masses, powers, rank);
    let binary64 = if status == LaneStatus::Success {
        let scale = t.iter().map(norm).fold(0.0, f64::max);
        t64.iter().zip(&t).map(|(a, b)| (a.re - b.re.to_f64()).hypot(a.im - b.im.to_f64())).fold(0.0, f64::max) / scale
    } else {
        f64::NAN
    };
    (worst / size, binary64)
}

/// `onia-collinear.json`'s points: lines 0 and 1 massless, separated by the
/// lightlike k = (3, 0, 0, 3), and three massive lines.
const COLLINEAR_SHIFTS: [[f64; 4]; 6] = [
    [0.0, 0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0, 3.0],
    [5.0, 1.0, 0.0, 2.0],
    [4.0, -1.0, 2.0, 1.0],
    [1.0, 0.5, -1.0, 2.0],
    [2.0, -0.5, 1.0, 0.25],
];
const COLLINEAR_MASSES: [f64; 6] = [0.0, 0.0, 4.0, 1.0, 2.0, 3.0];

/// Massless line 1 between two on-shell legs of lines 0 and 2 (m^2 = 4):
/// (r_1 - r_0)^2 = (r_2 - r_1)^2 = 4, a soft singularity.
const SOFT_SHIFTS: [[f64; 4]; 5] =
    [[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [4.5, 1.5, 0.0, 0.0], [1.0, 0.5, -1.0, 2.0], [-1.0, 0.25, 0.75, 0.5]];
const SOFT_MASSES: [f64; 5] = [4.0, 0.0, 4.0, 2.0, 1.0];

#[test]
fn doubled_massless_lines_keep_the_d_minus_4_terms_of_pentagons() {
    let mut worst = [0.0f64; 2];
    for (label, shifts, masses, powers) in [
        ("collinear pentagon, line 0 doubled", &COLLINEAR_SHIFTS[..5], &COLLINEAR_MASSES[..5], vec![2, 1, 1, 1, 1]),
        ("collinear pentagon, line 1 doubled", &COLLINEAR_SHIFTS[..5], &COLLINEAR_MASSES[..5], vec![1, 2, 1, 1, 1]),
        ("soft pentagon, line 1 doubled", &SOFT_SHIFTS[..], &SOFT_MASSES[..], vec![1, 2, 1, 1, 1]),
        ("collinear hexagon, line 1 doubled", &COLLINEAR_SHIFTS[..], &COLLINEAR_MASSES[..], vec![1, 2, 1, 1, 1, 1]),
    ] {
        for rank in 1..=5u16 {
            let (identity, binary64) = residuals(shifts, masses, &powers, rank, label);
            eprintln!("{label}, rank {rank}: identities {identity:.1e}, binary64 {binary64:.1e}");
            assert!(identity <= 1e-60, "{label}, rank {rank}: identity residual {identity:e}");
            worst = [worst[0].max(identity), worst[1].max(binary64)];
        }
    }
    eprintln!("worst: identities {:.1e}, binary64 {:.1e}", worst[0], worst[1]);
}

/// Jets other than one doubled line keep the terms at every order
/// (`PostOp::PentagonTerms`): a massless line tripled next to an on-shell
/// leg (the Coulomb gluon of two onia at threshold, `tests/seven_point_powers.rs`'s
/// pentagon [0, 2, 4, 5, 6]), the soft line doubled together with its
/// on-shell neighbour, a collinear pair member tripled, both pair members
/// doubled, a pair member and a massive line doubled, and the collinear
/// hexagon with two lines doubled. The same identities, ranks 1 to 5.
#[test]
fn raised_massless_lines_keep_the_d_minus_4_terms_at_every_order() {
    let threshold: [[f64; 4]; 5] =
        [[0.0, 0.0, 0.0, 0.0], [4.5, -3.0, -1.0, 8.0], [-1.5, -4.0, -2.0, 8.0], [-4.0, -2.0, -2.0, 8.0], [-7.0, 0.0, 0.0, 7.0]];
    let threshold_masses = [2.25, 0.0, 0.0, 2.25, 2.25];
    let mut worst = [0.0f64; 2];
    for (label, shifts, masses, powers) in [
        ("tripled gluon at threshold", &threshold[..], &threshold_masses[..], vec![1, 1, 3, 1, 1]),
        ("soft line and its neighbour doubled", &SOFT_SHIFTS[..], &SOFT_MASSES[..], vec![1, 2, 2, 1, 1]),
        ("collinear pentagon, line 1 tripled", &COLLINEAR_SHIFTS[..5], &COLLINEAR_MASSES[..5], vec![1, 3, 1, 1, 1]),
        ("collinear pentagon, lines 0 and 1 doubled", &COLLINEAR_SHIFTS[..5], &COLLINEAR_MASSES[..5], vec![2, 2, 1, 1, 1]),
        ("collinear pentagon, lines 1 and 2 doubled", &COLLINEAR_SHIFTS[..5], &COLLINEAR_MASSES[..5], vec![1, 2, 2, 1, 1]),
        ("collinear hexagon, lines 1 and 3 doubled", &COLLINEAR_SHIFTS[..], &COLLINEAR_MASSES[..], vec![1, 2, 1, 2, 1, 1]),
    ] {
        for rank in 1..=5u16 {
            let (identity, binary64) = residuals(shifts, masses, &powers, rank, label);
            eprintln!("{label}, rank {rank}: identities {identity:.1e}, binary64 {binary64:.1e}");
            assert!(identity <= 1e-60, "{label}, rank {rank}: identity residual {identity:e}");
            worst = [worst[0].max(identity), worst[1].max(binary64)];
        }
    }
    eprintln!("worst: identities {:.1e}, binary64 {:.1e}", worst[0], worst[1]);
}
