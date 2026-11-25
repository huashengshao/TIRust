//! Seven-point integrals with raised powers (two P-wave onia and a gluon):
//! the direct engine's mass-derivative jets through the seven-point thin
//! system and through the Coulomb relations of two onia at threshold,
//! against exact identities evaluated from six-point integrals at 256 bits.
use iregi_rust::{DirectReductionPlan, KinematicsBatch, LaneStatus, ReductionSettings, ScaleInput};
use symbolica::domains::float::{Complex, Float, Real, RealLike};

const BITS: u32 = 256;

fn big(x: f64) -> Float {
    Float::with_val(BITS, x)
}

struct Point {
    shifts: Vec<[f64; 4]>,
    masses: Vec<f64>,
}

/// Heavy lines of m^2 = 9/4 and a massless Coulomb gluon inside each onium:
/// lines 0..6 are heavy except 2 and 4.
const MASSES: [f64; 7] = [2.25, 2.25, 0.0, 2.25, 0.0, 2.25, 2.25];
const MU2: f64 = 4.0;

/// g g > (Q Qbar)(Q Qbar) g with each pair at P_i/2 (P_i^2 = 9 = 4 m^2), in
/// exactly representable momenta: legs (incoming) k1, -P1/2, -P1/2, -P2/2,
/// -P2/2, -k3, k2 between lines 0..6. Lines 1, 2, 3 and 3, 4, 5 are Coulomb
/// triples (D_1 - 2 D_2 + D_3 = 0), so no six of the seven points carry a
/// regular thin system.
fn threshold(variant: usize) -> Point {
    let (k1, p1, p2, k3): ([f64; 4], [f64; 4], [f64; 4], [f64; 4]) = match variant {
        0 => ([8.0, 0.0, 0.0, 8.0], [7.0, 6.0, 2.0, 0.0], [5.0, -4.0, 0.0, 0.0], [3.0, -2.0, -2.0, 1.0]),
        _ => ([12.5, 0.0, 0.0, 12.5], [5.0, 0.0, 4.0, 0.0], [10.5, -2.0, -5.0, 8.5], [3.0, 2.0, 1.0, -2.0]),
    };
    let half = |p: [f64; 4]| p.map(|x| -x / 2.0);
    let legs = [k1, half(p1), half(p1), half(p2), half(p2), k3.map(|x| -x)];
    let mut shifts = vec![[0.0; 4]];
    for leg in legs {
        let last = *shifts.last().unwrap();
        shifts.push(std::array::from_fn(|a| last[a] + leg[a]));
    }
    Point { shifts, masses: MASSES.to_vec() }
}

/// Seven generic points (dyadic shifts in [-8, 8)), the same masses: the
/// thin system over six of them is regular.
fn generic() -> Point {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) % 4096) as f64 / 256.0 - 8.0
    };
    let shifts = (0..7).map(|i| if i == 0 { [0.0; 4] } else { std::array::from_fn(|_| next()) }).collect();
    Point { shifts, masses: MASSES.to_vec() }
}

fn settings(n: usize, rank: u16, powers: &[u16]) -> ReductionSettings {
    let mut s = ReductionSettings::new(n, rank);
    s.powers = powers.to_vec();
    s.scale_input = ScaleInput::MuSquared;
    s
}

fn row<T: Clone>(p: &Point, members: &[usize], lift: impl Fn(f64) -> T, zero: T) -> Vec<Complex<T>> {
    let mut v = Vec::new();
    for &i in members {
        v.extend(p.shifts[i].iter().map(|&x| Complex::new(lift(x), zero.clone())));
    }
    for &i in members {
        v.push(Complex::new(lift(p.masses[i]), zero.clone()));
    }
    v.push(Complex::new(lift(MU2), zero.clone()));
    v
}

fn lane_256(p: &Point, members: &[usize], rank: u16, powers: &[u16]) -> Vec<Complex<Float>> {
    let mut w = DirectReductionPlan::new(settings(members.len(), rank, powers)).unwrap().workspace(Float::new(BITS)).unwrap();
    let r = w.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: row(p, members, big, big(0.0)) }).unwrap();
    assert_eq!(r.statuses[0], LaneStatus::Success, "256 bits, members {members:?}, powers {powers:?}, rank {rank}");
    r.lane(0).unwrap().to_vec()
}

fn lane_binary64(p: &Point, rank: u16, powers: &[u16]) -> Vec<Complex<f64>> {
    let members: Vec<usize> = (0..7).collect();
    let mut w = DirectReductionPlan::new(settings(7, rank, powers)).unwrap().workspace(0.0f64).unwrap();
    let r = w.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: row(p, &members, |x| x, 0.0) }).unwrap();
    assert_eq!(r.statuses[0], LaneStatus::Success, "binary64, powers {powers:?}, rank {rank}");
    r.lane(0).unwrap().to_vec()
}

fn combine(terms: Vec<(Float, Vec<Complex<Float>>)>) -> Vec<Complex<Float>> {
    let mut total: Option<Vec<Complex<Float>>> = None;
    for (weight, values) in terms {
        let scaled: Vec<Complex<Float>> = values.into_iter().map(|z| Complex::new(z.re * &weight, z.im * &weight)).collect();
        total = Some(match total {
            None => scaled,
            Some(t) => t.into_iter().zip(scaled).map(|(a, b)| Complex::new(a.re + b.re, a.im + b.im)).collect(),
        });
    }
    total.unwrap()
}

/// Weights of `1 = sum_i w_i D_i` over the six members other than `drop`
/// (`D_i = (l + q_i)^2 - m_i^2`), at 256 bits.
fn thin_weights(p: &Point, drop: usize) -> Vec<(usize, Float)> {
    let chosen: Vec<usize> = (0..7).filter(|&i| i != drop).collect();
    let o = chosen[0];
    let mut a: Vec<Vec<Float>> = vec![vec![big(0.0); 6]; 5];
    for (c, &i) in chosen[1..].iter().enumerate() {
        let d: Vec<Float> = (0..4).map(|k| big(p.shifts[i][k]) - big(p.shifts[o][k])).collect();
        for k in 0..4 {
            a[k][c] = d[k].clone();
        }
        let square = d[0].clone() * &d[0] - &(d[1].clone() * &d[1]) - &(d[2].clone() * &d[2]) - &(d[3].clone() * &d[3]);
        a[4][c] = square - big(p.masses[i]) + big(p.masses[o]);
    }
    a[4][5] = big(1.0);
    for col in 0..5 {
        let pivot = (col..5).max_by(|&x, &y| a[x][col].to_f64().abs().partial_cmp(&a[y][col].to_f64().abs()).unwrap()).unwrap();
        a.swap(col, pivot);
        for r in 0..5 {
            if r != col {
                let f = a[r][col].clone() / &a[col][col];
                for c in col..6 {
                    let t = a[col][c].clone() * &f;
                    a[r][c] -= &t;
                }
            }
        }
    }
    let w: Vec<Float> = (0..5).map(|r| a[r][5].clone() / &a[r][r]).collect();
    let mut sum = big(0.0);
    for x in &w {
        sum += x;
    }
    std::iter::once((o, -sum)).chain(chosen[1..].iter().copied().zip(w)).collect()
}

/// A generic heptagon with raised powers by the thin identity, one power at
/// a time, down to unit-power heptagons and hexagons with raised powers.
fn thin_reference(p: &Point, rank: u16, powers: &[u16], drop: usize) -> Vec<Complex<Float>> {
    if powers.iter().all(|&x| x == 1) {
        return lane_256(p, &(0..7).collect::<Vec<_>>(), rank, powers);
    }
    let terms = thin_weights(p, drop)
        .into_iter()
        .map(|(i, w)| {
            let values = if powers[i] == 1 {
                let members: Vec<usize> = (0..7).filter(|&k| k != i).collect();
                let sub: Vec<u16> = members.iter().map(|&k| powers[k]).collect();
                lane_256(p, &members, rank, &sub)
            } else {
                let mut lower = powers.to_vec();
                lower[i] -= 1;
                thin_reference(p, rank, &lower, drop)
            };
            (w, values)
        })
        .collect();
    combine(terms)
}

/// A threshold heptagon by the partial fraction of its first Coulomb triple:
/// `1/(D_1^a D_3^c) = [1/(D_1^a D_3^(c-1)) + 1/(D_1^(a-1) D_3^c)] / (2 D_2)`,
/// down to hexagons.
fn coulomb_reference(p: &Point, rank: u16, powers: &[u16]) -> Vec<Complex<Float>> {
    if powers[1] == 0 || powers[3] == 0 {
        let members: Vec<usize> = (0..7).filter(|&k| powers[k] > 0).collect();
        let sub: Vec<u16> = members.iter().map(|&k| powers[k]).collect();
        return lane_256(p, &members, rank, &sub);
    }
    let mut first = powers.to_vec();
    first[2] += 1;
    first[3] -= 1;
    let mut second = powers.to_vec();
    second[2] += 1;
    second[1] -= 1;
    combine(vec![(big(0.5), coulomb_reference(p, rank, &first)), (big(0.5), coulomb_reference(p, rank, &second))])
}

/// `max |value - reference| / max |reference|` over the lane.
fn error<T: Clone>(values: &[Complex<T>], reference: &[Complex<Float>], lift: impl Fn(&T) -> Float) -> f64 {
    let norm = |re: Float, im: Float| (re.clone() * &re + &(im.clone() * &im)).sqrt().to_f64();
    let scale = reference.iter().map(|z| norm(z.re.clone(), z.im.clone())).fold(0.0, f64::max);
    let worst = values
        .iter()
        .zip(reference)
        .map(|(v, r)| norm(lift(&v.re) - &r.re, lift(&v.im) - &r.im))
        .fold(0.0, f64::max);
    worst / scale
}

const DOTTED: [(&str, [u16; 7]); 6] = [
    ("heavy line 1 doubled", [1, 2, 1, 1, 1, 1, 1]),
    ("Coulomb gluon 2 doubled", [1, 1, 2, 1, 1, 1, 1]),
    ("heavy line 3 tripled", [1, 1, 1, 3, 1, 1, 1]),
    ("lines 1 and 5 doubled", [1, 2, 1, 1, 1, 2, 1]),
    ("gluons 2 and 4 doubled", [1, 1, 2, 1, 2, 1, 1]),
    ("lines 1 and 3 doubled", [1, 2, 1, 2, 1, 1, 1]),
];

/// Generic kinematics: the heptagon's thin system over the six members a lane
/// chooses carries the jets of its weights. Against the thin identity over a
/// different six-member system (drop of line 6, then of line 0).
#[test]
fn seven_point_thin_systems_carry_mass_derivative_jets() {
    let p = generic();
    let mut worst = [0.0f64; 3];
    for rank in 0..=1 {
        for (label, powers) in DOTTED {
            let reference = thin_reference(&p, rank, &powers, 6);
            let other = error(&thin_reference(&p, rank, &powers, 0), &reference, |x| x.clone());
            let mp = error(&lane_256(&p, &(0..7).collect::<Vec<_>>(), rank, &powers), &reference, |x| x.clone());
            let binary64 = error(&lane_binary64(&p, rank, &powers), &reference, |x| big(*x));
            eprintln!("generic rank {rank} {label}: 256 bits {mp:.1e}, binary64 {binary64:.1e} (identity drops 6 and 0 differ by {other:.1e})");
            assert!(other <= 1e-60, "{label}, rank {rank}: identity {other:e}");
            assert!(mp <= 1e-60, "{label}, rank {rank}: 256 bits {mp:e}");
            assert!(binary64 <= 1e-11, "{label}, rank {rank}: binary64 {binary64:e}");
            worst = [worst[0].max(other), worst[1].max(mp), worst[2].max(binary64)];
        }
    }
    eprintln!("worst: identity {:.1e}, 256 bits {:.1e}, binary64 {:.1e}", worst[0], worst[1], worst[2]);
}

/// Two onia at threshold: no thin system exists; the heptagon takes the
/// partial fractions of its Coulomb triples, whose raised members join the
/// dotted lines as jet variables. Against the first triple's partial
/// fraction, applied to the integrand down to hexagons (all but two lines
/// doubled: those hexagons would carry three raised lines; there, binary64
/// against 256 bits).
#[test]
fn two_onia_at_threshold_take_their_coulomb_partial_fractions_at_seven_points() {
    let mut worst = [0.0f64; 3];
    for variant in 0..2 {
        let p = threshold(variant);
        for rank in [0u16, 2] {
            let unit = [1u16; 7];
            let reference = lane_256(&p, &(0..7).collect::<Vec<_>>(), rank, &unit);
            let binary64 = error(&lane_binary64(&p, rank, &unit), &reference, |x| big(*x));
            assert!(binary64 <= 1e-11, "point {variant}, unit, rank {rank}: binary64 {binary64:e}");
            for (label, powers) in DOTTED {
                let mp = lane_256(&p, &(0..7).collect::<Vec<_>>(), rank, &powers);
                let binary64 = lane_binary64(&p, rank, &powers);
                if label == "lines 1 and 5 doubled" {
                    let own = error(&binary64, &mp, |x| big(*x));
                    eprintln!("point {variant} rank {rank} {label}: binary64 {own:.1e} from 256 bits");
                    assert!(own <= 1e-11, "point {variant}, {label}, rank {rank}: binary64 {own:e} from 256 bits");
                    worst[2] = worst[2].max(own);
                    continue;
                }
                let reference = coulomb_reference(&p, rank, &powers);
                let (mp, binary64) = (error(&mp, &reference, |x| x.clone()), error(&binary64, &reference, |x| big(*x)));
                eprintln!("point {variant} rank {rank} {label}: 256 bits {mp:.1e}, binary64 {binary64:.1e}");
                assert!(mp <= 1e-60, "point {variant}, {label}, rank {rank}: 256 bits {mp:e}");
                assert!(binary64 <= 1e-11, "point {variant}, {label}, rank {rank}: binary64 {binary64:e}");
                worst = [worst[0], worst[1].max(mp), worst[2].max(binary64)];
            }
        }
    }
    eprintln!("worst: 256 bits {:.1e}, binary64 {:.1e}", worst[1], worst[2]);
}
