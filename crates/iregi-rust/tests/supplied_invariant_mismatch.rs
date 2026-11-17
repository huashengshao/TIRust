//! Supplied invariants that disagree with the shifts beyond the working
//! precision's rounding (issue #40).
//!
//! Every carrier receives the shifts and invariants of momenta rounded to
//! binary64, cast once from their exact sums, with the on-shell entries the
//! legs' squared masses exactly (pyAmpliCol's inputs with its kinematics
//! improvement off). In double-double and MPFR these inputs disagree at
//! binary64's rounding. At an exactly planar 2 -> 3 point the shifts span
//! three dimensions while the invariants' pentagon Gram matrix is regular
//! only through that mismatch; above the Gram-free ranks (6 and up) its
//! inverse, applied to the shifts, gave moments wrong at O(1) with status
//! success, in every precision whose gate admitted it. Near-planar points
//! amplify the mismatch by the Gram matrix's condition number.
//!
//! Here: g g > H g g pentagons in exact rational kinematics, the inputs
//! formed that way, each carrier at pyAmpliCol's gate 1e-12^(bits/53). The
//! truth: the 512-bit value at the exact point. Every lane is refused or
//! within 1e-10 of it.
use iregi_masters::NativeFloat;
use iregi_rust::{
    Convention, DirectReductionPlan, InvariantInput, KinematicsBatch, LaneStatus,
    ReductionSettings, ScaleInput, TensorWeights,
};
use symbolica::domains::{
    float::{Complex, DoubleFloat, Float},
    integer::Integer,
    rational::Rational,
};

type Q = Rational;
type Leg = [Q; 4];

fn q(n: i64, d: i64) -> Q {
    Rational::from((Integer::from(n), Integer::from(d)))
}

fn square(v: &Leg) -> Q {
    &(&(&v[0] * &v[0]) - &(&v[1] * &v[1])) - &(&(&v[2] * &v[2]) + &(&v[3] * &v[3]))
}

/// g g > H g g at sqrt(s) = 1000, MH = 125: incoming along z, k4 along
/// (3/5, 0, 4/5) with |k4| = 150, k5 along (-5/13, 0, -12/13) rotated about
/// z by phi with tan(phi / 2) = `tilt` (zero: exactly planar), the Higgs
/// balancing. Legs outgoing; leg 2 is the Higgs.
fn point(tilt: &Q) -> Vec<Leg> {
    let t2 = tilt * tilt;
    let cos = &(&q(1, 1) - &t2) / &(&q(1, 1) + &t2);
    let sin = &(&q(2, 1) * tilt) / &(&q(1, 1) + &t2);
    let (d4, d5) = ((q(3, 5), q(4, 5)), (q(-5, 13), q(-12, 13)));
    let (roots, a, mh2) = (q(1000, 1), q(150, 1), q(15625, 1));
    let c = &(&(&d4.0 * &d5.0) * &cos) + &(&d4.1 * &d5.1);
    let num = &(&(&roots * &roots) - &(&(&q(2, 1) * &roots) * &a)) - &mh2;
    let den = &(&q(2, 1) * &roots) - &(&(&q(2, 1) * &a) * &(&q(1, 1) - &c));
    let b = &num / &den;
    let eh = &(&roots - &a) - &b;
    let e = q(500, 1);
    let p1 = [e.clone(), q(0, 1), q(0, 1), e.clone()];
    let p2 = [e.clone(), q(0, 1), q(0, 1), -e.clone()];
    let k4 = [a.clone(), &a * &d4.0, q(0, 1), &a * &d4.1];
    let k5 = [b.clone(), &(&b * &d5.0) * &cos, &(&b * &d5.0) * &sin, &b * &d5.1];
    let kh = [eh, -(&k4[1] + &k5[1]), -k5[2].clone(), -(&k4[3] + &k5[3])];
    assert_eq!(square(&kh), mh2);
    assert_eq!((square(&k4), square(&k5)), (q(0, 1), q(0, 1)));
    vec![p1.map(|x| -x), p2.map(|x| -x), kh, k4, k5]
}

/// `x` rounded to binary64, as an exact rational.
fn round64(x: &Q) -> Q {
    Float::with_val(53, x.to_f64()).try_to_rational().unwrap()
}

/// One row of the loop of `legs` in `order`: shifts (partial sums),
/// supplied invariants (with `on_shell`, the legs' own entries are their
/// squared masses `masses` exactly), massless propagators, mu^2 = MH^2.
fn row(legs: &[Leg], order: &[usize], on_shell: bool, masses: &[Q]) -> Vec<Q> {
    let n = legs.len();
    let mut shifts = vec![[q(0, 1), q(0, 1), q(0, 1), q(0, 1)]];
    for k in 0..n - 1 {
        let last = shifts.last().unwrap();
        shifts.push(std::array::from_fn(|a| &last[a] + &legs[order[k]][a]));
    }
    let mut values: Vec<Q> = shifts.iter().flatten().cloned().collect();
    for i in 0..n {
        for j in i + 1..n {
            values.push(if on_shell && j == i + 1 {
                masses[order[i]].clone()
            } else if on_shell && i == 0 && j == n - 1 {
                masses[order[n - 1]].clone()
            } else {
                square(&std::array::from_fn(|a| &shifts[i][a] - &shifts[j][a]))
            });
        }
    }
    values.extend((0..n).map(|_| q(0, 1)));
    values.push(q(15625, 1));
    values
}

fn settings(n: usize, rank: u16) -> ReductionSettings {
    let mut settings = ReductionSettings::new(n, rank);
    settings.convention = Convention::OneLoop;
    settings.tensor_weights = TensorWeights::Components;
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    settings
}

/// Statuses and values (rounded to binary64) of `rows` in the carrier of
/// `prototype` at the direct engine's stability threshold `gate`.
fn reduce<T: NativeFloat>(prototype: T, gate: T, n: usize, rank: u16, rows: &[Vec<Q>]) -> Vec<Result<Vec<Complex<f64>>, LaneStatus>>
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let mut workspace = DirectReductionPlan::new(settings(n, rank)).unwrap().workspace(prototype.clone()).unwrap();
    workspace.set_stability_threshold(gate);
    let zero = prototype.zero();
    let values = rows.iter().flatten().map(|x| Complex::new(prototype.from_rational(x), zero.clone())).collect();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: rows.len(), components: 1, values }).unwrap();
    (0..rows.len())
        .map(|lane| match result.lane(lane) {
            Ok(values) => Ok(values.iter().map(|z| Complex::new(z.re.to_f64(), z.im.to_f64())).collect()),
            Err(status) => Err(status.clone()),
        })
        .collect()
}

/// Largest |a - b| over each Laurent slot of the array, relative to the
/// slot's largest |b|.
fn deviation(a: &[Complex<f64>], b: &[Complex<f64>]) -> f64 {
    (0..4)
        .map(|slot| {
            let (mut error, mut scale) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b).skip(slot).step_by(4) {
                error = error.max((x.re - y.re).hypot(x.im - y.im));
                scale = scale.max(y.re.hypot(y.im));
            }
            if scale == 0.0 { error } else { error / scale }
        })
        .fold(0.0, f64::max)
}

/// pyAmpliCol's gate of a carrier of `bits` bits.
fn gate(bits: u32) -> f64 {
    1e-12f64.powf(f64::from(bits) / 53.0)
}

/// Per carrier and rank, (served, refused) over the lanes of `exact` in
/// `orders`, the inputs formed as pyAmpliCol forms them; panics on any served
/// lane more than 1e-10 off the truth.
fn served_or_refused(label: &str, exact: &[Leg], masses: &[Q], orders: &[Vec<usize>], ranks: &[u16]) -> Vec<(String, u16, usize, usize)> {
    let n = exact.len();
    let rounded: Vec<Leg> = exact.iter().map(|leg| leg.clone().map(|x| round64(&x))).collect();
    let truth_rows: Vec<Vec<Q>> = orders.iter().map(|order| row(exact, order, false, masses)).collect();
    let rows: Vec<Vec<Q>> = orders.iter().map(|order| row(&rounded, order, true, masses)).collect();
    let mut counts = Vec::new();
    for &rank in ranks {
        let truth = reduce(Float::new(512), Float::with_val(512, gate(512)), n, rank, &truth_rows);
        let carriers = [
            ("binary64", reduce(0.0f64, gate(53), n, rank, &rows)),
            ("double-double", reduce(DoubleFloat::from(0.0), DoubleFloat::from(gate(106)), n, rank, &rows)),
            ("MPFR 133", reduce(Float::new(133), Float::with_val(133, gate(133)), n, rank, &rows)),
            ("MPFR 200", reduce(Float::new(200), Float::with_val(200, gate(200)), n, rank, &rows)),
        ];
        for (name, lanes) in carriers {
            let (mut served, mut refused) = (0, 0);
            for (lane, result) in lanes.iter().enumerate() {
                let Ok(values) = result else {
                    refused += 1;
                    continue;
                };
                let expected = truth[lane].as_ref().expect("the exact point is served at 512 bits");
                let error = deviation(values, expected);
                assert!(
                    error <= 1e-10,
                    "{label}: {name}, rank {rank}, order {:?}: served {error:e} off the exact point",
                    orders[lane]
                );
                served += 1;
            }
            println!("{label}, rank {rank}, {name}: {served} served, {refused} refused");
            counts.push((name.to_string(), rank, served, refused));
        }
    }
    counts
}

/// The pentagon's masses (leg 2 is the Higgs).
fn pentagon_masses() -> Vec<Q> {
    vec![q(0, 1), q(0, 1), q(15625, 1), q(0, 1), q(0, 1)]
}

const ORDERS: [[usize; 5]; 6] = [
    [0, 1, 2, 3, 4],
    [0, 1, 3, 2, 4],
    [0, 2, 1, 3, 4],
    [0, 3, 1, 4, 2],
    [0, 3, 2, 4, 1],
    [0, 4, 3, 1, 2],
];

#[test]
fn exactly_planar_pentagons_take_their_affine_relation() {
    // Before the fix, double-double and MPFR served every rank-6 lane at
    // O(1) from the truth (the Gram recursion of a matrix regular only
    // through the mismatch), and MPFR 200 every rank-7 lane.
    let orders: Vec<Vec<usize>> = ORDERS.iter().map(|o| o.to_vec()).collect();
    let counts = served_or_refused("planar", &point(&q(0, 1)), &pentagon_masses(), &orders, &[6, 7]);
    // The extended carriers serve the planar lanes through the pentagon's
    // affine relation (partial fractions into boxes).
    for (name, rank, served, _) in &counts {
        if name != "binary64" && *rank == 6 {
            assert!(*served > 0, "{name}, rank {rank}: nothing served");
        }
    }
}

#[test]
fn near_planar_pentagons_are_refused_where_the_mismatch_is_amplified() {
    // tan(phi / 2) = 1e-5: the Gram matrix is regular, conditioned like
    // 1e10; before the fix every extended carrier served rank 6 up to 6e-6
    // off and rank 7 up to 0.17 off.
    let orders: Vec<Vec<usize>> = ORDERS.iter().map(|o| o.to_vec()).collect();
    served_or_refused("tilted", &point(&q(1, 100_000)), &pentagon_masses(), &orders, &[6, 7]);
}

#[test]
fn boxes_with_collinear_momenta_take_their_affine_relation() {
    // Every momentum along z: the Gram matrix of the box's three momenta is
    // exactly singular in the shifts, and regular in the invariants only
    // through the mismatch of components that round in binary64
    // (denominators 3, 7, 11). Before the fix MPFR 133 served rank 1 at
    // 1.8e-6 and MPFR 200 rank 2 at 3.9e-10 off the truth; now every carrier
    // serves them through the box's affine relation.
    let legs: Vec<Leg> = [(7, 3, 2, 3), (9, 7, -4, 7), (-10, 11, 3, 11)]
        .iter()
        .map(|&(e, de, z, dz)| [q(e, de), q(0, 1), q(0, 1), q(z, dz)])
        .collect();
    let last: Leg = std::array::from_fn(|a| -(&(&legs[0][a] + &legs[1][a]) + &legs[2][a]));
    let mut exact = legs;
    exact.push(last);
    let masses: Vec<Q> = exact.iter().map(square).collect();
    let orders: Vec<Vec<usize>> = vec![vec![0, 1, 2, 3], vec![0, 2, 1, 3], vec![0, 1, 3, 2]];
    let counts = served_or_refused("collinear box", &exact, &masses, &orders, &[1, 2, 3]);
    assert!(counts.iter().all(|(_, _, _, refused)| *refused == 0), "{counts:?}");
}
