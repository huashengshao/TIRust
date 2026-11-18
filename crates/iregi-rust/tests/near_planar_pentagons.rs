//! Nearly planar pentagons with consistent inputs in MPFR (issue #40).
//!
//! A g g > H g g point whose final state is tilted out of a plane containing
//! the beam by tan(phi / 2) = t has a pentagon Gram matrix conditioned like
//! 1/t^2. The rank-6 moment recursion applies its inverse once; the error
//! stays near kappa u, which the pentagon gate (2^-7 / kappa against the
//! threshold) bounds, only while rounding is that of a fixed precision.
//! Symbolica's `Float` operators track significance instead: a difference
//! is rounded to the bits its cancellation leaves, a product to its less
//! precise factor. The Gram system's large but correlated cancellations then
//! lose bits for good, and MPFR served these lanes about kappa^2 u off (34
//! digits at t = 1e-8 up to 27 times the value, 40 digits at 1e-12 4e11,
//! 50 and 60 digits at 1e-16 2e17 and 2e7), with status success, where
//! double-double at the same precision was right. The direct engine's
//! kernels now round MPFR at the working precision.
//!
//! Here: every invariant from the exact shifts (consistent to every
//! precision), each carrier at pyAmpliCol's gate 1e-12^(bits/53), the truth
//! the 512-bit value. Every lane is refused or within u / threshold of it,
//! the error the gate admits.
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
/// z by phi with tan(phi / 2) = `tilt`, the Higgs balancing. Legs outgoing;
/// leg 2 is the Higgs.
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

/// One row of the loop of `legs` in `order`: shifts (partial sums), every
/// invariant from them exactly, massless propagators, mu^2 = MH^2.
fn row(legs: &[Leg], order: &[usize]) -> Vec<Q> {
    let n = legs.len();
    let mut shifts = vec![[q(0, 1), q(0, 1), q(0, 1), q(0, 1)]];
    for k in 0..n - 1 {
        let last = shifts.last().unwrap();
        shifts.push(std::array::from_fn(|a| &last[a] + &legs[order[k]][a]));
    }
    let mut values: Vec<Q> = shifts.iter().flatten().cloned().collect();
    for i in 0..n {
        for j in i + 1..n {
            values.push(square(&std::array::from_fn(|a| &shifts[i][a] - &shifts[j][a])));
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

const ORDERS: [[usize; 5]; 6] = [
    [0, 1, 2, 3, 4],
    [0, 1, 3, 2, 4],
    [0, 2, 1, 3, 4],
    [0, 3, 1, 4, 2],
    [0, 3, 2, 4, 1],
    [0, 4, 3, 1, 2],
];

/// Rank-6 lanes of the point tilted by tan(phi / 2) = 10^-`digits`: every
/// served lane within u / threshold of the truth. Returns the lanes each
/// carrier served.
fn served_within_the_gate(digits: u32) -> Vec<(u32, usize)> {
    let tilt = Rational::from((Integer::from(1), Integer::from(10).pow(u64::from(digits))));
    let legs = point(&tilt);
    let rows: Vec<Vec<Q>> = ORDERS.iter().map(|order| row(&legs, order)).collect();
    let (n, rank) = (5, 6);
    let truth = reduce(Float::new(512), Float::with_val(512, gate(512)), n, rank, &rows);
    let mut served = Vec::new();
    for bits in [106u32, 113, 133, 167, 200] {
        let lanes = if bits == 106 {
            reduce(DoubleFloat::from(0.0), DoubleFloat::from(gate(106)), n, rank, &rows)
        } else {
            reduce(Float::new(bits), Float::with_val(bits, gate(bits)), n, rank, &rows)
        };
        let admitted = 2f64.powi(-(bits as i32)) / gate(bits);
        let mut count = 0;
        for (lane, result) in lanes.iter().enumerate() {
            let Ok(values) = result else { continue };
            let expected = truth[lane].as_ref().expect("the tilted point is served at 512 bits");
            let error = deviation(values, expected);
            assert!(
                error <= admitted,
                "tilt 1e-{digits}, {bits} bits, order {:?}: served {error:e} off, the gate admits {admitted:e}",
                ORDERS[lane]
            );
            count += 1;
        }
        println!("tilt 1e-{digits}, {bits} bits: {count} of {} served", rows.len());
        served.push((bits, count));
    }
    served
}

#[test]
fn nearly_planar_pentagons_stay_within_their_gate_in_mpfr() {
    // Before the fix MPFR 113 and 133 served every lane at t = 1e-8 up to 30
    // and 4e-5 off; at t = 1e-12 MPFR 133 and 167 up to 4e11 and 10; at
    // t = 1e-16 MPFR 167 and 200 up to 2e17 and 2e7.
    for digits in [8, 12, 16] {
        let served = served_within_the_gate(digits);
        // The recursion serves the lanes the gate admits; none is refused
        // by the fix itself.
        assert!(served.iter().any(|&(bits, count)| bits >= 133 && count > 0), "tilt 1e-{digits}: {served:?}");
    }
}
