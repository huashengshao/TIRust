//! The direct engine's former fixed caps (rank 32, 12 propagators, raised
//! power 5, dimension shift 3) are gone: its bounds come from its
//! representation and are reported by the capability query
//! (`DirectReductionPlan::capabilities`). Requests beyond the former caps are
//! evaluated and checked against exact or independent references; the
//! validated domain (N <= 6, R <= 6) is documentation, not a limit.
use iregi_rust::{
    DirectCapabilities, DirectReductionPlan, KinematicsBatch, LaneStatus, ReductionPlan, ReductionSettings,
    Workspace,
};
use symbolica::domains::{
    float::{Complex, DoubleFloat, Float, FloatLike, Real, RealLike, SingleFloat},
    integer::Integer,
    rational::Rational,
};

fn binomial(n: u64, k: u64) -> u64 {
    (0..k).fold(1u64, |acc, i| acc * (n - i) / (i + 1))
}

#[test]
fn capability_query_reports_bounds_derived_from_the_representation() {
    let caps = DirectReductionPlan::capabilities(&0.0f64);
    assert_eq!(caps, DirectCapabilities::of_representation(53));
    assert_eq!(DirectReductionPlan::capabilities(&DoubleFloat::from(0.0)).carrier_bits, 106);
    assert_eq!(DirectReductionPlan::capabilities(&Float::new(300)).carrier_bits, 300);
    assert_eq!(caps.max_propagators, 64);
    // Monomial indices are u32: C(R + 4, 4) fits, C(R + 5, 4) does not.
    let r = u64::from(caps.max_rank);
    assert!(r > 32);
    assert!(binomial(r + 4, 4) <= u64::from(u32::MAX) && binomial(r + 5, 4) > u64::from(u32::MAX));
    assert_eq!((caps.max_power(1), caps.max_power(4)), (u16::MAX, u16::MAX));
    assert_eq!((caps.max_power(5), caps.max_power(6)), (4095, 1023));
    assert_eq!((caps.max_dotted_propagators, caps.max_dotted_points, caps.max_shifted_points), (1, 6, 4));
    assert_eq!(caps.max_dimension_shift, i16::MAX);
    // `supports` reads the same bounds (no plan is built here).
    let supports = |n: usize, rank: u16, powers: Option<Vec<u16>>, h: i16| {
        let mut settings = ReductionSettings::new(n, rank);
        if let Some(powers) = powers {
            settings.powers = powers;
        }
        settings.dimension_shift = h;
        DirectReductionPlan::supports(&settings)
    };
    assert!(supports(13, 1, None, 0) && supports(64, 0, None, 0) && !supports(65, 0, None, 0));
    assert!(supports(2, caps.max_rank, None, 0) && !supports(2, caps.max_rank + 1, None, 0));
    assert!(supports(3, 2, Some(vec![9, 1, 1]), 0) && supports(6, 1, Some(vec![1, 1, 1, 1, 1, 1023]), 0));
    assert!(!supports(6, 1, Some(vec![1, 1, 1, 1, 1, 1024]), 0));
    assert!(!supports(7, 1, Some(vec![2, 1, 1, 1, 1, 1, 1]), 0), "raised powers stop at six points");
    assert!(!supports(3, 1, Some(vec![2, 2, 1]), 0), "one raised power");
    assert!(supports(3, 2, None, 9) && !supports(5, 2, None, 1), "shifts up to four points");
}

/// Tadpole moments at the origin, `int l^e / (l^2 - m^2)`: every odd
/// exponent vanishes, and `M(2a) = prod_i (2a_i - 1)!! (-1)^(a1+a2+a3)
/// A_{0^k}` with `k = |a|` and, from `int (l^2)^k / D = m^{2k} A0` in `D`
/// dimensions, `A_{0^k} = m^{2k} / (2^k (k+1)!) [A0 + UV(A0) H_k]`,
/// `H_k = sum_{j=1..k} 1/(j+1)` (the O(eps) part of `1/prod(D + 2j)`).
fn tadpole_reference(basis: &[iregi_rust::basis::Monomial], a0: [Rational; 2], uv: [Rational; 2], m2: &Rational) -> Vec<[Rational; 2]> {
    let mut out = Vec::with_capacity(4 * basis.len());
    let zero = Rational::from(Integer::from(0));
    for monomial in basis {
        let e = monomial.0;
        if e.iter().any(|&x| x % 2 == 1) {
            for _ in 0..4 {
                out.push([zero.clone(), zero.clone()]);
            }
            continue;
        }
        let a: Vec<u64> = e.iter().map(|&x| u64::from(x / 2)).collect();
        let k: u64 = a.iter().sum();
        let mut factor = Rational::from(Integer::from(1));
        for &ai in &a {
            for odd in (1..2 * ai).step_by(2) {
                factor = &factor * &Rational::from(Integer::from(odd as i64));
            }
        }
        if (a[1] + a[2] + a[3]) % 2 == 1 {
            factor = -factor;
        }
        let mut denominator = Integer::from(2).pow(k);
        for j in 2..=k + 1 {
            denominator *= &Integer::from(j as i64);
        }
        let mut power = Rational::from(Integer::from(1));
        for _ in 0..k {
            power = &power * m2;
        }
        factor = &(&factor * &power) / &Rational::from(denominator);
        let mut harmonic = zero.clone();
        for j in 1..=k {
            harmonic = &harmonic + &Rational::from((Integer::from(1), Integer::from((j + 1) as i64)));
        }
        let uv_value = [&factor * &uv[0], &factor * &uv[1]];
        let finite = [
            &factor * &(&a0[0] + &(&harmonic * &uv[0])),
            &factor * &(&a0[1] + &(&harmonic * &uv[1])),
        ];
        out.push(uv_value);
        out.push([zero.clone(), zero.clone()]);
        out.push([zero.clone(), zero.clone()]);
        out.push(finite);
    }
    out
}

fn tadpole<T: iregi_masters::NativeFloat>(prototype: T, rank: u16) -> iregi_rust::BatchResult<T>
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let real = |x: Rational| Complex::new(prototype.from_rational(&x), prototype.zero());
    let q = |x: i64| Rational::from(Integer::from(x));
    let values = vec![
        real(q(0)), real(q(0)), real(q(0)), real(q(0)),
        real(Rational::from((Integer::from(9), Integer::from(4)))),
        real(q(3)),
    ];
    let mut workspace = DirectReductionPlan::new(ReductionSettings::new(1, rank)).unwrap().workspace(prototype.clone()).unwrap();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    result
}

/// Largest deviation from the closed form, relative to the largest
/// reference entry of the same Laurent slot (weighted by mu^-rank), over the
/// whole moment array; its A0 and UV(A0) from the engine's own scalar.
fn tadpole_deviation<T: iregi_masters::NativeFloat>(result: &iregi_rust::BatchResult<T>) -> f64 {
    let lane = result.lane(0).unwrap();
    let exact = |x: &T| iregi_masters::NativeFloat::exact_rational(x).unwrap();
    let a0 = [exact(&lane[3].re), exact(&lane[3].im)];
    let uv = [exact(&lane[0].re), exact(&lane[0].im)];
    let reference = tadpole_reference(&result.basis, a0, uv, &Rational::from((Integer::from(9), Integer::from(4))));
    let bits = 600;
    let to = |x: &Rational| Float::with_val(bits, 0).from_rational(x);
    let mu = Float::with_val(bits, 3);
    let mut worst = 0.0f64;
    for slot in [0usize, 3] {
        let (mut error, mut scale) = (Float::with_val(bits, 0), Float::with_val(bits, 0));
        for (m, monomial) in result.basis.iter().enumerate() {
            let weight = mu.clone().pow(monomial.rank() as u64);
            let got = &lane[4 * m + slot];
            let want = &reference[4 * m + slot];
            let dr = to(&exact(&got.re)) - &to(&want[0]);
            let di = to(&exact(&got.im)) - &to(&want[1]);
            let e = (dr.clone() * &dr + di.clone() * &di).sqrt() / &weight;
            let s = (to(&want[0]) * &to(&want[0]) + to(&want[1]) * &to(&want[1])).sqrt() / &weight;
            if e > error {
                error = e;
            }
            if s > scale {
                scale = s;
            }
        }
        worst = worst.max((error / &scale).to_f64());
    }
    // Odd moments and the IR slots vanish exactly.
    for (m, monomial) in result.basis.iter().enumerate() {
        let odd = monomial.0.iter().any(|&x| x % 2 == 1);
        for slot in 0..4 {
            if odd || slot == 1 || slot == 2 {
                assert!(lane[4 * m + slot].re.is_zero() && lane[4 * m + slot].im.is_zero(), "{:?} slot {slot}", monomial.0);
            }
        }
    }
    worst
}

/// Largest relative deviation from the closed form of a single nonzero
/// entry (UV and finite slots) among the moments of rank above `above`: the
/// entries beyond the former rank cap, each against its own value (the
/// array-wide E/A above is dominated by the low ranks). `None` if every such
/// entry vanishes (odd ranks: checked to be exactly zero above).
fn tadpole_entry_deviation<T: iregi_masters::NativeFloat>(result: &iregi_rust::BatchResult<T>, above: u16) -> Option<f64> {
    let lane = result.lane(0).unwrap();
    let exact = |x: &T| iregi_masters::NativeFloat::exact_rational(x).unwrap();
    let a0 = [exact(&lane[3].re), exact(&lane[3].im)];
    let uv = [exact(&lane[0].re), exact(&lane[0].im)];
    let reference = tadpole_reference(&result.basis, a0, uv, &Rational::from((Integer::from(9), Integer::from(4))));
    let bits = 600;
    let to = |x: &Rational| Float::with_val(bits, 0).from_rational(x);
    let mut worst = 0.0f64;
    let mut entries = 0usize;
    for (m, monomial) in result.basis.iter().enumerate() {
        if monomial.rank() <= usize::from(above) {
            continue;
        }
        for slot in [0usize, 3] {
            let want = &reference[4 * m + slot];
            let size = (to(&want[0]) * &to(&want[0]) + to(&want[1]) * &to(&want[1])).sqrt();
            if size.is_zero() {
                continue;
            }
            let got = &lane[4 * m + slot];
            let dr = to(&exact(&got.re)) - &to(&want[0]);
            let di = to(&exact(&got.im)) - &to(&want[1]);
            worst = worst.max(((dr.clone() * &dr + di.clone() * &di).sqrt() / &size).to_f64());
            entries += 1;
        }
    }
    (entries > 0).then_some(worst)
}

#[test]
fn tadpole_beyond_rank_32_matches_its_closed_form() {
    // The closed form at a validated rank first (the oracle's conventions).
    let low = tadpole_deviation(&tadpole(Float::new(256), 6));
    assert!(low <= 1e-70, "{low:e}");
    for rank in [33u16, 36] {
        let start = std::time::Instant::now();
        let (mpfr_result, double_result, binary64_result) =
            (tadpole(Float::new(256), rank), tadpole(DoubleFloat::from(0.0), rank), tadpole(0.0f64, rank));
        let mpfr = tadpole_deviation(&mpfr_result);
        let double = tadpole_deviation(&double_result);
        let binary64 = tadpole_deviation(&binary64_result);
        println!(
            "tadpole rank {rank}: E/A against the closed form: binary64 {binary64:e}, double-double {double:e}, 256 bits {mpfr:e} ({:.2?})",
            start.elapsed()
        );
        assert!(mpfr <= 1e-70 && double <= 1e-28 && binary64 <= 1e-13, "{binary64:e} {double:e} {mpfr:e}");
        // Every nonzero entry above rank 32 (ranks 34 and 36 of the rank-36
        // array; the rank-33 entries are odd and exactly zero) against its own
        // closed form.
        let entry = [
            tadpole_entry_deviation(&binary64_result, 32),
            tadpole_entry_deviation(&double_result, 32),
            tadpole_entry_deviation(&mpfr_result, 32),
        ];
        println!("tadpole rank {rank}: largest relative entry deviation above rank 32 (binary64, double-double, 256 bits): {entry:?}");
        assert_eq!(entry.iter().all(Option::is_some), rank >= 34, "{entry:?}");
        if let [Some(b64), Some(dd), Some(mp)] = entry {
            assert!(b64 <= 1e-13 && dd <= 1e-28 && mp <= 1e-70, "{entry:?}");
        }
    }
}

/// Direct engine (double-double) against the generic symbolic engine at 256
/// bits on the same lane: E/A over the array.
fn direct_against_symbolic(settings: ReductionSettings, values: Vec<Complex<f64>>) -> f64 {
    let lift: Vec<Complex<DoubleFloat>> =
        values.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect();
    let direct = DirectReductionPlan::new(settings.clone())
        .unwrap()
        .workspace(DoubleFloat::from(0.0))
        .unwrap()
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lift })
        .unwrap();
    let lift: Vec<Complex<Float>> =
        values.iter().map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im))).collect();
    let plan = ReductionPlan::new(settings).unwrap();
    let symbolic = Workspace::new(&plan, Float::new(256))
        .unwrap()
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lift })
        .unwrap();
    assert_eq!(direct.statuses[0], LaneStatus::Success);
    assert_eq!(symbolic.statuses[0], LaneStatus::Success);
    let (mut error, mut scale) = (0.0f64, 0.0f64);
    for (a, b) in direct.lane(0).unwrap().iter().zip(symbolic.lane(0).unwrap()) {
        let exact = |x: &DoubleFloat| Float::with_val(256, 0).from_rational(&iregi_masters::NativeFloat::exact_rational(x).unwrap());
        let (dr, di) = ((exact(&a.re) - &b.re).to_f64(), (exact(&a.im) - &b.im).to_f64());
        error = error.max(dr.hypot(di));
        scale = scale.max(b.re.to_f64().hypot(b.im.to_f64()));
    }
    error / scale
}

fn row(shifts: &[[f64; 4]], masses: &[f64], mu: f64) -> Vec<Complex<f64>> {
    let mut values: Vec<Complex<f64>> = shifts.iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
    values.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
    values.push(Complex::new(mu, 0.0));
    values
}

#[test]
fn raised_powers_and_dimension_shifts_beyond_the_former_caps_match_independent_references() {
    let bubble = [[0.0, 0.0, 0.0, 0.0], [3.0, 0.5, 1.0, 2.0]];
    let triangle = [[0.0, 0.0, 0.0, 0.0], [3.0, 0.5, 1.0, 2.0], [5.0, -1.5, 2.0, 0.25]];
    // Euclidean triangle (every invariant negative, no threshold).
    let euclidean = [[0.0, 0.0, 0.0, 0.0], [0.3, 1.0, 0.0, 0.0], [0.1, 0.0, 1.2, 0.5]];
    // Dimension shift h = 5 (former cap 3): D = 14 - 2 eps.
    let mut settings = ReductionSettings::new(3, 2);
    settings.dimension_shift = 5;
    let deviation = direct_against_symbolic(settings, row(&triangle, &[1.5, 2.5, 0.75], 3.0));
    println!("triangle, h = 5, rank 2: direct (double-double) vs symbolic (256 bits) {deviation:e}");
    assert!(deviation <= 1e-26, "{deviation:e}");
    // Raised power 7 (former cap 5) on a bubble and 6 on a triangle.
    let mut settings = ReductionSettings::new(2, 2);
    settings.powers = vec![7, 1];
    let deviation = direct_against_symbolic(settings, row(&bubble, &[1.5, 2.5], 3.0));
    println!("bubble, power 7, rank 2: direct (double-double) vs symbolic (256 bits) {deviation:e}");
    assert!(deviation <= 1e-26, "{deviation:e}");
    // Power 6 on a triangle (the symbolic IBP refuses it, estimating
    // cancellation even at 256 bits): against the fifth squared-mass
    // derivative of the unit-power triangle, J(1,1,6) = d^5 J(1,1,1) / 5!
    // in m_2^2, by central differences at 512 bits (step 2^-64, error
    // O(2^-128)).
    let bits = 512;
    let unit = DirectReductionPlan::new(ReductionSettings::new(3, 1)).unwrap();
    let lane = |m2: Float| -> Vec<Complex<Float>> {
        let mut values: Vec<Complex<Float>> = euclidean
            .iter()
            .flatten()
            .map(|&x| Complex::new(Float::with_val(bits, x), Float::new(bits)))
            .collect();
        values.extend([Float::with_val(bits, 1.0), Float::with_val(bits, 2.0), m2].map(|m| Complex::new(m, Float::new(bits))));
        values.push(Complex::new(Float::with_val(bits, 3.0), Float::new(bits)));
        values
    };
    let evaluate = |values: Vec<Complex<Float>>, settings: ReductionSettings| -> Vec<Complex<Float>> {
        let mut workspace = DirectReductionPlan::new(settings).unwrap().workspace(Float::new(bits)).unwrap();
        workspace.set_stability_threshold(Float::with_val(bits, 2f64.powi(-400)));
        let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
        assert_eq!(result.statuses[0], LaneStatus::Success);
        result.lane(0).unwrap().to_vec()
    };
    let step = Float::with_val(bits, 2f64.powi(-64));
    let weights: [(i64, i64); 6] = [(3, 1), (2, -4), (1, 5), (-1, -5), (-2, 4), (-3, -1)];
    let mut derivative: Vec<Complex<Float>> = Vec::new();
    for (offset, weight) in weights {
        let m2 = Float::with_val(bits, 3.0) + &(step.clone() * &Float::with_val(bits, offset));
        let values = evaluate(lane(m2), unit.settings().clone());
        if derivative.is_empty() {
            derivative = values.iter().map(|_| Complex::new(Float::new(bits), Float::new(bits))).collect();
        }
        for (d, v) in derivative.iter_mut().zip(&values) {
            let w = Float::with_val(bits, weight);
            *d = Complex::new(d.re.clone() + &(v.re.clone() * &w), d.im.clone() + &(v.im.clone() * &w));
        }
    }
    // f^(5) ~ sum / (2 h^5); J(1,1,6) = f^(5) / 120.
    let denominator = step.clone() * &step * &step * &step * &step * &Float::with_val(bits, 240);
    let reference: Vec<Complex<Float>> =
        derivative.iter().map(|d| Complex::new(d.re.clone() / &denominator, d.im.clone() / &denominator)).collect();
    let mut settings = ReductionSettings::new(3, 1);
    settings.powers = vec![1, 1, 6];
    let lift: Vec<Complex<DoubleFloat>> = row(&euclidean, &[1.0, 2.0, 3.0], 3.0)
        .iter()
        .map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im)))
        .collect();
    let double = DirectReductionPlan::new(settings.clone())
        .unwrap()
        .workspace(DoubleFloat::from(0.0))
        .unwrap()
        .reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lift })
        .unwrap();
    assert_eq!(double.statuses[0], LaneStatus::Success);
    let exact = evaluate(lane(Float::with_val(bits, 3.0)), settings);
    let deviation = |a: &[Complex<Float>]| {
        let (mut error, mut scale) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(&reference) {
            error = error.max((x.re.clone() - &y.re).to_f64().hypot((x.im.clone() - &y.im).to_f64()));
            scale = scale.max(y.re.to_f64().hypot(y.im.to_f64()));
        }
        error / scale
    };
    let lifted: Vec<Complex<Float>> = double
        .lane(0)
        .unwrap()
        .iter()
        .map(|z| {
            let to = |x: &DoubleFloat| Float::with_val(bits, 0).from_rational(&iregi_masters::NativeFloat::exact_rational(x).unwrap());
            Complex::new(to(&z.re), to(&z.im))
        })
        .collect();
    let (at_512, at_106) = (deviation(&exact), deviation(&lifted));
    println!("triangle, power 6, rank 1: direct vs 5th m^2-derivative: 512 bits {at_512:e}, double-double {at_106:e}");
    assert!(at_512 <= 1e-30 && at_106 <= 1e-26, "{at_512:e} {at_106:e}");
}

/// Peak resident memory of this process so far (Linux; 0 elsewhere).
fn peak_rss_mib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmHWM:"))
                .and_then(|line| line.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .map_or(0, |kib| kib / 1024)
}

/// An N-point family above the former 12-propagator cap, planned and
/// evaluated in binary64 and at 256 bits (the achieved precision is their
/// difference); plan size, times and peak memory are printed for the record.
fn wide_family(n: usize, rank: u16) {
    // Deterministic generic kinematics: shifts in a unit box, masses 1..2.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut values = Vec::new();
    for i in 0..n {
        for _ in 0..4 {
            let x = if i == 0 { 0.0 } else { 2.0 * next() - 1.0 };
            values.push(Complex::new(x, 0.0));
        }
    }
    for _ in 0..n {
        values.push(Complex::new(1.0 + next(), 0.0));
    }
    values.push(Complex::new(1.0, 0.0));
    let start = std::time::Instant::now();
    let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank)).unwrap();
    let planned = start.elapsed();
    let (instructions, terms) = plan.instruction_counts();
    let start = std::time::Instant::now();
    let binary64 = plan.workspace(0.0f64).unwrap().reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: values.clone() }).unwrap();
    let evaluated = start.elapsed();
    let lift: Vec<Complex<Float>> =
        values.iter().map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im))).collect();
    let mut workspace = plan.workspace(Float::new(256)).unwrap();
    workspace.set_stability_threshold(Float::with_val(256, 2f64.powi(-200)));
    let start = std::time::Instant::now();
    let reference = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values: lift }).unwrap();
    let evaluated_256 = start.elapsed();
    assert_eq!(reference.statuses[0], LaneStatus::Success, "256 bits");
    let achieved = match binary64.lane(0) {
        Ok(lane) => {
            let (mut error, mut scale) = (0.0f64, 0.0f64);
            for (a, b) in lane.iter().zip(reference.lane(0).unwrap()) {
                error = error.max((a.re - b.re.to_f64()).hypot(a.im - b.im.to_f64()));
                scale = scale.max(b.re.to_f64().hypot(b.im.to_f64()));
            }
            format!("{:e}", error / scale)
        }
        Err(status) => format!("binary64 refused: {status:?}"),
    };
    println!(
        "N{n} R{rank}: {instructions} instructions, {terms} terms; plan {planned:.2?}, binary64 {evaluated:.2?}, \
         256 bits {evaluated_256:.2?}; binary64 vs 256 bits E/A {achieved}; peak RSS {} MiB",
        peak_rss_mib()
    );
}

#[test]
fn families_beyond_twelve_propagators_are_planned_and_evaluated() {
    wide_family(13, 1);
}
