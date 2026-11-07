//! Supplied invariants (`InvariantInput::SuppliedInvariants`, the S-matrix of
//! the shifts) and mu^2 input (`ScaleInput::MuSquared`).
//!
//! - Supplying the very invariants the direct engine derives from the shifts
//!   (its own arithmetic and lightlike rule) reproduces every opening case
//!   bitwise, through `DirectWorkspace` and through the facade with its
//!   loop-rotation test (which permutes the supplied table).
//! - mu^2 input equal to the square the engine forms reproduces mu input
//!   bitwise.
//! - The DESIGN 11.1 contract of pyAmpliCol at TIRust level: on the massive-b
//!   reproducer (b b~ > t t~, the IR-divergent (0, MB, MT) vertex triangle at
//!   a binary64-rounded point) with the on-shell invariants supplied exactly,
//!   double-double equals MPFR within 1e-24 and binary64 equals MPFR within
//!   1e-12, while the invariants derived from the rounded shifts leave the
//!   legs off shell and move MPFR by O(1).
use std::path::Path;

use iregi_rust::{
    DirectReductionPlan, InvariantInput, KinematicsBatch, LaneStatus, ReductionPlan,
    ReductionSettings, ScaleInput, TensorReductionPlan, TensorWorkspace, Workspace,
};
use iregi_masters::NativeFloat;
use serde_json::Value;
use symbolica::domains::{
    float::{Complex, DoubleFloat, Float, Real, RealLike, SingleFloat},
    integer::Integer,
    rational::Rational,
};

const SUITE: &str = "../../validation/benchmark/opening-rank6-20261026";

fn exact(text: &str) -> f64 {
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let q = Rational::from((n.parse::<Integer>().unwrap(), d.parse::<Integer>().unwrap()));
    0.0f64.from_rational(&q)
}

struct Case {
    id: String,
    settings: ReductionSettings,
    /// Shifts, squared masses, mu (the `FromShifts` row).
    row: Vec<Complex<f64>>,
}

fn opening_cases() -> Vec<Case> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(SUITE);
    let inputs: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("inputs.json")).unwrap()).unwrap();
    let mut cases = Vec::new();
    for case in inputs["cases"].as_array().unwrap() {
        let request = &case["driver_request"];
        let id = request["id"].as_str().unwrap().to_string();
        let sample = &request["samples"][0];
        let mut row = Vec::new();
        for shift in sample["shifts"].as_array().unwrap() {
            for component in shift.as_array().unwrap() {
                row.push(Complex::new(exact(component.as_str().unwrap()), 0.0));
            }
        }
        let n = row.len() / 4;
        for mass in sample["masses_squared"].as_array().unwrap() {
            row.push(Complex::new(exact(mass[0].as_str().unwrap()), exact(mass[1].as_str().unwrap())));
        }
        row.push(Complex::new(exact(sample["mu"].as_str().unwrap()), 0.0));
        let rank = request["rank"].as_u64().unwrap() as u16;
        let mut settings = ReductionSettings::new(n, rank);
        if let Some(powers) = request["powers"].as_array() {
            settings.powers = powers.iter().map(|p| p.as_u64().unwrap() as u16).collect();
        }
        settings.dimension_shift = request["dimension_shift"].as_i64().unwrap_or(0) as i16;
        if DirectReductionPlan::supports(&settings) {
            cases.push(Case { id, settings, row });
        }
    }
    cases
}

/// The direct engine's own binary64 invariants of real shifts: per component
/// one difference and one square, summed (+,-,-,-), and zero within the
/// rounding error of the represented shifts (its lightlike rule).
fn derived_invariants(row: &[Complex<f64>], n: usize) -> Vec<Complex<f64>> {
    let rounding = 8.0 * 2f64.powi(-53);
    let mut out = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            let (mut value, mut size) = (0.0f64, 0.0f64);
            for a in 0..4 {
                let (qi, qj) = (row[4 * i + a].re, row[4 * j + a].re);
                assert_eq!((row[4 * i + a].im, row[4 * j + a].im), (0.0, 0.0));
                let delta = qi - qj;
                let square = delta * delta;
                size += square.abs();
                size += delta.abs() * (qi.abs() + qj.abs()) * 2.0;
                value = if a == 0 { value + square } else { value - square };
            }
            if value.abs() <= size * rounding {
                value = 0.0;
            }
            out.push(Complex::new(value, 0.0));
        }
    }
    out
}

/// `row` (shifts, masses, scale) with `invariants` inserted after the shifts.
fn with_invariants<T: Clone>(row: &[Complex<T>], n: usize, invariants: &[Complex<T>]) -> Vec<Complex<T>> {
    let mut out = row[..4 * n].to_vec();
    out.extend_from_slice(invariants);
    out.extend_from_slice(&row[4 * n..]);
    out
}

fn supplied(settings: &ReductionSettings) -> ReductionSettings {
    let mut settings = settings.clone();
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings
}

fn bits(values: &[Complex<f64>]) -> Vec<(u64, u64)> {
    values.iter().map(|z| (z.re.to_bits(), z.im.to_bits())).collect()
}

fn batch<T: SingleFloat>(row: Vec<Complex<T>>) -> KinematicsBatch<T> {
    KinematicsBatch { rows: 1, components: 1, values: row }
}

#[test]
fn layout_packs_the_strict_upper_triangle() {
    let mut settings = supplied(&ReductionSettings::new(4, 2));
    let layout = iregi_rust::InputLayout::of(&settings);
    assert_eq!(layout.shifts, 0..16);
    assert_eq!(layout.scalar_products, Some(16..22));
    assert_eq!(layout.masses, 22..26);
    assert_eq!((layout.scale, layout.width()), (26, 27));
    let mut index = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            assert_eq!(InvariantInput::invariant_index(4, i, j), index);
            index += 1;
        }
    }
    settings.invariants = InvariantInput::FromShifts;
    assert_eq!(iregi_rust::InputLayout::of(&settings).width(), 21);
    let plan = TensorReductionPlan::new(supplied(&ReductionSettings::new(3, 1))).unwrap();
    assert_eq!(plan.engine_kind(), iregi_rust::TensorEngineKind::Direct);
    assert_eq!(plan.input_layout(), &iregi_rust::InputLayout::of(&supplied(&ReductionSettings::new(3, 1))));
}

#[test]
fn supplied_derived_invariants_reproduce_every_opening_case_bitwise() {
    let cases = opening_cases();
    assert!(cases.len() >= 18, "{} direct cases", cases.len());
    for case in &cases {
        let n = case.settings.propagators;
        let invariants = derived_invariants(&case.row, n);
        let row = with_invariants(&case.row, n, &invariants);
        let expected = DirectReductionPlan::new(case.settings.clone())
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&batch(case.row.clone()))
            .unwrap();
        let got = DirectReductionPlan::new(supplied(&case.settings))
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&batch(row.clone()))
            .unwrap();
        assert_eq!(got.statuses, expected.statuses, "{}", case.id);
        if expected.statuses[0] == LaneStatus::Success {
            assert_eq!(bits(got.lane(0).unwrap()), bits(expected.lane(0).unwrap()), "{}", case.id);
        }
        // The facade: loop-rotation test, retries and rescue on the
        // supplied table (rotated with the propagators).
        let plan = TensorReductionPlan::new(case.settings.clone()).unwrap();
        let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
        let expected = workspace.reduce_batch(&batch(case.row.clone())).unwrap();
        let plan = TensorReductionPlan::new(supplied(&case.settings)).unwrap();
        let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
        let got = workspace.reduce_batch(&batch(row)).unwrap();
        assert_eq!(got.statuses, expected.statuses, "{} (facade)", case.id);
        if expected.statuses[0] == LaneStatus::Success {
            assert_eq!(bits(got.lane(0).unwrap()), bits(expected.lane(0).unwrap()), "{} (facade)", case.id);
        }
    }
}

#[test]
fn mu_squared_input_reproduces_mu_input_bitwise() {
    for case in opening_cases() {
        let mut row = case.row.clone();
        let mu = row.last().unwrap().re;
        *row.last_mut().unwrap() = Complex::new(mu * mu, 0.0);
        let mut settings = case.settings.clone();
        settings.scale_input = ScaleInput::MuSquared;
        let expected = DirectReductionPlan::new(case.settings.clone())
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&batch(case.row.clone()))
            .unwrap();
        let got = DirectReductionPlan::new(settings.clone())
            .unwrap()
            .workspace(0.0f64)
            .unwrap()
            .reduce_batch(&batch(row.clone()))
            .unwrap();
        assert_eq!(got.statuses, expected.statuses, "{}", case.id);
        if expected.statuses[0] == LaneStatus::Success {
            assert_eq!(bits(got.lane(0).unwrap()), bits(expected.lane(0).unwrap()), "{}", case.id);
        }
        let plan = TensorReductionPlan::new(case.settings.clone()).unwrap();
        let expected = TensorWorkspace::new(&plan, 0.0f64).unwrap().reduce_batch(&batch(case.row.clone())).unwrap();
        let plan = TensorReductionPlan::new(settings).unwrap();
        let got = TensorWorkspace::new(&plan, 0.0f64).unwrap().reduce_batch(&batch(row)).unwrap();
        assert_eq!(got.statuses, expected.statuses, "{} (facade)", case.id);
        if expected.statuses[0] == LaneStatus::Success {
            assert_eq!(bits(got.lane(0).unwrap()), bits(expected.lane(0).unwrap()), "{} (facade)", case.id);
        }
    }
    // A non-positive mu^2 is invalid input, named as such.
    let mut settings = ReductionSettings::new(2, 1);
    settings.scale_input = ScaleInput::MuSquared;
    let mut row: Vec<Complex<f64>> = [0., 0., 0., 0., 5., 1., 0., 0., 4., 9., -3.]
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    let result = DirectReductionPlan::new(settings.clone()).unwrap().workspace(0.0f64).unwrap().reduce_batch(&batch(row.clone())).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::InvalidInput("mu^2 must be positive and real".into()));
    // The symbolic engines read mu: their own constructors refuse mu^2.
    assert!(matches!(ReductionPlan::new(settings.clone()), Err(iregi_rust::ReductionError::Unsupported(_))));
    // The facade forms mu = sqrt(mu^2) for them (here: double-double, whose
    // facade workspace is the symbolic engine).
    row[10] = Complex::new(9.0, 0.0);
    let plan = TensorReductionPlan::new(settings).unwrap();
    let mut workspace = TensorWorkspace::new(&plan, DoubleFloat::from(0.0)).unwrap();
    let lifted: Vec<Complex<DoubleFloat>> =
        row.iter().map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im))).collect();
    let got = workspace.reduce_batch(&batch(lifted.clone())).unwrap();
    let mut mu_row = lifted;
    mu_row[10] = Complex::new(DoubleFloat::from(3.0), DoubleFloat::from(0.0));
    let plan = TensorReductionPlan::new(ReductionSettings::new(2, 1)).unwrap();
    let expected = TensorWorkspace::new(&plan, DoubleFloat::from(0.0)).unwrap().reduce_batch(&batch(mu_row)).unwrap();
    assert_eq!(got.statuses, expected.statuses);
    for (a, b) in got.lane(0).unwrap().iter().zip(expected.lane(0).unwrap()) {
        assert_eq!((a.re.to_f64(), a.im.to_f64()), (b.re.to_f64(), b.im.to_f64()));
    }
}

/// `x` (binary64) exactly in the carrier of `prototype`.
fn lift<T: NativeFloat>(prototype: &T, x: f64) -> T {
    prototype.from_rational(&Float::with_val(53, x).try_to_rational().unwrap())
}

/// Minkowski square of `a - b` in the carrier.
fn square<T: NativeFloat>(a: &[T; 4], b: &[T; 4]) -> T {
    let d: Vec<T> = (0..4).map(|k| a[k].clone() - &b[k]).collect();
    d[0].clone() * &d[0] - d[1].clone() * &d[1] - d[2].clone() * &d[2] - d[3].clone() * &d[3]
}

/// The massive-b reproducer of pyAmpliCol's DESIGN 11.1 (b b~ > t t~ QED=2
/// [virt=QCD], E = 500 GeV, MB = 4.7, MT = 173, every momentum component
/// rounded to binary64, mu^2 = 250000): the vertex triangle of the b -> t W
/// line, propagators gluon (0), b (MB) and t (MT), its b and t legs on shell.
/// One row in the carrier of `prototype`: shifts, then (if `on_shell`) the
/// supplied invariants with the on-shell ones the carrier's squared masses
/// themselves, masses, mu^2.
fn reproducer_row<T: NativeFloat>(prototype: &T, on_shell: bool) -> Vec<Complex<T>> {
    let (energy, mb, mt) = (500.0f64, 4.7f64, 173.0f64);
    let pb = (energy * energy - mb * mb).sqrt();
    let pt = (energy * energy - mt * mt).sqrt();
    let b = [energy, 0.0, 0.0, pb];
    let t = [energy, pt * 0.8, 0.0, pt * 0.6];
    let lifted = |v: [f64; 4]| -> [T; 4] { v.map(|x| lift(prototype, x)) };
    let shifts = [lifted([0.0; 4]), lifted(b), lifted(t)];
    let zero = prototype.zero();
    let real = |x: T| Complex::new(x, zero.clone());
    let mb2 = prototype.from_rational(&Rational::from((Integer::from(2209), Integer::from(100))));
    let mt2 = prototype.from_rational(&Rational::from(Integer::from(29929)));
    let mut row: Vec<Complex<T>> = shifts.iter().flatten().cloned().map(real).collect();
    if on_shell {
        // (0,1): the b leg, (0,2): the t leg, (1,2): the W, from the shifts.
        row.extend([mb2.clone(), mt2.clone(), square(&shifts[1], &shifts[2])].map(real));
    }
    row.extend([prototype.zero(), mb2, mt2].map(real));
    row.push(real(prototype.from_rational(&Rational::from(Integer::from(250000)))));
    row
}

fn reproducer_settings(on_shell: bool) -> ReductionSettings {
    let mut settings = ReductionSettings::new(3, 2);
    settings.convention = iregi_rust::Convention::OneLoop;
    settings.scale_input = ScaleInput::MuSquared;
    if on_shell {
        settings.invariants = InvariantInput::SuppliedInvariants;
    }
    settings
}

fn reduce<T: NativeFloat>(prototype: T, on_shell: bool) -> Vec<Complex<f64>>
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let row = reproducer_row(&prototype, on_shell);
    let mut workspace = DirectReductionPlan::new(reproducer_settings(on_shell)).unwrap().workspace(prototype).unwrap();
    let result = workspace.reduce_batch(&batch(row)).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    result.lane(0).unwrap().iter().map(|z| Complex::new(z.re.to_f64(), z.im.to_f64())).collect()
}

/// Largest |a - b| over the array, relative to the largest |b|, per Laurent slot.
fn slot_deviations(a: &[Complex<f64>], b: &[Complex<f64>]) -> [f64; 4] {
    std::array::from_fn(|slot| {
        let (mut error, mut scale) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b).skip(slot).step_by(4) {
            error = error.max((x.re - y.re).hypot(x.im - y.im));
            scale = scale.max(y.re.hypot(y.im));
        }
        if scale == 0.0 { error } else { error / scale }
    })
}

/// Difference of two higher-precision arrays relative to the second's scale,
/// per Laurent slot, without rounding them to binary64 first.
fn precise_deviation<T: NativeFloat, U: NativeFloat>(a: &[Complex<T>], b: &[Complex<U>], bits: u32) -> f64 {
    let to = |x: &dyn Fn() -> Rational| Float::with_val(bits, 0).from_rational(&x());
    let mut worst = 0.0f64;
    for slot in 0..4 {
        let (mut error, mut scale) = (Float::with_val(bits, 0), Float::with_val(bits, 0));
        for (x, y) in a.iter().zip(b).skip(slot).step_by(4) {
            let xr = to(&|| x.re.exact_rational().unwrap());
            let xi = to(&|| x.im.exact_rational().unwrap());
            let yr = to(&|| y.re.exact_rational().unwrap());
            let yi = to(&|| y.im.exact_rational().unwrap());
            let dr = xr - &yr;
            let di = xi - &yi;
            let e = (dr.clone() * &dr + di.clone() * &di).sqrt();
            let s = (yr.clone() * &yr + yi.clone() * &yi).sqrt();
            if e > error {
                error = e;
            }
            if s > scale {
                scale = s;
            }
        }
        let relative = if scale.is_zero() { error.to_f64() } else { (error / &scale).to_f64() };
        worst = worst.max(relative);
    }
    worst
}

fn reduce_exact<T: NativeFloat>(prototype: T, on_shell: bool) -> Vec<Complex<T>>
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let row = reproducer_row(&prototype, on_shell);
    let mut workspace = DirectReductionPlan::new(reproducer_settings(on_shell)).unwrap().workspace(prototype).unwrap();
    let result = workspace.reduce_batch(&batch(row)).unwrap();
    assert_eq!(result.statuses[0], LaneStatus::Success);
    result.lane(0).unwrap().to_vec()
}

#[test]
fn massive_b_reproducer_double_double_equals_mpfr_on_shell() {
    let mpfr = reduce_exact(Float::new(256), true);
    let double = reduce_exact(DoubleFloat::from(0.0), true);
    let dd_deviation = precise_deviation(&double, &mpfr, 256);
    println!("double-double vs MPFR(256), on-shell invariants supplied: {dd_deviation:e}");
    assert!(dd_deviation <= 1e-24, "{dd_deviation:e}");
    // The soft pole of the IR-divergent triangle (massive legs: a single
    // pole, no double pole) is there.
    let mpfr64: Vec<Complex<f64>> = mpfr.iter().map(|z| Complex::new(z.re.to_f64(), z.im.to_f64())).collect();
    assert!(mpfr64[1].re != 0.0 && mpfr64[2].re == 0.0, "{:?}", &mpfr64[..4]);
    // Binary64 through the same supplied invariants.
    let binary64 = reduce(0.0f64, true);
    let b64 = slot_deviations(&binary64, &mpfr64);
    println!("binary64 vs MPFR(256), on-shell invariants supplied, per slot: {b64:?}");
    assert!(b64.iter().all(|&d| d <= 1e-12), "{b64:?}");
    // Invariants derived from the rounded shifts: the legs are off shell by
    // rounding, the masters take their off-shell branches and the soft pole
    // disappears (the single pole moves by 100 %, the finite part by 3e-4 of
    // the array): even MPFR is wrong at O(1).
    let off = reduce(Float::new(256), false);
    let moved = slot_deviations(&off, &mpfr64);
    println!("MPFR(256) from the rounded shifts vs on-shell, per slot: {moved:?}");
    assert!(off[1].re == 0.0 && moved[1] >= 0.5 && moved[3] > 1e-6, "{moved:?} {:?}", &off[..4]);
}

/// The generic symbolic plan of `settings` in the carrier of `prototype` on
/// the binary64 rows `rows` (each: shifts, squared masses, mu): its Cartesian
/// plan, then with the carrier's invariants of the same (exactly lifted)
/// shifts supplied. Returns both statuses and, per lane where both succeed, the
/// supplied-vs-Cartesian deviation.
fn symbolic_supplied_vs_cartesian<T: NativeFloat>(
    prototype: T,
    settings: &ReductionSettings,
    rows: &[Vec<Complex<f64>>],
) -> (Vec<LaneStatus>, Vec<LaneStatus>, Vec<f64>)
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    symbolic_supplied_vs_cartesian_at(prototype, None, settings, rows).0
}

/// [`symbolic_supplied_vs_cartesian`] with the workspaces' accuracy precision
/// `accuracy` (bits; `None`: the working precision), also returning the
/// Cartesian values.
#[allow(clippy::type_complexity)]
fn symbolic_supplied_vs_cartesian_at<T: NativeFloat>(
    prototype: T,
    accuracy: Option<u32>,
    settings: &ReductionSettings,
    rows: &[Vec<Complex<f64>>],
) -> ((Vec<LaneStatus>, Vec<LaneStatus>, Vec<f64>), Vec<Complex<T>>)
where
    Complex<T>: symbolica::evaluate::EvaluationDomain,
{
    let n = settings.propagators;
    let zero = prototype.zero();
    let (mut cartesian, mut supplied_rows) = (Vec::new(), Vec::new());
    for row in rows {
        let lifted: Vec<Complex<T>> =
            row.iter().map(|z| Complex::new(lift(&prototype, z.re), lift(&prototype, z.im))).collect();
        let shifts: Vec<[T; 4]> =
            (0..n).map(|i| std::array::from_fn(|a| lifted[4 * i + a].re.clone())).collect();
        let mut invariants = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                invariants.push(Complex::new(square(&shifts[i], &shifts[j]), zero.clone()));
            }
        }
        supplied_rows.extend(with_invariants(&lifted, n, &invariants));
        cartesian.extend(lifted);
    }
    let lanes = rows.len();
    let plan = ReductionPlan::new(settings.clone()).unwrap();
    let mut workspace = Workspace::new(&plan, prototype.clone()).unwrap();
    if let Some(bits) = accuracy {
        workspace.set_accuracy_precision(bits).unwrap();
    }
    let expected = workspace.reduce_batch(&KinematicsBatch { rows: lanes, components: 1, values: cartesian }).unwrap();
    let plan = ReductionPlan::new(supplied(settings)).unwrap();
    assert_eq!(plan.input_layout(), &iregi_rust::InputLayout::of(&supplied(settings)));
    let mut workspace = Workspace::new(&plan, prototype).unwrap();
    if let Some(bits) = accuracy {
        workspace.set_accuracy_precision(bits).unwrap();
    }
    let got = workspace.reduce_batch(&KinematicsBatch { rows: lanes, components: 1, values: supplied_rows }).unwrap();
    let deviations = (0..lanes)
        .filter(|&lane| got.statuses[lane] == LaneStatus::Success && expected.statuses[lane] == LaneStatus::Success)
        .map(|lane| precise_deviation(got.lane(lane).unwrap(), expected.lane(lane).unwrap(), 256))
        .collect();
    let width = expected.basis.len() * 4;
    let values = (0..lanes)
        .flat_map(|lane| expected.lane(lane).map_or_else(|_| vec![Complex::new(zero.clone(), zero.clone()); width], <[_]>::to_vec))
        .collect();
    ((got.statuses, expected.statuses.clone(), deviations), values)
}

#[test]
fn symbolic_engine_reads_supplied_invariants() {
    // The generic symbolic plan with supplied invariants equals its Cartesian
    // plan in double-double (supplied: the double-double invariants of the
    // same shifts): PV on a generic triangle and box, IBP (no self-alias of
    // the supplied symbols) on a dotted triangle.
    let shifts: [[f64; 4]; 4] = [
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.5, 2.0],
        [5.0, -1.5, 2.0, 0.25],
        [7.5, 0.5, -2.5, 1.0],
    ];
    let masses = [1.0f64, 2.5, 0.75, 4.0];
    let mut dotted = ReductionSettings::new(3, 1);
    dotted.powers = vec![2, 1, 1];
    dotted.algorithm = iregi_rust::Algorithm::Ibp;
    for settings in [ReductionSettings::new(3, 2), ReductionSettings::new(4, 1), dotted] {
        let n = settings.propagators;
        let mut row: Vec<Complex<f64>> = shifts[..n].iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
        row.extend(masses[..n].iter().map(|&m| Complex::new(m, 0.0)));
        row.push(Complex::new(3.0, 0.0));
        let (got, expected, deviations) = symbolic_supplied_vs_cartesian(DoubleFloat::from(0.0), &settings, &[row]);
        assert_eq!(got, expected);
        assert_eq!(got[0], LaneStatus::Success);
        println!("symbolic N{n} R{} powers {:?}: supplied vs Cartesian {:e}", settings.rank, settings.powers, deviations[0]);
        assert!(deviations[0] <= 1e-28, "{settings:?}: {:e}", deviations[0]);
    }
}

#[test]
fn symbolic_engine_reads_supplied_on_shell_invariants() {
    // The massive-b reproducer at 256 bits through the generic symbolic
    // engine: with the on-shell invariants supplied it equals the direct
    // engine and keeps the soft pole, which invariants recomputed from the
    // rounded shifts lose (an engine ignoring the supplied values fails).
    // The symbolic engine reads mu: 500 for the reproducer's mu^2 = 250000.
    let prototype = Float::new(256);
    let direct = reduce_exact(prototype.clone(), true);
    let with_mu = |on_shell: bool| {
        let mut settings = reproducer_settings(on_shell);
        settings.scale_input = ScaleInput::Mu;
        let mut row = reproducer_row(&prototype, on_shell);
        *row.last_mut().unwrap() = Complex::new(Float::with_val(256, 500), Float::new(256));
        (settings, row)
    };
    let (settings, row) = with_mu(true);
    let plan = ReductionPlan::new(settings).unwrap();
    let mut workspace = Workspace::new(&plan, prototype.clone()).unwrap();
    let got = workspace.reduce_batch(&batch(row)).unwrap();
    assert_eq!(got.statuses[0], LaneStatus::Success);
    let got = got.lane(0).unwrap();
    let deviation = precise_deviation(got, &direct, 256);
    println!("symbolic engine (256 bits), on-shell invariants supplied, vs the direct engine: {deviation:e}");
    assert!(deviation <= 1e-60, "{deviation:e}");
    assert!(!got[1].re.is_zero() && got[2].re.is_zero(), "the single soft pole, no double pole");
    // The Cartesian symbolic plan of the rounded shifts, for the record.
    let (settings, row) = with_mu(false);
    let cartesian = ReductionPlan::new(settings)
        .and_then(|plan| Workspace::new(&plan, prototype.clone())?.reduce_batch(&batch(row)));
    match cartesian {
        Ok(result) => match result.lane(0) {
            Ok(lane) => println!(
                "symbolic engine, Cartesian rounded shifts: single pole {:e} (on shell {:e})",
                lane[1].re.to_f64(),
                got[1].re.to_f64()
            ),
            Err(status) => println!("symbolic engine, Cartesian rounded shifts: {status:?}"),
        },
        Err(error) => println!("symbolic engine, Cartesian rounded shifts: refused ({error})"),
    }
}

/// Two lanes of `n <= 7` points: generic shifts and shifts routed by
/// timelike external momenta, squared masses 1 to 4, mu = 3 (dyadic values,
/// so their invariants are exact in every carrier here).
fn generic_rows(n: usize) -> Vec<Vec<Complex<f64>>> {
    let generic: [[f64; 4]; 7] = [
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.5, 2.0],
        [5.0, -1.5, 2.0, 0.25],
        [7.5, 0.5, -2.5, 1.0],
        [9.0, 2.25, 1.5, -3.0],
        [4.0, -0.75, -1.0, 2.5],
        [6.5, 1.75, -0.5, -1.5],
    ];
    let routed: [[f64; 4]; 7] = [
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 0.5, -0.75, 1.25],
        [9.5, 1.5, 0.25, 0.5],
        [15.0, 0.25, 1.0, -0.5],
        [19.25, -0.75, 0.5, -1.25],
        [24.0, -0.5, -0.25, 0.75],
        [28.5, 0.75, -1.0, 0.25],
    ];
    let masses = [1.0f64, 2.5, 0.75, 4.0, 1.5, 3.25, 2.0];
    [generic, routed]
        .iter()
        .map(|shifts| {
            let mut row: Vec<Complex<f64>> = shifts[..n].iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
            row.extend(masses[..n].iter().map(|&m| Complex::new(m, 0.0)));
            row.push(Complex::new(3.0, 0.0));
            row
        })
        .collect()
}

#[test]
#[ignore = "generic five- to seven-point symbolic preparation (two plans each; about 13 minutes in a debug build); run explicitly"]
fn symbolic_engine_reads_supplied_invariants_from_five_points_on() {
    // Five points (PV), six and seven points: the exact four-dimensional
    // denominator identities take their vector rows from the shifts and their
    // constant row from the supplied invariants, in PV and in the IBP routing.
    // At 256 bits with a requested accuracy of 64 bits (the generic plans'
    // cancellation estimate refuses these lanes at the full working
    // precision, for Cartesian shifts as well), the supplied-invariant plan
    // equals its Cartesian plan and both equal the direct engine.
    let mut six_ibp = ReductionSettings::new(6, 0);
    six_ibp.algorithm = iregi_rust::Algorithm::Ibp;
    for settings in [ReductionSettings::new(5, 1), ReductionSettings::new(6, 0), six_ibp, ReductionSettings::new(7, 0)] {
        let n = settings.propagators;
        let rows = generic_rows(n);
        let started = std::time::Instant::now();
        let ((got, expected, deviations), values) =
            symbolic_supplied_vs_cartesian_at(Float::new(256), Some(64), &settings, &rows);
        let elapsed = started.elapsed();
        assert_eq!(got, expected);
        assert!(got.iter().all(|status| *status == LaneStatus::Success), "{got:?}");
        assert!(deviations.iter().all(|&d| d <= 1e-60), "{deviations:?}");
        let mut direct_settings = settings.clone();
        direct_settings.algorithm = iregi_rust::Algorithm::Hybrid;
        let mut direct = DirectReductionPlan::new(direct_settings).unwrap().workspace(Float::new(256)).unwrap();
        let lifted = rows.iter().flatten().map(|z| Complex::new(Float::with_val(256, z.re), Float::with_val(256, z.im))).collect();
        let reference = direct.reduce_batch(&KinematicsBatch { rows: rows.len(), components: 1, values: lifted }).unwrap();
        let stride = values.len() / rows.len();
        let against_direct: Vec<f64> = (0..rows.len())
            .map(|lane| precise_deviation(&values[lane * stride..(lane + 1) * stride], reference.lane(lane).unwrap(), 256))
            .collect();
        println!(
            "symbolic N{n} R{} {:?} at 256 bits ({elapsed:?}): supplied vs Cartesian {deviations:?}, vs the direct engine {against_direct:?}",
            settings.rank, settings.algorithm
        );
        assert!(against_direct.iter().all(|&d| d <= 1e-60), "{against_direct:?}");
    }
    // The facade serves supplied invariants by the direct engine.
    let settings = supplied(&ReductionSettings::new(6, 1));
    assert_eq!(TensorReductionPlan::new(settings).unwrap().engine_kind(), iregi_rust::TensorEngineKind::Direct);
}
