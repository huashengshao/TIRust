//! Validation of oneloop's hand-written binary64 C0 for negative Källén
//! functions (the 't Hooft--Veltman form) against the generated expression
//! code it replaces. Random three-mass points; prints the maximum relative
//! difference per class and the worst point. usage: c0_hv_check [SAMPLES]
use iregi_masters::ScalarIntegral;
use symbolica::domains::float::Complex;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

fn main() {
    let samples: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(20000);
    let mut hand = oneloop::NativeEvaluator::<f64>::new(ScalarIntegral::C0).unwrap();
    let mut generated = oneloop::NativeEvaluator::<f64>::new(ScalarIntegral::C0).unwrap();
    generated.set_handwritten(false);
    let mut rng = Rng(0x9e3779b97f4a7c15);
    for (class, sign, complex) in [("euclidean real", -1.0, false), ("timelike real", 1.0, false),
                                   ("euclidean complex", -1.0, true), ("timelike complex", 1.0, true)] {
        let (mut worst, mut worst_point, mut count) = (0.0f64, None, 0);
        let (mut hand_max, mut generated_max) = (0.0f64, 0.0f64);
        let mut wide = oneloop::NativeEvaluator::<symbolica::domains::float::Float>::with_binary_precision(ScalarIntegral::C0, 128).unwrap();
        while count < samples {
            let p: [f64; 3] = std::array::from_fn(|_| sign * rng.range(0.05, 10.0));
            let lambda = (p[2] - p[0] - p[1]).powi(2) - 4.0 * p[0] * p[1];
            if lambda >= 0.0 {
                continue;
            }
            let m: [Complex<f64>; 3] = std::array::from_fn(|_| {
                Complex::new(rng.range(0.05, 10.0), if complex { -rng.range(0.0, 1.0) } else { 0.0 })
            });
            let mu2 = Complex::new(rng.range(0.5, 20.0), 0.0);
            let input = [Complex::new(p[0], 0.0), Complex::new(p[1], 0.0), Complex::new(p[2], 0.0), m[0], m[1], m[2], mu2];
            let mut a = [Complex::new(0.0, 0.0); 3];
            let mut b = [Complex::new(0.0, 0.0); 3];
            hand.evaluate(&input, &mut a).unwrap();
            generated.evaluate(&input, &mut b).unwrap();
            let diff = (a[0].re - b[0].re).hypot(a[0].im - b[0].im) / b[0].re.hypot(b[0].im).max(1e-300);
            if count < 2000 {
                use symbolica::domains::float::{Float, RealLike};
                let lifted: Vec<_> = input.iter().map(|z| Complex::new(Float::with_val(128, z.re), Float::with_val(128, z.im))).collect();
                let zero = Complex::new(Float::new(128), Float::new(128));
                let mut out = [zero.clone(), zero.clone(), zero];
                wide.evaluate(&lifted, &mut out).unwrap();
                let r = Complex::new(out[0].re.to_f64(), out[0].im.to_f64());
                let rel = |x: Complex<f64>| (x.re - r.re).hypot(x.im - r.im) / r.re.hypot(r.im);
                hand_max = hand_max.max(rel(a[0]));
                generated_max = generated_max.max(rel(b[0]));
            }
            if !(diff <= worst) {
                worst = diff;
                worst_point = Some((input, a[0], b[0]));
            }
            count += 1;
        }
        println!("{class:18}: max relative difference {worst:.3e} over {samples} points; vs 128-bit on 2000: hand-written {hand_max:.3e}, generated {generated_max:.3e}");
        if let Some((input, a, b)) = worst_point {
            // 128-bit generated reference at the worst point.
            use symbolica::domains::float::{Float, RealLike};
            let mut wide = oneloop::NativeEvaluator::<Float>::with_binary_precision(ScalarIntegral::C0, 128).unwrap();
            let lifted: Vec<_> = input.iter().map(|z| Complex::new(Float::with_val(128, z.re), Float::with_val(128, z.im))).collect();
            let zero = Complex::new(Float::new(128), Float::new(128));
            let mut out = [zero.clone(), zero.clone(), zero];
            wide.evaluate(&lifted, &mut out).unwrap();
            let r = Complex::new(out[0].re.to_f64(), out[0].im.to_f64());
            let rel = |x: Complex<f64>| (x.re - r.re).hypot(x.im - r.im) / r.re.hypot(r.im);
            println!("    vs 128-bit: hand-written {:.3e}, generated {:.3e}", rel(a), rel(b));
        }
    }
    // Timing of both on one point.
    let input = [-1.0, -2.0, -1.0, 1.0, 2.0, 3.0, 9.0].map(|x| Complex::new(x, 0.0));
    for (name, evaluator) in [("hand-written", &mut hand), ("generated", &mut generated)] {
        let mut out = [Complex::new(0.0, 0.0); 3];
        let started = std::time::Instant::now();
        for _ in 0..200000 {
            evaluator.evaluate(std::hint::black_box(&input), &mut out).unwrap();
        }
        println!("{name}: {:.3} us per C0", started.elapsed().as_secs_f64() / 200000.0 * 1e6);
    }
}
