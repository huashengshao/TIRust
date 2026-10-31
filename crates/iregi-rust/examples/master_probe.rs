//! One scalar master in binary64 (hand-written and generated kernels),
//! double-double and 256 bits (development aid).
//!
//! usage: master_probe FAMILY ARG... (OneLOop argument order, mu² last)
use oneloop::{NativeEvaluator, ScalarIntegral};
use symbolica::domains::float::{Complex, DoubleFloat, Float, RealLike};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let family = match args[1].as_str() {
        "A0" => ScalarIntegral::A0,
        "B0" => ScalarIntegral::B0,
        "C0" => ScalarIntegral::C0,
        "D0" => ScalarIntegral::D0,
        other => panic!("unknown family {other}"),
    };
    let values: Vec<f64> = args[2..].iter().map(|s| s.parse().unwrap()).collect();
    let show = |name: &str, out: Vec<(f64, f64)>| {
        println!("{name:>22}: {}", out.iter().map(|(r, i)| format!("({r:.17e}, {i:.17e})")).collect::<Vec<_>>().join(" "));
    };
    for handwritten in [true, false] {
        let mut native = NativeEvaluator::<f64>::new(family).unwrap();
        native.set_handwritten(handwritten);
        let input: Vec<_> = values.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let mut out = vec![Complex::new(0.0, 0.0); 3];
        let status = native.evaluate(&input, &mut out);
        show(if handwritten { "binary64 hand-written" } else { "binary64 generated" }, out.iter().map(|z| (z.re, z.im)).collect());
        if let Err(e) = status {
            println!("  error {e}");
        }
    }
    let mut native = NativeEvaluator::<DoubleFloat>::new(family).unwrap();
    let input: Vec<_> = values.iter().map(|&x| Complex::new(DoubleFloat::from(x), DoubleFloat::from(0.0))).collect();
    let mut out = vec![Complex::new(DoubleFloat::from(0.0), DoubleFloat::from(0.0)); 3];
    native.evaluate(&input, &mut out).unwrap();
    show("double-double", out.iter().map(|z| (z.re.to_f64(), z.im.to_f64())).collect());
    let bits = 256;
    let mut native = NativeEvaluator::<Float>::with_binary_precision(family, bits).unwrap();
    let input: Vec<_> = values.iter().map(|&x| Complex::new(Float::with_val(bits, x), Float::new(bits))).collect();
    let mut out = vec![Complex::new(Float::new(bits), Float::new(bits)); 3];
    native.evaluate(&input, &mut out).unwrap();
    show("256 bits", out.iter().map(|z| (z.re.to_f64(), z.im.to_f64())).collect());
}
