//! Investigate the preserved five-point rank-six binary64 discrepancy.
//! Uses exact Symbolica coefficients and native scalar masters at 128/256 bits.

#[path = "../src/ibp.rs"]
mod ibp;
#[path = "../src/pv.rs"]
mod pv;
use oneloop::{NativeEvaluator, ScalarIntegral};
use std::collections::HashMap;
use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    domains::float::{Complex, Float, Real, RealLike},
};
fn main() {
    let points = [
        [0i64, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ];
    let masses = [4i64, 9, 16, 25, 36];
    let mut compiler = pv::PvCompiler::new(
        points.iter().map(|p| p.map(Atom::num)).collect(),
        masses.map(Atom::num).to_vec(),
    )
    .unwrap();
    let expression = compiler.compile_moment([5, 0, 0, 1]).unwrap();
    eprintln!("compiled {} master leaves", compiler.leaves().len());
    for bits in [128, 256] {
        let zero = Float::new(bits);
        let complex = |x: i64| Complex::new(Float::with_val(bits, x), Float::new(bits));
        let mut values: [HashMap<Atom, Float>; 2] = std::array::from_fn(|_| HashMap::new());
        let mut evaluators: Vec<_> = [
            ScalarIntegral::A0,
            ScalarIntegral::B0,
            ScalarIntegral::C0,
            ScalarIntegral::D0,
        ]
        .map(|family| NativeEvaluator::<Float>::with_binary_precision(family, bits).unwrap())
        .into_iter()
        .collect();
        for leaf in compiler.leaves() {
            let a = &leaf.active;
            let mut args = Vec::new();
            let mut distance = |i: usize, j: usize| {
                let d = std::array::from_fn::<_, 4, _>(|mu| points[a[i]][mu] - points[a[j]][mu]);
                args.push(complex(
                    d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3],
                ));
            };
            match a.len() {
                1 => {}
                2 => distance(0, 1),
                3 => {
                    distance(0, 1);
                    distance(1, 2);
                    distance(0, 2);
                }
                4 => {
                    distance(0, 1);
                    distance(1, 2);
                    distance(2, 3);
                    distance(0, 3);
                    distance(0, 2);
                    distance(1, 3);
                }
                _ => unreachable!(),
            }
            args.extend(a.iter().map(|&i| complex(masses[i])));
            args.push(complex(9));
            let mut output = vec![complex(0); 3];
            evaluators[a.len() - 1]
                .evaluate(&args, &mut output)
                .unwrap();
            let uv = match a.len() {
                1 => complex(masses[a[0]]),
                2 => complex(1),
                _ => complex(0),
            };
            let slots = [
                uv.clone(),
                output[1].clone() - uv,
                output[2].clone(),
                output[0].clone(),
            ];
            for (slot, value) in leaf.slots.iter().zip(slots) {
                values[0].insert(slot.clone(), value.re);
                values[1].insert(slot.clone(), value.im);
            }
        }
        let moments: [Vec<_>; 2] = std::array::from_fn(|i| {
            expression
                .iter()
                .map(|e| e.evaluate_with_prec(&values[i], bits).unwrap())
                .collect()
        });
        // Diagnose the conditioning of the exact scalar-master combination.
        // These f64 conversions are confined to this validation experiment.
        let real_values_f64: HashMap<_, _> = values[0]
            .iter()
            .map(|(k, v)| (k.clone(), v.to_f64()))
            .collect();
        let rounded_values: HashMap<_, _> = real_values_f64
            .iter()
            .map(|(k, v)| (k.clone(), Float::with_val(bits, *v)))
            .collect();
        let rounded_only = expression[3]
            .evaluate_with_prec(&rounded_values, bits)
            .unwrap();
        let reduced_f64 = expression[3].evaluate(&real_values_f64).unwrap();
        let terms: Vec<_> = match expression[3].as_view() {
            AtomView::Add(sum) => sum.iter().map(|term| term.to_owned()).collect(),
            _ => vec![expression[3].clone()],
        };
        let absolute_sum = terms.iter().fold(Float::new(bits), |sum, term| {
            sum + term.evaluate_with_prec(&values[0], bits).unwrap().norm()
        });
        let pi = zero.pi();
        println!(
            "bits={bits} finite_term_abs_sum={absolute_sum} master_rounding_only_native={} rounded_master_f64_reduction_native={:.17e}",
            rounded_only * pi.clone() * pi.clone(),
            (reduced_f64
                - (pi.log() + zero.euler()).to_f64()
                    * (expression[0].evaluate(&real_values_f64).unwrap()
                        + expression[1].evaluate(&real_values_f64).unwrap()))
                * std::f64::consts::PI
                * std::f64::consts::PI
        );
        let conversion = pi.log() + zero.euler();
        for component in 0..2 {
            let finite = moments[component][3].clone()
                - conversion.clone()
                    * (moments[component][0].clone() + moments[component][1].clone());
            let native = finite * pi.clone() * pi.clone();
            println!(
                "bits={bits} internal_component={component} native_finite_before_i={native} uv={}",
                moments[component][0]
            );
        }
    }
}
