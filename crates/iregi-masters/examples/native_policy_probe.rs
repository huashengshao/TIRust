//! Controlled comparison of existing native batch policies.
//! Run an optimized build with the `native-batch` feature, a pinned CPU and an
//! exclusive runtime lease. Output is raw CSV; setup and parity checks are not
//! timed. A separate fresh evaluator for each policy avoids arena high-water
//! contamination. Override IREGI_PROBE_BLOCKS or IREGI_PROBE_MS for smoke runs.

use iregi_masters::{Complex, MasterBackend, MasterEvaluator, ScalarIntegral};
use oneloop::rust_codegen::runtime::BatchExecutionPolicy;
use std::{
    hint::black_box,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

fn point(values: &[f64], masses: std::ops::Range<usize>, complex: bool) -> Vec<Complex<f64>> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            Complex::new(
                value,
                if complex && masses.contains(&i) {
                    -0.1
                } else {
                    0.
                },
            )
        })
        .collect()
}

fn cases(family: ScalarIntegral) -> Vec<Vec<Complex<f64>>> {
    match family {
        ScalarIntegral::C0 => vec![
            point(&[1., 2., 3., 5., 7., 11., 1.], 3..6, false),
            point(&[-1., -2., 30., 2., 3., 5., 1.], 3..6, false),
            point(&[1., 2., 3., 0.5, 0.7, 1.4, 1.], 3..6, true),
            point(&[-1., -2., 30., 2., 3., 5., 1.], 3..6, true),
        ],
        ScalarIntegral::D0 => vec![
            point(
                &[1., 1., 1., 1., 8., -2., 20., 30., 40., 50., 1.],
                6..10,
                false,
            ),
            point(
                &[1., 1., 1., 1., 40., -8., 2., 3., 4., 5., 1.],
                6..10,
                false,
            ),
            point(
                &[-1., -2., -3., -4., 12., -5., 2., 3., 4., 5., 1.],
                6..10,
                true,
            ),
            point(
                &[-1., -2., -3., -4., 12., -5., 2., 2., 2., 2., 1.],
                6..10,
                true,
            ),
        ],
        _ => unreachable!(),
    }
}

fn main() {
    let blocks = std::env::var("IREGI_PROBE_BLOCKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(21usize);
    let millis = std::env::var("IREGI_PROBE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30u64);
    println!("family,ordering,policy,block,iterations,elapsed_ns,ns_per_batch,rows");
    for family in [ScalarIntegral::C0, ScalarIntegral::D0] {
        let cases = cases(family);
        for ordering in ["homogeneous", "interleaved", "grouped"] {
            let input: Vec<_> = (0..32)
                .flat_map(|i| {
                    let index = match ordering {
                        "homogeneous" => 0,
                        "interleaved" => i % 4,
                        _ => i / 8,
                    };
                    cases[index].iter().copied()
                })
                .collect();
            let policies = [None, Some(4), Some(8), Some(16), Some(32)];
            let mut evaluators: Vec<_> = policies
                .iter()
                .map(|rows| {
                    let mut evaluator =
                        MasterEvaluator::new_with_backend(family, 0., MasterBackend::RustBatch)
                            .unwrap();
                    let policy = rows.map_or(BatchExecutionPolicy::Whole, |rows| {
                        BatchExecutionPolicy::FixedRows(NonZeroUsize::new(rows).unwrap())
                    });
                    assert!(evaluator.set_native_batch_policy(policy));
                    evaluator
                })
                .collect();
            let expected = evaluators[0].evaluate_batch(&input, 32).unwrap();
            assert!(expected.errors().iter().all(Option::is_none));
            for evaluator in &mut evaluators {
                let actual = evaluator.evaluate_batch(&input, 32).unwrap();
                assert_eq!(actual.errors(), expected.errors());
                for lane in 0..32 {
                    let a = actual.lane(lane).unwrap();
                    let b = expected.lane(lane).unwrap();
                    for (a, b) in [a.finite, a.total_single, a.ir2, a.uv, a.ir]
                        .into_iter()
                        .zip([b.finite, b.total_single, b.ir2, b.uv, b.ir])
                    {
                        assert_eq!(a, b, "{} {ordering} lane{lane}", family.name());
                    }
                }
                for _ in 0..5 {
                    black_box(evaluator.evaluate_batch(black_box(&input), 32).unwrap());
                }
            }
            for block in 0..blocks {
                for offset in 0..policies.len() {
                    let index = (block + offset) % policies.len();
                    let start = Instant::now();
                    let mut iterations = 0u64;
                    loop {
                        black_box(
                            evaluators[index]
                                .evaluate_batch(black_box(&input), 32)
                                .unwrap(),
                        );
                        iterations += 1;
                        if start.elapsed() >= Duration::from_millis(millis) {
                            break;
                        }
                    }
                    let ns = start.elapsed().as_nanos();
                    let policy = policies[index].map_or("whole".to_string(), |n| n.to_string());
                    println!(
                        "{},{ordering},{policy},{block},{iterations},{ns},{:.6},32",
                        family.name(),
                        ns as f64 / iterations as f64
                    );
                }
            }
        }
    }
}
