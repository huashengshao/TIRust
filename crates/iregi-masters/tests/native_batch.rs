#![cfg(feature = "native-batch")]

use iregi_masters::{Complex, MasterBackend, MasterEvaluator, ScalarIntegral};
use oneloop::rust_codegen::runtime::BatchExecutionPolicy;
use std::num::NonZeroUsize;

fn point(values: &[f64], mass_range: std::ops::Range<usize>, widths: bool) -> Vec<Complex<f64>> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            Complex::new(
                value,
                if widths && mass_range.contains(&index) {
                    -0.1
                } else {
                    0.
                },
            )
        })
        .collect()
}

#[test]
fn native_mixed_physical_branches_and_block_tails_match_symbolica() {
    let cases = [
        (
            ScalarIntegral::DB0,
            vec![
                point(&[0., 2., 2., 1.], 1..3, false),
                point(&[0., 0., 2., 1.], 1..3, false),
                point(&[-3., 2., 5., 1.], 1..3, false),
                point(&[30., 2., 5., 1.], 1..3, true),
                point(&[0., 0., 0., 1.], 1..3, false),
            ],
        ),
        (
            ScalarIntegral::C0,
            vec![
                point(&[1., 2., 3., 5., 7., 11., 1.], 3..6, false),
                point(&[-1., -2., 30., 2., 3., 5., 1.], 3..6, false),
                point(&[1., 2., 3., 0.5, 0.7, 1.4, 1.], 3..6, true),
                point(&[-1., -2., 30., 2., 3., 5., 1.], 3..6, true),
                point(&[-1., -2., 30., 2., 3., 5., -1.], 3..6, false),
            ],
        ),
        (
            ScalarIntegral::D0,
            vec![
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
                point(
                    &[-1., -2., -3., -4., 12., -5., 2., 2., 2., 2., -1.],
                    6..10,
                    false,
                ),
            ],
        ),
    ];
    for (family, points) in cases {
        let input = points.concat();
        let mut reference =
            MasterEvaluator::new_with_backend(family, 0., MasterBackend::SymbolicaBatch).unwrap();
        let expected = reference.evaluate_batch(&input, points.len()).unwrap();
        for lane in 0..4 {
            assert!(
                expected.lane(lane).is_ok(),
                "{} lane{lane}: {:?}",
                family.name(),
                expected.errors()[lane]
            );
        }
        let mut native =
            MasterEvaluator::new_with_backend(family, 0., MasterBackend::RustBatch).unwrap();
        for policy in [
            BatchExecutionPolicy::Whole,
            BatchExecutionPolicy::FixedRows(NonZeroUsize::new(2).unwrap()),
            BatchExecutionPolicy::FixedRows(NonZeroUsize::new(3).unwrap()),
        ] {
            assert!(native.set_native_batch_policy(policy));
            let actual = native.evaluate_batch(&input, points.len()).unwrap();
            assert_eq!(actual.errors(), expected.errors());
            for lane in 0..points.len() {
                let Ok(actual) = actual.lane(lane) else {
                    continue;
                };
                let expected = expected.lane(lane).unwrap();
                for (actual, expected) in [
                    actual.finite,
                    actual.total_single,
                    actual.ir2,
                    actual.uv,
                    actual.ir,
                ]
                .into_iter()
                .zip([
                    expected.finite,
                    expected.total_single,
                    expected.ir2,
                    expected.uv,
                    expected.ir,
                ]) {
                    let error = (actual[0].re - expected[0].re).abs()
                        + (actual[0].im - expected[0].im).abs();
                    assert!(
                        error < 2e-10 * (1. + expected[0].re.abs() + expected[0].im.abs()),
                        "{} lane{lane}, policy{policy:?}: error={error}",
                        family.name()
                    );
                }
            }
        }
    }
}
