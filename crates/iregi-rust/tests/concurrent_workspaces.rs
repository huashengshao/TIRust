//! Numerical concurrency qualification is separate from exact-plan cache tests.
//! Run the ignored test only with a Symbolica license permitting two threads:
//! `cargo test -p iregi-rust --test concurrent_workspaces -- --ignored --nocapture`.
//! Preparation, lazy branch warmup, and sequential reference calls all finish
//! before either numerical worker starts. No license bypass is used.

use std::sync::Barrier;

use iregi_masters::{NativeFloat, default_master_backend};
use iregi_rust::{
    Algorithm, BatchResult, Convention, ExecutionStats, KinematicsBatch, LaneStatus, ReductionPlan,
    ReductionSettings, Workspace,
};
use symbolica::{
    domains::{
        dual::HyperDual,
        float::{Complex, Float},
    },
    evaluate::EvaluationDomain,
    license::LicenseManager,
};

struct Job<T: NativeFloat> {
    name: &'static str,
    workspace: Workspace<T>,
    batches: Vec<KinematicsBatch<T>>,
    expected: Vec<(BatchResult<T>, ExecutionStats)>,
}

impl<T: NativeFloat> Job<T>
where
    Complex<T>: EvaluationDomain,
{
    fn warmed(
        name: &'static str,
        mut workspace: Workspace<T>,
        batches: Vec<KinematicsBatch<T>>,
        must_have_unstable: bool,
    ) -> Self {
        // Exercise every lane pattern twice, including singleton/batch dispatch
        // and lazy constant-Dual/analytic master variants, before recording.
        for _ in 0..2 {
            for batch in &batches {
                workspace.reduce_batch(batch).unwrap();
            }
        }
        let expected: Vec<_> = batches
            .iter()
            .map(|batch| {
                let output = workspace.reduce_batch(batch).unwrap();
                (output, workspace.last_execution())
            })
            .collect();
        assert!(
            expected.iter().any(|(output, _)| output
                .statuses
                .iter()
                .any(|status| *status == LaneStatus::Success)),
            "no successful fixture lane in {name}: {:?}",
            expected
                .iter()
                .map(|(output, _)| &output.statuses)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            expected.iter().any(|(output, _)| output
                .statuses
                .iter()
                .any(|status| matches!(status, LaneStatus::Unstable(_)))),
            must_have_unstable,
            "unexpected unstable-lane coverage in {name}: {:?}",
            expected
                .iter()
                .map(|(output, _)| &output.statuses)
                .collect::<Vec<_>>()
        );
        for (output, _) in &expected {
            assert!(
                output
                    .statuses
                    .iter()
                    .all(|status| matches!(status, LaneStatus::Success | LaneStatus::Unstable(_))),
                "unexpected fixture-domain failure in {name}: {:?}",
                output.statuses
            );
        }
        Self {
            name,
            workspace,
            batches,
            expected,
        }
    }

    fn check(&mut self, round: usize) {
        for offset in 0..self.batches.len() {
            let index = (offset + round) % self.batches.len();
            let actual = self.workspace.reduce_batch(&self.batches[index]).unwrap();
            let (expected, statistics) = &self.expected[index];
            assert_eq!(
                actual.statuses, expected.statuses,
                "{} round{round}",
                self.name
            );
            assert_eq!(actual.basis, expected.basis);
            assert_eq!(actual.components, expected.components);
            assert_eq!(actual.metadata(), expected.metadata());
            assert_eq!(actual.fallback_used, expected.fallback_used);
            assert_eq!(actual.analytic_used, expected.analytic_used);
            assert_eq!(self.workspace.last_execution(), *statistics);
            for lane in 0..actual.rows() {
                match (actual.lane(lane), expected.lane(lane)) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(
                            actual, expected,
                            "{} lane{lane}, all Laurent/Taylor coefficients",
                            self.name
                        );
                        for (actual, expected) in actual.iter().zip(expected) {
                            assert_eq!(actual.re.get_precision(), expected.re.get_precision());
                            assert_eq!(actual.im.get_precision(), expected.im.get_precision());
                            assert_eq!(actual.re.negative_sign(), expected.re.negative_sign());
                            assert_eq!(actual.im.negative_sign(), expected.im.negative_sign());
                        }
                    }
                    (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                    _ => panic!("inconsistent {} lane{lane}", self.name),
                }
            }
        }
    }
}

fn plan(n: usize, rank: u16) -> ReductionPlan {
    let mut settings = ReductionSettings::new(n, rank);
    settings.algorithm = Algorithm::PassarinoVeltman;
    settings.convention = Convention::OneLoop;
    ReductionPlan::new(settings).unwrap()
}

fn c(re: f64) -> Complex<f64> {
    Complex::new(re, 0.0)
}

fn permutations<T: NativeFloat>(
    rows: &[Vec<Complex<T>>],
    components: usize,
    worker: usize,
) -> Vec<KinematicsBatch<T>> {
    let count = rows.len();
    let first: Vec<_> = (0..count).map(|i| (i + worker) % count).collect();
    let second: Vec<_> = first.iter().copied().rev().collect();
    let third = vec![0]; // Known healthy singleton, exercising a distinct evaluator path.
    [first, third, second]
        .into_iter()
        .map(|order| KinematicsBatch {
            rows: order.len(),
            components,
            values: order
                .into_iter()
                .flat_map(|row| rows[row].clone())
                .collect(),
        })
        .collect()
}

fn bubble_rows(dual: bool) -> Vec<Vec<Complex<f64>>> {
    let large = 134_217_728.0;
    let rows = [
        [0., 0., 0., 0., 0., 1., 0., 0., 2., 3., 1.], // Spacelike, unequal masses.
        [0., 0., 0., 0., 1., 1., 0., 0., 2., 2., 1.], // Analytic lightlike equal-mass branch.
        [0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1.], // Exact scaleless UV/IR cancellation.
        [
            0.,
            0.,
            0.,
            0.,
            large,
            large - 2_f64.powi(-26),
            2.,
            0.,
            0.,
            0.,
            1.,
        ], // Rounded-zero invariant.
    ];
    rows.into_iter()
        .enumerate()
        .map(|(lane, row)| {
            if !dual {
                return row.into_iter().map(c).collect();
            }
            row.into_iter()
                .enumerate()
                .flat_map(|(argument, primal)| {
                    let first = match (lane, argument) {
                        (0, 8) => 0.5,
                        (1, 4) => 0.25,
                        _ => 0.0,
                    };
                    [c(primal), c(first), c(0.0)]
                })
                .collect()
        })
        .collect()
}

fn triangle_rows() -> Vec<Vec<Complex<f64>>> {
    let mut ordinary: Vec<_> = [
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 2., 3., 5., 1.,
    ]
    .into_iter()
    .map(c)
    .collect();
    let mut complex = ordinary.clone();
    for (index, width) in [(12, 0.125), (13, 0.25), (14, 0.375)] {
        complex[index].im = -width;
    }
    ordinary[4] = c(4.0); // Physical massive branch; spatial components stay noncollinear.
    vec![complex, ordinary]
}

fn box_rows() -> Vec<Vec<Complex<f64>>> {
    let ordinary: Vec<_> = [
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 2., 3., 5., 7., 1.,
    ]
    .into_iter()
    .map(c)
    .collect();
    let mut complex = ordinary.clone();
    for index in 16..20 {
        complex[index].im = -0.125;
    }
    vec![ordinary, complex]
}

struct Worker {
    ordinary: Vec<Job<f64>>,
    dual: Job<f64>,
    precise: Job<Float>,
}

impl Worker {
    fn prepare(plans: &[ReductionPlan; 4], worker: usize) -> Self {
        let ordinary = vec![
            Job::warmed(
                "bubble",
                Workspace::new(&plans[0], 0.0).unwrap(),
                permutations(&bubble_rows(false), 1, worker),
                true,
            ),
            Job::warmed(
                "triangle",
                Workspace::new(&plans[1], 0.0).unwrap(),
                permutations(&triangle_rows(), 1, worker),
                false,
            ),
            Job::warmed(
                "scalar box",
                Workspace::new(&plans[2], 0.0).unwrap(),
                permutations(&box_rows(), 1, worker),
                false,
            ),
        ];
        let dual = Job::warmed(
            "second-order bubble",
            Workspace::new_dual(
                &plans[0],
                0.0,
                HyperDual::<f64>::new(vec![vec![0], vec![1], vec![2]]),
            )
            .unwrap(),
            permutations(&bubble_rows(true), 3, worker),
            true,
        );
        let rows: Vec<Vec<_>> = [(128, 2), (80, 2), (80, 0)]
            .into_iter()
            .map(|(bits, mass)| {
                [0, 0, 0, 0, mass, 1]
                    .into_iter()
                    .map(|value| Complex::new(Float::with_val(bits, value), Float::new(bits)))
                    .collect()
            })
            .collect();
        let mut workspace = Workspace::new(&plans[3], Float::new(128)).unwrap();
        workspace.set_accuracy_precision(128).unwrap();
        let precise = Job::warmed(
            "mixed-precision tadpole",
            workspace,
            permutations(&rows, 1, worker),
            true,
        );
        Self {
            ordinary,
            dual,
            precise,
        }
    }
    fn check(&mut self, round: usize) {
        let count = self.ordinary.len();
        for offset in 0..count {
            self.ordinary[(offset + round) % count].check(round);
        }
        self.dual.check(round);
        self.precise.check(round);
    }
}

fn prepared_workers() -> [Worker; 2] {
    // The box is a scalar D0 positive control. Tensor coverage comes from
    // the rank-2 triangle and second-order Dual bubble. The original rank-1
    // box fixture rejection is retained in validation/concurrent-workspaces.
    let plans = [plan(2, 0), plan(3, 2), plan(4, 0), plan(1, 0)];
    [Worker::prepare(&plans, 0), Worker::prepare(&plans, 1)]
}

#[test]
fn warmed_sequential_controls_cover_the_concurrency_corpus() {
    eprintln!(
        "Symbolica permitted threads={} master backend={:?}; sequential control only",
        LicenseManager::max_threads(2),
        default_master_backend()
    );
    let mut workers = prepared_workers();
    for round in 0..4 {
        for worker in &mut workers {
            worker.check(round);
        }
    }
}

#[test]
#[ignore = "requires Symbolica license allowing at least two threads"]
fn independent_warmed_numeric_workspaces_match_sequential_results() {
    assert!(
        LicenseManager::max_threads(2) >= 2,
        "concurrent numeric evaluation is unqualified: installed Symbolica license permits fewer than two threads"
    );
    let workers = prepared_workers();
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let handles: Vec<_> = workers
            .into_iter()
            .enumerate()
            .map(|(index, mut worker)| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    for round in 0..12 {
                        worker.check(round + index);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("numeric worker panicked");
        }
    });
}
