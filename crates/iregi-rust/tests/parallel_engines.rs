//! Lane-independent direct workspaces (`DirectWorkspace::set_lane_independent`)
//! as one engine per thread: every lane, its status, its conditioning
//! estimate, its small-Gram flag and its rotation copy are bit for bit what
//! one workspace gives on one thread in one batch, and what a new workspace
//! gives the lane alone, whatever the number of threads, the split of the
//! lanes between them, the batches and the calls the workspaces reduced
//! before; a tensor store shared by every thread changes nothing either.
//! The lanes include (near-)degenerate ones, whose result depends on the
//! previous lane in a workspace that is not lane-independent: affinely
//! dependent boxes, and copies of them perturbed below the relation's
//! tolerance, also inside pentagons and hexagons.
use std::thread;

use iregi_masters::NativeFloat;
use iregi_rust::{
    BatchResult, DirectReductionPlan, DirectWorkspace, KinematicsBatch, ReductionSettings,
    TensorStore, direct::EXPANSION_TOLERANCE,
};
use symbolica::{
    domains::float::{Complex, DoubleFloat, Float},
    evaluate::EvaluationDomain,
};

mod common;
use common::hexagon::{Lcg, POINTS};

type Row = Vec<Complex<f64>>;

fn row(shifts: &[[f64; 4]], masses: &[f64], scale: f64) -> Row {
    let mut values: Row = shifts
        .iter()
        .flatten()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    values.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
    values.push(Complex::new(scale, 0.0));
    values
}

/// The `n`-point sub-topologies of the physical hexagon points (a massive
/// top loop and a massless one: lightlike bubbles, gate-refused lanes),
/// every `step`-th.
fn physical(n: usize, step: usize) -> Vec<Row> {
    let mut rows = Vec::new();
    for mass2 in [29929.0, 0.0] {
        for members in (1u32..1 << 6).filter(|m| m.count_ones() as usize == n) {
            let members: Vec<usize> = (0..6).filter(|i| members & (1 << i) != 0).collect();
            for point in &POINTS {
                let shifts: Vec<[f64; 4]> = members.iter().map(|&i| point[i]).collect();
                rows.push(row(&shifts, &vec![mass2; n], 1000.0));
            }
        }
    }
    rows.into_iter().step_by(step).collect()
}

/// Generic massive lanes.
fn generic(n: usize, count: usize) -> Vec<Row> {
    let mut rng = Lcg(31 + n as u64);
    (0..count)
        .map(|_| {
            let shifts: Vec<[f64; 4]> = (0..n)
                .map(|i| {
                    if i == 0 {
                        [0.0; 4]
                    } else {
                        [0.0; 4].map(|_| 600.0 * rng.next() - 300.0)
                    }
                })
                .collect();
            let masses: Vec<f64> = (0..n).map(|_| 1000.0 + 40000.0 * rng.next()).collect();
            row(&shifts, &masses, 500.0)
        })
        .collect()
}

/// Parallelogram boxes (`q3 = q1 + q2 - q0`, an exact affine relation),
/// each followed by copies whose last shift is moved by `eps` (the relation
/// then holds only to its tolerance, or not at all), completed to `n`
/// propagators by generic ones.
fn degenerate(n: usize) -> Vec<Row> {
    const BOXES: [([[f64; 4]; 4], [f64; 4]); 2] = [
        (
            [
                [-1.0, -1.0, 0.0, 0.0],
                [0.0, -3.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -2.0, -1.0, 0.0],
            ],
            [0.0, 2.0, 0.0, 2.0],
        ),
        (
            [
                [1.0, -2.0, 0.0, 2.0],
                [-1.0, -3.0, 0.0, 0.0],
                [1.0, -3.0, -2.0, 0.0],
                [-1.0, -4.0, -2.0, -2.0],
            ],
            [0.0, 3.0, 0.0, 4.0],
        ),
    ];
    const EXTRA: [([f64; 4], f64); 2] = [
        ([0.3, 0.5, -0.35, -0.15], 0.6),
        ([1.1, -0.25, 0.4, 0.6], 1.7),
    ];
    let mut rows = Vec::new();
    for (shifts, masses) in BOXES {
        for eps in [0.0, 1e-15, 1e-13, 3e-13, 1e-12, 1e-9] {
            let mut s: Vec<[f64; 4]> = shifts.to_vec();
            s[3][2] += eps;
            let mut m: Vec<f64> = masses.to_vec();
            for (shift, mass) in EXTRA.iter().take(n - 4) {
                s.push(*shift);
                m.push(*mass);
            }
            rows.push(row(&s, &m, 3.0));
        }
    }
    rows
}

/// Physical, generic and degenerate lanes, interleaved (the degenerate
/// ones keep their order: an exact box before its perturbed copies).
fn lanes(n: usize, step: usize, count: usize) -> Vec<Row> {
    let (a, b) = (physical(n, step), generic(n, count));
    let mut all = Vec::new();
    let (mut a, mut b) = (a.into_iter(), b.into_iter());
    loop {
        match (a.next(), b.next()) {
            (None, None) => break,
            (x, y) => all.extend(x.into_iter().chain(y)),
        }
    }
    let degenerate = degenerate(n);
    let every = all.len() / degenerate.len() + 1;
    let mut out = Vec::new();
    let mut rest = degenerate.into_iter();
    for (k, lane) in all.into_iter().enumerate() {
        if k % every == 0 {
            out.extend(rest.next());
        }
        out.push(lane);
    }
    out.extend(rest);
    out
}

fn batch<T: NativeFloat>(rows: &[&Row], lift: &impl Fn(f64) -> T) -> KinematicsBatch<T> {
    KinematicsBatch {
        rows: rows.len(),
        components: 1,
        values: rows
            .iter()
            .flat_map(|row| row.iter().map(|z| Complex::new(lift(z.re), lift(z.im))))
            .collect(),
    }
}

/// One lane exactly: its array (`{:?}` of the carrier is exact), status,
/// conditioning estimate and small-Gram flag; and its rotation copy's array
/// and status.
type Lane = (String, String, Option<u64>, bool, Option<(String, String)>);

fn exact<T: NativeFloat>(result: &BatchResult<T>, lane: usize) -> (String, String) {
    let values = result
        .lane(lane)
        .map_or_else(|_| "refused".into(), |values| format!("{values:?}"));
    (values, format!("{:?}", result.statuses[lane]))
}

/// A workspace setting: its name and how to configure a new workspace.
type Setting<T> = (&'static str, fn(&mut DirectWorkspace<T>, &T));

struct Case<'a, T: NativeFloat> {
    plan: &'a DirectReductionPlan,
    prototype: T,
    setting: Setting<T>,
    rotation: Option<usize>,
}

impl<T: NativeFloat + Send> Case<'_, T>
where
    Complex<T>: EvaluationDomain,
{
    fn workspace(&self, independent: bool) -> DirectWorkspace<T> {
        let mut workspace = self.plan.workspace(self.prototype.clone()).unwrap();
        (self.setting.1)(&mut workspace, &self.prototype);
        workspace.set_lane_independent(independent);
        workspace
    }

    /// The lanes `part` of `rows` in one call of `workspace`.
    fn call(
        &self,
        workspace: &mut DirectWorkspace<T>,
        rows: &[&Row],
        lift: &impl Fn(f64) -> T,
    ) -> Vec<Lane> {
        let batch = batch(rows, lift);
        let (primary, copy) = match self.rotation {
            Some(rotation) => {
                let (primary, copy) = workspace
                    .reduce_batch_with_rotation(&batch, rotation)
                    .unwrap();
                (primary, Some(copy))
            }
            None => (workspace.reduce_batch(&batch).unwrap(), None),
        };
        (0..rows.len())
            .map(|lane| {
                let (values, status) = exact(&primary, lane);
                (
                    values,
                    status,
                    workspace.last_conditions()[lane].map(f64::to_bits),
                    workspace.last_gram_lanes()[lane],
                    copy.as_ref().map(|copy| exact(copy, lane)),
                )
            })
            .collect()
    }

    /// Every lane through `threads` threads, each with its own workspace and
    /// its share of the lanes (contiguous or round robin) in calls of at most
    /// `chunk` lanes, `repeats` times with the same workspaces; optionally
    /// one store for every thread. Returns every repeat's lanes in order and
    /// its scalar-master requests (`last_master_rows`, summed over calls).
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        rows: &[Row],
        lift: &(impl Fn(f64) -> T + Sync),
        independent: bool,
        threads: usize,
        round_robin: bool,
        chunk: usize,
        repeats: usize,
        store: Option<&TensorStore>,
    ) -> Vec<(Vec<Lane>, u64)>
    where
        Self: Sync,
    {
        let owner = |lane: usize| {
            if round_robin {
                lane % threads
            } else {
                lane * threads / rows.len()
            }
        };
        type Repeat = (Vec<(usize, Lane)>, u64);
        let parts: Vec<Vec<Repeat>> = thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    scope.spawn(move || {
                        let mine: Vec<usize> =
                            (0..rows.len()).filter(|&lane| owner(lane) == t).collect();
                        let mut workspace = self.workspace(independent);
                        workspace.set_tensor_store(store);
                        (0..repeats)
                            .map(|_| {
                                let (mut done, mut masters) = (Vec::new(), 0);
                                for part in mine.chunks(chunk.max(1)) {
                                    let rows: Vec<&Row> =
                                        part.iter().map(|&lane| &rows[lane]).collect();
                                    done.extend(part.iter().copied().zip(self.call(
                                        &mut workspace,
                                        &rows,
                                        lift,
                                    )));
                                    masters += workspace.last_master_rows() as u64;
                                }
                                (done, masters)
                            })
                            .collect()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        });
        (0..repeats)
            .map(|repeat| {
                let mut out: Vec<Option<Lane>> = vec![None; rows.len()];
                let mut masters = 0;
                for part in &parts {
                    for (lane, value) in &part[repeat].0 {
                        out[*lane] = Some(value.clone());
                    }
                    masters += part[repeat].1;
                }
                (out.into_iter().map(Option::unwrap).collect(), masters)
            })
            .collect()
    }

    fn check(
        &self,
        name: &str,
        rows: &[Row],
        lift: &(impl Fn(f64) -> T + Sync),
        threads: &[usize],
        chunks: &[usize],
    ) where
        Self: Sync,
    {
        // One lane-independent workspace, one thread, one call.
        let reference = self
            .run(rows, lift, true, 1, false, rows.len(), 1, None)
            .remove(0);
        let stored = |store: &TensorStore| {
            let s = store.statistics();
            [
                s.entries,
                s.subsets_stored,
                s.masters_stored,
                s.moments_stored,
                s.dd_stored,
            ]
        };
        let alone_store = TensorStore::new();
        let with_store = self
            .run(
                rows,
                lift,
                true,
                1,
                false,
                rows.len(),
                1,
                Some(&alone_store),
            )
            .remove(0);
        assert_eq!(with_store, reference, "{name}: one call with a store");
        let reference_stored = stored(&alone_store);
        let served = reference
            .0
            .iter()
            .filter(|lane| lane.1 == "Success")
            .count();
        let flagged = reference.0.iter().filter(|lane| lane.3).count();
        eprintln!(
            "{name}: {} lanes, {served} served, {flagged} with small-Gram methods, {} master requests, stored {reference_stored:?}",
            reference.0.len(),
            reference.1
        );
        assert!(served > 0, "{name}: no lane served");
        // A lane in a new workspace alone.
        for (k, row) in rows.iter().enumerate() {
            let alone = self.call(&mut self.workspace(true), &[row], lift).remove(0);
            assert_eq!(alone, reference.0[k], "{name}: lane {k} alone");
        }
        for &threads in threads {
            for round_robin in [false, true] {
                for &chunk in chunks {
                    for (repeat, lanes) in self
                        .run(rows, lift, true, threads, round_robin, chunk, 2, None)
                        .into_iter()
                        .enumerate()
                    {
                        assert_eq!(
                            lanes, reference,
                            "{name}: {threads} threads, round robin {round_robin}, calls of {chunk}, repeat {repeat}"
                        );
                    }
                }
            }
            let store = TensorStore::new();
            for lanes in self.run(rows, lift, true, threads, true, 3, 2, Some(&store)) {
                assert_eq!(
                    lanes, reference,
                    "{name}: {threads} threads sharing one store"
                );
            }
            // The distinct results stored do not depend on the threads either
            // (the reused / computed lookup counts do).
            assert_eq!(
                stored(&store),
                reference_stored,
                "{name}: {threads} threads, stored entries"
            );
        }
    }
}

fn gates<T: NativeFloat>(_: &mut DirectWorkspace<T>, _: &T) {}

/// The last resort of pyAmpliCol's checked and unchecked modes.
fn gate_zero<T: NativeFloat>(workspace: &mut DirectWorkspace<T>, prototype: &T)
where
    Complex<T>: EvaluationDomain,
{
    workspace.set_stability_threshold(prototype.zero());
}

/// pyAmpliCol's binary64 method choice.
fn method_choice(workspace: &mut DirectWorkspace<f64>, _: &f64) {
    workspace.set_stability_threshold(1e-15);
    workspace.enable_estimated_method_choice(12, 1e-15);
    workspace.set_alternative_first(Some(EXPANSION_TOLERANCE));
}

fn partial_rescue(workspace: &mut DirectWorkspace<f64>, _: &f64) {
    workspace.enable_partial_rescue(1e-20);
}

#[test]
fn binary64_lane_independent_engines_are_bitwise_independent_of_the_threads() {
    for (n, rank, step) in [(4usize, 4u16, 3usize), (5, 5, 2), (6, 6, 1)] {
        let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank)).unwrap();
        let rows = lanes(n, step, 12);
        let settings: [Setting<f64>; 4] = [
            ("gates", gates),
            ("gate 0", gate_zero),
            ("method choice", method_choice),
            ("partial rescue", partial_rescue),
        ];
        for setting in settings {
            for rotation in [None, Some(1)] {
                let case = Case {
                    plan: &plan,
                    prototype: 0.0f64,
                    setting,
                    rotation,
                };
                let name = format!("N{n} R{rank} {} rotation {rotation:?}", setting.0);
                case.check(&name, &rows, &|x| x, &[1, 2, 8], &[rows.len(), 3, 1]);
            }
        }
    }
}

#[test]
fn double_double_lane_independent_engines_are_bitwise_independent_of_the_threads() {
    let plan = DirectReductionPlan::new(ReductionSettings::new(5, 5)).unwrap();
    let rows = lanes(5, 4, 4);
    let settings: [Setting<DoubleFloat>; 2] = [("gates", gates), ("gate 0", gate_zero)];
    for setting in settings {
        for rotation in [None, Some(1)] {
            let case = Case {
                plan: &plan,
                prototype: DoubleFloat::from(0.0),
                setting,
                rotation,
            };
            let name = format!("N5 R5 double-double {} rotation {rotation:?}", setting.0);
            case.check(
                &name,
                &rows,
                &DoubleFloat::from,
                &[1, 2, 8],
                &[rows.len(), 3],
            );
        }
    }
}

#[test]
fn mpfr_lane_independent_engines_are_bitwise_independent_of_the_threads() {
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, 3)).unwrap();
    // The degenerate lanes and every twelfth physical and generic box: MPFR
    // is slow.
    let mut rows = degenerate(4);
    rows.extend(physical(4, 12));
    rows.extend(generic(4, 2));
    let settings: [Setting<Float>; 2] = [("gates", gates), ("gate 0", gate_zero)];
    for setting in settings {
        let case = Case {
            plan: &plan,
            prototype: Float::new(128),
            setting,
            rotation: Some(1),
        };
        let name = format!("N4 R3 MPFR 128 bits {}", setting.0);
        case.check(
            &name,
            &rows,
            &|x| Float::with_val(128, x),
            &[1, 2, 8],
            &[rows.len(), 3],
        );
    }
}

/// The lanes above exercise what the setting removes: without it some
/// degenerate lane differs between one call of all lanes and alone.
#[test]
fn without_the_setting_degenerate_lanes_depend_on_their_predecessors() {
    let plan = DirectReductionPlan::new(ReductionSettings::new(4, 2)).unwrap();
    let rows = degenerate(4);
    let settings: [Setting<f64>; 2] = [("gate 0", gate_zero), ("method choice", method_choice)];
    for setting in settings {
        let case = Case {
            plan: &plan,
            prototype: 0.0f64,
            setting,
            rotation: None,
        };
        let all: Vec<&Row> = rows.iter().collect();
        let together = case.call(&mut case.workspace(false), &all, &|x| x);
        let differs = (0..rows.len())
            .filter(|&k| {
                case.call(&mut case.workspace(false), &[&rows[k]], &|x| x)
                    .remove(0)
                    != together[k]
            })
            .count();
        assert!(
            differs > 0,
            "{}: no degenerate lane depends on its predecessor",
            setting.0
        );
        let together = case.call(&mut case.workspace(true), &all, &|x| x);
        for (k, row) in rows.iter().enumerate() {
            assert_eq!(
                case.call(&mut case.workspace(true), &[row], &|x| x)
                    .remove(0),
                together[k],
                "{}: lane {k}",
                setting.0
            );
        }
    }
}
