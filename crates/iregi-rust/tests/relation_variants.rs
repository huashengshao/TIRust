//! Relation variants of the direct engine: affinely dependent propagators
//! with a nonzero constant relation `sum_i c_i D_i = K`. A parallelogram box
//! (`q3 = q1 + q2 - q0`, `c = (1, -1, -1, 1)`) at rank 2 is reduced through
//! triangles whose rank-2 tensors need the A0 family, which the base rank-2
//! box plan does not create: the variant must evaluate those masters itself,
//! alone and next to generic lanes, never read the previous lane's values.
use iregi_rust::{DirectReductionPlan, KinematicsBatch, LaneStatus, ReductionSettings};
use symbolica::domains::float::Complex;

type Lane = Vec<Complex<f64>>;

fn lane(shifts: &[[f64; 4]], masses: &[f64]) -> Lane {
    let mut values: Lane = shifts.iter().flatten().map(|&x| Complex::new(x, 0.0)).collect();
    values.extend(masses.iter().map(|&m| Complex::new(m, 0.0)));
    values.push(Complex::new(3.0, 0.0));
    values
}

fn reduce(n: usize, rank: u16, lanes: &[Lane]) -> Vec<Lane> {
    let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank)).unwrap();
    let mut workspace = plan.workspace(0.0f64).unwrap();
    let batch = KinematicsBatch { rows: lanes.len(), components: 1, values: lanes.concat() };
    let result = workspace.reduce_batch(&batch).unwrap();
    (0..lanes.len())
        .map(|k| {
            assert_eq!(result.statuses[k], LaneStatus::Success, "lane {k}");
            result.lane(k).unwrap().to_vec()
        })
        .collect()
}

fn minkowski(a: &[f64; 4]) -> f64 {
    a[0] * a[0] - a[1] * a[1] - a[2] * a[2] - a[3] * a[3]
}

fn error(found: &[Complex<f64>], reference: &[Complex<f64>]) -> f64 {
    let (mut e, mut a) = (0.0f64, 0.0f64);
    for (x, y) in found.iter().zip(reference) {
        e = e.max((x.re - y.re).hypot(x.im - y.im));
        a = a.max(y.re.hypot(y.im));
    }
    e / a
}

/// Parallelogram boxes (integer kinematics, two massless lines) with
/// `K = sum_i c_i (q_i^2 - m_i^2)` of 6 and -11.
const BOXES: [([[f64; 4]; 4], [f64; 4]); 2] = [
    (
        [[-1.0, -1.0, 0.0, 0.0], [0.0, -3.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, -2.0, -1.0, 0.0]],
        [0.0, 2.0, 0.0, 2.0],
    ),
    (
        [[1.0, -2.0, 0.0, 2.0], [-1.0, -3.0, 0.0, 0.0], [1.0, -3.0, -2.0, 0.0], [-1.0, -4.0, -2.0, -2.0]],
        [0.0, 3.0, 0.0, 4.0],
    ),
];

/// Generic boxes (massive lines, affinely independent shifts).
const GENERIC: [([[f64; 4]; 4], [f64; 4]); 2] = [
    (
        [[0.0, 0.0, 0.0, 0.0], [1.3, 0.2, -0.7, 0.4], [2.1, -0.5, 0.3, 1.1], [0.6, 0.9, 1.4, -0.8]],
        [1.5, 2.2, 3.1, 1.7],
    ),
    (
        [[0.0, 0.0, 0.0, 0.0], [-0.8, 1.6, 0.5, -0.3], [1.7, 0.4, -1.2, 0.9], [2.4, -1.1, 0.6, 0.2]],
        [2.8, 1.1, 4.0, 2.5],
    ),
];

#[test]
fn parallelogram_boxes_agree_with_their_denominator_identity() {
    let c = [1.0, -1.0, -1.0, 1.0];
    let generic: Vec<Lane> = GENERIC.iter().map(|(s, m)| lane(s, m)).collect();
    for (shifts, masses) in BOXES {
        let k: f64 = (0..4).map(|i| c[i] * (minkowski(&shifts[i]) - masses[i])).sum();
        assert_ne!(k, 0.0);
        let dependent = lane(&shifts, &masses);
        for rank in 0..=4u16 {
            // 1/(D0 D1 D2 D3) = sum_i (c_i / K) / prod_{j != i} D_j, the
            // triangles affinely independent.
            let mut reference = Vec::new();
            for i in 0..4 {
                let keep: Vec<usize> = (0..4).filter(|&j| j != i).collect();
                let pinch = lane(
                    &keep.iter().map(|&j| shifts[j]).collect::<Vec<_>>(),
                    &keep.iter().map(|&j| masses[j]).collect::<Vec<_>>(),
                );
                let values = reduce(3, rank, &[pinch]).remove(0);
                reference.resize(values.len(), Complex::new(0.0, 0.0));
                let w = c[i] / k;
                for (r, v) in reference.iter_mut().zip(&values) {
                    *r = Complex::new(r.re + w * v.re, r.im + w * v.im);
                }
            }
            let alone = reduce(4, rank, std::slice::from_ref(&dependent)).remove(0);
            let after = reduce(4, rank, &[generic[0].clone(), generic[1].clone(), dependent.clone()]).remove(2);
            let before = reduce(4, rank, &[dependent.clone(), generic[0].clone()]).remove(0);
            for (what, found) in [("alone", &alone), ("after generic lanes", &after), ("before a generic lane", &before)] {
                let e = error(found, &reference);
                assert!(e < 1e-12, "K = {k}, rank {rank}, {what}: E/A {e:e}");
            }
        }
    }
}
