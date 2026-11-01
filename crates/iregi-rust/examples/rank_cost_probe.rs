//! Cost of the binary64 direct engine by point count and rank, split into
//! geometry, masters, recurrence, gather and output (development aid for the
//! expansion cost). Kinematics: the first N shifts of the opening N7R7 lane.
use std::time::{Duration, Instant};

use iregi_rust::{DirectReductionPlan, KinematicsBatch, ReductionSettings};
use symbolica::domains::float::Complex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shifts = [
        [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.25, 0.375, 0.125, 0.625],
    ];
    let masses = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0];
    for n in [2usize, 3, 4] {
        for rank in [4u16, 8, 12, 16] {
            let mut values = Vec::new();
            for shift in &shifts[..n] {
                values.extend(shift.iter().map(|&x| Complex::new(x, 0.0)));
            }
            values.extend(masses[..n].iter().map(|&m| Complex::new(m, 0.0)));
            values.push(Complex::new(3.0, 0.0));
            let batch = KinematicsBatch { rows: 1, components: 1, values };
            let plan = DirectReductionPlan::new(ReductionSettings::new(n, rank))?;
            let mut workspace = plan.workspace(0.0f64)?;
            workspace.set_stability_threshold(0.0);
            workspace.reduce_batch(&batch)?;
            workspace.set_profiling(true);
            let calls = 50;
            let start = Instant::now();
            let mut total = [Duration::ZERO; 5];
            for _ in 0..calls {
                std::hint::black_box(workspace.reduce_batch(&batch)?);
                let p = workspace.last_profile();
                for (t, d) in total.iter_mut().zip([p.geometry, p.masters, p.recurrence, p.gather, p.output]) {
                    *t += d;
                }
            }
            let per = |d: Duration| d.as_secs_f64() * 1e6 / calls as f64;
            println!("N{n} R{rank:2}: {:8.1} us/call (geometry {:.1}, masters {:.1}, recurrence {:.1}, gather {:.1}, output {:.1})",
                per(start.elapsed()), per(total[0]), per(total[1]), per(total[2]), per(total[3]), per(total[4]));
        }
    }
    // The opening N5R5 lane with the a priori method choice: total against
    // the parent's own buckets (the rest is in the auxiliary executors).
    let mut values = Vec::new();
    for shift in &shifts[..5] {
        values.extend(shift.iter().map(|&x| Complex::new(x, 0.0)));
    }
    values.extend(masses[..5].iter().map(|&m| Complex::new(m, 0.0)));
    values.push(Complex::new(3.0, 0.0));
    let batch = KinematicsBatch { rows: 1, components: 1, values };
    for (label, estimated) in [("plain", false), ("estimated", true)] {
        let plan = DirectReductionPlan::new(ReductionSettings::new(5, 5))?;
        let mut workspace = plan.workspace(0.0f64)?;
        workspace.set_stability_threshold(1e-15);
        if estimated {
            workspace.enable_estimated_method_choice(12, 1e-15);
        }
        workspace.reduce_batch(&batch)?;
        workspace.set_profiling(true);
        let calls = 10;
        let start = Instant::now();
        let mut total = [Duration::ZERO; 5];
        for _ in 0..calls {
            std::hint::black_box(workspace.reduce_batch(&batch)?);
            let p = workspace.last_profile();
            for (t, d) in total.iter_mut().zip([p.geometry, p.masters, p.recurrence, p.gather, p.output]) {
                *t += d;
            }
        }
        let per = |d: Duration| d.as_secs_f64() * 1e6 / calls as f64;
        println!("N5R5 {label}: {:8.1} us/call (geometry {:.1}, masters {:.1}, recurrence {:.1}, gather {:.1}, output {:.1}), gram {:?}",
            per(start.elapsed()), per(total[0]), per(total[1]), per(total[2]), per(total[3]), per(total[4]), workspace.last_gram_lanes());
    }
    Ok(())
}
