//! Accuracy and timing check of the direct engine on a float fixture.
//!
//! Usage: direct_check FIXTURE.json [ID-SUBSTRING] [--batch N] [--profile]
//! The fixture is a list of `{id, n, rank, powers, h, shifts, masses, mu,
//! basis, reference}` records, with `reference` a flat `[re, im]` list in
//! `4 * moment + slot` order (IREGI-native convention, raw components).
//! Accuracy is the dimensionally rescaled whole-array error `E/A`.
use std::{error::Error, hint::black_box, time::Instant};

use iregi_masters::MasterExecutionPolicy;
use iregi_rust::{DirectMasters, DirectReductionPlan, KinematicsBatch, ReductionSettings};
use serde::Deserialize;
use symbolica::domains::float::Complex;

type Fallible<T> = Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
struct Case {
    id: String,
    n: usize,
    rank: u16,
    powers: Vec<u16>,
    h: i16,
    shifts: Vec<[f64; 4]>,
    masses: Vec<[f64; 2]>,
    mu: f64,
    basis: Vec<[u16; 4]>,
    reference: Vec<[f64; 2]>,
}

fn main() -> Fallible<()> {
    let args: Vec<String> = std::env::args().collect();
    let fixture: Vec<Case> = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let filter = args.get(2).filter(|a| !a.starts_with("--")).cloned();
    let batch_size = args
        .iter()
        .position(|a| a == "--batch")
        .map(|i| args[i + 1].parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let profile = args.iter().any(|a| a == "--profile");
    let masters = if args.iter().any(|a| a == "--policy-masters") {
        DirectMasters::Policy(if cfg!(feature = "native-f64-abc") {
            MasterExecutionPolicy::NativeF64Abc
        } else {
            MasterExecutionPolicy::Symbolica
        })
    } else {
        DirectMasters::Native
    };
    println!(
        "{:<22} {:>9} {:>9} {:>12} {:>12} {:>11} {:>10}",
        "id", "plan ms", "ws ms", "E/A", "max|ref|", "us/call", "us/point"
    );
    for case in &fixture {
        if let Some(filter) = &filter {
            if !case.id.contains(filter.as_str()) {
                continue;
            }
        }
        let mut settings = ReductionSettings::new(case.n, case.rank);
        settings.powers = case.powers.clone();
        settings.dimension_shift = case.h;
        if !DirectReductionPlan::supports(&settings) {
            println!("{:<22} unsupported by the direct engine", case.id);
            continue;
        }
        let started = Instant::now();
        let plan = DirectReductionPlan::new(settings)?;
        let plan_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let mut workspace = plan.workspace_with_masters(0.0f64, masters)?;
        let workspace_ms = started.elapsed().as_secs_f64() * 1e3;
        assert_eq!(plan.basis().len(), case.basis.len());
        for (a, b) in plan.basis().iter().zip(&case.basis) {
            assert_eq!(&a.0, b, "basis order");
        }
        let mut values = Vec::new();
        for lane in 0..batch_size {
            // Lane 0 is the reference point; others are mild perturbations.
            let scale = 1.0 + lane as f64 / 256.0;
            for shift in &case.shifts {
                for &c in shift {
                    values.push(Complex::new(c * scale, 0.0));
                }
            }
            for mass in &case.masses {
                values.push(Complex::new(mass[0] + lane as f64 / 64.0, mass[1]));
            }
            values.push(Complex::new(case.mu, 0.0));
        }
        let batch = KinematicsBatch {
            rows: batch_size,
            components: 1,
            values,
        };
        let result = workspace.reduce_batch(&batch)?;
        let lane = match result.lane(0) {
            Ok(lane) => lane,
            Err(status) => {
                println!("{:<22} lane 0 failed: {status:?}", case.id);
                continue;
            }
        };
        let base = 4 + 2 * i32::from(case.h) - 2 * case.powers.iter().map(|&p| i32::from(p)).sum::<i32>();
        let (mut error, mut amplitude) = (0.0f64, 0.0f64);
        let mut worst = (0usize, 0usize);
        for (m, e) in case.basis.iter().enumerate() {
            let degree: i32 = e.iter().map(|&x| i32::from(x)).sum();
            let weight = case.mu.powi(degree + base);
            for slot in 0..4 {
                let value = lane[4 * m + slot];
                let reference = case.reference[4 * m + slot];
                let difference =
                    (value.re - reference[0]).hypot(value.im - reference[1]) / weight;
                if difference > error {
                    error = difference;
                    worst = (m, slot);
                }
                amplitude = amplitude.max(reference[0].hypot(reference[1]) / weight);
            }
        }
        // Warm timing: blocks of about 20 ms, report the median block mean.
        let mut per_call = Vec::new();
        let mut iterations = 1usize;
        loop {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(workspace.reduce_batch(black_box(&batch))?);
            }
            let elapsed = started.elapsed().as_secs_f64();
            if elapsed > 0.02 || iterations > 1 << 20 {
                break;
            }
            iterations = (iterations * 2).max((0.02 / elapsed.max(1e-9) * iterations as f64) as usize);
        }
        for _ in 0..7 {
            let started = Instant::now();
            for _ in 0..iterations {
                black_box(workspace.reduce_batch(black_box(&batch))?);
            }
            per_call.push(started.elapsed().as_secs_f64() / iterations as f64);
        }
        per_call.sort_by(f64::total_cmp);
        let median = per_call[per_call.len() / 2] * 1e6;
        let failed = result.statuses.iter().filter(|s| **s != iregi_rust::LaneStatus::Success).count();
        println!(
            "{:<22} {:>9.2} {:>9.2} {:>12.3e} {:>12.3e} {:>11.3} {:>10.3}{}",
            case.id,
            plan_ms,
            workspace_ms,
            error / amplitude,
            amplitude,
            median,
            median / batch_size as f64,
            if failed > 0 { format!("  ({failed} failed lanes)") } else { String::new() }
        );
        if error / amplitude > 1e-10 {
            println!("    worst moment {:?} slot {}", case.basis[worst.0], worst.1);
        }
        if profile {
            workspace.set_profiling(true);
            workspace.reduce_batch(&batch)?;
            let p = workspace.last_profile();
            println!(
                "    profile us: masters {:.2} geometry {:.2} recurrence {:.2} gather {:.2} output {:.2}",
                p.masters.as_secs_f64() * 1e6,
                p.geometry.as_secs_f64() * 1e6,
                p.recurrence.as_secs_f64() * 1e6,
                p.gather.as_secs_f64() * 1e6,
                p.output.as_secs_f64() * 1e6
            );
            workspace.set_profiling(false);
        }
    }
    Ok(())
}
