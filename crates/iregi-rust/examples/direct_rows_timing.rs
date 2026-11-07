//! Warm binary64 timing of direct-engine moment rows (`DirectWorkspace<f64>`,
//! conditioning gates only, as an embedding host reduces its rows) and of the
//! facade's default binary64 workspace, on the opening inputs.
//!
//! Usage: `direct_rows_timing <inputs.json> <rows> <blocks> <calls>`; one JSON
//! line per supported case and path: nanoseconds per row of every block
//! (`calls` complete `reduce_batch` calls of `rows` identical rows each) after
//! three warm-up calls, plus a checksum of the first row's array.
use std::{hint::black_box, time::Instant};

use iregi_rust::{DirectReductionPlan, KinematicsBatch, ReductionSettings, TensorReductionPlan, TensorWorkspace};
use serde_json::{Value, json};
use symbolica::domains::{
    float::{Complex, SingleFloat},
    integer::Integer,
    rational::Rational,
};

fn exact(text: &str) -> f64 {
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let q = Rational::from((n.parse::<Integer>().unwrap(), d.parse::<Integer>().unwrap()));
    0.0f64.from_rational(&q)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inputs: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let rows: usize = args[2].parse().unwrap();
    let blocks: usize = args[3].parse().unwrap();
    let calls: usize = args[4].parse().unwrap();
    for case in inputs["cases"].as_array().unwrap() {
        let request = &case["driver_request"];
        let id = request["id"].as_str().unwrap();
        let sample = &request["samples"][0];
        let mut row = Vec::new();
        for shift in sample["shifts"].as_array().unwrap() {
            for component in shift.as_array().unwrap() {
                row.push(Complex::new(exact(component.as_str().unwrap()), 0.0));
            }
        }
        let n = row.len() / 4;
        for mass in sample["masses_squared"].as_array().unwrap() {
            row.push(Complex::new(exact(mass[0].as_str().unwrap()), exact(mass[1].as_str().unwrap())));
        }
        row.push(Complex::new(exact(sample["mu"].as_str().unwrap()), 0.0));
        let mut settings = ReductionSettings::new(n, request["rank"].as_u64().unwrap() as u16);
        settings.powers = request["powers"].as_array().unwrap().iter().map(|p| p.as_u64().unwrap() as u16).collect();
        settings.dimension_shift = request["dimension_shift"].as_i64().unwrap() as i16;
        if n > 6 || !DirectReductionPlan::supports(&settings) {
            continue;
        }
        let batch = KinematicsBatch { rows, components: 1, values: row.repeat(rows) };
        let direct_plan = DirectReductionPlan::new(settings.clone()).unwrap();
        let mut direct = direct_plan.workspace(0.0f64).unwrap();
        let facade_plan = TensorReductionPlan::new(settings).unwrap();
        let mut facade = TensorWorkspace::new(&facade_plan, 0.0f64).unwrap();
        for (path, run) in [
            ("direct", &mut (|b: &KinematicsBatch<f64>| direct.reduce_batch(b).unwrap()) as &mut dyn FnMut(&KinematicsBatch<f64>) -> iregi_rust::BatchResult<f64>),
            ("facade", &mut |b: &KinematicsBatch<f64>| facade.reduce_batch(b).unwrap()),
        ] {
            let mut checksum = 0.0f64;
            for _ in 0..3 {
                let result = run(&batch);
                checksum = result.lane(0).map_or(f64::NAN, |lane| lane.iter().map(|z| z.re.abs() + z.im.abs()).sum());
            }
            let mut per_row = Vec::with_capacity(blocks);
            for _ in 0..blocks {
                let start = Instant::now();
                for _ in 0..calls {
                    black_box(run(black_box(&batch)));
                }
                per_row.push(start.elapsed().as_nanos() as f64 / (calls * rows) as f64);
            }
            println!("{}", json!({"id": id, "path": path, "n": n, "ns_per_row": per_row, "checksum": checksum}));
        }
    }
}
