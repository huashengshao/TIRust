//! Timing of the seven-point paths on one heptagon point (development aid).
//!
//! usage: n7_probe POINTS_JSON INDEX RANK
use std::time::Instant;

use iregi_rust::{KinematicsBatch, ReductionSettings, TensorEngineExecution, TensorReductionPlan, TensorWorkspace};
use symbolica::domains::float::Complex;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let points: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let index: usize = args[2].parse().unwrap();
    let rank: u16 = args[3].parse().unwrap();
    let integral = points["integrals"].as_array().unwrap().iter().find(|i| i["id"] == "heptagon-r7").unwrap();
    let mut values = Vec::new();
    for shift in integral["points"][index].as_array().unwrap() {
        values.extend(shift.as_array().unwrap().iter().map(|x| Complex::new(x.as_f64().unwrap(), 0.0)));
    }
    values.extend((0..7).map(|_| Complex::new(173.0f64 * 173.0, 0.0)));
    values.push(Complex::new(1000.0, 0.0));
    let batch = KinematicsBatch { rows: 1, components: 1, values };
    let started = Instant::now();
    let plan = TensorReductionPlan::new(ReductionSettings::new(7, rank)).unwrap();
    println!("facade plan {:?}", started.elapsed());
    let started = Instant::now();
    let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
    println!("facade workspace {:?}", started.elapsed());
    for repeat in 0..2 {
        let started = Instant::now();
        let result = workspace.reduce_batch(&batch).unwrap();
        println!("facade call {repeat}: {:?} {:?}", started.elapsed(), result.statuses[0]);
        if let TensorEngineExecution::Direct(stats) = workspace.engine_execution() {
            println!("  {stats:?}");
        }
        println!("  diagnostics {:?}", workspace.last_diagnostics());
    }
    if std::env::var("WRAPPER").is_ok() {
        let started = Instant::now();
        let plan = TensorReductionPlan::new_without_direct(ReductionSettings::new(7, rank)).unwrap();
        let mut workspace = TensorWorkspace::new(&plan, 0.0f64).unwrap();
        println!("wrapper plan+workspace {:?}", started.elapsed());
        let started = Instant::now();
        let result = workspace.reduce_batch(&batch).unwrap();
        println!("wrapper call {:?} {:?}", started.elapsed(), result.statuses[0]);
    }
}
