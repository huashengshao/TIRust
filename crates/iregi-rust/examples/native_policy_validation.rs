//! Untimed Cargo-linked low-point validation example. No private capture cfg.
use std::{env, fs};
use iregi_masters::{MasterBackend, MasterEvaluator, MasterExecutionPolicy, NativeFloat, ScalarIntegral};
use iregi_rust::{KinematicsBatch, LaneStatus, ReductionPlan, ReductionSettings, Workspace};
use serde::Deserialize;
use serde_json::{Value, json};
use symbolica::domains::float::{Complex, Float, FloatLike};

#[derive(Deserialize)]
struct Sample { shifts: Vec<[f64; 4]>, masses: Vec<[f64; 2]>, mu: f64 }
#[derive(Deserialize)]
struct Input { id: String, samples: Vec<Sample> }

fn represented(z: &Complex<f64>) -> Value {
    json!([&z.re, &z.im].into_iter().map(|x| json!({
        "rational":x.exact_rational().unwrap().to_string(), "precision":x.get_precision(),
        "negative_sign":x.negative_sign(), "finite":x.is_finite()
    })).collect::<Vec<_>>())
}

fn observe(plan: &ReductionPlan, batch: &KinematicsBatch<f64>, policy: MasterExecutionPolicy) -> Value {
    let mut workspace = Workspace::new_with_master_policy(plan, 0., policy).unwrap();
    assert_eq!(workspace.master_execution_policy(), policy);
    workspace.set_profiling(false);
    let input_before: Vec<_> = batch.values.iter().map(represented).collect();
    let result = workspace.reduce_batch(batch).unwrap();
    assert_eq!(result.basis.as_slice(), plan.basis());
    assert_eq!(result.components, 1);
    assert_eq!(result.statuses.len(), 5);
    let lanes: Vec<_> = (0..5).map(|lane| {
        if lane == 3 {
            assert!(matches!(&result.statuses[lane], LaneStatus::InvalidInput(reason) if reason == "mu must be positive and real"));
            assert!(result.lane(lane).is_err());
            json!({"status":format!("{:?}",result.statuses[lane]),"moments":null,"represented":null,
                "fallback_used":result.fallback_used[lane],"analytic_used":result.analytic_used[lane]})
        } else {
            assert_eq!(result.statuses[lane], LaneStatus::Success);
            let values = result.lane(lane).unwrap();
            assert_eq!(values.len(), 20);
            assert!(values.iter().all(|z|z.re.is_finite() && z.im.is_finite()));
            json!({"status":"success","moments":values.chunks_exact(4).map(|moment|moment.iter()
                .map(|z|[z.re,z.im]).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "represented":values.iter().map(represented).collect::<Vec<_>>(),
                "fallback_used":result.fallback_used[lane],"analytic_used":result.analytic_used[lane]})
        }
    }).collect();
    assert_eq!(result.lane(0).unwrap(),result.lane(4).unwrap());
    let input_after: Vec<_> = batch.values.iter().map(represented).collect();
    assert_eq!(input_before,input_after);
    let stats=workspace.last_execution();
    assert_eq!((stats.input_rows,stats.argument_rows,stats.master_family_batches,stats.master_rows,stats.reduction_rows),(5,4,2,16,4));
    assert_eq!((stats.fallback_rows,stats.fallback_successes,stats.analytic_rows,stats.pentagon_box_rows),(0,0,0,0));
    assert!(workspace.last_profile().is_none());
    json!({"requested_master_policy":format!("{policy:?}"),"basis":result.basis.clone(),"metadata":{
            "convention":format!("{:?}",result.metadata().convention()),
            "tensor_weights":format!("{:?}",result.metadata().tensor_weights()),"dual_shape":result.metadata().dual_shape()},
        "components":1,"lanes":lanes,"diagnostics":workspace.last_diagnostics(),"input":input_before,
        "profiling_enabled":false,"input_unchanged":true,"execution":{
            "input_rows":stats.input_rows,"argument_rows":stats.argument_rows,"master_family_batches":stats.master_family_batches,
            "master_rows":stats.master_rows,"pentagon_box_rows":stats.pentagon_box_rows,"reduction_rows":stats.reduction_rows,
            "fallback_rows":stats.fallback_rows,"fallback_successes":stats.fallback_successes,"analytic_rows":stats.analytic_rows}})
}

fn main() {
    let path=env::args().nth(1).expect("one frozen input JSON file is required");
    assert!(env::args().nth(2).is_none(),"no timing/engine arguments are supported");
    let input: Input=serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(input.id,"n3-r1-massive-five-lane-current-cargo");
    assert_eq!(input.samples.len(),5);
    assert_eq!(iregi_masters::default_master_backend(),MasterBackend::SymbolicaBatch);
    let selective_available=MasterExecutionPolicy::NativeF64Abc.validate_available().is_ok();
    assert_eq!(selective_available,cfg!(feature="native-f64-abc"),"Cargo forwarding/feature unification differs from this example cfg");
    if selective_available {
        // Float53 remains a Float and must never select the ordinary-f64 emitter.
        let float53=MasterEvaluator::new_with_policy(ScalarIntegral::A0,Float::new(53),MasterExecutionPolicy::NativeF64Abc).unwrap();
        assert_eq!(float53.backend(),MasterBackend::SymbolicaBatch);
    }
    let mut values=Vec::new();
    for sample in &input.samples {
        assert_eq!(sample.shifts.len(),3);assert_eq!(sample.masses.len(),3);
        values.extend(sample.shifts.iter().flatten().map(|&x|Complex::new(x,0.)));
        values.extend(sample.masses.iter().map(|z|Complex::new(z[0],z[1])));
        values.push(Complex::new(sample.mu,0.));
    }
    let plan=ReductionPlan::new(ReductionSettings::new(3,1)).unwrap();
    let batch=KinematicsBatch{rows:5,components:1,values};
    let default=observe(&plan,&batch,MasterExecutionPolicy::Default);
    let symbolica=observe(&plan,&batch,MasterExecutionPolicy::Symbolica);
    for field in ["basis","metadata","components","lanes","diagnostics","input","execution"] {
        assert_eq!(default[field],symbolica[field],"default changed: {field}");
    }
    let selected=selective_available.then(||observe(&plan,&batch,MasterExecutionPolicy::NativeF64Abc));
    println!("RESULT {}",json!({"id":input.id,"status":"passed","numeric_type":"f64","rows":5,
        "compiled_selective_feature":cfg!(feature="native-f64-abc"),"master_default_backend":"SymbolicaBatch",
        "capture_cfg":false,"timing":false,"default":default,"symbolica":symbolica,"selected":selected}));
}
