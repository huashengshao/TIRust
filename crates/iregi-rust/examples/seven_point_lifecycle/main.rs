//! Whole-array lifecycle capture for the proposed public seven-point scalar API.
//! No reference kernel or independent algebra is implemented here.

use std::{error::Error, path::Path, time::Instant};
use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, BatchResult, ContractionWorkspace, ExecutionStats, KinematicsBatch,
    LaneStatus, NumeratorBatch, ReductionSettings, SevenPointReductionPlan,
    SevenPointWorkspace, basis::Monomial,
};
use serde_json::{Value, json};
use symbolica::domains::{
    float::{Complex, Float, FloatLike, SingleFloat}, integer::Integer,
    rational::Rational,
};

fn number(value: &Value) -> Result<Rational, Box<dyn Error>> {
    let text = value.as_str().ok_or("expected exact rational string")?;
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let d = d.parse::<Integer>()?;
    if d == Integer::from(0) { return Err("zero rational denominator".into()); }
    Ok(Rational::from((n.parse::<Integer>()?, d)))
}

fn save(path: &Path, record: &Value) -> Result<(), Box<dyn Error>> {
    std::fs::write(path, serde_json::to_string_pretty(record)? + "\n")?;
    Ok(())
}

fn slots(values: &[Complex<Float>]) -> Result<Value, Box<dyn Error>> {
    values.iter().map(|v| Ok(json!({
        "real":v.re.exact_rational().ok_or("nonfinite successful real output")?.to_string(),
        "imaginary":v.im.exact_rational().ok_or("nonfinite successful imaginary output")?.to_string(),
        "real_precision":v.re.get_precision(), "imaginary_precision":v.im.get_precision(),
        "real_sign_negative":v.re.is_sign_negative(), "imaginary_sign_negative":v.im.is_sign_negative(),
    }))).collect::<Result<Vec<_>, Box<dyn Error>>>().map(Value::Array)
}

fn work(stats: ExecutionStats) -> Value {
    json!({"input_rows":stats.input_rows,"argument_rows":stats.argument_rows,
        "master_family_batches":stats.master_family_batches,"master_rows":stats.master_rows,
        "pentagon_box_rows":stats.pentagon_box_rows,"reduction_rows":stats.reduction_rows,
        "fallback_rows":stats.fallback_rows,"fallback_successes":stats.fallback_successes,
        "analytic_rows":stats.analytic_rows})
}

fn snapshot(
    workspace: &SevenPointWorkspace<Float>, result: &BatchResult<Float>,
    contraction: &mut ContractionWorkspace<Float>, prototype: &Float,
) -> Result<Value, Box<dyn Error>> {
    let one = Complex::new(prototype.from_rational(&Rational::from(1)), prototype.zero());
    let contracted = contraction.contract_batch(result, &NumeratorBatch {
        rows:result.rows(), components:1, values:vec![one;result.rows()],
    })?;
    let mut lanes = Vec::new();
    for row in 0..result.rows() {
        match result.lane(row) {
            Ok(values) => {
                if values.len()!=4 {return Err("incomplete rank-zero array".into());}
                let contracted = contracted.lane(row).map_err(|_| "successful moment failed contraction")?;
                if values != contracted {return Err("unit contraction changed scalar slots".into());}
                lanes.push(json!({"status":"Success","moments":[slots(values)?],"contraction":slots(contracted)?}));
            }
            Err(status) => {
                if contracted.lane(row).is_ok() {return Err("failed lane became successful contraction".into());}
                lanes.push(json!({"status":format!("{status:?}"),"moments":null,"contraction":null}));
            }
        }
    }
    let stats=workspace.last_execution();
    Ok(json!({
        "basis":result.basis,"components":result.components,
        "metadata":{"convention":format!("{:?}",result.metadata().convention()),
            "tensor_weights":format!("{:?}",result.metadata().tensor_weights()),
            "dual_shape":result.metadata().dual_shape()},
        "statuses":result.statuses.iter().map(|s|format!("{s:?}")).collect::<Vec<_>>(),
        "fallback_used":result.fallback_used,"analytic_used":result.analytic_used,
        "diagnostics":workspace.last_diagnostics(),"lanes":lanes,
        "execution":{"parent_input_rows":stats.parent_input_rows,
            "child_batch_calls":stats.child_batch_calls,"packed_child_rows":stats.packed_child_rows,
            "parent_batch_calls":stats.parent_batch_calls,"packed_parent_rows":stats.packed_parent_rows,
            "child_execution":work(stats.child_execution),"parent_execution":work(stats.parent_execution),
            "child_certificate_execution":format!("{:?}",stats.child_certificate_execution),
            "parent_certificate_execution":format!("{:?}",stats.parent_certificate_execution)},
        "contraction_evaluated_rows":contraction.last_evaluated_rows(),
    }))
}

fn capture(input: &Path, bits: u32, output: &Path) -> Result<(), Box<dyn Error>> {
    let started=Instant::now();
    let mut record=json!({"schema_version":1,"status":"reading_exact_inputs",
        "interface":"SevenPointReductionPlan; SevenPointWorkspace<Float>",
        "working_bits":bits,"requested_bits":53,"fixture":input,
        "scope":"One retained real-mass seven-point rank-zero family, ordinary components, one structural propagator permutation, mixed invalid lanes, empty/chunked/reordered/repeated calls, independent cloned owned scratch and constant-one contraction. No independent oracle, broad domain, Dual, high rank or performance claim.",
        "calls":[]});
    save(output,&record)?;
    let outcome=(|| -> Result<(),Box<dyn Error>> {
        let fixture:Value=serde_json::from_str(&std::fs::read_to_string(input)?)?;
        let samples=fixture["samples"].as_array().ok_or("missing samples")?;
        if samples.len()!=1 {return Err("exactly one retained sample required".into());}
        let props=samples[0]["propagators"].as_array().ok_or("missing propagators")?;
        if props.len()!=7 {return Err("seven propagators required".into());}
        let prototype=Float::new(bits);
        let mut base=Vec::new();
        let mut exact=Vec::new();
        for prop in props {
            let shift=prop["shift"].as_array().ok_or("missing shift")?;
            if shift.len()!=4 || prop["power"]!=1 {return Err("invalid propagator".into());}
            for v in shift {exact.push((number(v)?,Rational::from(0)));}
        }
        for prop in props {
            let mass=prop["mass_squared"].as_array().ok_or("missing squared mass")?;
            if mass.len()!=2 {return Err("invalid squared mass".into());}
            exact.push((number(&mass[0])?,number(&mass[1])?));
        }
        exact.push((number(&samples[0]["mu"])?,Rational::from(0)));
        for (re,im) in &exact {
            let v=Complex::new(prototype.from_rational(re),prototype.from_rational(im));
            if v.re.exact_rational().as_ref()!=Some(re) || v.im.exact_rational().as_ref()!=Some(im) {
                return Err("inputs must round-trip exactly at working precision".into());
            }
            base.push(v);
        }
        let permutation=[0,2,1,3,4,5,6];
        let mut permuted=Vec::new();
        for i in permutation {permuted.extend_from_slice(&base[4*i..4*(i+1)]);}
        for i in permutation {permuted.push(base[28+i].clone());}
        permuted.push(base[35].clone());
        let mut invalid=base.clone();
        invalid[35]=Complex::new(prototype.zero(),prototype.zero());
        record["input_exact_rational"]=json!(exact.iter().map(|(re,im)|[re.to_string(),im.to_string()]).collect::<Vec<_>>());
        record["permutation"]=json!(permutation);
        record["status"]=json!("preparing_plan");save(output,&record)?;
        let mut settings=ReductionSettings::new(7,0);settings.algorithm=Algorithm::PassarinoVeltman;
        let plan=SevenPointReductionPlan::new(settings)?;
        if plan.parameter_count()!=36 || plan.basis()!=[Monomial([0;4])] {return Err("public plan layout mismatch".into());}
        record["plan_settings"]=json!(plan.settings());
        record["input_layout"]=json!(format!("{:?}",plan.input_layout()));
        record["parameter_count"]=json!(plan.parameter_count());
        record["status"]=json!("preparing_workspace");save(output,&record)?;
        let mut workspace=SevenPointWorkspace::new(&plan,prototype.clone())?;
        workspace.set_accuracy_precision(53)?;
        let low_rejected=workspace.set_accuracy_precision(0).is_err();
        let high_rejected=workspace.set_accuracy_precision(bits+1).is_err();
        if !low_rejected || !high_rejected {return Err("invalid accuracy precision accepted".into());}
        workspace.set_accuracy_precision(53)?;
        let mut independent=workspace.clone();
        let contraction_plan=plan.contraction_plan(&[Monomial([0;4])])?;
        if plan.contraction_plan(&[Monomial([1,0,0,0])]).is_ok() {return Err("non-scalar monomial accepted".into());}
        let mut contraction=ContractionWorkspace::new(&contraction_plan,prototype.clone())?;
        record["precision_bound_rejections"]=json!([low_rejected,high_rejected]);
        let sources=[base,permuted,invalid];
        let specs:Vec<(&str,Vec<usize>,bool)>=vec![
            ("empty_before",vec![],false),
            ("singleton_base",vec![0],false),
            ("odd_mixed",vec![0,2,1],false),
            ("reordered",vec![1,0,2],false),
            ("chunk_two",vec![1,0],false),
            ("chunk_invalid",vec![2],false),
            ("independent_owned_scratch",vec![0],true),
            ("empty_after",vec![],false),
        ];
        for (label,ids,other) in specs {
            record["status"]=json!("reducing");record["active_call"]=json!(label);save(output,&record)?;
            let batch=KinematicsBatch {rows:ids.len(),components:1,
                values:ids.iter().flat_map(|&i|sources[i].iter().cloned()).collect()};
            let input_before=slots(&batch.values)?;
            let w=if other {&mut independent} else {&mut workspace};
            let result=w.reduce_batch(&batch)?;
            let input_after=slots(&batch.values)?;
            if input_before!=input_after {return Err("reduction changed represented inputs, precision or sign".into());}
            let expected=ids.iter().filter(|&&id|id!=2).count();
            if result.statuses.iter().filter(|s|**s==LaneStatus::Success).count()!=expected {
                let snapshot=snapshot(w,&result,&mut contraction,&prototype)?;
                record["calls"].as_array_mut().unwrap().push(json!({"label":label,"source_rows":ids,"result":snapshot}));
                return Err("unexpected lane success/rejection pattern".into());
            }
            for (row,&id) in ids.iter().enumerate() {
                if id==2 && !matches!(result.statuses[row],LaneStatus::InvalidInput(_)) {
                    return Err("invalid scale did not retain InvalidInput".into());
                }
            }
            let stats=w.last_execution();
            if stats.child_batch_calls!=usize::from(expected!=0) || stats.packed_child_rows!=21*expected
                || stats.parent_batch_calls!=usize::from(expected!=0) || stats.packed_parent_rows!=expected {
                return Err("N7 packed-batch execution counts mismatch".into());
            }
            let snapshot=snapshot(w,&result,&mut contraction,&prototype)?;
            record["calls"].as_array_mut().unwrap().push(json!({"label":label,"source_rows":ids,"independent_workspace":other,
                "input_before":input_before,"input_after":input_after,"inputs_unchanged":true,"result":snapshot}));
            save(output,&record)?;
        }
        record["status"]=json!("checking_malformed_call");save(output,&record)?;
        let malformed_batch=KinematicsBatch {rows:1,components:1,values:Vec::new()};
        let malformed_input_before=slots(&malformed_batch.values)?;
        let malformed=workspace.reduce_batch(&malformed_batch);
        let malformed_input_after=slots(&malformed_batch.values)?;
        let stats=workspace.last_execution();
        let reset=stats.child_batch_calls==0 && stats.parent_batch_calls==0 && stats.packed_child_rows==0
            && stats.packed_parent_rows==0 && stats.child_execution==ExecutionStats::default()
            && stats.parent_execution==ExecutionStats::default() && workspace.last_diagnostics().is_empty();
        record["malformed_call"]=json!({"error":malformed.err().map(|e|format!("{e:?}")),"counters_reset":reset,
            "input_before":malformed_input_before,"input_after":malformed_input_after,
            "inputs_unchanged":malformed_input_before==malformed_input_after});
        if record["malformed_call"]["error"].is_null() || !reset || malformed_input_before!=malformed_input_after {
            return Err("malformed call did not reset or preserve inputs".into());
        }
        record["status"]=json!("reducing_after_malformed");save(output,&record)?;
        let final_batch=KinematicsBatch {rows:1,components:1,values:sources[0].clone()};
        let input_before=slots(&final_batch.values)?;
        let result=workspace.reduce_batch(&final_batch)?;
        let input_after=slots(&final_batch.values)?;
        let stats=workspace.last_execution();
        if stats.parent_input_rows!=1 || stats.child_batch_calls!=1 || stats.packed_child_rows!=21
            || stats.parent_batch_calls!=1 || stats.packed_parent_rows!=1 || input_before!=input_after {
            return Err("valid call after malformed had incorrect counts or changed inputs".into());
        }
        let snapshot=snapshot(&workspace,&result,&mut contraction,&prototype)?;
        record["calls"].as_array_mut().unwrap().push(json!({"label":"valid_after_malformed","source_rows":[0],
            "input_before":input_before,"input_after":input_after,"inputs_unchanged":true,"result":snapshot}));
        if result.statuses != [LaneStatus::Success] {return Err("valid call after malformed failed".into());}
        Ok(())
    })();
    record["elapsed_seconds"]=json!(started.elapsed().as_secs_f64());
    match outcome {
        Ok(()) => {record["status"]=json!("passed");save(output,&record)},
        Err(error) => {record["status"]=json!("failed");record["error"]=json!(error.to_string());save(output,&record)?;Err(error)},
    }
}

fn run() -> Result<(),Box<dyn Error>> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3 {return Err("lifecycle fixture.json working_bits output.json".into());}
    let bits:u32=args[1].parse()?;
    if bits!=128 {return Err("frozen lifecycle captures Float128 only".into());}
    capture(Path::new(&args[0]),bits,Path::new(&args[2]))
}

fn main() -> std::process::ExitCode {
    match run() {Ok(())=>std::process::ExitCode::SUCCESS,
        Err(error)=>{eprintln!("{error}");std::process::ExitCode::FAILURE}}
}
