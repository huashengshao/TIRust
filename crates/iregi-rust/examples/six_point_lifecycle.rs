//! Source-only rank-seven public SixPoint lifecycle and numerator capture.
//! All reduction and contraction algebra belongs to the existing library APIs.
use std::{error::Error, path::Path, time::Instant};

use iregi_masters::NativeFloat;
use iregi_rust::{
    Algorithm, BatchResult, CertificateExecutionStats, ContractionPlan,
    ContractionWorkspace, Convention, ExecutionStats, KinematicsBatch, LaneStatus,
    NumeratorBatch, ReductionError, ReductionSettings, SixPointExecutionStats,
    SixPointReductionPlan, SixPointWorkspace, TensorWeights, basis::Monomial,
};
use serde_json::{Value, json};
use symbolica::{
    domains::{float::{Complex, Float, FloatLike}, integer::Integer, rational::Rational},
    evaluate::EvaluationDomain,
};

type Answer<T> = Result<T, Box<dyn Error>>;

fn rational(value: &Value) -> Answer<Rational> {
    let text = value.as_str().ok_or("exact rational string required")?;
    let (n, d) = text.split_once('/').unwrap_or((text, "1"));
    let d = d.parse::<Integer>()?;
    if d == Integer::from(0) { return Err("zero denominator".into()); }
    Ok(Rational::from((n.parse::<Integer>()?, d)))
}

fn save(path: &Path, record: &Value) -> Answer<()> {
    std::fs::write(path, serde_json::to_string_pretty(record)? + "\n")?;
    Ok(())
}

fn represented<T: NativeFloat>(values: &[Complex<T>]) -> Answer<Value> {
    values.iter().map(|v| Ok(json!({
        "real":v.re.exact_rational().ok_or("nonfinite represented real")?.to_string(),
        "imaginary":v.im.exact_rational().ok_or("nonfinite represented imaginary")?.to_string(),
        "real_precision":v.re.get_precision(), "imaginary_precision":v.im.get_precision(),
        "real_sign_negative":v.re.negative_sign(),
        "imaginary_sign_negative":v.im.negative_sign(),
    }))).collect::<Answer<Vec<_>>>().map(Value::Array)
}

fn execution(s: ExecutionStats) -> Value {
    json!({"input_rows":s.input_rows,"argument_rows":s.argument_rows,
        "master_family_batches":s.master_family_batches,"master_rows":s.master_rows,
        "pentagon_box_rows":s.pentagon_box_rows,"reduction_rows":s.reduction_rows,
        "fallback_rows":s.fallback_rows,"fallback_successes":s.fallback_successes,
        "analytic_rows":s.analytic_rows})
}
fn certificate(s: CertificateExecutionStats) -> Value {
    json!({"input_rows":s.input_rows,"evaluated_rows":s.evaluated_rows,
        "passes":s.passes,"reduction_batches":s.reduction_batches})
}
fn trace<T: NativeFloat>(w: &SixPointWorkspace<T>) -> Value
where Complex<T>: EvaluationDomain {
    let s=w.last_execution();
    json!({"parent_input_rows":s.parent_input_rows,"child_batch_calls":s.child_batch_calls,
        "packed_child_rows":s.packed_child_rows,"parent_batch_calls":s.parent_batch_calls,
        "packed_parent_rows":s.packed_parent_rows,"child_execution":execution(s.child_execution),
        "parent_execution":execution(s.parent_execution),
        "child_certificate_execution":certificate(s.child_certificate_execution),
        "parent_certificate_execution":certificate(s.parent_certificate_execution),
        "diagnostics":w.last_diagnostics()})
}
fn reset<T: NativeFloat>(w: &SixPointWorkspace<T>, rows: usize) -> bool
where Complex<T>: EvaluationDomain {
    let s=w.last_execution();
    s.parent_input_rows==rows && s.child_batch_calls==0 && s.packed_child_rows==0
        && s.parent_batch_calls==0 && s.packed_parent_rows==0
        && s.child_execution==ExecutionStats::default()
        && s.parent_execution==ExecutionStats::default()
        && s.child_certificate_execution==CertificateExecutionStats::default()
        && s.parent_certificate_execution==CertificateExecutionStats::default()
}

struct Polynomial<T: NativeFloat> {
    name: &'static str,
    plan: ContractionPlan,
    coefficients: Vec<Complex<T>>,
    exact_coefficients: Vec<[String;2]>,
    workspace: ContractionWorkspace<T>,
}
impl<T: NativeFloat> Clone for Polynomial<T> {
    fn clone(&self) -> Self {
        Self {name:self.name,plan:self.plan.clone(),coefficients:self.coefficients.clone(),
            exact_coefficients:self.exact_coefficients.clone(),workspace:self.workspace.clone()}
    }
}

fn polynomial<T: NativeFloat>(name: &'static str, reduction: &SixPointReductionPlan,
    support: &[Monomial], exact: Vec<[Rational;2]>, prototype: &T) -> Answer<Polynomial<T>>
where Complex<T>: EvaluationDomain {
    if support.len()!=exact.len() {return Err("polynomial support/coefficients mismatch".into());}
    let plan=reduction.contraction_plan(support)?;
    let coefficients=exact.iter().map(|[re,im]|{
        let v=Complex::new(prototype.from_rational(re),prototype.from_rational(im));
        if v.re.exact_rational().as_ref()!=Some(re) || v.im.exact_rational().as_ref()!=Some(im) {
            return Err("coefficient not exactly represented at working precision".into());
        }
        Ok(v)
    }).collect::<Answer<Vec<_>>>()?;
    let workspace=ContractionWorkspace::new(&plan,prototype.clone())?;
    Ok(Polynomial {name,plan,coefficients,
        exact_coefficients:exact.iter().map(|[re,im]|[re.to_string(),im.to_string()]).collect(),workspace})
}

fn contract<T: NativeFloat>(p: &mut Polynomial<T>, result: &BatchResult<T>) -> Answer<Value>
where Complex<T>: EvaluationDomain {
    let numerator=NumeratorBatch {rows:result.rows(),components:1,
        values:(0..result.rows()).flat_map(|_|p.coefficients.iter().cloned()).collect()};
    let before=represented(&numerator.values)?;
    let contracted=p.workspace.contract_batch(result,&numerator)?;
    let after=represented(&numerator.values)?;
    if before!=after || contracted.rows()!=result.rows() || contracted.components!=1
        || contracted.statuses!=result.statuses || contracted.fallback_used!=result.fallback_used
        || contracted.analytic_used!=result.analytic_used {
        return Err("contraction changed coefficients/status/layout/route".into());
    }
    let active=result.statuses.iter().filter(|s|**s==LaneStatus::Success).count();
    if p.workspace.last_evaluated_rows()!=active {return Err("contraction active-row count mismatch".into());}
    let lanes=(0..result.rows()).map(|row| {
        let slots=match contracted.lane(row) {
            Ok(values) => {
                if result.lane(row).is_err() || values.len()!=4 {return Err("incomplete/unexpected contracted lane".into());}
                represented(values)?
            }
            Err(_) => {
                if result.lane(row).is_ok() {return Err("successful input became failed contraction".into());}
                Value::Null
            }
        };
        Ok(json!({"row":row,"status":format!("{:?}",contracted.statuses[row]),"slots":slots}))
    }).collect::<Answer<Vec<_>>>()?;
    Ok(json!({"name":p.name,"support":p.plan.monomials(),"exact_coefficients":p.exact_coefficients,
        "coefficient_input_before":before,"coefficient_input_after":after,"inputs_unchanged":true,
        "evaluated_rows":p.workspace.last_evaluated_rows(),"lanes":lanes,
        "fallback_used":contracted.fallback_used,"analytic_used":contracted.analytic_used}))
}

fn capture_call<T: NativeFloat>(w: &mut SixPointWorkspace<T>, plan: &SixPointReductionPlan,
    polynomials: &mut [Polynomial<T>], sources: &[Vec<Complex<T>>], ids: &[usize]) -> Answer<Value>
where Complex<T>: EvaluationDomain {
    let batch=KinematicsBatch {rows:ids.len(),components:1,
        values:ids.iter().flat_map(|&id|sources[id].iter().cloned()).collect()};
    let before=represented(&batch.values)?;
    let reduced=w.reduce_batch(&batch);
    let after=represented(&batch.values)?;
    let base=json!({"source_rows":ids,"rows":ids.len(),"input_before":before,
        "input_after":after,"inputs_unchanged":before==after,"execution":trace(w)});
    // Preserve the actual failing call, including lane diagnostics, before any
    // terminal assertion is returned to the outer ledger.
    let mut record=base;
    let result=match reduced {
        Ok(result)=>result,
        Err(error)=>{record["api_error"]=json!(error.to_string());record["contract_pass"]=json!(false);return Ok(record);}
    };
    let shape=result.rows()==ids.len() && result.components==1 && result.basis.as_slice()==plan.basis()
        && result.statuses.len()==ids.len() && result.fallback_used.len()==ids.len()
        && result.analytic_used.len()==ids.len() && result.metadata().convention()==Convention::IregiNative
        && result.metadata().tensor_weights()==TensorWeights::Components && result.metadata().dual_shape().is_none();
    let active=ids.iter().filter(|&&id|id!=1).count();
    let s=w.last_execution();
    let packing=s.parent_input_rows==ids.len() && s.child_batch_calls==usize::from(active!=0)
        && s.packed_child_rows==6*active && s.parent_batch_calls==usize::from(active!=0)
        && s.packed_parent_rows==active;
    record["basis"]=json!(result.basis);
    record["components"]=json!(result.components);
    record["metadata"]=json!({"convention":format!("{:?}",result.metadata().convention()),
        "tensor_weights":format!("{:?}",result.metadata().tensor_weights()),"dual_shape":result.metadata().dual_shape()});
    record["fallback_used"]=json!(result.fallback_used);
    record["analytic_used"]=json!(result.analytic_used);
    record["shape_pass"]=json!(shape);
    record["packing_pass"]=json!(packing);
    if !shape {record["contract_pass"]=json!(false);return Ok(record);}
    let mut statuses=true;
    let mut lanes=Vec::new();
    for (row,&id) in ids.iter().enumerate() {
        let mut lane=json!({"row":row,"source_row":id,"status":format!("{:?}",result.statuses[row]),
            "fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row],"moments":null});
        match result.lane(row) {
            Ok(values)=>{
                let complete=values.len()==330*4;
                statuses &= id!=1 && complete;
                if complete {
                    lane["moments"]=Value::Array(values.chunks_exact(4).map(represented).collect::<Answer<Vec<_>>>()?);
                }
            }
            Err(status)=>statuses &= id==1 && matches!(status,LaneStatus::InvalidInput(_)),
        }
        lanes.push(lane);
    }
    record["lanes"]=json!(lanes);
    let mut contractions=Vec::new();
    let mut contractions_pass=true;
    for polynomial in polynomials {
        match contract(polynomial,&result) {
            Ok(capture)=>contractions.push(capture),
            Err(error)=>{
                contractions_pass=false;
                contractions.push(json!({"name":polynomial.name,"error":error.to_string()}));
            }
        }
    }
    record["contractions"]=json!(contractions);
    let idle=if active==0 {reset(w,ids.len())} else {true};
    record["idle_counters_reset"]=json!(idle);
    record["contract_pass"]=json!(shape && packing && statuses && idle && contractions_pass && before==after);
    Ok(record)
}

fn read_sources<T: NativeFloat>(fixture: &Value, prototype: &T) -> Answer<Vec<Vec<Complex<T>>>> {
    let samples=fixture["samples"].as_array().ok_or("missing samples")?;
    if samples.len()<2 || samples[0]["id"]!="base" || samples[1]["id"]!="scaled_by_two" {
        return Err("unchanged retained base/scaled fixture required".into());
    }
    let mut sources=Vec::new();
    for sample in &samples[..2] {
        let props=sample["propagators"].as_array().ok_or("missing propagators")?;
        if props.len()<6 {return Err("six propagators required".into());}
        let mut exact=Vec::new();
        for prop in &props[..6] {
            let shift=prop["shift"].as_array().ok_or("missing shift")?;
            if shift.len()!=4 || prop["power"]!=1 {return Err("invalid unit-power propagator".into());}
            for v in shift {exact.push((rational(v)?,Rational::from(0)));}
        }
        for prop in &props[..6] {
            let mass=prop["mass_squared"].as_array().ok_or("missing complex squared mass")?;
            if mass.len()!=2 {return Err("invalid squared mass pair".into());}
            exact.push((rational(&mass[0])?,rational(&mass[1])?));
        }
        exact.push((rational(&sample["mu"])?,Rational::from(0)));
        let row=exact.iter().map(|(re,im)|{
            let v=Complex::new(prototype.from_rational(re),prototype.from_rational(im));
            if v.re.exact_rational().as_ref()!=Some(re) || v.im.exact_rational().as_ref()!=Some(im) {
                return Err("input failed exact represented rational roundtrip".into());
            }
            Ok(v)
        }).collect::<Answer<Vec<_>>>()?;
        sources.push(row);
    }
    let mut invalid=sources[0].clone();
    invalid[30]=Complex::new(prototype.zero(),prototype.zero());
    sources.insert(1,invalid);
    Ok(sources)
}

fn capture<T: NativeFloat>(input: &Path, prototype: T, output: &Path) -> Answer<()>
where Complex<T>: EvaluationDomain {
    let started=Instant::now();
    let mut record=json!({"schema_version":1,"status":"reading_fixture","working_bits":prototype.get_precision(),
        "requested_bits":53,"fixture":input,"rank":7,"ordinary_components":1,"calls":[],
        "source_row_ids":["base","invalid_scale_zero","scaled_by_two"],
        "scope":"Public SixPoint rank7 ordinary lifecycle and Symbolica sparse/dense numerator contractions; one structural plan, one prepared workspace and one clone; no reference kernel, new algebra, broad domain, Dual or performance claim.",
        "numerical_qualification":"pending separate saved-quad comparator"});
    save(output,&record)?;
    let outcome=(|| -> Answer<()> {
        let fixture:Value=serde_json::from_str(&std::fs::read_to_string(input)?)?;
        let sources=read_sources(&fixture,&prototype)?;
        record["source_inputs"]=json!(sources.iter().map(|row|represented(row)).collect::<Answer<Vec<_>>>()?);
        let mut settings=ReductionSettings::new(6,7);settings.algorithm=Algorithm::PassarinoVeltman;
        record["settings"]=json!(settings);
        record["status"]=json!("preparing_one_plan");save(output,&record)?;
        let plan=SixPointReductionPlan::new(settings)?;
        if plan.parameter_count()!=31 || plan.basis().len()!=330 {return Err("rank7 public plan layout mismatch".into());}
        record["basis"]=json!(plan.basis());
        record["input_layout"]=json!(format!("{:?}",plan.input_layout()));
        record["status"]=json!("preparing_one_workspace");save(output,&record)?;
        let mut workspace=SixPointWorkspace::new(&plan,prototype.clone())?;
        workspace.set_accuracy_precision(53)?;
        let rejected=[workspace.set_accuracy_precision(0).is_err(),
            workspace.set_accuracy_precision(prototype.get_precision()+1).is_err()];
        if rejected!=[true,true] {return Err("invalid accuracy bound accepted".into());}
        workspace.set_accuracy_precision(53)?;
        let mut independent=workspace.clone();
        let sparse=[Monomial([0,0,0,0]),Monomial([1,0,0,0]),Monomial([0,1,0,6]),
            Monomial([1,3,2,1]),Monomial([1,2,4,0])];
        let sparse_coefficients=[[2,0],[-3,1],[5,-2],[-7,3],[11,-1]]
            .map(|c|c.map(Rational::from)).to_vec();
        let dense_coefficients=(0..330).map(|i|{
            let sign=if i%2==0 {1} else {-1};
            [Rational::from((sign*(1+i%7),if i%3==0 {2} else {1})),
                Rational::from((sign*(1+(i+2)%5),4))]
        }).collect();
        record["status"]=json!("preparing_two_symbolica_contractions");save(output,&record)?;
        let mut polynomials=[polynomial("sparse_odd_mixed",&plan,&sparse,sparse_coefficients,&prototype)?,
            polynomial("dense_all330",&plan,plan.basis(),dense_coefficients,&prototype)?];
        let mut independent_polynomials=polynomials.clone();
        record["precision_bound_rejections"]=json!(rejected);
        record["workspace_components"]=json!(workspace.component_count());
        record["workspace_working_bits"]=json!(workspace.binary_precision());
        let specs:[(&str,&[usize],bool);8]=[
            ("empty_before",&[],false),("odd_mixed",&[0,1,2],false),
            ("reordered",&[2,0,1],false),("chunk_first",&[0,1],false),
            ("invalid_only",&[1],false),("empty_after_failure",&[],false),
            ("chunk_second",&[2],false),("independent_owned_scratch",&[0],true)];
        record["schedule"]=json!(specs.iter().map(|(name,ids,other)|
            json!({"label":name,"source_rows":ids,"independent_workspace":other})).collect::<Vec<_>>());
        for (label,ids,other) in specs {
            record["status"]=json!("reducing");record["active_call"]=json!(label);save(output,&record)?;
            let before_main=trace(&workspace);
            let before_contracts=polynomials.each_ref().map(|p|p.workspace.last_evaluated_rows());
            let mut call=if other {capture_call(&mut independent,&plan,&mut independent_polynomials,&sources,ids)?}
                else {capture_call(&mut workspace,&plan,&mut polynomials,&sources,ids)?};
            call["label"]=json!(label);call["independent_workspace"]=json!(other);
            if other {
                call["main_trace_before"]=before_main.clone();call["main_trace_after"]=trace(&workspace);
                call["main_contraction_rows_before"]=json!(before_contracts);
                call["main_contraction_rows_after"]=json!(polynomials.each_ref().map(|p|p.workspace.last_evaluated_rows()));
                let unchanged=call["main_trace_before"]==call["main_trace_after"]
                    && call["main_contraction_rows_before"]==call["main_contraction_rows_after"];
                call["independent_scratch_preserved_main"]=json!(unchanged);
                if !unchanged {call["contract_pass"]=json!(false);}
            }
            let passed=call["contract_pass"]==true;
            record["calls"].as_array_mut().unwrap().push(call);save(output,&record)?;
            if !passed {return Err("lifecycle call failed complete status/layout/packing contract".into());}
        }
        record["status"]=json!("checking_malformed_reset");save(output,&record)?;
        // Main last performed chunk_second successfully; the independent clone
        // cannot replace these nonzero counters. This makes reset nonvacuous.
        let execution_before=trace(&workspace);
        let stale=workspace.last_execution();
        if stale.child_batch_calls!=1 || stale.parent_batch_calls!=1
            || stale.packed_child_rows!=6 || stale.packed_parent_rows!=1 {
            return Err("malformed reset probe requires preceding nonzero main work".into());
        }
        let malformed=KinematicsBatch {rows:1,components:1,values:Vec::new()};
        let before=represented(&malformed.values)?;
        let error=workspace.reduce_batch(&malformed).err();
        let malformed_record=json!({"error":error.as_ref().map(ToString::to_string),
            "invalid_input_error":matches!(error,Some(ReductionError::InvalidInput(_))),
            "execution_before":execution_before,
            "input_before":before,"input_after":represented(&malformed.values)?,
            "execution":trace(&workspace),"counters_reset":reset(&workspace,1),
            "diagnostics_reset":workspace.last_diagnostics().is_empty()});
        let passed=malformed_record["invalid_input_error"]==true && malformed_record["counters_reset"]==true
            && malformed_record["diagnostics_reset"]==true && malformed_record["input_before"]==malformed_record["input_after"];
        record["malformed_call"]=malformed_record;save(output,&record)?;
        if !passed {return Err("malformed input did not reject/reset/preserve".into());}
        record["status"]=json!("reducing_valid_after_malformed");save(output,&record)?;
        let mut call=capture_call(&mut workspace,&plan,&mut polynomials,&sources,&[0])?;
        call["label"]=json!("valid_after_malformed");call["independent_workspace"]=json!(false);
        let passed=call["contract_pass"]==true;
        record["calls"].as_array_mut().unwrap().push(call);save(output,&record)?;
        if !passed {return Err("valid recovery after malformed call failed".into());}
        Ok(())
    })();
    record["elapsed_seconds"]=json!(started.elapsed().as_secs_f64());
    record["active_call"]=Value::Null;
    match outcome {
        Ok(())=>{record["status"]=json!("capture_complete_contracts_passed");save(output,&record)},
        Err(error)=>{record["status"]=json!("failed");record["error"]=json!(error.to_string());save(output,&record)?;Err(error)},
    }
}

fn main() -> std::process::ExitCode {
    let result=(|| -> Answer<()> {
        let args:Vec<_>=std::env::args().skip(1).collect();
        if args.len()!=3 {return Err("six_lifecycle retained-fixture.json 128|256 new-output.json".into());}
        let bits:u32=args[1].parse()?;
        if ![128,256].contains(&bits) {return Err("this frozen cohort admits ordinary Float128/256 only".into());}
        let output=Path::new(&args[2]);
        if output.exists() {return Err("preserve existing captures".into());}
        capture(Path::new(&args[0]),Float::new(bits),output)
    })();
    match result {Ok(())=>std::process::ExitCode::SUCCESS,
        Err(error)=>{eprintln!("{error}");std::process::ExitCode::FAILURE}}
}
