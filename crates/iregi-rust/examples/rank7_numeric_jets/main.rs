//! Source-only proposed example: ordinary rank-seven generic and Dual arrays.
//! Input preparation is separate from reduction and uses Symbolica exclusively.
mod jets;
mod rescue;
mod facade;
use iregi_masters::NativeFloat;
use iregi_rust::{Algorithm,BatchResult,ExecutionStats,KinematicsBatch,LaneStatus,
                 ReductionPlan,ReductionSettings,Workspace};
use serde_json::{Value,json};
use std::{error::Error,path::Path};
use symbolica::{
    domains::{dual::{DualNumber,HyperDual},float::{Complex,DoubleFloat,Float},
              integer::Integer,rational::Rational},
    evaluate::EvaluationDomain,
};
symbolica::create_hyperdual_from_components!(FirstJet,[[0,0],[1,0]]);
symbolica::create_hyperdual_from_components!(MixedJet,[[0,0],[1,0],[0,1],[2,0],[1,1],[0,2]]);
type Fallible<T>=Result<T,Box<dyn Error>>;
fn rational(value:&Value)->Fallible<Rational> {
    let text=value.as_str().ok_or("input numbers must be exact rational strings")?;
    let (n,d)=text.split_once('/').unwrap_or((text,"1"));
    let denominator=d.parse::<Integer>()?;
    if denominator==Integer::from(0) {return Err("zero rational denominator".into());}
    Ok(Rational::from((n.parse::<Integer>()?,denominator)))
}
fn base(path:&Path,n:usize)->Fallible<Vec<Rational>> {
    let fixture:Value=serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let sample=&fixture["samples"][0];
    let propagators=sample["propagators"].as_array().ok_or("missing propagators")?;
    if propagators.len()<n {return Err("fixture has too few propagators".into());}
    let mut input=Vec::with_capacity(5*n+1);
    for p in &propagators[..n] {
        let shift=p["shift"].as_array().ok_or("missing Cartesian shift")?;
        if shift.len()!=4 {return Err("Cartesian shift must have four components".into());}
        for v in shift {input.push(rational(v)?);}
    }
    for p in &propagators[..n] {
        let pair=p["mass_squared"].as_array().ok_or("missing squared mass")?;
        if pair.len()!=2 || rational(&pair[1])?!=Rational::from(0) || p["power"]!=1 {
            return Err("this first cohort requires real unit-power fixture masses".into());
        }
        input.push(rational(&pair[0])?);
    }
    input.push(rational(&sample["mu"])?);Ok(input)
}
fn write(path:&Path,record:&Value)->Fallible<()> {
    std::fs::write(path,serde_json::to_string_pretty(record)?+"\n")?;Ok(())
}
fn operand<T:NativeFloat>(v:&T)->Value {
    json!({"rational":v.exact_rational().map(|x|x.to_string()),"precision":v.get_precision(),
           "negative_sign":v.negative_sign(),"finite":v.is_finite()})
}
fn represented<T:NativeFloat>(v:&[Complex<T>])->Value {
    json!(v.iter().map(|z|json!([operand(&z.re),operand(&z.im)])).collect::<Vec<_>>())
}
fn convert<T:NativeFloat>(prototype:&T,exact:&[Rational])->Fallible<Vec<Complex<T>>> {
    exact.iter().map(|q| {
        let re=prototype.from_rational(q);
        if re.exact_rational().as_ref()!=Some(q) {return Err("input rounding would change this exact cohort".into());}
        Ok(Complex::new(re,prototype.zero()))
    }).collect()
}
fn counters(s:ExecutionStats)->Value {
    json!({"input_rows":s.input_rows,"argument_rows":s.argument_rows,
        "master_family_batches":s.master_family_batches,"master_rows":s.master_rows,
        "pentagon_box_rows":s.pentagon_box_rows,"reduction_rows":s.reduction_rows,
        "fallback_rows":s.fallback_rows,"fallback_successes":s.fallback_successes,"analytic_rows":s.analytic_rows})
}
fn describe<T:NativeFloat>(workspace:&Workspace<T>,result:Fallible<BatchResult<T>>,
    name:&str,before:Value,after:Value,expected:&[&str])->Fallible<Value>
where Complex<T>:EvaluationDomain {
    let mut call=json!({"name":name,"input_before":before,"input_after":after,
        "input_preserved":before==after,"execution":counters(workspace.last_execution()),
        "diagnostics":workspace.last_diagnostics(),"expected_lane_statuses":expected});
    if let Some(p)=workspace.last_profile() {
        call["diagnostic_timings"]=json!({"total":p.total.as_secs_f64(),"arguments":p.arguments.as_secs_f64(),
            "scalar_masters":p.scalar_masters.as_secs_f64(),"reduction":p.reduction.as_secs_f64()});
    }
    describe_result(call,result,expected)
}
fn describe_result<T:NativeFloat>(mut call:Value,result:Fallible<BatchResult<T>>,expected:&[&str])->Fallible<Value> {
    let result=match result {
        Ok(value)=>value,
        Err(error)=> {call["status"]=json!("batch_error");call["error"]=json!(error.to_string());return Ok(call);}
    };
    if result.statuses.len()!=expected.len() {return Err("unexpected public status array length".into());}
    let mut lanes=Vec::new();let mut good=true;
    for (lane,expected) in expected.iter().enumerate() {
        let expected_ok=match expected {
            &"Success"=>result.statuses[lane]==LaneStatus::Success,
            &"InvalidInput"=>matches!(result.statuses[lane],LaneStatus::InvalidInput(_)),
            _=>false,
        };
        good &= expected_ok;
        let values=match result.lane(lane) {
            Ok(values)=> {
                if values.len()!=result.basis.len()*4*result.components {return Err("incomplete successful tensor array".into());}
                if values.iter().any(|z|z.re.exact_rational().is_none()||z.im.exact_rational().is_none()) {
                    return Err("successful tensor array contains nonfinite components".into());
                }
                represented(values)
            },
            Err(_)=>Value::Null,
        };
        lanes.push(json!({"lane":lane,"status":format!("{:?}",result.statuses[lane]),
            "expected_status":expected,"expected_outcome":expected_ok,"values":values,
            "fallback_used":result.fallback_used[lane],"analytic_used":result.analytic_used[lane]}));
    }
    call["status"]=json!("captured");call["expected_outcomes"]=json!(good);
    call["basis"]=json!(result.basis.iter().map(|m|m.0).collect::<Vec<_>>());
    call["components"]=json!(result.components);
    call["metadata"]=json!({"convention":result.metadata().convention(),
        "tensor_weights":result.metadata().tensor_weights(),"dual_shape":result.metadata().dual_shape()});
    call["lanes"]=json!(lanes);Ok(call)
}
fn typed<T:NativeFloat,D:DualNumber<Complex<T>>>(workspace:&mut Workspace<T>,name:&str,
    rows:&[Vec<Complex<T>>],components:usize,make:impl Fn(&[Complex<T>])->D,
    expected:&[&str])->Fallible<Value>
where Complex<T>:EvaluationDomain {
    let input:Vec<_>=rows.iter().flat_map(|r|r.chunks_exact(components).map(&make)).collect();
    let before=represented(&input.iter().flat_map(|d|d.values().iter().cloned()).collect::<Vec<_>>());
    let result=workspace.reduce_typed_dual_batch(&input,rows.len()).map_err(|e|e.into());
    let after=represented(&input.iter().flat_map(|d|d.values().iter().cloned()).collect::<Vec<_>>());
    describe(workspace,result,name,before,after,expected)
}
fn plain<T:NativeFloat>(workspace:&mut Workspace<T>,name:&str,rows:&[Vec<Complex<T>>],expected:&[&str])->Fallible<Value>
where Complex<T>:EvaluationDomain {
    let input=KinematicsBatch {rows:rows.len(),components:1,values:rows.iter().flatten().cloned().collect()};
    let before=represented(&input.values);
    let result=workspace.reduce_batch(&input).map_err(|e|e.into());
    let after=represented(&input.values);describe(workspace,result,name,before,after,expected)
}
fn save_call(record:&mut Value,output:&Path,name:&str,run:impl FnOnce()->Fallible<Value>)->Fallible<()> {
    record["status"]=json!("evaluating_batch");record["active_call"]=json!(name);write(output,record)?;
    record["calls"].as_array_mut().unwrap().push(run()?);write(output,record)
}
fn capture<T:NativeFloat>(domain:&str,prototype:T,fixture:&Path,n:usize,rank:u16,
    requested_bits:u32,profile:&str,algorithm:Algorithm,first_power:u16,output:&Path)->Fallible<u8>
where Complex<T>:EvaluationDomain {
    if n==6 {return facade::capture(domain,prototype,fixture,n,rank,requested_bits,profile,algorithm,first_power,output);}
    if requested_bits>prototype.get_precision() {return Err("requested accuracy exceeds working precision".into());}
    let base=base(fixture,n)?;
    let shape:Vec<Vec<usize>>=match profile {
        "first"=>jets::FIRST.map(Vec::from).to_vec(),
        "mixed"=>jets::MIXED.map(Vec::from).to_vec(),
        _=>vec![vec![0,0]],
    };
    let mut settings=ReductionSettings::new(n,rank);settings.algorithm=algorithm;settings.powers[0]=first_power;
    let mut record=json!({"schema_version":1,"status":"preparing_input","domain":domain,
        "working_bits":prototype.get_precision(),"requested_bits":requested_bits,"settings":settings,
        "profile":profile,"ordered_shape":shape,"fixture":fixture.to_string_lossy(),
        "base_exact_rational":base.iter().map(ToString::to_string).collect::<Vec<_>>(),"calls":[],
        "limits":"Public ordinary Workspace, N1..5 only. Retains every coefficient, sign, represented precision and failure. Input Dualizer and response derivatives are Symbolica. FD full tensors use ordinary reduction, an independent derivative route but not independent integration. No performance, derivative error enclosure or blanket branch qualification claim."});
    if profile=="plain"||profile=="fd" {record["ordered_shape"]=Value::Null;}
    write(output,&record)?;
    let outcome=(||->Fallible<()> {
        let mut exact_rows=Vec::new();
        if profile=="fd" {
            let mut grid=Vec::new();let mut exact_points=Vec::new();
            for exponent in [12,18,24] {
                let h=Rational::from((1_i64,1_i64<<exponent));
                for a in -2_i64..=2 {for b in -2_i64..=2 {
                    let x=&h*&Rational::from(a);let y=&h*&Rational::from(b);
                    grid.push(json!({"h":h.to_string(),"mass_node":a,"scale_node":b}));
                    exact_points.push((x,y));
                }}
            }
            exact_rows=jets::points(&base,n,&exact_points)?;
            record["fd_grid"]=json!(grid);
            record["fd_exact_taylor_weights"]=json!((0..=2).map(jets::stencil)
                .collect::<Result<Vec<_>,_>>()?.iter().map(|w|w.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>());
        } else {
            let scaled=jets::points(&base,n,&[(Rational::from(0),Rational::from(1))])?.remove(0);
            for source in [&base,&scaled] {
                exact_rows.push(if profile=="plain" {(*source).clone()} else {jets::input_jets(source,n,&shape)?});
            }
        }
        record["input_exact_rational"]=json!(exact_rows.iter().map(|r|r.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>());
        if profile=="mixed"||profile=="first" {
            record["exact_response_columns"]=json!((0..=rank as usize).map(|r|jets::response(n,r))
                .collect::<Result<Vec<_>,_>>()?.iter().map(|rows|rows.iter()
                .map(|row|row.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>());
        }
        let mut rows=exact_rows.iter().map(|r|convert(&prototype,r)).collect::<Fallible<Vec<_>>>()?;
        record["status"]=json!("preparing_structural_plan");write(output,&record)?;
        let plan=ReductionPlan::new(settings)?;
        record["basis"]=json!(plan.basis().iter().map(|m|m.0).collect::<Vec<_>>());
        record["algorithm_used"]=json!(plan.algorithm_used());record["preparation_notes"]=json!(plan.preparation_notes());
        record["status"]=json!("preparing_workspace");write(output,&record)?;
        let mut workspace=if profile=="first"||profile=="mixed" {
            Workspace::new_dual(&plan,prototype.clone(),HyperDual::<u8>::new(shape.clone()))?
        } else {Workspace::new(&plan,prototype.clone())?};
        workspace.set_accuracy_precision(requested_bits)?;workspace.set_profiling(true);
        if profile=="fd" {
            let expected=vec!["Success";rows.len()];
            save_call(&mut record,output,"ordinary_fd75",||plain(&mut workspace,"ordinary_fd75",&rows,&expected))?;
        } else {
            let components=if profile=="plain" {1} else {shape.len()};
            let mut invalid=rows[0].clone();
            for value in &mut invalid[5*n*components..(5*n+1)*components] {
                *value=Complex::new(prototype.zero(),prototype.zero());
            }
            if profile=="plain" {
                rows.insert(1,invalid);
                save_call(&mut record,output,"ordinary_odd3",||plain(&mut workspace,"ordinary_odd3",&rows,&["Success","InvalidInput","Success"]))?;
            } else {
                let static_rows=vec![rows[0].clone()];
                save_call(&mut record,output,"static_base",||if profile=="first" {
                    typed(&mut workspace,"static_base",&static_rows,components,
                        |v|FirstJet {values:v.to_vec().try_into().unwrap()},&["Success"])
                } else {
                    typed(&mut workspace,"static_base",&static_rows,components,
                        |v|MixedJet {values:v.to_vec().try_into().unwrap()},&["Success"])
                })?;
                let mixed=vec![rows[1].clone(),invalid,rows[0].clone()];
                save_call(&mut record,output,"dynamic_odd3",||typed(&mut workspace,"dynamic_odd3",&mixed,components,
                    |v|HyperDual::from_values(shape.clone(),v.to_vec()),&["Success","InvalidInput","Success"]))?;
                save_call(&mut record,output,"dynamic_empty",||typed(&mut workspace,"dynamic_empty",&[],components,
                    |v|HyperDual::from_values(shape.clone(),v.to_vec()),&[]))?;
            }
        }
        Ok(())
    })();
    if let Err(error)=outcome {
        record["status"]=json!("capture_error");record["error"]=json!(error.to_string());write(output,&record)?;return Ok(2);
    }
    let all_expected=record["calls"].as_array().unwrap().iter().all(|call|
        call["status"]=="captured"&&call["expected_outcomes"]==true&&call["input_preserved"]==true);
    record["status"]=json!("capture_complete_not_yet_qualified");record["all_expected_outcomes"]=json!(all_expected);
    write(output,&record)?;Ok(if all_expected {0} else {2})
}
fn run()->Fallible<u8> {
    let mut args:Vec<_>=std::env::args().collect();
    let rescue_flag=args.last().is_some_and(|a|a=="--rescue");
    if rescue_flag {args.pop();}
    let rescue_env=match std::env::var("IREGI_NUMERIC_JETS_RESCUE") {
        Err(std::env::VarError::NotPresent)=>false,
        Ok(value) if value=="0"=>false,
        Ok(value) if value=="1"=>true,
        _=>return Err("IREGI_NUMERIC_JETS_RESCUE must be 0 or 1".into()),
    };
    let usage="rank7_numeric_jets fixture.json N rank f64|dd|float128|float256|float384 requested_bits plain|first|mixed|fd pv|ibp|hybrid first_power output.json";
    if args.len()!=10 {return Err(usage.into());}
    let n:usize=args[2].parse()?;let rank:u16=args[3].parse()?;let requested_bits:u32=args[5].parse()?;let profile=args[6].as_str();
    let algorithm=match args[7].as_str() {"pv"=>Algorithm::PassarinoVeltman,"ibp"=>Algorithm::Ibp,"hybrid"=>Algorithm::Hybrid,_=>return Err(usage.into())};
    let first_power:u16=args[8].parse()?;
    if !(1..=6).contains(&n)||rank>7||n==6&&(rank>6||profile=="fd"||first_power!=1)
        ||!["plain","first","mixed","fd"].contains(&profile)
        || !(1..=3).contains(&first_power)||first_power!=1&&profile!="plain"||requested_bits<53
        ||profile=="fd"&&(args[4]!="float384"||requested_bits<160) {
        return Err(usage.into());
    }
    let fixture=Path::new(&args[1]);let output=Path::new(&args[9]);
    if output.exists() {return Err("refusing to overwrite capture".into());}
    if rescue_flag||rescue_env {
        if args[4]!="f64"||n!=4||rank!=6||profile!="mixed"||algorithm!=Algorithm::PassarinoVeltman
            ||first_power!=1||requested_bits!=53 {
            return Err("first rescue cohort requires N4 rank6 f64 requested53 mixed PV first_power1".into());
        }
        return rescue::capture(fixture,n,rank,requested_bits,algorithm,output);
    }
    match args[4].as_str() {
        "f64"=>capture("f64",0f64,fixture,n,rank,requested_bits,profile,algorithm,first_power,output),
        "dd"=>capture("DoubleFloat",DoubleFloat::from(0f64),fixture,n,rank,requested_bits,profile,algorithm,first_power,output),
        "float128"=>capture("Float128",Float::new(128),fixture,n,rank,requested_bits,profile,algorithm,first_power,output),
        "float256"=>capture("Float256",Float::new(256),fixture,n,rank,requested_bits,profile,algorithm,first_power,output),
        "float384"=>capture("Float384",Float::new(384),fixture,n,rank,requested_bits,profile,algorithm,first_power,output),
        _=>Err(usage.into()),
    }
}
fn main()->std::process::ExitCode {
    match run() {Ok(code)=>std::process::ExitCode::from(code),Err(error)=> {
        eprintln!("{error}");std::process::ExitCode::FAILURE
    }}
}
