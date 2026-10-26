//! Example-only public precision-rescue observer. Numerator/input jets stay in jets.rs.
use super::{Fallible,MixedJet,base,counters,describe_result,jets,represented,save_call,write};
use iregi_rust::{Algorithm,RescueSettings,RescueTrigger,TensorReductionPlan,TensorRescueWorkspace};
use serde_json::{Value,json};
use std::path::Path;
use symbolica::domains::{dual::{DualNumber,HyperDual},float::Complex,rational::Rational};

fn typed<D:DualNumber<Complex<f64>>>(workspace:&mut TensorRescueWorkspace<f64>,name:&str,
    rows:&[Vec<Complex<f64>>],shape:&[Vec<usize>],make:impl Fn(&[Complex<f64>])->D,
    expected:&[&str])->Fallible<Value> {
    let input:Vec<_>=rows.iter().flat_map(|r|r.chunks_exact(shape.len()).map(&make)).collect();
    let before=represented(&input.iter().flat_map(|d|d.values().iter().cloned()).collect::<Vec<_>>());
    let result=workspace.reduce_typed_dual_batch(&input,rows.len()).map_err(|e|e.into());
    let after=represented(&input.iter().flat_map(|d|d.values().iter().cloned()).collect::<Vec<_>>());
    let stats=workspace.last_execution();
    let primary=workspace.last_primary_statuses();
    let rescue=workspace.last_rescue_statuses();
    if primary.len()!=expected.len()||rescue.len()!=expected.len() {
        return Err("rescue did not retain every primary and rescue lane status".into());
    }
    let call=json!({"name":name,"input_before":before,"input_after":after,
        "input_preserved":before==after,"expected_lane_statuses":expected,
        "diagnostics":workspace.last_diagnostics(),
        "execution":{"primary":counters(stats.primary),"working":counters(stats.working),
            "verification":counters(stats.verification),"requested_rows":stats.requested_rows,
            "accepted_rows":stats.accepted_rows,"corrected_successful_rows":stats.corrected_successful_rows},
        "rescue":{"engine_kind":format!("{:?}",workspace.engine_kind()),
            "algorithm_used":format!("{:?}",workspace.algorithm_used()),
            "primary_statuses":primary.iter().map(|s|format!("{s:?}")).collect::<Vec<_>>(),
            "statuses":rescue.iter().map(|s|format!("{s:?}")).collect::<Vec<_>>(),
            "raw_primary_diagnostics":workspace.last_diagnostics(),
            "primary_engine_execution":workspace.primary_engine_execution().map(|e|format!("{e:?}")),
            "working_engine_execution":workspace.working_engine_execution().map(|e|format!("{e:?}")),
            "verification_engine_execution":workspace.verification_engine_execution().map(|e|format!("{e:?}")),
            "generic_detailed_stage_reports_exposed":false,
            "accepted_component_sensitivity_envelope_claimed":false}});
    describe_result(call,result,expected)
}

pub(super) fn capture(fixture:&Path,n:usize,rank:u16,requested_bits:u32,
    algorithm:Algorithm,output:&Path)->Fallible<u8> {
    let base=base(fixture,n)?;
    let shape=jets::MIXED.map(Vec::from).to_vec();
    let mut settings=iregi_rust::ReductionSettings::new(n,rank);settings.algorithm=algorithm;
    let policy=RescueSettings {working_bits:128,verification_bits:256,
        accuracy_bits:Some(requested_bits),trigger:RescueTrigger::UnstableOnly};
    let mut record=json!({"schema_version":1,"status":"preparing_input","domain":"f64",
        "working_bits":53,"requested_bits":requested_bits,"settings":settings,"profile":"mixed",
        "ordered_shape":shape,"fixture":fixture.to_string_lossy(),
        "base_exact_rational":base.iter().map(ToString::to_string).collect::<Vec<_>>(),"calls":[],
        "precision_rescue":{"working_bits":128,"verification_bits":256,
            "accuracy_bits":requested_bits,"trigger":"UnstableOnly","public_constructor":"TensorRescueWorkspace::new_dual"},
        "limits":"One public N4R6 f64 mixed Dual rescue cohort. Every native component is retained, together with the actual primary failure and stage counters. Converged results are rounded to f64 by the library. Generic rescue does not expose detailed tagged stage reports or transport accepted sensitivity envelopes; no such claim is made. No new algebra, branch certificate, rigorous accuracy bound or performance claim."});
    write(output,&record)?;
    let outcome=(||->Fallible<()> {
        let scaled=jets::points(&base,n,&[(Rational::from(0),Rational::from(1))])?.remove(0);
        let exact_rows=[&base,&scaled].into_iter().map(|r|jets::input_jets(r,n,&shape))
            .collect::<Result<Vec<_>,String>>()?;
        record["input_exact_rational"]=json!(exact_rows.iter().map(|r|r.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>());
        record["exact_response_columns"]=json!((0..=rank as usize).map(|r|jets::response(n,r))
            .collect::<Result<Vec<_>,_>>()?.iter().map(|rows|rows.iter()
                .map(|row|row.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>());
        let rows=exact_rows.iter().map(|r|super::convert(&0f64,r)).collect::<Fallible<Vec<_>>>()?;
        record["status"]=json!("preparing_structural_plan");write(output,&record)?;
        let plan=TensorReductionPlan::new(settings)?;
        record["basis"]=json!(plan.basis().iter().map(|m|m.0).collect::<Vec<_>>());
        record["status"]=json!("preparing_workspace");write(output,&record)?;
        let mut workspace=TensorRescueWorkspace::new_dual(&plan,0f64,policy,HyperDual::<u8>::new(shape.clone()))?;
        record["algorithm_used"]=json!(workspace.algorithm_used());
        record["preparation_notes"]=json!(workspace.preparation_notes());
        let mut invalid=rows[0].clone();
        for value in &mut invalid[5*n*shape.len()..(5*n+1)*shape.len()] {*value=Complex::new(0.,0.);}
        let static_rows=vec![rows[0].clone()];
        save_call(&mut record,output,"static_base",||typed(&mut workspace,"static_base",&static_rows,&shape,
            |v|MixedJet {values:v.to_vec().try_into().unwrap()},&["Success"]))?;
        let odd=vec![rows[1].clone(),invalid,rows[0].clone()];
        save_call(&mut record,output,"dynamic_odd3",||typed(&mut workspace,"dynamic_odd3",&odd,&shape,
            |v|HyperDual::from_values(shape.clone(),v.to_vec()),&["Success","InvalidInput","Success"]))?;
        save_call(&mut record,output,"dynamic_empty",||typed(&mut workspace,"dynamic_empty",&[],&shape,
            |v|HyperDual::from_values(shape.clone(),v.to_vec()),&[]))?;
        Ok(())
    })();
    if let Err(error)=outcome {
        record["status"]=json!("capture_error");record["error"]=json!(error.to_string());write(output,&record)?;return Ok(2);
    }
    let expected=record["calls"].as_array().unwrap().iter().all(|call|
        call["status"]=="captured"&&call["expected_outcomes"]==true&&call["input_preserved"]==true);
    record["status"]=json!("capture_complete_not_yet_qualified");record["all_expected_outcomes"]=json!(expected);
    write(output,&record)?;Ok(if expected {0} else {2})
}
