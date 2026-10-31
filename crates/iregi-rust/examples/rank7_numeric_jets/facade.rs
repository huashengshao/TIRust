//! Example-only public facade observer; Symbolica input preparation stays in jets.rs.
use super::{base, counters, describe_result, jets, represented, save_call, write,
    Fallible, FirstJet, MixedJet};
use iregi_masters::{MasterExecutionPolicy, NativeFloat};
use iregi_rust::{Algorithm, BatchResult, KinematicsBatch, ReductionSettings,
    TensorEngineExecution, TensorReductionPlan, TensorWorkspace, TensorWorkspaceOptions};
use serde_json::{json, Value};
use std::path::Path;
use symbolica::{domains::{dual::{DualNumber, HyperDual}, float::Complex,
    rational::Rational}, evaluate::EvaluationDomain};

fn describe<T: NativeFloat>(workspace: &TensorWorkspace<T>,
    result: Fallible<BatchResult<T>>, name: &str, before: Value, after: Value,
    expected: &[&str]) -> Fallible<Value>
where Complex<T>: EvaluationDomain {
    let actual = workspace.engine_execution();
    let nested = match &actual {
        TensorEngineExecution::AffinePower(s) => json!({
            "input_rows":s.input_rows,
            "certificate":{"input_rows":s.certificate.input_rows,
                "evaluated_rows":s.certificate.evaluated_rows,"passes":s.certificate.passes,
                "reduction_batches":s.certificate.reduction_batches},
            "profiles":s.profiles,"dag_nodes":s.dag_nodes,"dag_edges":s.dag_edges,
            "packed_child_rows":s.packed_child_rows,"child_batch_calls":s.child_batch_calls,
            "parent_batch_calls":s.parent_batch_calls,"parent_rows":s.parent_rows,
            "quadrature_work_rows":s.quadrature_work_rows,
            "child_execution":counters(s.child_execution),"parent_execution":counters(s.parent_execution),
            "child_methods":s.child_methods.iter().map(|m|json!({
                "propagators":m.propagators,"powers":m.powers,"requested":m.requested,
                "actual":m.actual,"rows":m.rows,"notes":m.notes})).collect::<Vec<_>>()
        }),
        _ => return Err(format!("N6 facade observer expected AffinePower execution, received {actual:?}").into()),
    };
    let mut call = json!({"name":name,"input_before":before,"input_after":after,
        "input_preserved":before==after,"expected_lane_statuses":expected,
        "execution":counters(actual.summary()?),
        "engine_kind":format!("{:?}",workspace.engine_kind()),
        "raw_engine_execution":format!("{actual:?}"),
        "affine_power_execution":nested,
        "diagnostics":workspace.last_diagnostics()});
    if let Some(p) = workspace.last_profile() {
        call["diagnostic_timings"] = json!({"total":p.total.as_secs_f64(),
            "arguments":p.arguments.as_secs_f64(),"scalar_masters":p.scalar_masters.as_secs_f64(),
            "reduction":p.reduction.as_secs_f64()});
    }
    describe_result(call, result, expected)
}

fn typed<T: NativeFloat, D: DualNumber<Complex<T>>>(
    workspace: &mut TensorWorkspace<T>, name: &str, rows: &[Vec<Complex<T>>],
    shape: &[Vec<usize>], parameters: usize, make: impl Fn(&[Complex<T>]) -> D,
    expected: &[&str]) -> Fallible<Value>
where Complex<T>: EvaluationDomain {
    let components = shape.len();
    let width = parameters.checked_mul(components).ok_or("typed row size overflow")?;
    if components == 0 || rows.iter().any(|row| row.len() != width)
        || workspace.dual_shape() != Some(shape) {
        return Err("typed preparation disagrees with the public ordered layout".into());
    }
    let input: Vec<_> = rows.iter().flat_map(|row|row.chunks_exact(components).map(&make)).collect();
    for dual in &input {
        iregi_masters::validate_dual_structure(dual)?;
        if dual.get_shape().iter().map(|r|r.to_vec()).collect::<Vec<_>>() != shape
            || dual.values().len() != components {
            return Err("typed Symbolica input has a different ordered shape".into());
        }
    }
    let before = represented(&input.iter().flat_map(|d|d.values().iter().cloned()).collect::<Vec<_>>());
    // TensorWorkspace's public numerical API accepts explicit components. This
    // is input packaging of existing Symbolica values, not another Dual algebra
    // or a claim that the facade supplies a typed convenience adapter.
    let batch = KinematicsBatch {rows:rows.len(),components,
        values:input.iter().flat_map(|d|d.values().iter().cloned()).collect()};
    let batch_before = represented(&batch.values);
    let result = workspace.reduce_batch(&batch).map_err(|e|e.into());
    let after = represented(&input.iter().flat_map(|d|d.values().iter().cloned()).collect::<Vec<_>>());
    if batch_before != represented(&batch.values) {
        return Err("public facade modified the prepared component input".into());
    }
    let mut call = describe(workspace, result, name, before, after, expected)?;
    call["typed_transport"] = json!("Existing Symbolica static/dynamic Dual values repacked into public KinematicsBatch; no facade typed convenience method is claimed");
    Ok(call)
}

fn plain<T: NativeFloat>(workspace: &mut TensorWorkspace<T>, name: &str,
    rows: &[Vec<Complex<T>>], expected: &[&str]) -> Fallible<Value>
where Complex<T>: EvaluationDomain {
    let input = KinematicsBatch {rows:rows.len(),components:1,
        values:rows.iter().flatten().cloned().collect()};
    let before = represented(&input.values);
    let result = workspace.reduce_batch(&input).map_err(|e|e.into());
    describe(workspace,result,name,before,represented(&input.values),expected)
}

pub(super) fn capture<T: NativeFloat>(domain: &str, prototype: T,
    fixture: &Path, n: usize, rank: u16, requested_bits: u32,
    profile: &str, algorithm: Algorithm, first_power: u16, output: &Path) -> Fallible<u8>
where Complex<T>: EvaluationDomain {
    if n != 6 || rank > 6 || first_power != 1
        || !["plain","first","mixed"].contains(&profile) {
        return Err("this public facade cohort requires N6 rank<=6, unit powers, plain/first/mixed".into());
    }
    if requested_bits > prototype.get_precision() {
        return Err("requested accuracy exceeds working precision".into());
    }
    let base = base(fixture,n)?;
    let shape = if profile == "first" {jets::FIRST.map(Vec::from).to_vec()}
        else {jets::MIXED.map(Vec::from).to_vec()};
    let mut settings = ReductionSettings::new(n,rank);
    settings.algorithm = algorithm;
    let mut record = json!({"schema_version":1,"status":"preparing_input","domain":domain,
        "working_bits":prototype.get_precision(),"requested_bits":requested_bits,
        "settings":settings,"profile":profile,"ordered_shape":shape,
        "fixture":fixture.to_string_lossy(),
        "base_exact_rational":base.iter().map(ToString::to_string).collect::<Vec<_>>(),"calls":[],
        "public_constructor":"TensorWorkspace::new_dual_with_options",
        "typed_transport":"Symbolica values repacked into KinematicsBatch",
        "limits":"One public N6 unit-power h0 rank<=6 facade cohort. Existing Symbolica static/dynamic Dual input preparation remains separate. Every value, represented precision, sign, lane status and actual composite counter is retained. Public release results do not expose child sensitivity contexts; those are qualified separately by the frozen cfg(test) observer. A precision guard failure is retained. No rigorous derivative bound, arbitrary-branch or performance claim."});
    if profile == "plain" {
        record["ordered_shape"] = Value::Null;
        record["public_constructor"] = json!("TensorWorkspace::new_with_options");
        record["typed_transport"] = Value::Null;
    }
    write(output,&record)?;
    let outcome = (|| -> Fallible<()> {
        let scaled = jets::points(&base,n,&[(Rational::from(0),Rational::from(1))])?.remove(0);
        let exact_rows = [&base,&scaled].into_iter().map(|source| {
            if profile == "plain" {Ok(source.clone())}
            else {jets::input_jets(source,n,&shape)}
        }).collect::<Result<Vec<_>,String>>()?;
        record["input_exact_rational"] = json!(exact_rows.iter().map(|r|
            r.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>());
        if profile != "plain" {
            record["exact_response_columns"] = json!((0..=rank as usize).map(|r|jets::response(n,r))
                .collect::<Result<Vec<_>,_>>()?.iter().map(|rows|rows.iter()
                    .map(|row|row.iter().map(ToString::to_string).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>());
        }
        let rows = exact_rows.iter().map(|r|super::convert(&prototype,r)).collect::<Fallible<Vec<_>>>()?;
        record["status"] = json!("preparing_structural_plan");write(output,&record)?;
        let plan = TensorReductionPlan::new(settings)?;
        record["basis"] = json!(plan.basis().iter().map(|m|m.0).collect::<Vec<_>>());
        record["plan_engine_kind"] = json!(format!("{:?}",plan.engine_kind()));
        let parameters = plan.parameter_count();
        record["status"] = json!("preparing_workspace");write(output,&record)?;
        let options = TensorWorkspaceOptions {master_policy:MasterExecutionPolicy::Symbolica,profiling:true,..Default::default()};
        let mut workspace = if profile == "plain" {
            TensorWorkspace::new_with_options(&plan,prototype.clone(),options)?
        } else {
            TensorWorkspace::new_dual_with_options(&plan,prototype.clone(),
                HyperDual::<u8>::new(shape.clone()),options)?
        };
        workspace.set_accuracy_precision(requested_bits)?;
        record["workspace_engine_kind"] = json!(format!("{:?}",workspace.engine_kind()));
        record["preparation_notes"] = json!(workspace.preparation_notes());
        let components = if profile == "plain" {1} else {shape.len()};
        let mut invalid = rows[0].clone();
        for value in &mut invalid[5*n*components..(5*n+1)*components] {
            *value = Complex::new(prototype.zero(),prototype.zero());
        }
        if profile == "plain" {
            let odd = vec![rows[0].clone(),invalid,rows[1].clone()];
            save_call(&mut record,output,"facade_ordinary_odd3",||plain(&mut workspace,
                "facade_ordinary_odd3",&odd,&["Success","InvalidInput","Success"]))?;
        } else {
            let static_rows = vec![rows[0].clone()];
            save_call(&mut record,output,"facade_static_base",||if profile == "first" {
                typed(&mut workspace,"facade_static_base",&static_rows,&shape,parameters,
                    |v|FirstJet {values:v.to_vec().try_into().unwrap()},&["Success"])
            } else {
                typed(&mut workspace,"facade_static_base",&static_rows,&shape,parameters,
                    |v|MixedJet {values:v.to_vec().try_into().unwrap()},&["Success"])
            })?;
            let odd = vec![rows[1].clone(),invalid,rows[0].clone()];
            save_call(&mut record,output,"facade_dynamic_odd3",||typed(&mut workspace,
                "facade_dynamic_odd3",&odd,&shape,parameters,
                |v|HyperDual::from_values(shape.clone(),v.to_vec()),&["Success","InvalidInput","Success"]))?;
            save_call(&mut record,output,"facade_dynamic_empty",||typed(&mut workspace,
                "facade_dynamic_empty",&[],&shape,parameters,
                |v|HyperDual::from_values(shape.clone(),v.to_vec()),&[]))?;
        }
        Ok(())
    })();
    if let Err(error) = outcome {
        record["status"] = json!("capture_error");record["error"] = json!(error.to_string());
        write(output,&record)?;return Ok(2);
    }
    let expected = record["calls"].as_array().unwrap().iter().all(|call|
        call["status"] == "captured" && call["expected_outcomes"] == true && call["input_preserved"] == true);
    record["status"] = json!("capture_complete_not_yet_qualified");
    record["all_expected_outcomes"] = json!(expected);write(output,&record)?;
    Ok(if expected {0} else {2})
}
