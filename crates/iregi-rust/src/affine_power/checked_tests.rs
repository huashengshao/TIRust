use super::*;
use symbolica::prelude::Real;
use symbolica::domains::float::{Float, FloatLike, SingleFloat};

fn checked(n:usize)->ReductionSettings {
    let mut settings=ReductionSettings::new(n,0);
    settings.algorithm=Algorithm::Ibp;
    settings.invariants=InvariantInput::SuppliedScalarProducts;
    settings
}
fn bubble()->Vec<Complex<Float>> {
    // p0=0, p1=(0,1,0,0), packed G=(0,0,-1), m²=(1,2), mu=3.
    [0,0,0,0,0,1,0,0,0,0,-1,1,2,3].into_iter()
        .map(|v|Complex::new(Float::with_val(128,v),Float::with_val(128,0)))
        .collect()
}
#[test]
fn checked_outer_admission_keeps_public_cartesian_domain_and_private_layouts_distinct() {
    let settings=checked(6);
    assert!(AffinePowerReductionPlan::new(settings.clone()).is_err());
    let plan=AffinePowerReductionPlan::new_checked_supplied(settings).unwrap();
    assert!(plan.checked_supplied);assert_eq!(plan.parameter_count(),52);
    assert_eq!(plan.input_layout().scalar_products,Some(24..45));
    assert_eq!(plan.input_layout().masses,45..51);
    assert!(AffinePowerReductionPlan::new_checked_supplied(checked(5)).is_err());
    let recursive=AffinePowerReductionPlan::new_checked_internal(checked(5),AffinePowerBudget::default()).unwrap();
    assert_eq!(recursive.parameter_count(),41);
    assert!(recursive.children.lock().unwrap().is_empty());
    assert!(recursive.guards.lock().unwrap().is_empty());
    assert!(recursive.profiles.lock().unwrap().is_empty());
}
#[test]
fn checked_incompatible_gram_is_rejected_before_any_direct_or_recursive_child_prepare() {
    let plan=AffinePowerReductionPlan::new_checked_internal(checked(2),AffinePowerBudget::default()).unwrap();
    let mut workspace=AffinePowerWorkspace::new(&plan,Float::new(128)).unwrap();
    let mut row=bubble();row[10]=Complex::new(Float::with_val(128,0),Float::with_val(128,0));
    let result=workspace.reduce_coupled_batch(&KinematicsBatch{values:row,rows:1,components:1}).unwrap();
    assert!(matches!(result.result().statuses[0],LaneStatus::InvalidInput(_)));
    assert!(workspace.children.is_empty());assert!(workspace.profiles.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls,0);
    assert!(result.quadrature_estimates(0).is_err());
}
#[test]
fn checked_regular_rows_use_one_batched_lowpoint_route_and_scatter_its_owned_context() {
    let settings=checked(2);
    let plan=AffinePowerReductionPlan::new_checked_internal(settings.clone(),AffinePowerBudget::default()).unwrap();
    let mut workspace=AffinePowerWorkspace::new(&plan,Float::new(128)).unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    let row=bubble();let mut bad=row.clone();bad[10]=Complex::new(Float::with_val(128,0),Float::with_val(128,0));
    let values=row.iter().cloned().chain(bad).chain(row.iter().cloned()).collect();
    let result=workspace.reduce_coupled_batch(&KinematicsBatch{values,rows:3,components:1}).unwrap();
    assert_eq!(result.result().statuses[0],LaneStatus::Success);
    assert!(matches!(result.result().statuses[1],LaneStatus::InvalidInput(_)));
    assert_eq!(result.result().statuses[2],LaneStatus::Success);
    let raw_plan=ReductionPlan::new(settings).unwrap();
    let mut raw=Workspace::new(&raw_plan,Float::new(128)).unwrap();raw.set_accuracy_precision(53).unwrap();
    let expected=raw.reduce_batch(&KinematicsBatch{values:row,rows:1,components:1}).unwrap();
    assert_eq!(result.result().lane(0).unwrap(),expected.lane(0).unwrap());
    assert_eq!(result.result().lane(2).unwrap(),expected.lane(0).unwrap());
    for row in [0,2] {
        assert_eq!(result.owned.origin(row,128,53).unwrap(),QuadratureErrorOrigin::KnownNoQuadrature);
        assert_eq!(result.quadrature_estimates(row).unwrap().len(),4);
        assert!(result.quadrature_estimates(row).unwrap().iter().all(|z|z.re.is_zero()&&z.im.is_zero()));
    }
    assert!(result.quadrature_estimates(1).is_err());
    assert!(workspace.profiles.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls,1);
    assert_eq!(workspace.last_execution().packed_child_rows,2);
    assert_eq!(workspace.last_execution().child_methods[0].powers,vec![1,1]);
    let empty=workspace.reduce_coupled_batch(&KinematicsBatch{values:vec![],rows:0,components:1}).unwrap();
    assert_eq!(empty.result().rows(),0);assert_eq!(workspace.last_execution().child_batch_calls,0);
}
#[test]
fn checked_nilpotent_only_coordinate_rank_change_never_prepares_regular_lowpoint() {
    let plan=AffinePowerReductionPlan::new_checked_internal(checked(2),AffinePowerBudget::default()).unwrap();
    let shape=HyperDual::<u8>::new(vec![vec![0],vec![1]]);
    let mut workspace=AffinePowerWorkspace::new_dual(&plan,Float::new(128),shape).unwrap();
    // p1 = t*e_x and G11=-t² has zero represented orders0/1. Compatibility
    // holds at every represented order, but the nonzero coordinate direction
    // is nilpotent only and cannot certify a regular primal direct route.
    let mut values=Vec::new();
    for parameter in 0..14 {
        let primal=match parameter {11=>1,12=>2,13=>3,_=>0};
        values.push(Complex::new(Float::with_val(128,primal),Float::with_val(128,0)));
        values.push(Complex::new(Float::with_val(128,i64::from(parameter==5)),Float::with_val(128,0)));
    }
    let result=workspace.reduce_coupled_batch(&KinematicsBatch{values,rows:1,components:2}).unwrap();
    assert!(matches!(result.result().statuses[0],LaneStatus::Unstable(_)));
    assert!(workspace.children.is_empty());assert!(workspace.profiles.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls,0);
    assert!(result.quadrature_estimates(0).is_err());
}


// Strategy-selection tests use opaque physical rows. They establish transport
// and branch selection only; the owning runtime still proves p/G and regularity.
fn dotted_checked_five() -> ReductionSettings {
    let mut settings = checked(5);
    settings.powers[0] = 2;
    settings
}
fn strategy_batch(components: usize, masses: &[[i64; 5]]) -> KinematicsBatch<Float> {
    let mut values = Vec::new();
    for row in masses {
        for parameter in 0..41 {
            for component in 0..components {
                let value = if component == 0 {
                    if (35..40).contains(&parameter) {
                        row[parameter - 35]
                    } else if parameter == 40 {
                        3
                    } else {
                        0
                    }
                } else {
                    (100 * parameter + component) as i64
                };
                values.push(Complex::new(
                    Float::with_val(128, value),
                    Float::with_val(128, 0),
                ));
            }
        }
    }
    KinematicsBatch {
        rows: masses.len(),
        components,
        values,
    }
}
#[test]
fn checked_zero_mass_strategy_keeps_original_ordinary_lanes_and_two_separate_groups() {
    let settings = dotted_checked_five();
    let layout = checked_layout(5).unwrap();
    let input = strategy_batch(1, &[[0, 2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 2, 3, 4, 5]]);
    let saved = input.values.clone();
    let (auxiliary, direct) =
        checked_regular_mass_groups(&settings, &layout, &input, &[2, 1, 0]).unwrap();
    assert_eq!(auxiliary, vec![1]);
    assert_eq!(direct, vec![2, 0]);
    assert_eq!(input.values, saved);
    assert_eq!(settings.powers, vec![2, 1, 1, 1, 1]);
    assert_eq!(settings.invariants, InvariantInput::SuppliedScalarProducts);
}
#[test]
fn checked_zero_mass_strategy_retains_every_ordered_dual_component_and_original_branch() {
    let settings = dotted_checked_five();
    let layout = checked_layout(5).unwrap();
    // Six user Taylor components are not six sample lanes. A nonzero user
    // derivative at the zero primal still goes to the original-request branch;
    // only the original master's analytic-branch validator may accept it.
    let input = strategy_batch(6, &[[0, 2, 3, 4, 5], [1, 2, 3, 4, 5]]);
    let saved = input.values.clone();
    let groups = checked_regular_mass_groups(&settings, &layout, &input, &[0, 1]).unwrap();
    assert_eq!(groups, (vec![1], vec![0]));
    assert!(!input.values[35 * 6 + 1].re.is_zero());
    assert_eq!(input.values, saved);
}
#[test]
fn checked_zero_mass_strategy_only_tests_differentiated_complex_mass_primals() {
    let settings = dotted_checked_five();
    let layout = checked_layout(5).unwrap();
    // A zero undotted mass does not create an auxiliary mass direction.
    let mut input = strategy_batch(1, &[[1, 0, 3, 4, 5], [0, 2, 3, 4, 5]]);
    input.values[41 + 35].im = Float::with_val(128, -1);
    assert_eq!(
        checked_regular_mass_groups(&settings, &layout, &input, &[0, 1]).unwrap(),
        (vec![0, 1], vec![])
    );
    let mut shifted = settings.clone();
    shifted.dimension_shift = 1;
    input.values[41 + 35].im = Float::with_val(128, 0);
    assert_eq!(
        checked_regular_mass_groups(&shifted, &layout, &input, &[1]).unwrap(),
        (vec![1], vec![])
    );
}
#[test]
fn checked_zero_mass_strategy_empty_and_malformed_inputs_never_create_placeholder_rows() {
    let settings = dotted_checked_five();
    let layout = checked_layout(5).unwrap();
    let empty = strategy_batch(2, &[]);
    assert_eq!(
        checked_regular_mass_groups(&settings, &layout, &empty, &[]).unwrap(),
        (vec![], vec![])
    );
    let mut input = strategy_batch(2, &[[0, 2, 3, 4, 5]]);
    assert!(checked_regular_mass_groups(&settings, &layout, &input, &[1]).is_err());
    input.values.pop();
    assert!(checked_regular_mass_groups(&settings, &layout, &input, &[0]).is_err());
    input.components = 0;
    assert!(checked_regular_mass_groups(&settings, &layout, &input, &[]).is_err());
}

fn regular_five_zero_mass_input(components: usize, user_mass_axis: bool) -> KinematicsBatch<Float> {
    use symbolica::{
        domains::rational::{Q, Rational},
        tensors::matrix::Matrix,
    };
    let r = |n: i64, d: i64| Rational::from((n, d));
    let points = Matrix::from_nested_vec(
        vec![
            vec![r(0, 1), r(0, 1), r(0, 1), r(0, 1)],
            vec![r(0, 1), r(1, 1), r(0, 1), r(0, 1)],
            vec![r(0, 1), r(0, 1), r(1, 1), r(0, 1)],
            vec![r(0, 1), r(0, 1), r(0, 1), r(1, 1)],
            vec![r(1, 2), r(0, 1), r(0, 1), r(0, 1)],
        ],
        Q,
    )
    .unwrap();
    let eta = Matrix::from_nested_vec(
        vec![
            vec![r(1, 1), r(0, 1), r(0, 1), r(0, 1)],
            vec![r(0, 1), r(-1, 1), r(0, 1), r(0, 1)],
            vec![r(0, 1), r(0, 1), r(-1, 1), r(0, 1)],
            vec![r(0, 1), r(0, 1), r(0, 1), r(-1, 1)],
        ],
        Q,
    )
    .unwrap();
    let gram = &(&points * &eta) * &points.transpose();
    let mut exact = Vec::new();
    for i in 0..5 {
        for mu in 0..4 {
            exact.push(points[(i, mu)].clone());
        }
    }
    for i in 0..5 {
        for j in i..5 {
            exact.push(gram[(i, j)].clone());
        }
    }
    exact.extend([r(0, 1), r(2, 1), r(3, 1), r(4, 1), r(2, 1), r(3, 1)]);
    let prototype = Float::new(128);
    let mut values = Vec::new();
    for (parameter, primal) in exact.iter().enumerate() {
        for component in 0..components {
            let value = if component == 0 {
                primal.clone()
            } else {
                r(
                    i64::from(user_mass_axis && parameter == 36 && component == 1),
                    1,
                )
            };
            values.push(Complex::new(
                prototype.from_rational(&value),
                Float::with_val(128, 0),
            ));
        }
    }
    KinematicsBatch {
        rows: 1,
        components,
        values,
    }
}
#[test]
fn checked_zero_mass_constructors_and_bad_full_g_rows_never_prepare_either_strategy() {
    let plan = AffinePowerReductionPlan::new_checked_internal(
        dotted_checked_five(),
        AffinePowerBudget::default(),
    )
    .unwrap();
    assert!(plan.children.lock().unwrap().is_empty());
    for components in [1, 2] {
        let mut workspace = if components == 1 {
            AffinePowerWorkspace::new(&plan, Float::new(128)).unwrap()
        } else {
            AffinePowerWorkspace::new_dual(
                &plan,
                Float::new(128),
                HyperDual::<u8>::new(vec![vec![0], vec![1]]),
            )
            .unwrap()
        };
        let mut input = regular_five_zero_mass_input(components, false);
        input.values[25 * components].re = Float::with_val(128, 0); // G11 differs from p1.p1=-1.
        let output = workspace.reduce_coupled_batch(&input).unwrap();
        assert!(matches!(
            output.result().statuses[0],
            LaneStatus::InvalidInput(_)
        ));
        assert!(workspace.children.is_empty() && workspace.checked_zero_mass_child.is_none());
        assert_eq!(workspace.last_execution().child_batch_calls, 0);
        let empty = KinematicsBatch {
            rows: 0,
            components,
            values: vec![],
        };
        assert_eq!(workspace.reduce_batch(&empty).unwrap().rows(), 0);
        assert!(workspace.checked_zero_mass_child.is_none());
    }
}

fn zero_mass_scalar_json(v: &Complex<Float>) -> serde_json::Value {
    serde_json::json!({"real":v.re.exact_rational().expect("finite capture real").to_string(),
        "imaginary":v.im.exact_rational().expect("finite capture imaginary").to_string(),
        "real_precision":v.re.get_precision(),"imaginary_precision":v.im.get_precision(),
        "real_negative":v.re.is_sign_negative(),"imaginary_negative":v.im.is_sign_negative()})
}
fn zero_mass_ir_control(input: &KinematicsBatch<Float>) -> (Vec<Complex<Float>>,serde_json::Value) {
    use symbolica::atom::{Atom,AtomCore,Symbol};
    use symbolica::domains::rational::Rational;
    // Symbolica constructs the local q^-4 coefficient and its user-mass series.
    // No custom polynomial, derivative or Laurent manipulation is used.
    let delta_symbol=symbolica::symbol!("iregi_zero_mass_control::delta");
    let delta=Atom::var(delta_symbol);let mut diagonal=Vec::new();let mut index=20;
    for i in 0..5 {for j in i..5 {if i==j {
        diagonal.push(input.values[index*input.components].re.exact_rational().unwrap());
    }index+=1;}}
    let mut denominator=Atom::num(1);
    for line in 1..5 {
        let mass=input.values[(35+line)*input.components].re.exact_rational().unwrap();
        let mut factor=Atom::num(mass)-Atom::num(diagonal[line].clone());
        if line==1 {factor=&factor+&delta;}
        denominator=&denominator*factor;
    }
    let coefficient=-Atom::num(1)/&denominator;
    let series=coefficient.series(delta_symbol,0,1).unwrap();
    let coefficients=[series.coefficient(0.into()).unwrap(),series.coefficient(1.into()).unwrap()];
    let exact=coefficients.iter().map(|v|Rational::try_from(v).unwrap()).collect::<Vec<_>>();
    assert_eq!(exact[0],Rational::from((-1,105)));assert_eq!(exact[1],Rational::from((1,315)));
    let pi=Float::new(128).pi();let pi_squared=pi.pow(2);let prototype=Float::new(128);
    let values=exact.iter().map(|v|Complex::new(Float::with_val(128,0),pi_squared.clone()*prototype.from_rational(v))).collect::<Vec<_>>();
    let expression=symbolica::parse!("𝑖")*Atom::var(Symbol::PI).pow(2)*coefficient;
    let note=serde_json::json!({"identity":"local scaleless q^-4 IR residue, native normalization",
        "symbolica_expression":expression.to_string(),"symbolica_series_coefficients":exact.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "expected":["-i*pi^2/105","+i*pi^2/315"],"UV1_zero":true,"IR2_zero":true,
        "values":values.iter().map(zero_mass_scalar_json).collect::<Vec<_>>() });
    (values,note)
}
fn zero_mass_capture_call(
    tag:&str,input:&KinematicsBatch<Float>,value:&AffinePowerResult<Float>,workspace:&AffinePowerWorkspace<Float>,
)->serde_json::Value {
    let result=value.result();let stats=workspace.last_execution();
    let lanes=(0..result.rows()).map(|row| {
        let moments=result.lane(row).ok().map(|v|v.chunks_exact(result.components)
            .map(|slot|slot.iter().map(zero_mass_scalar_json).collect::<Vec<_>>()).collect::<Vec<_>>());
        let envelope=result.validated_primal_envelope(row,128,53).map(|e|serde_json::json!({
            "working_precision":e.working_precision,"accuracy_bits":e.accuracy_bits,"components":e.components,
            "routes":format!("{:?}",e.routes),"output_primals":e.output_primals.iter().map(zero_mass_scalar_json).collect::<Vec<_>>(),
            "output_components":e.output_components.as_ref().map(|v|v.iter().map(zero_mass_scalar_json).collect::<Vec<_>>()),
            "component_amplitudes":e.component_amplitudes.as_ref().map(|v|v.iter().map(zero_mass_scalar_json).collect::<Vec<_>>())})).map_err(|e|e.to_string());
        serde_json::json!({"status":format!("{:?}",result.statuses[row]),"moments":moments,"envelope":envelope,
            "quadrature_origin":value.owned.origin(row,128,53).map(|v|format!("{v:?}")),
            "quadrature_errors":value.quadrature_estimates(row).map(|v|v.iter().map(zero_mass_scalar_json).collect::<Vec<_>>()),
            "fallback_used":result.fallback_used[row],"analytic_used":result.analytic_used[row]})
    }).collect::<Vec<_>>();
    serde_json::json!({"tag":tag,"rows":input.rows,"components":input.components,
        "inputs":input.values.iter().map(zero_mass_scalar_json).collect::<Vec<_>>(),"lanes":lanes,
        "dual_shape":result.metadata().dual_shape(),"convention":result.metadata().convention(),
        "tensor_weights":result.metadata().tensor_weights(),"diagnostics":workspace.last_diagnostics(),
        "execution":{"input_rows":stats.input_rows,"packed_child_rows":stats.packed_child_rows,"child_batch_calls":stats.child_batch_calls,
            "certificate":{"input_rows":stats.certificate.input_rows,"evaluated_rows":stats.certificate.evaluated_rows,"passes":stats.certificate.passes,"reduction_batches":stats.certificate.reduction_batches},
            "child_execution":format!("{:?}",stats.child_execution),"parent_execution":format!("{:?}",stats.parent_execution),
            "child_methods":stats.child_methods.iter().map(|m|serde_json::json!({"propagators":m.propagators,"powers":m.powers,
                "requested":m.requested,"actual":m.actual,"rows":m.rows,"notes":m.notes})).collect::<Vec<_>>()},
        "cache":{"auxiliary_child_present":!workspace.children.is_empty(),"direct_child_present":workspace.checked_zero_mass_child.is_some()}})
}
fn persist_zero_mass_capture(record:&serde_json::Value) {
    if let Some(path)=std::env::var_os("IREGI_CHECKED_ZERO_MASS_OUTPUT") {
        std::fs::write(path,serde_json::to_string_pretty(record).unwrap()+"\n").unwrap();
    }
}
fn zero_mass_scale_response() -> (Vec<symbolica::domains::rational::Rational>,serde_json::Value) {
    use symbolica::atom::{Atom,AtomCore};
    use symbolica::domains::rational::Rational;
    use symbolica::id::Replacement;
    // Reuse the exact Laurent-series construction in masters::scale_transport.
    // Public mu=3*(1+t), so the scalar-master scale ratio is mu(t)^2/mu(0)^2.
    let eps_symbol=symbolica::symbol!("iregi_zero_mass_scale_control::epsilon");
    let t_symbol=symbolica::symbol!("iregi_zero_mass_scale_control::t");
    let eps=Atom::var(eps_symbol);let t=Atom::var(t_symbol);
    let finite=symbolica::symbol!("iregi_zero_mass_scale_control::finite").to_atom();
    let ir=symbolica::symbol!("iregi_zero_mass_scale_control::ir1").to_atom();
    let formal_log=symbolica::symbol!("iregi_zero_mass_scale_control::formal_log").to_atom();
    let mu=Atom::num(3)*(Atom::num(1)+&t);
    let logarithm=(mu.pow(2)/Atom::num(9)).log();
    let original=&finite+&ir/&eps;
    let laurent=((&eps*&formal_log).exp()*original).series(eps_symbol,0,0).unwrap();
    let mut responses=Vec::new();
    for power in [-2,-1,0] {
        let coefficient=laurent.coefficient(power.into()).unwrap_or_else(||Atom::num(0))
            .replace_multiple([Replacement::new(formal_log.clone(),logarithm.clone())].iter());
        let derivative=coefficient.series(t_symbol,0,1).unwrap().coefficient(1.into()).unwrap_or_else(||Atom::num(0));
        responses.push(derivative);
    }
    let response=[Atom::num(0),responses[1].clone(),responses[0].clone(),responses[2].clone()];
    let factors=response.iter().map(|v|Rational::try_from(&(v/&ir)).unwrap()).collect::<Vec<_>>();
    assert_eq!(factors,[Rational::from(0),Rational::from(0),Rational::from(0),Rational::from(2)]);
    let note=serde_json::json!({"convention":"public mu^(2epsilon), mu(t)=3*(1+t); UV1=IR2=0 for this request",
        "laurent_series":laurent.to_string(),"mu_ratio_log":logarithm.to_string(),
        "symbolica_response":response.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "response_factors_of_primal_IR1":factors.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "other_mass_direction_qualified":false});
    (factors,note)
}
fn actual_zero_mass_direct_delegation(dual: bool, scale_only: bool) {
    assert!(!scale_only||dual);
    let settings = dotted_checked_five();
    let mut record=serde_json::json!({"schema_version":1,"status":"preparing","working_bits":128,"requested_bits":53,
        "settings":settings,"components":if dual{2}else{1},"dual_direction":if scale_only{"scale_only_mu_jet3"}else if dual{"second_nonzero_mass_squared_jet1"}else{"ordinary"},"calls":[],"performance_qualified":false});
    persist_zero_mass_capture(&record);
    let plan=AffinePowerReductionPlan::new_checked_internal(settings.clone(),AffinePowerBudget::default()).unwrap();
    let mut workspace=if dual {AffinePowerWorkspace::new_dual(&plan,Float::new(128),HyperDual::<u8>::new(vec![vec![0],vec![1]])).unwrap()}
        else {AffinePowerWorkspace::new(&plan,Float::new(128)).unwrap()};
    workspace.set_accuracy_precision(53).unwrap();
    let mut input=regular_five_zero_mass_input(if dual{2}else{1},dual&&!scale_only);
    if scale_only {input.values[40*input.components+1].re=Float::with_val(128,3);}
    let before=input.values.clone();
    let (mut ir,mut ir_record)=zero_mass_ir_control(&input);
    let scale_response=if scale_only {
        ir[1]=Complex::new(Float::with_val(128,0),Float::with_val(128,0));
        let(factors,note)=zero_mass_scale_response();
        ir_record=serde_json::json!({"identity":"ordinary local IR residue with scale-only tangent; pole response derived by the separate Symbolica mu^(2epsilon) control",
            "ordinary_and_unselected_mass_direction_control":ir_record,
            "symbolica_series_coefficients":["-1/105",factors[1].to_string()],
            "expected":["-i*pi^2/105","0"],
            "values":ir.iter().map(zero_mass_scalar_json).collect::<Vec<_>>()});
        record["scale_response_control"]=note;Some(factors)
    }else{None};
    record["local_IR_control"]=ir_record;
    let mut calls=Vec::new();
    let first=workspace.reduce_coupled_batch(&input).unwrap();
    calls.push(zero_mass_capture_call("zero_singleton",&input,&first,&workspace));record["calls"]=serde_json::json!(calls);persist_zero_mass_capture(&record);
    assert_eq!(first.result().statuses[0],LaneStatus::Success,"{:?}",workspace.last_diagnostics());
    assert!(workspace.children.is_empty()&&workspace.checked_zero_mass_child.is_some());assert_eq!(input.values,before);
    let direct=ReductionPlan::new(settings).unwrap();
    assert!(direct.leaves.iter().all(|leaf|leaf.kind==crate::ibp::MasterLeafKind::FourDimensional),"rank-zero h0 dotted witness must avoid every genuine E6 leaf");
    let mut control=if dual {Workspace::new_dual_with_master_policy(&direct,Float::new(128),HyperDual::<u8>::new(vec![vec![0],vec![1]]),MasterExecutionPolicy::Default).unwrap()}
        else {Workspace::new(&direct,Float::new(128)).unwrap()};
    control.set_accuracy_precision(53).unwrap();let expected=control.reduce_batch(&input).unwrap();
    assert_eq!(expected.statuses[0],LaneStatus::Success,"direct control must be successful");
    let tolerance=Float::new(128).from_rational(&symbolica::domains::rational::Rational::from((1,10_000_000_000i64)));
    use symbolica::atom::{Atom,AtomCore};
    let independent_preparation_tolerance=Float::new(128).from_rational(&symbolica::domains::rational::Rational::try_from(&Atom::num(2).pow(-80)).unwrap());
    for (slot,(actual,control)) in first.result().lane(0).unwrap().iter().zip(expected.lane(0).unwrap()).enumerate() {
        use symbolica::atom::{Atom,AtomCore};
        let component=slot%input.components;
        let unit=symbolica::domains::rational::Rational::try_from(&Atom::num(3).pow(-8-if scale_only{0}else{2*component as i64})).unwrap();
        let scale=Float::new(128).from_rational(&unit)+control.re.norm()+control.im.norm();
        let difference=(actual.re.clone()-&control.re).norm()+(actual.im.clone()-&control.im).norm();
        assert!(difference<=scale*&independent_preparation_tolerance,"dimensionally normalized2^-80 direct-control slot{slot}");
    }
    for component in 0..input.components {
        let actual=&first.result().lane(0).unwrap()[input.components+component];
        assert!(actual.re.norm()+(actual.im.clone()-&ir[component].im).norm()<=ir[component].im.norm()*&tolerance,"independent local IR control component{component}");
        use symbolica::atom::{Atom,AtomCore};
        let exact_unit=Atom::num(3).pow(-8-if scale_only{0}else{2*component as i64});
        let unit=symbolica::domains::rational::Rational::try_from(&exact_unit).unwrap();
        let zero_tolerance=Float::new(128).from_rational(&unit)*&tolerance;
        for pole in [0,2] {let actual=&first.result().lane(0).unwrap()[pole*input.components+component];
            assert!(actual.re.norm()+actual.im.norm()<=zero_tolerance,"independent UV1/IR2 zero control");}
    }
    if let Some(factors)=&scale_response {
        let lane=first.result().lane(0).unwrap();let primal_ir=&lane[input.components];
        for pole in 0..4 {
            let factor=Float::new(128).from_rational(&factors[pole]);
            let predicted=Complex::new(primal_ir.re.clone()*&factor,primal_ir.im.clone()*&factor);
            let actual=&lane[pole*input.components+1];
            let difference=(actual.re.clone()-&predicted.re).norm()+(actual.im.clone()-&predicted.im).norm();
            let unit=symbolica::domains::rational::Rational::try_from(&Atom::num(3).pow(-8)).unwrap();
            let scale=Float::new(128).from_rational(&unit)+predicted.re.norm()+predicted.im.norm();
            assert!(difference<=scale*&independent_preparation_tolerance,"Symbolica scale-only Laurent identity pole{pole}");
        }
    }
    let mut nonzero=input.values.clone();nonzero[35*input.components].re=Float::with_val(128,1);
    let mut invalid=input.values.clone();invalid[25*input.components].re=Float::with_val(128,0);
    let mixed=KinematicsBatch{rows:4,components:input.components,values:input.values.iter().cloned().chain(nonzero).chain(invalid).chain(input.values.iter().cloned()).collect()};
    let saved=mixed.values.clone();let mixed_value=workspace.reduce_coupled_batch(&mixed).unwrap();
    calls.push(zero_mass_capture_call("mixed_zero_nonzero_invalid_zero",&mixed,&mixed_value,&workspace));record["calls"]=serde_json::json!(calls);persist_zero_mass_capture(&record);
    for row in [0,1,3] {assert_eq!(mixed_value.result().statuses[row],LaneStatus::Success,"{:?}",workspace.last_diagnostics());}
    assert!(matches!(mixed_value.result().statuses[2],LaneStatus::InvalidInput(_)));assert!(mixed_value.result().lane(2).is_err());
    assert_eq!(mixed.values,saved);assert!(workspace.checked_zero_mass_child.is_some()&&!workspace.children.is_empty());
    assert_eq!(workspace.last_execution().child_batch_calls,2);assert_eq!(workspace.last_execution().packed_child_rows,3);
    assert_eq!(mixed_value.result().lane(0).unwrap(),first.result().lane(0).unwrap());assert_eq!(mixed_value.result().lane(3).unwrap(),first.result().lane(0).unwrap());
    let empty=KinematicsBatch{rows:0,components:input.components,values:vec![]};let empty_value=workspace.reduce_coupled_batch(&empty).unwrap();
    calls.push(zero_mass_capture_call("empty_reset",&empty,&empty_value,&workspace));record["calls"]=serde_json::json!(calls);persist_zero_mass_capture(&record);
    assert_eq!(empty_value.result().rows(),0);assert_eq!(workspace.last_execution().child_batch_calls,0);
    let recovered=workspace.reduce_coupled_batch(&input).unwrap();
    calls.push(zero_mass_capture_call("zero_recovery",&input,&recovered,&workspace));record["calls"]=serde_json::json!(calls);persist_zero_mass_capture(&record);
    assert_eq!(recovered.result().lane(0).unwrap(),first.result().lane(0).unwrap());
    assert_eq!(workspace.last_execution().child_batch_calls,1);assert_eq!(workspace.last_execution().packed_child_rows,1);
    record["status"]=serde_json::json!("passed");record["four_dimensional_leaves_only"]=serde_json::json!(true);
    record["input_preserved"]=serde_json::json!(input.values==before&&mixed.values==saved);
    persist_zero_mass_capture(&record);println!("CHECKED_ZERO_MASS_RESULT {}",serde_json::to_string(&record).unwrap());
}

#[test]
#[ignore = "genuine zero-mass dotted N5 recurrence; requires separate finite numeric authorization"]
fn actual_checked_zero_mass_direct_ordinary_matches_original_request_recurrence() {
    actual_zero_mass_direct_delegation(false,false);
}
#[test]
#[ignore = "genuine zero-mass dotted N5 user-Dual recurrence; requires separate finite numeric authorization"]
fn actual_checked_zero_mass_direct_dual_keeps_nonzero_other_mass_direction() {
    actual_zero_mass_direct_delegation(true,false);
}

#[test]
#[ignore = "genuine scale-only Dual zero-mass N5 recurrence; requires separate finite numeric authorization"]
fn actual_checked_zero_mass_scale_only_dual_uses_existing_mu_transport() {
    actual_zero_mass_direct_delegation(true,true);
}
