#[test]
fn public_planar_statistics_profiles_reordering_and_error_reuse() {
    let configuration = settings(0);
    let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
    let tetrahedron = [
        [0, -1, -1, -1],
        [0, 1, 1, -1],
        [0, 1, -1, 1],
        [0, -1, 1, 1],
        [0, 0, 0, 0],
    ];
    let mut valid = row(&0_f64, &tetrahedron, 1);
    for value in &mut valid[20..26] {
        value.re = 1.;
    }
    let mut bad_scale = valid.clone();
    bad_scale[25].re = -1.;
    let mut nonplanar = valid.clone();
    nonplanar[16].re = 1.;
    let mut bad_mass = valid.clone();
    bad_mass[20].im = 1.;
    let mut bad_shift = valid.clone();
    bad_shift[0].im = 1.;
    let input = KinematicsBatch {
        rows: 7,
        components: 1,
        values: [
            valid.clone(),
            bad_scale,
            nonplanar,
            valid.clone(),
            bad_mass,
            bad_shift,
            valid.clone(),
        ]
        .concat(),
    };
    let original = input.values.clone();
    let mut workspace = CertifiedPlanarWorkspace::new(&plan, 0_f64).unwrap();
    workspace.set_profiling(true);
    let result = workspace.reduce_batch(&input).unwrap();
    assert_eq!(input.values, original);
    assert_eq!(
        result.analytic_used,
        [true, false, false, true, false, false, true]
    );
    assert_eq!(result.fallback_used, [false; 7]);
    for lane in [0, 3, 6] {
        assert_eq!(result.statuses[lane], LaneStatus::Success);
    }
    for lane in [1, 2, 4, 5] {
        assert!(matches!(result.statuses[lane], LaneStatus::InvalidInput(_)));
        assert!(result.lane(lane).is_err());
    }
    assert!(
        matches!(&result.statuses[1],LaneStatus::InvalidInput(s) if s=="mu must be positive and real")
    );
    assert!(
        matches!(&result.statuses[4],LaneStatus::InvalidInput(s) if s=="squared masses must have nonpositive imaginary parts")
    );
    assert!(
        matches!(&result.statuses[5],LaneStatus::InvalidInput(s) if s=="momentum-shift and scalar-product primal components must be real")
    );
    assert_eq!(workspace.last_execution().input_rows, 7);
    assert_eq!(workspace.last_execution().analytic_rows, 3);
    assert_eq!(workspace.last_execution().argument_rows, 3);
    assert_eq!(workspace.last_execution().reduction_rows, 3);
    assert_eq!(workspace.last_certificate_execution().reduction_batches, 1);
    let profile = workspace.last_profile().unwrap();
    assert!(profile.total > std::time::Duration::ZERO);
    assert!(profile.arguments + profile.scalar_masters + profile.reduction <= profile.total);
    let expected = result.lane(0).unwrap().to_vec();
    workspace.set_accuracy_precision(52).unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    assert!(workspace.set_accuracy_precision(54).is_err());
    let reordered = KinematicsBatch {
        rows: 3,
        components: 1,
        values: [6, 0, 3]
            .into_iter()
            .flat_map(|lane| input.values[lane * 26..(lane + 1) * 26].iter().cloned())
            .collect(),
    };
    let repeated = workspace.reduce_batch(&reordered).unwrap();
    for lane in 0..3 {
        assert_eq!(repeated.lane(lane).unwrap(), expected);
    }
    assert_eq!(repeated.metadata(), result.metadata());
    assert_eq!(workspace.last_execution().input_rows, 3);
    assert_eq!(workspace.last_execution().analytic_rows, 3);
    let mut independent = workspace.clone();
    let empty = workspace
        .reduce_batch(&KinematicsBatch {
            rows: 0,
            components: 1,
            values: vec![],
        })
        .unwrap();
    assert!(empty.statuses.is_empty());
    assert_eq!(workspace.last_execution(), ExecutionStats::default());
    assert_eq!(
        workspace.last_certificate_execution(),
        CertificateExecutionStats::default()
    );
    assert!(
        workspace
            .reduce_batch(&KinematicsBatch {
                rows: 1,
                components: 1,
                values: vec![]
            })
            .is_err()
    );
    let error_profile = workspace.last_profile().unwrap();
    assert!(error_profile.total > std::time::Duration::ZERO);
    assert_eq!(error_profile.arguments, std::time::Duration::ZERO);
    assert_eq!(workspace.last_execution().master_rows, 0);
    let singleton = KinematicsBatch {
        rows: 1,
        components: 1,
        values: valid,
    };
    workspace.set_profiling(false);
    assert_eq!(
        workspace.reduce_batch(&singleton).unwrap().lane(0).unwrap(),
        expected
    );
    assert!(workspace.last_profile().is_none());
    assert_eq!(
        independent
            .reduce_batch(&singleton)
            .unwrap()
            .lane(0)
            .unwrap(),
        expected
    );
    assert_eq!(independent.last_execution().input_rows, 1);
}

#[test]
fn shared_input_validator_preserves_full_jet_check_order() {
    let plan = CertifiedPlanarReductionPlan::new(settings(0)).unwrap();
    let mut input = row(&0_f64, &points(), 6);
    input[25 * 6].re = -1.;
    input[20 * 6].im = 1.;
    input[0].im = 1.;
    input[4 * 6 + 5].re = f64::NAN;
    assert!(
        matches!(validate_input_row(&input,plan.input_layout(),6),Some(LaneStatus::InvalidInput(s)) if s=="nonfinite input at argument 4, Taylor component 5")
    );
    input[4 * 6 + 5].re = 0.;
    assert!(
        matches!(validate_input_row(&input,plan.input_layout(),6),Some(LaneStatus::InvalidInput(s)) if s=="mu must be positive and real")
    );
    input[25 * 6].re = 1.;
    assert!(
        matches!(validate_input_row(&input,plan.input_layout(),6),Some(LaneStatus::InvalidInput(s)) if s=="momentum-shift and scalar-product primal components must be real")
    );
    input[0].im = 0.;
    assert!(
        matches!(validate_input_row(&input,plan.input_layout(),6),Some(LaneStatus::InvalidInput(s)) if s=="squared masses must have nonpositive imaginary parts")
    );
    input[20 * 6].im = 0.;
    input[1].im = 1.; // Complex tangent components remain allowed.
    assert_eq!(validate_input_row(&input, plan.input_layout(), 6), None);
}

fn exact_components(values: &[Complex<Float>]) -> serde_json::Value {
    serde_json::Value::Array(
        values
            .iter()
            .map(|v| {
                serde_json::json!({
                    "real":v.re.exact_rational().unwrap().to_string(),
                    "imaginary":v.im.exact_rational().unwrap().to_string(),
                    "real_precision":v.re.get_precision(),
                    "imaginary_precision":v.im.get_precision()
                })
            })
            .collect(),
    )
}
fn represented_component(prototype: &Float, value: &serde_json::Value) -> Complex<Float> {
    let parse = |key: &str| {
        let expression = Atom::parse(
            value[key].as_str().unwrap(),
            "iregi_planar_artifact",
            Default::default(),
        )
        .unwrap();
        prototype.from_rational(&Rational::try_from(expression).unwrap())
    };
    Complex::new(parse("real"), parse("imaginary"))
}
fn write_precision_record(path: &std::path::Path, record: &serde_json::Value) {
    let temporary = path.with_extension("pending.json");
    std::fs::write(&temporary, serde_json::to_vec_pretty(record).unwrap()).unwrap();
    std::fs::rename(temporary, path).unwrap();
}

#[test]
#[ignore = "bounded rank-seven qualification; requires exact-output artifact paths"]
fn public_planar_rank7_precision_artifacts_and_convergence() {
    let bits: u32 = std::env::var("IREGI_PLANAR_PRECISION")
        .unwrap()
        .parse()
        .unwrap();
    assert!([128, 256].contains(&bits));
    let destination = std::path::PathBuf::from(std::env::var("IREGI_PLANAR_OUTPUT").unwrap());
    assert!(
        !destination.exists(),
        "preserve previous precision artifacts"
    );
    let prototype = Float::new(bits);
    let configuration = settings(7);
    let input = KinematicsBatch {
        rows: 1,
        components: 1,
        values: row(&prototype, &points(), 1),
    };
    let basis = tensor_basis(7);
    let mut record = serde_json::json!({"schema_version":1,"fixture":"planar-spatial-constant-one-v1","status":"started","working_precision":bits,"requested_precision":53,"components":1,"settings":configuration,"basis":basis.iter().map(|m|m.0).collect::<Vec<_>>(),"inputs":exact_components(&input.values)});
    write_precision_record(&destination, &record);
    let started = Instant::now();
    eprintln!("PLANAR_PRECISION bits={bits} plan START");
    let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
    let mut workspace = CertifiedPlanarWorkspace::new(&plan, prototype.clone()).unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    workspace.set_profiling(true);
    eprintln!(
        "PLANAR_PRECISION bits={bits} reduction START elapsed={:?}",
        started.elapsed()
    );
    let reduced = workspace.reduce_batch(&input).unwrap();
    eprintln!(
        "PLANAR_PRECISION bits={bits} reduction END elapsed={:?} statuses={:?}",
        started.elapsed(),
        reduced.statuses
    );
    assert_eq!(reduced.statuses, [LaneStatus::Success]);
    assert_eq!(reduced.basis, basis);
    assert_eq!(reduced.components, 1);
    assert_eq!(reduced.metadata().convention(), configuration.convention);
    assert_eq!(
        reduced.metadata().tensor_weights(),
        configuration.tensor_weights
    );
    assert_eq!(reduced.metadata().dual_shape(), None);
    let actual = reduced.lane(0).unwrap();
    assert_eq!(actual.len(), 1320);
    record["status"] = "primary_complete".into();
    record["values"] = exact_components(actual);
    record["primary_statuses"] = serde_json::json!(["Success"]);
    write_precision_record(&destination, &record);
    eprintln!(
        "PLANAR_PRECISION bits={bits} independent boxes START elapsed={:?}",
        started.elapsed()
    );
    let expected = box_identity(
        &configuration,
        prototype.clone(),
        53,
        &None,
        &input,
        &constant_weights(&prototype, 6, 1),
    );
    assert_eq!(expected.len(), 1320);
    eprintln!(
        "PLANAR_PRECISION bits={bits} independent boxes END elapsed={:?}",
        started.elapsed()
    );
    let tolerance = prototype.from_rational(&Rational::from((1, 10_000_000_000_i64)));
    let mut maximum = prototype.zero();
    for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
        let error = NativeFloat::hypot(&(a.re.clone() - &b.re), &(a.im.clone() - &b.im));
        let natural = dimensional_reference(&prototype, &input.values, 1, &basis[index / 4], b);
        let normalized = error / natural;
        assert!(
            normalized <= tolerance,
            "independent box normalized slot {index}"
        );
        if normalized > maximum {
            maximum = normalized;
        }
    }
    record["independent_box_values"] = exact_components(&expected);
    record["independent_box_statuses"] = serde_json::json!(vec!["Success"; 5]);
    record["independent_max_normalized_error"] =
        maximum.exact_rational().unwrap().to_string().into();
    record["independent_tolerance"] = "1/10000000000".into();
    record["status"] = "independent_compared".into();
    write_precision_record(&destination, &record);
    if bits == 256 {
        let previous_path =
            std::path::PathBuf::from(std::env::var("IREGI_PLANAR_PREVIOUS").unwrap());
        let previous: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&previous_path).unwrap()).unwrap();
        assert_eq!(previous["status"], "passed");
        assert_eq!(previous["working_precision"], 128);
        for key in [
            "fixture",
            "schema_version",
            "requested_precision",
            "settings",
            "basis",
            "components",
        ] {
            assert_eq!(previous[key], record[key], "precision artifact {key}");
        }
        let prior_inputs = previous["inputs"].as_array().unwrap();
        assert_eq!(prior_inputs.len(), input.values.len());
        for (a, b) in input.values.iter().zip(prior_inputs) {
            assert_eq!(*a, represented_component(&prototype, b));
        }
        let prior_values = previous["values"].as_array().unwrap();
        assert_eq!(prior_values.len(), 1320);
        let convergence = prototype.from_rational(&Rational::from((1, 2)).pow(65));
        let mut maximum = prototype.zero();
        for (index, (high, stored)) in actual.iter().zip(prior_values).enumerate() {
            let low = represented_component(&prototype, stored);
            let error = NativeFloat::hypot(&(low.re - &high.re), &(low.im - &high.im));
            let natural =
                dimensional_reference(&prototype, &input.values, 1, &basis[index / 4], high);
            let normalized = error / natural;
            assert!(normalized <= convergence, "128→256 normalized slot {index}");
            if normalized > maximum {
                maximum = normalized;
            }
        }
        record["convergence"] = serde_json::json!({"status":"passed","previous_artifact":previous_path,"previous_working_precision":128,"compared_complex_slots":1320,"max_normalized_error":maximum.exact_rational().unwrap().to_string(),"tolerance":Rational::from((1,2)).pow(65).to_string()});
    }
    record["status"] = "passed".into();
    record["input_rows"] = workspace.last_execution().input_rows.into();
    record["analytic_rows"] = workspace.last_execution().analytic_rows.into();
    record["reduction_rows"] = workspace.last_execution().reduction_rows.into();
    record["master_rows"] = workspace.last_execution().master_rows.into();
    record["certificate_batches"] = workspace
        .last_certificate_execution()
        .reduction_batches
        .into();
    write_precision_record(&destination, &record);
    eprintln!(
        "PLANAR_PRECISION bits={bits} COMPLETE elapsed={:?} artifact={}",
        started.elapsed(),
        destination.display()
    );
}

#[test]
#[ignore = "bounded rank-seven mixed qualification; see docs/research/certified-planar-api.md"]
fn public_planar_rank7_rectangular_mixed_artifact() {
    let destination = std::path::PathBuf::from(std::env::var("IREGI_PLANAR_MIXED_OUTPUT").unwrap());
    assert!(!destination.exists(), "preserve previous mixed artifacts");
    let shape = Some(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
    let components = 4;
    let prototype = Float::new(128);
    let mut configuration = settings(7);
    configuration.convention = Convention::MadLoop;
    configuration.tensor_weights = TensorWeights::Multinomial;
    let mut tangent = row(&prototype, &points(), components);
    tangent[4 * components + 2].re = prototype.one();
    tangent[8 * components + 1].re = prototype.one();
    tangent[16 * components + 2].re = prototype.one();
    tangent[16 * components + 1].re = prototype.one();
    for mass in 20..25 {
        tangent[mass * components + 3].re = prototype.one();
    }
    let mut transverse = tangent.clone();
    transverse[16 * components + 3].re = prototype.one();
    let input = KinematicsBatch {
        rows: 3,
        components,
        values: [tangent.clone(), transverse, tangent.clone()].concat(),
    };
    let basis = tensor_basis(7);
    let mut record = serde_json::json!({
        "schema_version":1,"fixture":"planar-spatial-rotating-rectangular-mixed-v1",
        "status":"started","working_precision":128,"requested_precision":53,
        "rows":3,"components":components,"shape":shape,"settings":configuration,
        "basis":basis.iter().map(|m|m.0).collect::<Vec<_>>(),"inputs":exact_components(&input.values)
    });
    write_precision_record(&destination, &record);
    let started = Instant::now();
    eprintln!("PLANAR_MIXED rank7 rectangular plan START");
    let plan = CertifiedPlanarReductionPlan::new(configuration.clone()).unwrap();
    let mut workspace = CertifiedPlanarWorkspace::new_dual(
        &plan,
        prototype.clone(),
        HyperDual::<u8>::new(shape.clone().unwrap()),
    )
    .unwrap();
    workspace.set_accuracy_precision(53).unwrap();
    workspace.set_profiling(true);
    eprintln!(
        "PLANAR_MIXED rank7 reduction START elapsed={:?}",
        started.elapsed()
    );
    let reduced = workspace.reduce_batch(&input).unwrap();
    eprintln!(
        "PLANAR_MIXED rank7 reduction END elapsed={:?} statuses={:?}",
        started.elapsed(),
        reduced.statuses
    );
    record["status"] = "primary_complete".into();
    record["primary_statuses"] = serde_json::json!(
        reduced
            .statuses
            .iter()
            .map(|s| format!("{s:?}"))
            .collect::<Vec<_>>()
    );
    record["returned_metadata"] = serde_json::json!({
        "convention":reduced.metadata().convention(),
        "tensor_weights":reduced.metadata().tensor_weights(),
        "dual_shape":reduced.metadata().dual_shape(),
        "components":reduced.components,
        "basis":reduced.basis.iter().map(|m|m.0).collect::<Vec<_>>()
    });
    record["lane_values"] = serde_json::Value::Array(
        (0..3)
            .map(|lane| {
                reduced
                    .lane(lane)
                    .map_or(serde_json::Value::Null, exact_components)
            })
            .collect(),
    );
    record["analytic_used"] = serde_json::json!(reduced.analytic_used);
    record["fallback_used"] = serde_json::json!(reduced.fallback_used);
    write_precision_record(&destination, &record);
    assert_eq!(reduced.basis, basis);
    assert_eq!(reduced.components, components);
    assert_eq!(reduced.metadata().convention(), configuration.convention);
    assert_eq!(
        reduced.metadata().tensor_weights(),
        configuration.tensor_weights
    );
    assert_eq!(reduced.metadata().dual_shape(), shape.as_deref());
    assert_eq!(reduced.statuses[0], LaneStatus::Success);
    assert_eq!(reduced.statuses[2], LaneStatus::Success);
    assert!(
        matches!(&reduced.statuses[1], LaneStatus::InvalidInput(message)
        if message.contains("Taylor component 3"))
    );
    assert!(reduced.lane(1).is_err());
    assert_eq!(reduced.analytic_used, [true, false, true]);
    assert_eq!(reduced.fallback_used, [false; 3]);
    assert_eq!(reduced.lane(0).unwrap(), reduced.lane(2).unwrap());
    assert_eq!(reduced.lane(0).unwrap().len(), 5280);
    // The independently known denominator identity is C(s,t)=6+2st.
    // Its inverse through this requested rectangular order is1/6-st/18.
    let mut coefficients = constant_weights(&prototype, 6, components);
    for (index, weight) in [2_i64, -1, -1, -1, 1].into_iter().enumerate() {
        coefficients[index * components + 3].re =
            prototype.from_rational(&Rational::from((-weight, 18)));
    }
    record["independent_coefficients"] = exact_components(&coefficients);
    write_precision_record(&destination, &record);
    eprintln!(
        "PLANAR_MIXED independent boxes START elapsed={:?}",
        started.elapsed()
    );
    let expected = box_identity(
        &configuration,
        prototype.clone(),
        53,
        &shape,
        &KinematicsBatch {
            rows: 1,
            components,
            values: tangent,
        },
        &coefficients,
    );
    eprintln!(
        "PLANAR_MIXED independent boxes END elapsed={:?}",
        started.elapsed()
    );
    assert_eq!(expected.len(), 5280);
    let tolerance = prototype.from_rational(&Rational::from((1, 10_000_000_000_i64)));
    let mut maximum = prototype.zero();
    for (index, (actual, reference)) in reduced.lane(0).unwrap().iter().zip(&expected).enumerate() {
        let error = NativeFloat::hypot(
            &(actual.re.clone() - &reference.re),
            &(actual.im.clone() - &reference.im),
        );
        let natural = dimensional_reference(
            &prototype,
            &input.values[..26 * components],
            components,
            &basis[index / (4 * components)],
            reference,
        );
        let normalized = error / natural;
        assert!(
            normalized <= tolerance,
            "rectangular mixed normalized slot {index}"
        );
        if normalized > maximum {
            maximum = normalized;
        }
    }
    assert_eq!(workspace.last_certificate_execution().reduction_batches, 1);
    assert_eq!(workspace.last_execution().input_rows, 3);
    assert_eq!(workspace.last_execution().analytic_rows, 2);
    assert_eq!(workspace.last_execution().reduction_rows, 2);
    record["independent_box_values"] = exact_components(&expected);
    record["independent_box_statuses"] = serde_json::json!(vec!["Success"; 5]);
    record["independent_max_normalized_error"] =
        maximum.exact_rational().unwrap().to_string().into();
    record["independent_tolerance"] = "1/10000000000".into();
    record["execution"] = serde_json::json!({
        "input_rows":workspace.last_execution().input_rows,
        "analytic_rows":workspace.last_execution().analytic_rows,
        "master_rows":workspace.last_execution().master_rows,
        "reduction_rows":workspace.last_execution().reduction_rows,
        "certificate_batches":workspace.last_certificate_execution().reduction_batches
    });
    record["status"] = "passed".into();
    write_precision_record(&destination, &record);
    eprintln!(
        "PLANAR_MIXED rank7 COMPLETE elapsed={:?} artifact={}",
        started.elapsed(),
        destination.display()
    );
}
