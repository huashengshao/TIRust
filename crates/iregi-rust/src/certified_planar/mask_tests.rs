// Private source-prototype tests. These check certificate routing and cache
// isolation; the separate retained N6 capture qualifies numerical integrals.
#[test]
fn cofactor_masks_remain_private_to_explicit_six_point_children() {
    let ordinary = CertifiedPlanarReductionPlan::new(settings(0)).unwrap();
    let child = CertifiedPlanarReductionPlan::new_for_six_point_child(settings(0)).unwrap();
    assert!(!ordinary.cofactor_masks);
    assert_eq!(ordinary.constraints.len(), 5);
    assert!(ordinary.chart(3, 8).is_err());
    assert!(child.cofactor_masks);
    assert_eq!(child.constraints.len(), 25);
    assert!(!Arc::ptr_eq(&ordinary.charts, &child.charts));
    assert!(!Arc::ptr_eq(&ordinary.certificates, &child.certificates));
}

fn retained_mask_row(omitted: usize, components: usize, mixed: bool) -> Vec<Complex<Rational>> {
    let points = [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
        [1, 0, 0, 0],
        [0, 2, 2, 0],
    ];
    let masses = [1, 2, 3, 4, 2, 3];
    let active: Vec<_> = (0..6).filter(|&line| line != omitted).collect();
    let zero = Complex::new(Rational::from(0), Rational::from(0));
    let mut row = vec![zero; 26 * components];
    for (local, &original) in active.iter().enumerate() {
        for mu in 0..4 {
            row[(4 * local + mu) * components].re = Rational::from((points[original][mu], 2));
        }
        row[(20 + local) * components].re = Rational::from(masses[original]);
    }
    row[25 * components].re = Rational::from(3);
    if mixed {
        assert_eq!(components, 4);
        let mu = match omitted {
            3 => 0,
            4 => 3,
            _ => unreachable!("the retained mask fixtures omit line 3 or 4"),
        };
        // The original last line is local line 4 in either selected pentagon.
        row[(16 + mu) * components + 3].re = Rational::from(1);
    }
    row
}

#[test]
fn planar_mask_certificates_keep_mixed_only_children_and_lane_order() {
    let shape = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
    let plan = CertifiedPlanarReductionPlan::new_for_six_point_child(settings(1)).unwrap();
    let mut workspace = CertifiedPlanarWorkspace::new_dual(
        &plan, 0_f64, HyperDual::<u8>::new(shape.clone()),
    ).unwrap();
    let rows = [
        retained_mask_row(3, 4, false),
        retained_mask_row(3, 4, true),
        retained_mask_row(4, 4, false),
        retained_mask_row(4, 4, true),
    ];
    let exact = rows.concat();
    let charts = [3, 3, 0, 0];
    let rank = vec![(0..4).map(|component| (0, component)).collect(); 4];
    assert_eq!(
        workspace.certificate.check_selected_batch(&exact, 4, &rank).unwrap(),
        vec![ExactZeroCertificate::Zero; 4],
    );
    let primal = charts.iter().map(|chart| vec![(1 + chart, 0)]).collect::<Vec<_>>();
    let chart_certificates = workspace.certificate.check_selected_batch(&exact, 4, &primal).unwrap();
    for (certificate, chart) in chart_certificates.into_iter().zip(charts) {
        assert_eq!(certificate, ExactZeroCertificate::Nonzero { constraint: 1 + chart, component: 0 });
    }
    assert_eq!(workspace.certify_inactive_masks(&exact, &charts).unwrap(), [8, 0, 8, 0]);
    assert_eq!(workspace.last_certificate_execution().passes, 5);
    assert_eq!(workspace.last_certificate_execution().evaluated_rows, 20);
    let order = [3, 0, 2, 1, 0];
    let reordered: Vec<_> = order.iter().flat_map(|&i| rows[i].iter().cloned()).collect();
    let reordered_charts: Vec<_> = order.iter().map(|&i| charts[i]).collect();
    let expected = [0, 8, 8, 0, 8];
    assert_eq!(workspace.certify_inactive_masks(&reordered, &reordered_charts).unwrap(), expected);
    let mut chunked = Vec::new();
    for range in [0..2, 2..5] {
        chunked.extend(workspace.certify_inactive_masks(
            &reordered[range.start * 104..range.end * 104],
            &reordered_charts[range],
        ).unwrap());
    }
    assert_eq!(chunked, expected);
    let mut independent = CertifiedPlanarWorkspace::new_dual(
        &plan, 0_f64, HyperDual::<u8>::new(shape),
    ).unwrap();
    assert_eq!(independent.certify_inactive_masks(&exact, &charts).unwrap(), [8, 0, 8, 0]);
    // Certificates are fresh through zero -> mixed-only nonzero -> empty -> zero.
    assert_eq!(workspace.certify_inactive_masks(&rows[0], &[3]).unwrap(), [8]);
    assert_eq!(workspace.certify_inactive_masks(&rows[1], &[3]).unwrap(), [0]);
    assert!(workspace.reduce_batch(&KinematicsBatch { rows: 0, components: 4, values: vec![] }).unwrap().statuses.is_empty());
    assert_eq!(workspace.last_certificate_execution().passes, 0);
    assert_eq!(workspace.last_certificate_execution().evaluated_rows, 0);
    assert!(workspace.last_chart_masks().is_empty());
    assert_eq!(workspace.certify_inactive_masks(&rows[0], &[3]).unwrap(), [8]);
}

#[test]
fn planar_mask_unresolved_proof_retains_child_and_harvests_attempts() {
    // A controlled polynomial isolates routing semantics from the Cramer
    // construction: only child 3 depends on m0-m1, all others are nonzero.
    let mut plan = CertifiedPlanarReductionPlan::new_for_six_point_child(settings(0)).unwrap();
    plan.constraints[5..].fill(Atom::num(1));
    plan.constraints[5 + 3 * 5 + 3] = &plan.parameters[20] - &plan.parameters[21];
    let mut workspace = CertifiedPlanarWorkspace::new(&plan, 0_f64).unwrap();
    workspace.set_certificate_precision_budget(128, 128).unwrap();
    let mut equal = retained_mask_row(3, 1, false);
    equal[20].re = Rational::from(1);
    equal[21].re = Rational::from(1);
    let mut tiny = equal.clone();
    tiny[20].re += Rational::from((1, 2)).pow(300);
    let mut nonzero = equal.clone();
    nonzero[20].re = Rational::from(2);
    let exact = [equal.clone(), tiny, nonzero].concat();
    assert_eq!(
        workspace.certificate.check_selected_batch(&exact, 3, &vec![vec![(23, 0)]; 3]).unwrap(),
        [ExactZeroCertificate::Zero, ExactZeroCertificate::Unresolved { working_bits: 128 },
         ExactZeroCertificate::Nonzero { constraint: 23, component: 0 }],
    );
    assert_eq!(workspace.certify_inactive_masks(&exact, &[3, 3, 3]).unwrap(), [8, 0, 0]);
    assert_eq!(workspace.last_certificate_execution().passes, 5);
    assert_eq!(workspace.last_certificate_execution().evaluated_rows, 15);
    let before = workspace.last_certificate_execution();
    assert!(workspace.certify_inactive_masks(&equal[..25], &[3]).is_err());
    // A rejected malformed pass performs no evaluation and cannot replay the
    // preceding certificate's counters or its zero-mask authorization.
    assert_eq!(workspace.last_certificate_execution().passes, before.passes);
    assert_eq!(workspace.last_certificate_execution().evaluated_rows, before.evaluated_rows);
    assert!(workspace.certify_inactive_masks(&equal, &[4]).is_err());
}

#[test]
fn planar_mask_full_mask_and_invalid_batch_reset_fail_closed() {
    let mut plan = CertifiedPlanarReductionPlan::new_for_six_point_child(settings(0)).unwrap();
    // This impossible cofactor family tests the explicit consistency guard.
    plan.constraints[5..].fill(Atom::Zero);
    let mut workspace = CertifiedPlanarWorkspace::new(&plan, 0_f64).unwrap();
    let exact = retained_mask_row(3, 1, false);
    let values = exact.iter().map(|v| Complex::new(0_f64.from_rational(&v.re), 0.)).collect();
    let result = workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 1, values }).unwrap();
    assert!(matches!(&result.statuses[0], LaneStatus::Unstable(message) if message.contains("every child is inactive")));
    assert!(result.lane(0).is_err());
    assert_eq!(workspace.last_chart_attempts(), [false]);
    assert_eq!(workspace.last_chart_masks(), [None]);
    assert_eq!(workspace.last_certificate_execution().passes, 7);
    assert_eq!(workspace.last_certificate_execution().evaluated_rows, 7);
    assert_eq!(workspace.last_certificate_execution().reduction_batches, 0);
    assert_eq!(workspace.last_execution().master_rows, 0);
    assert_eq!(result.analytic_used, [false]);
    assert!(workspace.charts.is_empty());
    assert!(workspace.reduce_batch(&KinematicsBatch { rows: 1, components: 2, values: vec![] }).is_err());
    assert!(workspace.last_chart_masks().is_empty());
    assert!(workspace.last_chart_attempts().is_empty());
    assert_eq!(workspace.last_certificate_execution().passes, 0);
    assert_eq!(workspace.last_execution().master_rows, 0);
}

#[test]
fn planar_mask_plan_cache_separates_masks_charts_and_ordered_shapes() {
    let plan = CertifiedPlanarReductionPlan::new_for_six_point_child(settings(0)).unwrap();
    let unmasked = plan.chart(3, 0).unwrap();
    let masked = plan.chart(3, 8).unwrap();
    assert!(!Arc::ptr_eq(&unmasked, &masked));
    assert!(Arc::ptr_eq(&masked, &plan.chart(3, 8).unwrap()));
    assert!(Arc::ptr_eq(&masked, &plan.clone().chart(3, 8).unwrap()));
    assert!(!Arc::ptr_eq(&masked, &plan.chart(0, 8).unwrap()));
    assert!(plan.chart(4, 0).is_err());
    for invalid in [31, 32, 63, 128, 255] {
        assert!(plan.chart(3, invalid).is_err());
        assert!(ReductionPlan::new_certified_planar_masked(settings(0), 3, invalid).is_err());
    }
    let first = Some(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);
    let second = Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]);
    let certificate = plan.certificate(&first).unwrap();
    assert!(Arc::ptr_eq(&certificate, &plan.certificate(&first).unwrap()));
    assert!(!Arc::ptr_eq(&certificate, &plan.certificate(&second).unwrap()));
}
