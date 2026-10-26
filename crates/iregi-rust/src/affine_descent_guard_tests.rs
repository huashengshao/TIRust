use super::*;

fn c(value: i64) -> Complex<Rational> {
    Complex::new(Rational::from(value), Rational::from(0))
}

fn fixture(components: usize) -> Vec<Complex<Rational>> {
    let mut row = vec![c(0); PARAMETERS * components];
    for (parameter, value) in [(5, 1), (9, 2), (12, 1), (13, 2), (14, 3), (15, 3)] {
        row[parameter * components] = c(value);
    }
    row
}

fn dual() -> HyperDual<u8> {
    // Deliberately keep mixed components in this explicitly ordered shape.
    HyperDual::new(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]])
}

#[test]
fn batch_guards_isolate_bad_chart_and_off_family_without_evaluating_children() {
    let plan = triangle_plan();
    assert_eq!(plan.parameters().len(), 16);
    let mut workspace = plan.instantiate();
    let good = fixture(1);
    let mut off = good.clone();
    off[10] = c(1);
    let mut singular = good.clone();
    singular[14] = c(1);
    let input = [good.clone(), off, singular, good].concat();
    let actual = workspace.select_batch_exact(&input, 4).unwrap();
    assert!(
        matches!(&actual[0], AffineDescentSelection::Chart { index: 1, inactive } if inactive.is_empty())
    );
    assert!(matches!(
        actual[1],
        AffineDescentSelection::OffAffineFamily { component: 0, .. }
    ));
    assert_eq!(actual[2], AffineDescentSelection::NoRegularChart);
    assert_eq!(actual[3], actual[0]);
    assert!(workspace.select_batch_exact(&[], 0).unwrap().is_empty());
}

#[test]
fn translated_routing_and_independent_mass_jets_are_accepted_at_all_orders() {
    let plan = dual_triangle_plan();
    let mut workspace = plan.instantiate();
    let mut row = fixture(4);
    for i in 0..3 {
        for mu in 0..4 {
            // A common routing translation, including a mixed translation,
            // must cancel before the exact full-Taylor rank test.
            row[(4 * i + mu) * 4 + 1] = c((mu + 1) as i64);
            row[(4 * i + mu) * 4 + 3] = c(-(mu as i64) - 2);
        }
    }
    // A tangent change of relative collinear spacing remains admissible.
    row[5 * 4 + 2] = c(1);
    row[9 * 4 + 2] = c(2);
    for mass in 12..15 {
        row[mass * 4 + 1] = c((mass - 10) as i64);
        row[mass * 4 + 2] = c(-1);
        row[mass * 4 + 3] = c((mass - 11) as i64);
    }
    assert_eq!(
        workspace.select_batch_exact(&row, 1).unwrap(),
        [AffineDescentSelection::Chart {
            index: 1,
            inactive: vec![]
        }]
    );
}

#[test]
fn transverse_and_mixed_off_chart_jets_are_rejected_even_with_valid_primal() {
    let plan = dual_triangle_plan();
    let mut workspace = plan.instantiate();
    let good = fixture(4);
    let mut transverse = good.clone();
    transverse[10 * 4 + 1] = c(1);
    let mut mixed = good.clone();
    mixed[10 * 4 + 3] = c(1);
    let actual = workspace
        .select_batch_exact(&[good, transverse, mixed].concat(), 3)
        .unwrap();
    assert!(
        matches!(&actual[0], AffineDescentSelection::Chart { index: 1, inactive } if inactive.is_empty())
    );
    assert!(matches!(
        actual[1],
        AffineDescentSelection::OffAffineFamily { component: 1, .. }
    ));
    assert!(matches!(
        actual[2],
        AffineDescentSelection::OffAffineFamily { component: 3, .. }
    ));
}

#[test]
fn a_nonzero_nilpotent_chart_never_authorizes_an_inverse() {
    let plan = dual_triangle_plan();
    let mut workspace = plan.instantiate();
    let mut row = fixture(4);
    row[14 * 4] = c(1); // Every chart primal vanishes.
    row[14 * 4 + 1] = c(1); // A higher component is nonzero but noninvertible.
    assert_eq!(
        workspace.select_batch_exact(&row, 1).unwrap(),
        [AffineDescentSelection::NoRegularChart]
    );
}

#[test]
fn empty_chunked_reordered_batches_and_axis_charts_preserve_lane_mapping() {
    let mut workspace = triangle_plan().instantiate();
    let mut rows = Vec::new();
    for axis in 0..4 {
        let mut row = fixture(1);
        row[5] = c(0);
        row[9] = c(0);
        row[4 + axis] = c(1);
        row[8 + axis] = c(2);
        rows.push(row);
    }
    let expected = workspace.select_batch_exact(&rows.concat(), 4).unwrap();
    for (axis, selection) in expected.iter().enumerate() {
        assert_eq!(
            *selection,
            AffineDescentSelection::Chart {
                index: axis,
                inactive: vec![]
            }
        );
    }
    for (row, expected) in rows.iter().zip(&expected) {
        assert_eq!(
            workspace.select_batch_exact(row, 1).unwrap(),
            [expected.clone()]
        );
    }
    let reordered = [rows[3].clone(), rows[0].clone(), rows[2].clone()].concat();
    assert_eq!(
        workspace.select_batch_exact(&reordered, 3).unwrap(),
        [
            expected[3].clone(),
            expected[0].clone(),
            expected[2].clone()
        ]
    );
}

fn generic_boundary<T: NativeFloat>(prototype: T) {
    let good = fixture(1);
    let mut invalid = good.clone();
    invalid[15] = c(-3);
    let values = [good.clone(), invalid, good]
        .concat()
        .iter()
        .map(|value| {
            Complex::new(
                prototype.from_rational(&value.re),
                prototype.from_rational(&value.im),
            )
        })
        .collect();
    let input = KinematicsBatch {
        rows: 3,
        components: 1,
        values,
    };
    let mut workspace = triangle_plan().instantiate();
    let actual = workspace.select_batch(&input).unwrap();
    assert_eq!(
        actual[0],
        AffineDescentSelection::Chart {
            index: 1,
            inactive: vec![]
        }
    );
    assert!(
        matches!(&actual[1], AffineDescentSelection::InvalidInput(reason) if reason.contains("mu must be positive"))
    );
    assert_eq!(actual[2], actual[0]);
}

#[test]
fn existing_generic_exact_conversion_handles_f64_doublefloat_and_float() {
    generic_boundary(0_f64);
    generic_boundary(symbolica::domains::float::DoubleFloat::from(0.0));
    generic_boundary(symbolica::domains::float::Float::new(256));
}

const PARAMETERS: usize = 16;

#[test]
fn rank_one_four_line_profile_selects_an_alternate_column_chart() {
    // The first x-chart uses relative columns[0,1] and is singular:
    // its f-row is proportional to its coordinate row. Column[0,2]
    // supplies a regular chart. The temporal charts are also singular.
    // The existing Matrix and exact full-Taylor guard derive every value.
    let plan = AffineDescentGuardPlan::new(4, 1, crate::pv::AffineDescentBudget::default()).unwrap();
    assert_eq!(plan.charts().len(), 12);
    assert_eq!(plan.charts()[3].coordinate_rows, [1]);
    assert_eq!(plan.charts()[3].relative_columns, [0, 1]);
    assert_eq!(plan.charts()[4].coordinate_rows, [1]);
    assert_eq!(plan.charts()[4].relative_columns, [0, 2]);
    let mut input = vec![c(0); 21];
    for (i, x) in [0, 1, 2, 3].into_iter().enumerate() {
        input[4 * i + 1] = c(x);
    }
    for (i, mass) in [1, 2, 1, 3].into_iter().enumerate() {
        input[16 + i] = c(mass);
    }
    input[20] = c(3);
    let selected = plan.instantiate().select_batch_exact(&input, 1).unwrap();
    assert_eq!(selected, [AffineDescentSelection::Chart { index: 4, inactive: vec![2] }]);
}

fn triangle_plan() -> AffineDescentGuardPlan {
    AffineDescentGuardPlan::new(3, 1, crate::pv::AffineDescentBudget::default()).unwrap()
}
fn dual_triangle_plan() -> AffineDescentGuardPlan {
    AffineDescentGuardPlan::new_dual(3, 1, crate::pv::AffineDescentBudget::default(), &dual())
        .unwrap()
}

#[test]
#[ignore="expensive N7/r4 symbolic guard stress; prior fixed90 timeout retained; run only with an explicit finite stress budget"]
fn a_vanishing_first_pivot_selects_an_alternate_generic_column_basis() {
    let plan =
        AffineDescentGuardPlan::new(7, 4, crate::pv::AffineDescentBudget::default()).unwrap();
    assert_eq!(plan.charts()[0].relative_columns, [0, 1, 2, 3, 4]);
    assert_eq!(plan.charts()[1].relative_columns, [0, 1, 2, 3, 5]);
    let mut row = vec![c(0); 36];
    for (i, p) in [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [2, 0, 0, 0],
        [0, 2, 0, 0],
    ]
    .iter()
    .enumerate()
    {
        for mu in 0..4 {
            row[4 * i + mu] = c(p[mu]);
        }
    }
    for (i, mass) in [1, 2, 3, 4, 5, 5, 7].into_iter().enumerate() {
        row[28 + i] = c(mass);
    }
    row[35] = c(3);
    let result = plan.instantiate().select_batch_exact(&row, 1).unwrap();
    assert!(
        matches!(&result[0], AffineDescentSelection::Chart { index:1, inactive }
        if inactive.contains(&5)),
        "failed first pivot is never inverted"
    );
}

#[test]
fn full_jet_zero_cofactors_are_masked_but_primal_only_zeros_are_retained() {
    let mut workspace = dual_triangle_plan().instantiate();
    let mut exact_zero = fixture(4);
    exact_zero[5 * 4] = c(0); // p1==p0, p2 remains distinct.
    // Arbitrary mass jets cannot change the exact repeated-routing relation.
    exact_zero[12 * 4 + 1] = c(1);
    exact_zero[13 * 4 + 3] = c(2);
    let mut primal_zero = exact_zero.clone();
    primal_zero[5 * 4 + 3] = c(1); // Full rank-one variety, but child2 cofactor has a mixed jet.
    let result = workspace
        .select_batch_exact(&[exact_zero, primal_zero].concat(), 2)
        .unwrap();
    assert!(
        matches!(&result[0],AffineDescentSelection::Chart {inactive,..} if inactive.contains(&2))
    );
    assert!(
        matches!(&result[1],AffineDescentSelection::Chart {inactive,..} if !inactive.contains(&2)),
        "a primal-only zero may not remove a child carrying a derivative"
    );
}

#[test]
fn cancellation_smaller_than_binary64_is_unresolved_then_proved_nonzero() {
    let mut workspace = triangle_plan().instantiate();
    let mut row = fixture(1);
    row[6] = c(1);
    row[10] = c(2);
    row[10].re += Rational::from((1, 2)).pow(300);
    workspace.set_precision_budget(128, 128).unwrap();
    assert!(matches!(
        workspace.select_batch_exact(&row, 1).unwrap()[0],
        AffineDescentSelection::Unresolved { working_bits: 128 }
    ));
    workspace.set_precision_budget(128, 1024).unwrap();
    assert!(matches!(
        workspace.select_batch_exact(&row, 1).unwrap()[0],
        AffineDescentSelection::OffAffineFamily { component: 0, .. }
    ));
}
