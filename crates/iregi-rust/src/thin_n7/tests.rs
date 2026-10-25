//! Independent exact indexing, lifecycle and cheap constructor/backend checks.
//! The constructor check prepares N1/N2 master evaluators without reducing
//! its deliberately synthetic wrapper; these tests do not evaluate N7 integrals.
//! Symbolica performs all expression arithmetic and Taylor multiplication.

use super::*;
use crate::{
    basis::tensor_basis,
    plan::{apply_final_uv_power_counting, assemble_thin_seven_moments},
    pv::{ThinSevenChart, ThinSevenGeometry, ThinSixGeometry},
    types::{PrimalEnvelope, PrimalEnvelopeRoute, PrimalOperand, PrimalOperandKind},
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        dual::HyperDual,
        rational::{Q, Rational},
    },
    evaluate::{Dualizer, ExpressionEvaluator},
};

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("thin_seven_test::{name}")))
}

fn synthetic_geometry(omitted: usize) -> ThinSevenGeometry {
    let active: Vec<_> = (0..7).collect();
    let weights = std::array::from_fn(|column| {
        (0..7)
            .map(|line| {
                if line == omitted + 1 {
                    Atom::Zero
                } else {
                    Atom::num((1 + column * 7 + line) as i64)
                }
            })
            .collect()
    });
    let six = (0..7)
        .map(|outer| ThinSixGeometry {
            active: active
                .iter()
                .copied()
                .filter(|&line| line != outer)
                .collect(),
            determinant: Atom::num(1),
            weights: Some(std::array::from_fn(|column| {
                (0..6)
                    .map(|local| Atom::num((1 + (outer * 5 + column) * 6 + local) as i64))
                    .collect()
            })),
        })
        .collect();
    ThinSevenGeometry {
        active,
        charts: vec![ThinSevenChart {
            omitted,
            determinant: Atom::num(1),
            weights,
        }],
        six,
    }
}

// Scan the defining exponent relation instead of mutating exponents using the
// implementation's two-step dispatch. A sorted list of Cartesian indices fixes
// the outer and inner columns independently, including scalar and rank one.
fn expected_child(parent: Monomial, children: &[Monomial]) -> (usize, usize, usize) {
    let indices: Vec<_> = parent
        .0
        .iter()
        .enumerate()
        .flat_map(|(axis, &power)| std::iter::repeat_n(axis, usize::from(power)))
        .collect();
    let outer = indices.first().copied();
    let inner = indices.get(1).copied();
    let matches: Vec<_> = children
        .iter()
        .enumerate()
        .filter_map(|(index, child)| {
            (0..4)
                .all(|axis| {
                    child.0[axis] + u16::from(outer == Some(axis)) + u16::from(inner == Some(axis))
                        == parent.0[axis]
                })
                .then_some(index)
        })
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "unique twice-lowered child for {parent:?}"
    );
    (
        outer.map_or(0, |axis| axis + 1),
        inner.map_or(0, |axis| axis + 1),
        matches[0],
    )
}

// Pair-major oracle: each canonical N5 subset has two incoming N6 paths.
// Assembly is expected to be outer-child-major, so this exercises a different
// indexing traversal and distinguishes both paths without a hard-coded stride.
fn expected_slot(
    parent: Monomial,
    children: &[Monomial],
    geometry: &ThinSevenGeometry,
    chart: &ThinSevenChart,
    values: &[Atom],
    pole: usize,
) -> Atom {
    let (outer_column, inner_column, child) = expected_child(parent, children);
    let slots = children.len() * 4;
    let mut terms = Vec::new();
    for (subset, retained) in five_subsets().iter().enumerate() {
        let omitted: Vec<_> = geometry
            .active
            .iter()
            .copied()
            .filter(|line| !retained.contains(line))
            .collect();
        assert_eq!(omitted.len(), 2);
        for (&outer, &inner) in [
            (omitted.first().unwrap(), omitted.last().unwrap()),
            (omitted.last().unwrap(), omitted.first().unwrap()),
        ] {
            let outer_index = geometry
                .active
                .iter()
                .position(|&line| line == outer)
                .unwrap();
            let outer_weight = &chart.weights[outer_column][outer_index];
            if outer_weight.is_zero() {
                continue;
            }
            let six = &geometry.six[outer_index];
            let local_inner = six.active.iter().position(|&line| line == inner).unwrap();
            let inner_weight = &six.weights.as_ref().expect("demanded synthetic geometry")
                [inner_column][local_inner];
            terms.push(outer_weight * inner_weight * &values[subset * slots + child * 4 + pole]);
        }
    }
    Atom::add_many(&terms)
}

fn assemble(
    basis: &[Monomial],
    children: &[Monomial],
    geometry: &ThinSevenGeometry,
    values: &[Atom],
) -> Result<Vec<[Atom; 4]>, String> {
    assemble_thin_seven_moments(basis, children, geometry, values, &mut |value| value)
}

#[test]
fn seven_point_subsets_have_canonical_unique_original_line_identity() {
    let expected = vec![
        [2, 3, 4, 5, 6],
        [1, 3, 4, 5, 6],
        [1, 2, 4, 5, 6],
        [1, 2, 3, 5, 6],
        [1, 2, 3, 4, 6],
        [1, 2, 3, 4, 5],
        [0, 3, 4, 5, 6],
        [0, 2, 4, 5, 6],
        [0, 2, 3, 5, 6],
        [0, 2, 3, 4, 6],
        [0, 2, 3, 4, 5],
        [0, 1, 4, 5, 6],
        [0, 1, 3, 5, 6],
        [0, 1, 3, 4, 6],
        [0, 1, 3, 4, 5],
        [0, 1, 2, 5, 6],
        [0, 1, 2, 4, 6],
        [0, 1, 2, 4, 5],
        [0, 1, 2, 3, 6],
        [0, 1, 2, 3, 5],
        [0, 1, 2, 3, 4],
    ];
    assert_eq!(five_subsets(), expected);
    assert_eq!(
        expected
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        21
    );
    for line in 0..7 {
        assert_eq!(
            expected
                .iter()
                .filter(|subset| subset.contains(&line))
                .count(),
            15
        );
    }
}

#[test]
fn seven_point_all_ranks_all_four_slots_follow_two_lowerings_and_both_paths() {
    let geometry = synthetic_geometry(5);
    let mut checked = 0;
    for rank in 0_u16..=7 {
        let basis = tensor_basis(rank);
        let children = tensor_basis(rank.saturating_sub(2));
        let slots = children.len() * 4;
        let values: Vec<_> = (0..21 * slots)
            .map(|slot| variable(&format!("all_slots_{rank}_{slot}")))
            .collect();
        let actual = assemble(&basis, &children, &geometry, &values).unwrap();
        assert_eq!(actual.len(), basis.len());
        for (parent, output) in basis.iter().zip(&actual) {
            for pole in 0..4 {
                let expected = expected_slot(
                    *parent,
                    &children,
                    &geometry,
                    &geometry.charts[0],
                    &values,
                    pole,
                );
                assert!(
                    (&output[pole] - expected).expand().is_zero(),
                    "rank={rank}, parent={parent:?}, Laurent slot={pole}"
                );
                checked += 1;
            }
        }
        let reversed: Vec<_> = children.iter().copied().rev().collect();
        let mut reordered = Vec::with_capacity(values.len());
        for subset in 0..21 {
            for child in (0..children.len()).rev() {
                reordered.extend_from_slice(
                    &values[subset * slots + child * 4..subset * slots + child * 4 + 4],
                );
            }
        }
        assert_eq!(
            assemble(&basis, &reversed, &geometry, &reordered).unwrap(),
            actual,
            "child records follow exponents rather than an implicit basis order"
        );
    }
    assert_eq!(checked, 3168);
}

#[test]
fn seven_point_each_chart_keeps_origin_and_omitted_column_mapping() {
    let basis = tensor_basis(3);
    let children = tensor_basis(1);
    let values: Vec<_> = (0..21 * children.len() * 4)
        .map(|slot| variable(&format!("chart_slot_{slot}")))
        .collect();
    for omitted in 0..6 {
        let geometry = synthetic_geometry(omitted);
        let actual = assemble(&basis, &children, &geometry, &values).unwrap();
        for (parent, output) in basis.iter().zip(actual) {
            for pole in 0..4 {
                let expected = expected_slot(
                    *parent,
                    &children,
                    &geometry,
                    &geometry.charts[0],
                    &values,
                    pole,
                );
                assert!(
                    (&output[pole] - expected).expand().is_zero(),
                    "omitted={omitted}, parent={parent:?}, pole={pole}"
                );
            }
        }
    }
}

#[test]
fn seven_point_unavailable_inner_geometry_is_not_silent_zero() {
    let basis = tensor_basis(2);
    let children = tensor_basis(0);
    let values: Vec<_> = (0..84).map(|slot| Atom::num((slot + 1) as i64)).collect();
    let mut geometry = synthetic_geometry(5);
    let expected = assemble(&basis, &children, &geometry, &values).unwrap();
    // The omitted outer child is structurally absent in every RHS column.
    geometry.six[6].determinant = Atom::Zero;
    geometry.six[6].weights = None;
    assert_eq!(
        assemble(&basis, &children, &geometry, &values).unwrap(),
        expected
    );
    geometry.six[2].determinant = Atom::Zero;
    geometry.six[2].weights = None;
    assert!(
        assemble(&basis, &children, &geometry, &values).is_err(),
        "an active unavailable inverse must not become zero data"
    );
}

#[test]
fn seven_point_complete_chart_if_keeps_inactive_inner_poles_lazy_with_duals() {
    let x = variable("lazy_chart");
    let z = variable("inactive_inner_denominator");
    let basis = tensor_basis(2);
    let children = tensor_basis(0);
    let values: Vec<_> = (0..84).map(|slot| Atom::num((slot + 1) as i64)).collect();
    let mut geometry = synthetic_geometry(5);
    geometry.charts[0].determinant = x.clone();
    for column in &mut geometry.charts[0].weights {
        for weight in column {
            *weight = &*weight / &x;
        }
    }
    for column in geometry.six[1].weights.as_mut().unwrap() {
        for weight in column {
            *weight = &*weight / &z;
        }
    }
    // The fallback chart omits line1, so its complete expression does not use
    // the singular six-line inverse. This is an evaluator-routing witness,
    // not a claim that the synthetic chart geometry is physically realizable.
    geometry.charts.push(synthetic_geometry(0).charts.remove(0));
    let mut aliases = Vec::new();
    let actual = assemble_thin_seven_moments(&basis, &children, &geometry, &values, &mut |body| {
        let name = variable(&format!("lazy_inner_{}", aliases.len()));
        aliases.push((name.clone(), body));
        name
    })
    .unwrap();
    let mut expected = Vec::new();
    for parent in &basis {
        for pole in 0..4 {
            expected.push(expected_slot(
                *parent,
                &children,
                &geometry,
                &geometry.charts[1],
                &values,
                pole,
            ));
        }
    }
    let count = expected.len();
    let expressions: Vec<_> = actual.into_iter().flatten().chain(expected).collect();
    let exact: ExpressionEvaluator<Complex<Rational>> =
        Atom::evaluator_multiple(&expressions, &[x, z])
            .add_aliases(aliases)
            .unwrap()
            .direct_translation(true)
            .cores(1)
            .horner_iterations(0)
            .build()
            .unwrap();
    let dual = exact
        .vectorize(&Dualizer::new(
            HyperDual::<f64>::new(vec![vec![0], vec![1]]),
            vec![],
        ))
        .unwrap();
    let mut evaluator = dual.map_coeff(&|coefficient| f64::from(coefficient.re.clone()));
    let mut output = vec![0.; count * 4];
    evaluator.evaluate(&[0., 1., 0., 0.], &mut output);
    assert!(output.iter().all(|value| value.is_finite()));
    assert_eq!(&output[..count * 2], &output[count * 2..]);
    for slot in 0..count {
        assert!(output[slot * 2] > 0.);
        assert_eq!(output[slot * 2 + 1], 0.);
    }
}

#[test]
fn seven_point_assembly_rejects_malformed_geometry_and_child_layouts() {
    let basis = tensor_basis(3);
    let children = tensor_basis(1);
    let geometry = synthetic_geometry(5);
    let values = vec![Atom::num(1); 21 * children.len() * 4];
    for fault in 0..12 {
        let mut bad_geometry = geometry.clone();
        let mut bad_children = children.clone();
        let mut bad_values = values.clone();
        match fault {
            0 => bad_geometry.active.swap(0, 1),
            1 => {
                bad_geometry.six.pop();
            }
            2 => bad_geometry.charts.clear(),
            3 => {
                bad_values.pop();
            }
            4 => bad_children[1] = bad_children[0],
            5 => bad_geometry.six[2].active.swap(0, 1),
            6 => {
                bad_geometry.six[2].weights.as_mut().unwrap()[0].pop();
            }
            7 => bad_geometry.charts.push(bad_geometry.charts[0].clone()),
            8 => bad_geometry.charts[0].omitted = 6,
            9 => {
                bad_geometry.charts[0].weights[0].pop();
            }
            10 => bad_geometry.charts[0].weights[0][6] = Atom::num(1),
            11 => {
                bad_children = tensor_basis(0);
                bad_values = vec![Atom::num(1); 84];
            }
            _ => unreachable!(),
        }
        assert!(
            assemble(&basis, &bad_children, &bad_geometry, &bad_values).is_err(),
            "malformed assembly control {fault}"
        );
    }
    assert!(assemble(&basis, &[], &geometry, &[]).is_err());
}

#[test]
fn seven_point_synthetic_uv_cancellation_occurs_before_the_final_uv_rule() {
    let a = variable("uv_outer");
    let b = variable("uv_inner_b");
    let c = variable("uv_inner_c");
    let x = variable("uv_child_x");
    let basis = tensor_basis(0);
    let children = tensor_basis(0);
    let mut geometry = synthetic_geometry(5);
    for column in &mut geometry.charts[0].weights {
        column.fill(Atom::Zero);
        column[0] = a.clone();
    }
    for column in geometry.six[0].weights.as_mut().unwrap() {
        column.fill(Atom::Zero);
        column[0] = b.clone(); // second omitted line1 -> canonical subset0
        column[1] = c.clone(); // second omitted line2 -> canonical subset1
    }
    let mut values: Vec<_> = (0..84)
        .map(|slot| variable(&format!("uv_other_slot_{slot}")))
        .collect();
    values[0] = &c * &x;
    values[4] = -&b * &x;
    assert!(!values[0].is_zero() && !values[4].is_zero());
    let actual = assemble(&basis, &children, &geometry, &values).unwrap();
    assert!(
        actual[0][0].expand().is_zero(),
        "synthetic nonzero UV operands cancel algebraically"
    );
    for pole in 1..4 {
        let expected = &a * (&b * &values[pole] + &c * &values[4 + pole]);
        assert!((&actual[0][pole] - expected).expand().is_zero());
        assert!(!actual[0][pole].is_zero());
    }
    let mut noncanceling = values.clone();
    noncanceling[4] = &b * &x;
    let before_rule = assemble(&basis, &children, &geometry, &noncanceling).unwrap();
    assert!(
        !before_rule[0][0].expand().is_zero(),
        "assembly must not hide the negative control by applying the final UV rule early"
    );
    let mut after_rule = before_rule.clone();
    apply_final_uv_power_counting(&ReductionSettings::new(7, 0), &basis, &mut after_rule).unwrap();
    assert!(after_rule[0][0].is_zero());
    assert_eq!(&after_rule[0][1..], &before_rule[0][1..]);
}

#[test]
fn seven_point_mixed_only_outer_and_two_level_weights_keep_all_taylor_terms() {
    let x = variable("mixed_x");
    let y = variable("mixed_y");
    let mut actual_roots = Vec::new();
    let mut expected_roots = Vec::new();
    for mode in 0..2 {
        let mut geometry = synthetic_geometry(5);
        for column in &mut geometry.charts[0].weights {
            for weight in column {
                *weight = if mode == 0 {
                    &*weight * &x
                } else {
                    &*weight * &x * &y
                };
            }
        }
        if mode == 0 {
            for six in &mut geometry.six {
                for column in six.weights.as_mut().unwrap() {
                    for weight in column {
                        *weight = &*weight * &y;
                    }
                }
            }
        }
        for rank in 0_u16..=7 {
            let basis = tensor_basis(rank);
            let children = tensor_basis(rank.saturating_sub(2));
            let values: Vec<_> = (0..21 * children.len() * 4)
                .map(|slot| Atom::num((slot + 1) as i64))
                .collect();
            let actual = assemble(&basis, &children, &geometry, &values).unwrap();
            for (index, parent) in basis.iter().enumerate().filter(|(_, parent)| {
                parent.rank() == usize::from(rank)
                    && (parent.0.iter().filter(|&&power| power != 0).count() <= 1
                        || parent.0 == [rank.saturating_sub(1), 0, 1, 0])
            }) {
                for pole in 0..4 {
                    actual_roots.push(actual[index][pole].clone());
                    expected_roots.push(expected_slot(
                        *parent,
                        &children,
                        &geometry,
                        &geometry.charts[0],
                        &values,
                        pole,
                    ));
                }
            }
        }
    }
    let roots = actual_roots.len();
    let expressions: Vec<_> = actual_roots.into_iter().chain(expected_roots).collect();
    let exact: ExpressionEvaluator<Complex<Rational>> =
        Atom::evaluator_multiple(&expressions, &[x, y])
            .direct_translation(true)
            .cores(1)
            .horner_iterations(0)
            .build()
            .unwrap();
    for shape in [
        vec![
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![2, 0],
            vec![1, 1],
            vec![0, 2],
        ],
        vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 1],
            vec![2, 0],
        ],
    ] {
        let components = shape.len();
        let x_index = shape.iter().position(|power| power == &[1, 0]).unwrap();
        let y_index = shape.iter().position(|power| power == &[0, 1]).unwrap();
        let mixed_index = shape.iter().position(|power| power == &[1, 1]).unwrap();
        let mut evaluator = exact
            .clone()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(shape),
                vec![],
            ))
            .unwrap()
            .map_to_ring(&Q)
            .unwrap();
        let mut inputs = vec![Rational::from(0); 2 * components];
        inputs[x_index] = Rational::from(1);
        inputs[components + y_index] = Rational::from(1);
        let mut output = vec![Rational::from(0); 2 * roots * components];
        evaluator.evaluate_in_ring(&inputs, &mut output, &Q);
        assert_eq!(&output[..roots * components], &output[roots * components..]);
        for root in 0..roots {
            for component in 0..components {
                if component == mixed_index {
                    assert!(
                        output[root * components + component] > Rational::from(0),
                        "nonzero mixed-only contribution root={root}"
                    );
                } else {
                    assert_eq!(output[root * components + component], Rational::from(0));
                }
            }
        }
    }
}

#[test]
fn seven_point_final_uv_rule_changes_only_uv_and_retains_logarithmic_boundary() {
    for rank in [0_u16, 1, 2, 3, 4, 5, 6, 7, 10] {
        let settings = ReductionSettings::new(7, rank);
        let basis = tensor_basis(rank);
        let before: Vec<[Atom; 4]> = basis
            .iter()
            .enumerate()
            .map(|(moment, _)| {
                std::array::from_fn(|pole| Atom::num((4 * moment + pole + 1) as i64))
            })
            .collect();
        let mut after = before.clone();
        apply_final_uv_power_counting(&settings, &basis, &mut after).unwrap();
        for (index, monomial) in basis.iter().enumerate() {
            if monomial.rank() < 10 {
                assert!(after[index][0].is_zero());
            } else {
                assert_eq!(after[index][0], before[index][0]);
            }
            assert_eq!(&after[index][1..], &before[index][1..]);
        }
    }
}

fn successful_children(parent_rows: usize) -> BatchResult<f64> {
    let basis = tensor_basis(1);
    let components = 4;
    let metadata = ResultMetadata::new(
        Convention::IregiNative,
        TensorWeights::Components,
        Some(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]),
    );
    let rows = 21 * parent_rows;
    let outputs = basis.len() * 4;
    let values: Vec<_> = (0..rows * outputs * components)
        .map(|index| Complex::new((index + 1) as f64, -(index as f64)))
        .collect();
    let primal_envelopes = (0..rows)
        .map(|row| {
            Some(PrimalEnvelope {
                basis: basis.clone(),
                metadata: metadata.clone(),
                components,
                working_precision: 53,
                accuracy_bits: 53,
                operands: (0..outputs)
                    .map(|index| PrimalOperand {
                        output_index: index,
                        amplitude: Complex::new((100_000 + index) as f64, 0.),
                        kind: PrimalOperandKind::UncancelledMasters,
                    })
                    .collect(),
                output_primals: (0..outputs)
                    .map(|index| values[(row * outputs + index) * components])
                    .collect(),
                routes: vec![PrimalEnvelopeRoute::Ordinary],
            })
        })
        .collect();
    BatchResult {
        basis,
        components,
        metadata,
        values,
        statuses: vec![LaneStatus::Success; rows],
        fallback_used: vec![false; rows],
        analytic_used: vec![false; rows],
        primal_envelopes,
        primal_envelope_errors: vec![None; rows],
    }
}

#[test]
fn seven_point_mandatory_children_preserve_subset_lane_failure_and_route_flags() {
    let clean = successful_children(2);
    for compact in 0..2 {
        assert!(
            inspect_required_children(&clean, 2, compact, 53, 53)
                .failure
                .is_none()
        );
    }
    for subset in 0..21 {
        let mut failed = clean.clone();
        let row = subset * 2 + 1;
        failed.statuses[row] = LaneStatus::MasterFailure("required child witness".into());
        failed.fallback_used[41] = true;
        failed.analytic_used[39] = true;
        let state = inspect_required_children(&failed, 2, 1, 53, 53);
        let message = state
            .failure
            .expect("every one of the 21 children is mandatory");
        assert!(message.contains(&format!("subset {subset}")), "{message}");
        assert!(
            state.fallback_used && state.analytic_used,
            "later child route flags must survive an earlier child failure"
        );
        assert!(
            inspect_required_children(&failed, 2, 0, 53, 53)
                .failure
                .is_none(),
            "subset-major mapping must isolate the neighboring parent lane"
        );
    }
}

#[test]
fn seven_point_child_success_without_current_actual_route_envelope_is_rejected() {
    let clean = successful_children(2);
    for fault in 0..7 {
        let mut malformed = clean.clone();
        let row = 10 * 2 + 1;
        match fault {
            0 => malformed.primal_envelopes[row] = None,
            1 => {
                malformed.primal_envelopes[row]
                    .as_mut()
                    .unwrap()
                    .working_precision = 128
            }
            2 => {
                malformed.primal_envelopes[row]
                    .as_mut()
                    .unwrap()
                    .accuracy_bits = 52
            }
            3 => malformed.primal_envelopes[row]
                .as_mut()
                .unwrap()
                .routes
                .clear(),
            4 => {
                malformed.primal_envelopes[row]
                    .as_mut()
                    .unwrap()
                    .output_primals[0]
                    .re += 1.
            }
            5 => malformed.primal_envelopes[row].as_mut().unwrap().operands[0].output_index = 1,
            6 => malformed.primal_envelope_errors[row] = Some("unsupported actual route".into()),
            _ => unreachable!(),
        }
        assert_eq!(malformed.statuses[row], LaneStatus::Success);
        let state = inspect_required_children(&malformed, 2, 1, 53, 53);
        let message = state
            .failure
            .expect("successful child still needs a valid envelope");
        assert!(message.contains("subset 10"), "{message}");
        assert!(
            inspect_required_children(&malformed, 2, 0, 53, 53)
                .failure
                .is_none()
        );
    }
}

#[test]
fn seven_point_complete_layout_binds_ordered_shape_and_all_lane_buffers() {
    let clean = successful_children(2);
    let basis = clean.basis.clone();
    let metadata = clean.metadata.clone();
    validate_result_layout(&clean, &basis, 4, &metadata, 42).unwrap();
    for fault in 0..17 {
        let mut malformed = clean.clone();
        match fault {
            0 => {
                malformed.values.pop();
            }
            1 => malformed.values.push(Complex::new(0., 0.)),
            2 => {
                malformed.statuses.pop();
            }
            3 => malformed.statuses.push(LaneStatus::Success),
            4 => {
                malformed.fallback_used.pop();
            }
            5 => malformed.fallback_used.push(false),
            6 => {
                malformed.analytic_used.pop();
            }
            7 => malformed.analytic_used.push(false),
            8 => {
                malformed.primal_envelopes.pop();
            }
            9 => malformed.primal_envelopes.push(None),
            10 => {
                malformed.primal_envelope_errors.pop();
            }
            11 => malformed.primal_envelope_errors.push(None),
            12 => malformed.basis[0] = Monomial([1, 0, 0, 0]),
            13 => malformed.components = 3,
            14 => {
                malformed.metadata = ResultMetadata::new(
                    Convention::OneLoop,
                    TensorWeights::Components,
                    metadata.dual_shape().map(<[Vec<usize>]>::to_vec),
                )
            }
            15 => {
                malformed.metadata = ResultMetadata::new(
                    Convention::IregiNative,
                    TensorWeights::Multinomial,
                    metadata.dual_shape().map(<[Vec<usize>]>::to_vec),
                )
            }
            16 => {
                malformed.metadata = ResultMetadata::new(
                    Convention::IregiNative,
                    TensorWeights::Components,
                    Some(vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]]),
                )
            }
            _ => unreachable!(),
        }
        assert!(
            validate_result_layout(&malformed, &basis, 4, &metadata, 42).is_err(),
            "malformed complete layout {fault}"
        );
    }
    assert!(validate_result_layout(&clean, &basis, usize::MAX, &metadata, 42).is_err());
    assert!(validate_result_layout(&clean, &basis, 4, &metadata, usize::MAX).is_err());
    let mut empty = clean.clone();
    empty.values.clear();
    empty.statuses.clear();
    empty.fallback_used.clear();
    empty.analytic_used.clear();
    empty.primal_envelopes.clear();
    empty.primal_envelope_errors.clear();
    validate_result_layout(&empty, &basis, 4, &metadata, 0).unwrap();
}

// Deliberately substitute cheap tadpole scratch. Every call below exits before
// dispatching child or parent numerical work. This isolates the N7 whole-buffer
// lifecycle from expensive symbolic N7 preparation and from scalar evaluation.
fn validation_only_workspace() -> ThinSevenWorkspace<f64> {
    let plan = ReductionPlan::new(ReductionSettings::new(1, 0)).unwrap();
    let scratch = Workspace::new(&plan, 0.).unwrap();
    ThinSevenWorkspace {
        prototype: 0.,
        accuracy_bits: 53,
        components: 1,
        metadata: ResultMetadata::new(Convention::IregiNative, TensorWeights::Components, None),
        basis: tensor_basis(0),
        child_basis: tensor_basis(0),
        child: scratch.clone(),
        parent: scratch,
        stats: SevenPointExecutionStats::default(),
        diagnostics: Vec::new(),
    }
}

fn seed_previous_call(workspace: &mut ThinSevenWorkspace<f64>) {
    workspace.stats.parent_input_rows = 9;
    workspace.stats.child_batch_calls = 1;
    workspace.stats.packed_child_rows = 189;
    workspace.stats.parent_batch_calls = 1;
    workspace.stats.packed_parent_rows = 9;
    workspace.stats.child_execution.master_rows = 945;
    workspace.stats.parent_execution.reduction_rows = 9;
    workspace.stats.child_certificate_execution.input_rows = 7;
    workspace.stats.parent_certificate_execution.evaluated_rows = 3;
    workspace.diagnostics = vec![vec!["previous call".into()]; 9];
}

fn assert_no_dispatch(workspace: &ThinSevenWorkspace<f64>, rows: usize) {
    let stats = workspace.last_execution();
    assert_eq!(stats.parent_input_rows, rows);
    assert_eq!(stats.child_batch_calls, 0);
    assert_eq!(stats.packed_child_rows, 0);
    assert_eq!(stats.parent_batch_calls, 0);
    assert_eq!(stats.packed_parent_rows, 0);
    assert_eq!(stats.child_execution, ExecutionStats::default());
    assert_eq!(stats.parent_execution, ExecutionStats::default());
    assert_eq!(stats.child_certificate_execution, CertificateExecutionStats::default());
    assert_eq!(stats.parent_certificate_execution, CertificateExecutionStats::default());
}

#[test]
fn seven_point_empty_and_malformed_calls_reset_all_owned_call_state() {
    let mut workspace = validation_only_workspace();
    seed_previous_call(&mut workspace);
    let empty = workspace.reduce_batch(&KinematicsBatch {
        rows: 0, components: 1, values: Vec::new(),
    }).unwrap();
    assert_eq!(empty.rows(), 0);
    assert_eq!(empty.basis, tensor_basis(0));
    assert!(empty.values.is_empty());
    assert!(workspace.last_diagnostics().is_empty());
    assert_no_dispatch(&workspace, 0);

    for (rows, components, len) in [(1, 1, 0), (0, 1, 1), (1, 2, 72), (1, 0, 0), (usize::MAX, 1, 0)] {
        seed_previous_call(&mut workspace);
        let batch = KinematicsBatch { rows, components, values: vec![Complex::new(0., 0.); len] };
        assert!(matches!(workspace.reduce_batch(&batch), Err(ReductionError::InvalidInput(_))));
        assert!(workspace.last_diagnostics().is_empty());
        assert_no_dispatch(&workspace, rows);
    }

    // Internal corruption must fail before chunks_exact(0) or row allocation.
    workspace.components = 0;
    assert!(matches!(workspace.reduce_batch(&KinematicsBatch {
        rows: usize::MAX, components: 0, values: Vec::new(),
    }), Err(ReductionError::InvalidInput(_))));
    assert!(workspace.last_diagnostics().is_empty());
    assert_no_dispatch(&workspace, usize::MAX);

    workspace.components = usize::MAX;
    assert!(matches!(workspace.reduce_batch(&KinematicsBatch {
        rows: 0, components: usize::MAX, values: Vec::new(),
    }), Err(ReductionError::InvalidInput(_))));
    assert!(workspace.last_diagnostics().is_empty());
    assert_no_dispatch(&workspace, 0);
}

#[test]
fn seven_point_all_invalid_lanes_are_inaccessible_and_do_not_dispatch() {
    let mut workspace = validation_only_workspace();
    let zero = Complex::new(0., 0.);
    let mut batch = KinematicsBatch { rows: 5, components: 1, values: vec![zero; 5 * 36] };
    for row in batch.values.chunks_exact_mut(36) { row[35] = Complex::new(1., 0.); }
    batch.values[0] = Complex::new(f64::NAN, 0.);
    batch.values[36 + 35] = Complex::new(0., 0.);
    batch.values[72 + 35] = Complex::new(-1., 0.);
    batch.values[108] = Complex::new(0., 1.);
    batch.values[144 + 28] = Complex::new(1., 1.);
    seed_previous_call(&mut workspace);
    let result = workspace.reduce_batch(&batch).unwrap();
    assert_eq!(result.rows(), 5);
    for lane in 0..5 {
        assert!(matches!(result.statuses[lane], LaneStatus::InvalidInput(_)));
        assert!(result.lane(lane).is_err());
        assert!(result.primal_envelopes[lane].is_none());
    }
    assert_eq!(workspace.last_diagnostics().len(), 5);
    assert!(workspace.last_diagnostics().iter().all(Vec::is_empty));
    assert_no_dispatch(&workspace, 5);
}

// Constructor-only policy regression: a deliberately synthetic plan supplies
// cheap N1/N2 preparation. Never evaluate it as a seven-point integral. This
// exercises the actual ThinSevenWorkspace::new path, not a hand-built workspace
// or a duplicate of its policy constants, without preparing N7 geometry.
#[test]
fn seven_point_constructor_pins_symbolica_with_cheap_plan_fixture() {
    let parent = ReductionPlan::new(ReductionSettings::new(1, 0)).unwrap();
    let mut child_settings = ReductionSettings::new(2, 1);
    child_settings.algorithm = Algorithm::Hybrid;
    let plan = ThinSevenReductionPlan {
        parent,
        child: ReductionPlan::new(child_settings).unwrap(),
    };
    let workspace = ThinSevenWorkspace::new(&plan, 0.).unwrap();
    assert!(crate::execution::policy_composition_tests::assert_context(
        &workspace.child, WorkspacePolicy::SIX_POINT_CHILD,
        MasterExecutionPolicy::Symbolica, true,
    ) > 1, "real eager child master evaluators must be inspected");
    assert!(crate::execution::policy_composition_tests::assert_context(
        &workspace.parent, WorkspacePolicy::RETAIN_ENVELOPES,
        MasterExecutionPolicy::Symbolica, true,
    ) > 0, "a real parent master evaluator must be inspected");
    assert_eq!(workspace.child.dual_shape(), None);
    assert_eq!(workspace.parent.dual_shape(), None);
    assert_eq!(workspace.component_count(), 1);
    assert_eq!(workspace.binary_precision(), 53);

    // Explicit N7 policy selection must not change ordinary public defaults.
    let ordinary = Workspace::new(&plan.parent, 0.).unwrap();
    assert!(crate::execution::policy_composition_tests::assert_context(
        &ordinary, WorkspacePolicy::ORDINARY,
        MasterExecutionPolicy::Default, true,
    ) > 0);

    #[cfg(feature = "native-f64-abc")]
    {
        // This positive control proves that the linked feature can actually
        // select native evaluators. The N7 constructor above remains Symbolica.
        let selected = Workspace::new_with_master_policy(
            &plan.child, 0., MasterExecutionPolicy::NativeF64Abc,
        ).unwrap();
        assert!(crate::execution::policy_composition_tests::assert_context(
            &selected, WorkspacePolicy::ORDINARY,
            MasterExecutionPolicy::NativeF64Abc, true,
        ) > 1);
    }
}
