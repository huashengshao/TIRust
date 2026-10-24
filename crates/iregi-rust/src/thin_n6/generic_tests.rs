//! Exact tests for generic thin N6 assembly. No numerical master or high-rank
//! workspace is prepared here. Integer operations below enumerate tensor and
//! buffer indices; Symbolica owns every polynomial operation and Taylor product.

use super::*;
use crate::plan::{apply_final_uv_power_counting, assemble_thin_six_moments, thin_six_external_primal_seeds};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{dual::HyperDual, rational::{Rational, Q}},
    evaluate::{Dualizer, ExpressionEvaluator},
};

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("generic_thin_test::{name}")))
}

// This index oracle scans the supplied child basis using the defining exponent
// relation. It does not mutate a parent exponent or assume any child stride.
fn expected_child(parent: Monomial, children: &[Monomial]) -> (usize, usize) {
    if parent.rank() == 0 {
        let indices: Vec<_> = children.iter().enumerate()
            .filter_map(|(i, child)| (child.rank() == 0).then_some(i)).collect();
        assert_eq!(indices.len(), 1);
        return (0, indices[0]);
    }
    let mu = (0..4).find(|&mu| parent.0[..mu].iter().all(|&n| n == 0)
        && parent.0[mu] != 0).unwrap();
    let indices: Vec<_> = children.iter().enumerate().filter_map(|(i, child)| {
        (child.rank() + 1 == parent.rank() && (0..4).all(|axis|
            child.0[axis] + u16::from(axis == mu) == parent.0[axis])).then_some(i)
    }).collect();
    assert_eq!(indices.len(), 1, "unique child for {parent:?}");
    (mu + 1, indices[0])
}

fn assert_basis_order(basis: &[Monomial], rank: u16) {
    let degree_counts: [usize; 8] = [1, 4, 10, 20, 35, 56, 84, 120];
    assert_eq!(basis.len(), degree_counts[..=usize::from(rank)].iter().sum::<usize>());
    let mut offset = 0;
    for degree in 0..=usize::from(rank) {
        let count = degree_counts[degree];
        let block = &basis[offset..offset + count];
        assert!(block.iter().all(|m| m.rank() == degree));
        assert_eq!(block[0], Monomial([degree as u16, 0, 0, 0]));
        assert_eq!(block[count - 1], Monomial([0, 0, 0, degree as u16]));
        assert!(block.windows(2).all(|pair| pair[0].0 > pair[1].0));
        offset += count;
    }
    assert_eq!(basis.iter().collect::<std::collections::HashSet<_>>().len(), basis.len());
}

#[test]
fn generic_thin_all_ranks_all_native_slots_follow_dynamic_child_basis() {
    let weights: [Vec<Atom>; 5] = std::array::from_fn(|column|
        (0..6).map(|subset| variable(&format!("w_{column}_{subset}"))).collect());
    let mut checked = 0;
    for rank in 0_u16..=7 {
        let basis = tensor_basis(rank);
        let children = tensor_basis(rank.saturating_sub(1));
        assert_basis_order(&basis, rank);
        assert_basis_order(&children, rank.saturating_sub(1));
        let slots = children.len() * 4;
        let values: Vec<_> = (0..6 * slots)
            .map(|index| variable(&format!("v_{rank}_{index}"))).collect();
        let actual = assemble_thin_six_moments(&basis, &children, &weights, &values);
        assert_eq!(actual.len(), basis.len());
        for (monomial, output) in basis.iter().zip(&actual) {
            let (column, child) = expected_child(*monomial, &children);
            for pole in 0..4 {
                let expected = Atom::add_many(&(0..6).map(|subset|
                    &weights[column][subset] * &values[subset * slots + 4 * child + pole]
                ).collect::<Vec<_>>());
                assert!((&output[pole] - expected).expand().is_zero(),
                    "rank={rank}, parent={monomial:?}, pole={pole}");
                checked += 1;
            }
        }
        // Reorder the supplied child records, preserving each entire four-slot
        // block; assembly must follow their exponents and not an implicit order.
        let reversed: Vec<_> = children.iter().copied().rev().collect();
        let mut reordered = Vec::with_capacity(values.len());
        for subset in 0..6 {
            for child in (0..children.len()).rev() {
                reordered.extend_from_slice(&values[subset * slots + child * 4..subset * slots + child * 4 + 4]);
            }
        }
        assert_eq!(assemble_thin_six_moments(&basis, &reversed, &weights, &reordered), actual);
    }
    assert_eq!(checked, 3168);
    eprintln!("generic thin exact indexing: 8 parent ranks, 3168 native slots, reversed child ordering");
}

#[test]
fn generic_thin_final_uv_rule_preserves_ir_and_finite_and_the_boundary() {
    // Nonzero UV sentinels exercise the final rule separately from the exact
    // cancellation test above. The shared rule also serves ordinary plans, so
    // its logarithmic boundary is tested even though thin N6 rejects rank eight.
    for rank in 0_u16..=8 {
        let settings = ReductionSettings::new(6, rank);
        let basis = tensor_basis(rank);
        let before: Vec<[Atom; 4]> = basis.iter().enumerate().map(|(moment, _)|
            std::array::from_fn(|pole| variable(&format!("final_uv_{rank}_{moment}_{pole}")))
        ).collect();
        let mut after = before.clone();
        apply_final_uv_power_counting(&settings, &basis, &mut after).unwrap();
        for (index, monomial) in basis.iter().enumerate() {
            if monomial.rank() < 8 {
                assert!(after[index][0].is_zero());
            } else {
                assert_eq!(after[index][0], before[index][0], "logarithmic UV boundary");
            }
            assert_eq!(&after[index][1..], &before[index][1..], "IR/finite preservation");
        }
    }
    // Dimensional shifts and repeated powers must retain the existing ordinary
    // rule when it is extracted; neither belongs to the thin caller domain.
    for (shift, first_power, threshold) in [(1, 1, 6), (-1, 1, 10), (1, 2, 8)] {
        let mut settings = ReductionSettings::new(6, 10);
        settings.dimension_shift = shift;
        settings.powers[0] = first_power;
        let basis = tensor_basis(10);
        let before: Vec<[Atom; 4]> = basis.iter().enumerate().map(|(moment, _)|
            std::array::from_fn(|pole| Atom::num((4 * moment + pole + 1) as i64))
        ).collect();
        let mut after = before.clone();
        apply_final_uv_power_counting(&settings, &basis, &mut after).unwrap();
        for (index, monomial) in basis.iter().enumerate() {
            if monomial.rank() < threshold {
                assert!(after[index][0].is_zero());
            } else {
                assert_eq!(after[index][0], before[index][0]);
            }
            assert_eq!(&after[index][1..], &before[index][1..]);
        }
    }
}

#[test]
fn generic_thin_mixed_taylor_and_odd_rank_outputs_keep_raw_components() {
    let x = variable("mixed_x");
    let y = variable("mixed_y");
    let weights: [Vec<Atom>; 5] = std::array::from_fn(|column| (0..6).map(|subset|
        Atom::num((column + 1) as i64)
            + &x * Atom::num((subset + 1) as i64)
            + &y * Atom::num((column + subset + 2) as i64)
    ).collect());
    let mut actual_roots = Vec::new();
    let mut expected_roots = Vec::new();
    let mut odd_roots = Vec::new();
    for rank in 0_u16..=7 {
        let basis = tensor_basis(rank);
        let children = tensor_basis(rank.saturating_sub(1));
        let slots = children.len() * 4;
        let values: Vec<_> = (0..6 * slots).map(|index|
            Atom::num((index + 1) as i64)
                + &x * Atom::num((index + 2) as i64)
                + &y * Atom::num((index + 3) as i64)
                + &x * &y * Atom::num((index + 4) as i64)
        ).collect();
        let actual = assemble_thin_six_moments(&basis, &children, &weights, &values);
        let selected: Vec<_> = basis.iter().enumerate().filter_map(|(index, monomial)| {
            (monomial.rank() == usize::from(rank)
                && (monomial.0.iter().filter(|&&n| n != 0).count() <= 1
                    || monomial.0 == [rank.saturating_sub(1), 0, 1, 0]))
                .then_some(index)
        }).collect();
        for index in selected {
            let (column, child) = expected_child(basis[index], &children);
            for pole in 0..4 {
                if rank % 2 == 1 { odd_roots.push(actual_roots.len()); }
                actual_roots.push(actual[index][pole].clone());
                expected_roots.push(Atom::add_many(&(0..6).map(|subset|
                    &weights[column][subset] * &values[subset * slots + child * 4 + pole]
                ).collect::<Vec<_>>()));
            }
        }
    }
    let count = actual_roots.len();
    let expressions: Vec<_> = actual_roots.into_iter().chain(expected_roots).collect();
    let exact: ExpressionEvaluator<Complex<Rational>> = Atom::evaluator_multiple(&expressions, &[x, y])
        .direct_translation(true).cores(1).horner_iterations(0).build().unwrap();
    // Deliberately non-lexicographic derivative order, with pure second and
    // mixed components. Symbolica owns the derivative multiplication table.
    let shape = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![2, 0], vec![1, 1], vec![0, 2]];
    let mut evaluator = exact.vectorize(&Dualizer::new(
        HyperDual::<Complex<Rational>>::new(shape), vec![],
    )).unwrap().map_to_ring(&Q).unwrap();
    let mut input = vec![Rational::from(0); 12];
    input[2] = Rational::from(1); // x derivative
    input[7] = Rational::from(1); // y derivative
    let mut output = vec![Rational::from(0); 2 * count * 6];
    evaluator.evaluate_in_ring(&input, &mut output, &Q);
    assert_eq!(&output[..count * 6], &output[count * 6..]);
    for root in 0..count {
        assert_ne!(output[root * 6 + 4], Rational::from(0), "mixed component {root}");
    }
    for root in odd_roots {
        assert!(output[root * 6] > Rational::from(0),
            "native raw odd-rank tensor acquired a sign or vanished at root {root}");
    }
    eprintln!("generic thin ordered mixed Taylor: {count} outputs, six components, all parent ranks");
}

#[test]
fn generic_thin_uv_cancellation_uses_nonzero_child_poles_all_ranks() {
    let points = [[0,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2],[1,0,0,0],[0,2,2,0]];
    let shifts = points.iter().map(|p| p.map(|n| Atom::num(n) / Atom::num(2))).collect();
    let masses = [1,2,3,4,2,3].into_iter().map(Atom::num).collect();
    let mut compiler = crate::pv::PvCompiler::new(shifts, masses).unwrap();
    // One exact numeric geometry solve, shared by every rank. No generic
    // symbolic K matrix and no tensor reduction workspace are rebuilt.
    let weights = compiler.thin_six_weights().unwrap();
    for column in &weights {
        assert!(Atom::add_many(column).expand().together().is_zero());
        assert!(column.iter().any(|weight| !weight.is_zero()));
    }
    let mut checked = 0;
    for rank in 0_u16..=7 {
        let basis = tensor_basis(rank);
        let children = tensor_basis(rank.saturating_sub(1));
        let slots = children.len() * 4;
        let values: Vec<_> = (0..6).flat_map(|subset| (0..slots).map(move |slot|
            if slot % 4 == 0 {
                // Artificial UV poles are nonzero and identical across
                // subsets, so the physical identity must cancel them.
                Atom::num((slot / 4 + 1) as i64)
            } else {
                Atom::num((subset * slots + slot + 1) as i64)
            }
        )).collect();
        assert!(values.iter().step_by(4).all(|pole| !pole.is_zero()));
        let actual = assemble_thin_six_moments(&basis, &children, &weights, &values);
        for (parent, output) in basis.iter().zip(&actual) {
            assert!(output[0].expand().together().is_zero(), "rank={rank}, parent={parent:?}");
            let (column, child) = expected_child(*parent, &children);
            for pole in 1..4 {
                let expected = Atom::add_many(&(0..6).map(|subset|
                    &weights[column][subset] * &values[subset * slots + child * 4 + pole]
                ).collect::<Vec<_>>());
                assert!((&output[pole] - expected).expand().together().is_zero());
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 792);
    eprintln!("generic thin UV cancellation: 792 parent UV slots with nonzero imported UV poles");
}

#[test]
fn generic_thin_integrand_identity_all_rank_seven_moments_and_routings() {
    fn square(p: &[Atom; 4]) -> Atom {
        &p[0] * &p[0] - &p[1] * &p[1] - &p[2] * &p[2] - &p[3] * &p[3]
    }
    fn monomial(q: &[Atom; 4], exponents: Monomial) -> Atom {
        q.iter().zip(exponents.0).fold(Atom::num(1), |product, (q, exponent)|
            product * q.pow(i64::from(exponent)))
    }
    let points = [[0,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2],[1,0,0,0],[0,2,2,0]];
    let q: [Atom; 4] = std::array::from_fn(|mu| variable(&format!("integrand_q{mu}")));
    let q_perp_squared = variable("integrand_q_perp_squared");
    let basis = tensor_basis(7);
    let children = tensor_basis(6);
    let child_monomials: Vec<_> = children.iter().map(|&child| monomial(&q, child)).collect();
    let expected_monomials: Vec<_> = basis.iter().map(|&parent| monomial(&q, parent)).collect();
    let mut checked = 0;
    for variant in 0..5 {
        let mut shifts: Vec<[Atom;4]> = points.iter().map(|p| p.map(|n| Atom::num(n) / Atom::num(2))).collect();
        let mut masses: Vec<_> = [1,2,3,4,2,3].into_iter().map(Atom::num).collect();
        match variant {
            1 => for p in &mut shifts { for (mu, component) in p.iter_mut().enumerate() {
                *component = &*component + Atom::num((mu + 1) as i64) / Atom::num(3);
            }},
            2 => for p in &mut shifts { for component in p { *component = -&*component; }},
            3 => { shifts.reverse(); masses.reverse(); },
            4 => for (i, mass) in masses.iter_mut().enumerate() {
                *mass = &*mass - Atom::i() * Atom::num((i + 1) as i64) / Atom::num(16);
            },
            _ => (),
        }
        let mut compiler = crate::pv::PvCompiler::new(shifts.clone(), masses.clone()).unwrap();
        let weights = compiler.thin_six_weights().unwrap();
        let denominators: Vec<_> = shifts.iter().zip(&masses).map(|(p, mass)| {
            let shifted = std::array::from_fn(|mu| &q[mu] + &p[mu]);
            square(&shifted) + &q_perp_squared - mass
        }).collect();
        // Multiply each child integrand by its pinched denominator. This puts
        // every term over the parent's common product of all six denominators.
        // Slot multipliers distinguish all four assembly outputs; this is an
        // integrand identity, not a fabricated Laurent value of an integral.
        let mut values = Vec::with_capacity(6 * children.len() * 4);
        for denominator in &denominators {
            for child in &child_monomials {
                for pole in 0..4 {
                    values.push(denominator * child * Atom::num((pole + 1) as i64));
                }
            }
        }
        let actual = assemble_thin_six_moments(&basis, &children, &weights, &values);
        for (index, output) in actual.iter().enumerate() {
            for pole in 0..4 {
                let expected = &expected_monomials[index] * Atom::num((pole + 1) as i64);
                assert!((&output[pole] - expected).expand().together().is_zero(),
                    "variant={variant}, monomial={:?}, slot={pole}", basis[index]);
                checked += 1;
            }
        }
        eprintln!("generic thin physical integrand: variant {variant}, 330 moments, four assembly slots, explicit transverse square");
    }
    assert_eq!(checked, 6600);
}

#[test]
fn generic_thin_external_amplitudes_keep_child_dimensions_and_slot_identity() {
    let cumulative_counts = [1, 5, 15, 35, 70, 126, 210];
    let mut checked = 0;
    for rank in 0_u16..=7 {
        let child_rank = rank.saturating_sub(1);
        let children = tensor_basis(child_rank);
        assert_eq!(children.len(), cumulative_counts[usize::from(child_rank)]);
        let width = 4 * children.len();
        let values: Vec<_> = (0..6 * width).map(|i| variable(&format!("dimension_value_{rank}_{i}"))).collect();
        let amplitudes: Vec<_> = (0..6 * width).map(|i| variable(&format!("dimension_amplitude_{rank}_{i}"))).collect();
        let seeds = thin_six_external_primal_seeds(&children, &values, &amplitudes);
        assert_eq!(seeds.len(), 6 * width);
        if rank == 7 {
            let highest_child = children.iter().position(|monomial| monomial.rank() == 6).unwrap();
            assert_eq!(seeds[highest_child * 4].2, 0);
            let parent = tensor_basis(rank).into_iter().find(|monomial| monomial.rank() == 7).unwrap();
            assert_eq!(4 + parent.rank() as i32 - 2 * 6, -1);
        }
        for subset in 0..6 {
            for (child, monomial) in children.iter().enumerate() {
                for pole in 0..4 {
                    let index = subset * width + child * 4 + pole;
                    assert_eq!(seeds[index].0, values[index]);
                    assert_eq!(seeds[index].1, amplitudes[index]);
                    // Five unit propagators: measure dimension four minus ten.
                    assert_eq!(seeds[index].2, monomial.rank() as i32 - 6);
                    assert!((-6..=0).contains(&seeds[index].2));
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 11112);
    eprintln!("generic thin imported amplitude dimensions: 11112 subset/moment/pole records");
}

#[test]
fn generic_thin_rejects_unsupported_callers_before_geometry() {
    for case in 0..11 {
        let mut settings = ReductionSettings::new(6, 7);
        settings.algorithm = Algorithm::PassarinoVeltman;
        match case {
            0 => settings.rank = 8,
            1 => settings.propagators = 7,
            2 => settings.powers[2] = 2,
            3 => settings.powers.pop().map(|_| ()).unwrap(),
            4 => settings.dimension_shift = 1,
            5 => settings.invariants = crate::InvariantInput::SuppliedScalarProducts,
            6 => settings.input_policy = crate::InputPolicy::legacy_onshell(),
            7 => settings.algorithm = Algorithm::Ibp,
            8 => settings.convention = Convention::MadLoop,
            9 => settings.tensor_weights = TensorWeights::Multinomial,
            10 => settings.algorithm = Algorithm::Hybrid,
            _ => unreachable!(),
        }
        assert!(matches!(ReductionPlan::new_thin_six(settings, 0),
            Err(ReductionError::Unsupported(_))), "unsupported caller {case}");
    }
}

#[test]
fn generic_thin_borrowed_dual_preserves_ordered_shape_and_rejects_bad_metadata() {
    let first = vec![vec![0,0],vec![1,0],vec![0,1],vec![2,0],vec![1,1],vec![0,2]];
    let second = vec![vec![0,0],vec![0,1],vec![1,0],vec![0,2],vec![1,1],vec![2,0]];
    for shape in [first, second] {
        let canonical = HyperDual::<u8>::new(shape.clone());
        let borrowed = BorrowedDual(&canonical);
        iregi_masters::validate_dual_structure(&borrowed).unwrap();
        assert_eq!(borrowed.get_shape(), shape.iter().map(Vec::as_slice).collect::<Vec<_>>());
        assert_eq!(borrowed.get_multiplication_table(), canonical.get_multiplication_table());
        assert_eq!(borrowed.get_max_depth(), canonical.get_max_depth());
    }
    struct WrongLength(HyperDual<u8>);
    impl DualNumberStructure for WrongLength {
        fn get_len(&self) -> usize { self.0.get_len() + 1 }
        fn get_shape(&self) -> Vec<&[usize]> { self.0.get_shape() }
        fn get_multiplication_table(&self) -> &[(usize,usize,usize)] { self.0.get_multiplication_table() }
        fn get_max_depth(&self) -> usize { self.0.get_max_depth() }
    }
    let invalid = WrongLength(HyperDual::<u8>::new(vec![vec![0],vec![1]]));
    assert!(iregi_masters::validate_dual_structure(&BorrowedDual(&invalid)).is_err());
}

#[test]
fn generic_thin_complete_result_layout_rejects_malformed_buffers_and_shapes() {
    let mixed_shape = vec![vec![0,0],vec![0,1],vec![1,0],vec![1,1]];
    let reordered_shape = vec![vec![0,0],vec![1,0],vec![0,1],vec![1,1]];
    let mut rejected = 0;
    for rank in [0, 3, 7] {
        let basis = tensor_basis(rank);
        for components in [1, 4] {
            let shape = (components == 4).then(|| mixed_shape.clone());
            let metadata = ResultMetadata::new(Convention::IregiNative, TensorWeights::Components, shape.clone());
            let outputs = basis.len() * 4;
            let result = BatchResult {
                basis: basis.clone(), components, metadata: metadata.clone(),
                values: (0..3 * outputs * components)
                    .map(|index| Complex::new(index as f64, -(index as f64))).collect(),
                statuses: vec![LaneStatus::Success, LaneStatus::Unstable("demanded child failed".into()), LaneStatus::Success],
                fallback_used: vec![false, true, false], analytic_used: vec![true, true, false],
                primal_envelopes: vec![None; 3], primal_envelope_errors: vec![None; 3],
            };
            validate_result_layout(&result, &basis, components, &metadata, 3).unwrap();
            assert_eq!(result.lane(0).unwrap().len(), outputs * components);
            assert!(result.lane(1).is_err(), "shape validation does not turn a failure into data");
            assert_eq!(result.lane(2).unwrap().len(), outputs * components);
            assert!(result.validated_primal_envelope(0, 53, 53).is_err(),
                "layout validation alone must not authorize a missing amplitude context");
            for fault in 0..17 {
                let mut malformed = result.clone();
                match fault {
                    0 => { malformed.values.pop(); },
                    1 => malformed.values.push(Complex::new(0., 0.)),
                    2 => { malformed.statuses.pop(); },
                    3 => malformed.statuses.push(LaneStatus::Success),
                    4 => { malformed.fallback_used.pop(); },
                    5 => malformed.fallback_used.push(false),
                    6 => { malformed.analytic_used.pop(); },
                    7 => malformed.analytic_used.push(false),
                    8 => { malformed.primal_envelopes.pop(); },
                    9 => malformed.primal_envelopes.push(None),
                    10 => { malformed.primal_envelope_errors.pop(); },
                    11 => malformed.primal_envelope_errors.push(None),
                    12 => malformed.basis[0] = Monomial([1, 0, 0, 0]),
                    13 => malformed.components += 1,
                    14 => malformed.metadata = ResultMetadata::new(Convention::OneLoop, TensorWeights::Components, shape.clone()),
                    15 => malformed.metadata = ResultMetadata::new(Convention::IregiNative, TensorWeights::Multinomial, shape.clone()),
                    16 => malformed.metadata = ResultMetadata::new(Convention::IregiNative, TensorWeights::Components,
                        Some(if components == 4 { reordered_shape.clone() } else { vec![vec![0]] })),
                    _ => unreachable!(),
                }
                assert!(validate_result_layout(&malformed, &basis, components, &metadata, 3).is_err(),
                    "rank={rank}, components={components}, fault={fault}");
                rejected += 1;
            }
            assert!(validate_result_layout(&result, &basis, components, &metadata, usize::MAX).is_err());
            assert!(validate_result_layout(&result, &basis, usize::MAX, &metadata, 3).is_err());
            let mut empty = result.clone();
            empty.values.clear(); empty.statuses.clear(); empty.fallback_used.clear();
            empty.analytic_used.clear(); empty.primal_envelopes.clear(); empty.primal_envelope_errors.clear();
            validate_result_layout(&empty, &basis, components, &metadata, 0).unwrap();
        }
    }
    assert_eq!(rejected, 102);
    eprintln!("generic thin result layout: six rank/component layouts, 102 malformed controls, overflow and empty controls");
}
