//! Private source-only proposal tests for the real Cayley geometry caller.
//!
//! The matched reference below uses the original AtomField inverse directly.
//! It seeds only the five-line Geometry entry; both integral calculations still
//! enter ScalarCompiler::compile and execute all descendant recurrences.

use super::*;
use symbolica::id::Replacement;

fn clear_domains() {
    routing_rational::TEST_DOMAINS.with(|domains| domains.borrow_mut().clear());
}

fn domains() -> Vec<&'static str> {
    routing_rational::TEST_DOMAINS.with(|domains| domains.borrow().clone())
}

fn assert_equal(actual: &Atom, expected: &Atom, context: &str) {
    let residual = (actual - expected).together();
    assert!(residual.is_zero(), "{context}: residual {residual}");
}

fn expand_aliases(value: &Atom, aliases: &[(Atom, Atom)]) -> Atom {
    aliases
        .iter()
        .rev()
        .fold(value.clone(), |value, (alias, body)| {
            value.replace(alias.clone()).with(body.clone())
        })
}

fn expand_slots(compiler: &ScalarCompiler, slots: &[Atom; 4]) -> [Atom; 4] {
    std::array::from_fn(|slot| expand_aliases(&slots[slot], compiler.aliases()))
}

/// The retained conditioned-mass fixture, with only m_0^2 made symbolic.
/// All numbers, including the timelike half-unit shift, stay exact Atoms.
fn physical_fixture(n: usize, mass_offset: Atom) -> (Vec<[Atom; 4]>, Vec<Atom>, Vec<Vec<Atom>>) {
    assert!((5..=7).contains(&n));
    let points = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ];
    let mut shifts: Vec<[Atom; 4]> = points[..n]
        .iter()
        .map(|point| point.map(Atom::num))
        .collect();
    shifts[4][0] = Atom::one() / Atom::num(2);
    let mut masses: Vec<Atom> = [1, 2, 3, 4, 2, 3, 4][..n]
        .iter()
        .map(|&mass| Atom::num(mass))
        .collect();
    masses[0] = Atom::one() + mass_offset;
    let r = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let delta: [Atom; 4] =
                        std::array::from_fn(|mu| &shifts[i][mu] - &shifts[j][mu]);
                    // Match the expanded affine R syntax of the public plan's
                    // independent mass/distance aliases. Symbolica performs
                    // this fixture preparation; the physical point is unchanged.
                    let raw = delta[0].pow(2)
                        - delta[1].pow(2)
                        - delta[2].pow(2)
                        - delta[3].pow(2)
                        - &masses[i]
                        - &masses[j];
                    let expanded = raw.expand();
                    assert_equal(&expanded, &raw, "physical fixture affine expansion");
                    expanded
                })
                .collect()
        })
        .collect();
    (shifts, masses, r)
}

fn selected_rows(r: &[Vec<Atom>], active: &[usize]) -> Vec<Vec<Atom>> {
    active
        .iter()
        .map(|&i| active.iter().map(|&j| r[i][j].clone()).collect())
        .collect()
}

/// Independent reference: exactly the original AtomField Matrix::inv path.
/// This deliberately does not call geometry(), PreparedRoutingSystem, or the
/// proposed inverse operation, so it cannot select the candidate field.
fn original_atom_geometry(r: &[Vec<Atom>], active: &[usize]) -> Geometry {
    let matrix = Matrix::from_nested_vec(selected_rows(r, active), exact_field()).unwrap();
    let original_inverse = matrix.inv().unwrap();
    let inverse: Vec<Vec<Atom>> = (0..active.len())
        .map(|i| {
            (0..active.len())
                .map(|j| original_inverse[(i as u32, j as u32)].clone())
                .collect()
        })
        .collect();
    let b: Vec<Atom> = inverse
        .iter()
        .map(|row| {
            row.iter()
                .fold(Atom::new(), |sum, entry| sum + entry)
                .together()
        })
        .collect();
    let sum_b = b
        .iter()
        .fold(Atom::new(), |sum, entry| sum + entry)
        .together();
    Geometry {
        inverse: Some(inverse),
        b,
        sum_b,
    }
}

fn assert_inverse_identity(r: &[Vec<Atom>], active: &[usize], geometry: &Geometry) {
    let inverse = geometry.inverse.as_ref().expect("full geometry inverse");
    assert_eq!(inverse.len(), active.len());
    assert!(inverse.iter().all(|row| row.len() == active.len()));
    let matrix = Matrix::from_nested_vec(selected_rows(r, active), exact_field()).unwrap();
    let inverse = Matrix::from_nested_vec(inverse.clone(), exact_field()).unwrap();
    let left_identity = &matrix * &inverse;
    let right_identity = &inverse * &matrix;
    for i in 0..active.len() {
        for j in 0..active.len() {
            let expected = Atom::num(i64::from(i == j));
            assert_equal(
                &left_identity[(i as u32, j as u32)],
                &expected,
                &format!("R*Rinv entry ({i},{j})"),
            );
            assert_equal(
                &right_identity[(i as u32, j as u32)],
                &expected,
                &format!("Rinv*R entry ({i},{j})"),
            );
        }
    }
}

fn master_signatures(compiler: &ScalarCompiler) -> Vec<(u8, Vec<usize>)> {
    let mut signatures: Vec<_> = compiler
        .leaves
        .iter()
        .map(|leaf| {
            let kind = match leaf.kind {
                MasterLeafKind::FourDimensional => 0,
                MasterLeafKind::SixDimensionalPentagon => 1,
            };
            (kind, leaf.active.clone())
        })
        .collect();
    signatures.sort();
    signatures
}

fn cache_sizes(compiler: &ScalarCompiler) -> [usize; 9] {
    [
        compiler.node_count(),
        compiler.cache.len(),
        compiler.geometry.len(),
        compiler.exceptional_geometry.len(),
        compiler.regular_bubble_geometry.len(),
        compiler.degenerate_families.len(),
        compiler.routing_relations.len(),
        compiler.aliases.len(),
        compiler.leaves.len(),
    ]
}

fn massive_master_poles(compiler: &ScalarCompiler, masses: &[Atom]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    for leaf in &compiler.leaves {
        assert_eq!(leaf.kind, MasterLeafKind::FourDimensional);
        let uv = match leaf.active.len() {
            1 => masses[leaf.active[0]].clone(),
            2 => Atom::one(),
            3 | 4 => Atom::new(),
            _ => panic!("unexpected master support {:?}", leaf.active),
        };
        replacements.extend([
            Replacement::new(leaf.slots[0].clone(), uv),
            Replacement::new(leaf.slots[1].clone(), Atom::new()),
            Replacement::new(leaf.slots[2].clone(), Atom::new()),
        ]);
    }
    replacements
}

#[test]
fn physical_dotted_pentagon_compile_matches_original_atom_inverse_all_native_slots() {
    let mass_symbol = symbolica::symbol!("cayley_actual_compile::mass_offset");
    let (shifts, masses, r) = physical_fixture(5, Atom::var(mass_symbol));
    let active: Vec<usize> = (0..5).collect();
    let request = ScalarRequest {
        powers: vec![2, 1, 1, 1, 1],
        dimension_shift: 0,
    };

    // Only this five-line entry is seeded in the independent old-field path.
    // Descendant requests run the same ordinary compile entry point below.
    let reference_geometry = original_atom_geometry(&r, &active);
    assert_inverse_identity(&r, &active, &reference_geometry);
    let mut reference = ScalarCompiler::with_aliases(r.clone()).unwrap();
    reference.set_routing(shifts.clone()).unwrap();
    reference
        .geometry
        .insert(active.clone(), reference_geometry.clone());
    clear_domains();
    let reference_slots = reference.compile(request.clone()).unwrap();
    assert!(
        domains().is_empty(),
        "the reference must use its seeded inverse"
    );

    let mut candidate = ScalarCompiler::with_aliases(r.clone()).unwrap();
    candidate.set_routing(shifts).unwrap();
    clear_domains();
    let candidate_slots = candidate.compile(request.clone()).unwrap();
    assert_eq!(domains(), ["rational", "rational_inverse"]);
    assert!(candidate.cache.contains_key(&request));
    let actual_geometry = candidate.geometry.get(&active).unwrap();
    assert_inverse_identity(&r, &active, actual_geometry);
    for (actual_row, expected_row) in actual_geometry
        .inverse
        .as_ref()
        .unwrap()
        .iter()
        .zip(reference_geometry.inverse.as_ref().unwrap())
    {
        for (actual, expected) in actual_row.iter().zip(expected_row) {
            assert_equal(actual, expected, "actual compile cached inverse");
        }
    }

    let expected = expand_slots(&reference, &reference_slots);
    let actual = expand_slots(&candidate, &candidate_slots);
    // Check UV, IR single, IR double and finite before any pole specialization.
    for slot in 0..4 {
        assert_equal(
            &actual[slot],
            &expected[slot],
            &format!("native slot {slot}"),
        );
    }
    assert!(!actual[3].together().is_zero(), "finite output must exist");
    assert!(
        actual[3].contains_symbol(mass_symbol),
        "retain symbolic mass dependence"
    );
    assert!(
        !candidate.aliases().is_empty(),
        "exercise actual alias preparation"
    );
    assert!(!candidate.leaves.is_empty());
    assert_eq!(master_signatures(&candidate), master_signatures(&reference));
    assert!(candidate.leaves.iter().all(|leaf| {
        leaf.kind == MasterLeafKind::FourDimensional
            && (1..=4).contains(&leaf.active.len())
            && leaf.active.windows(2).all(|pair| pair[0] < pair[1])
    }));
    assert!(candidate.leaves.iter().any(|leaf| leaf.active.len() == 4));
    assert!(!candidate.exceptional_geometry.contains_key(&active));
    assert!(!reference.exceptional_geometry.contains_key(&active));
    let mut actual_exceptional: Vec<_> = candidate.exceptional_geometry.keys().cloned().collect();
    let mut expected_exceptional: Vec<_> = reference.exceptional_geometry.keys().cloned().collect();
    actual_exceptional.sort();
    expected_exceptional.sort();
    assert_eq!(actual_exceptional, expected_exceptional);

    // This massive six-denominator-power integral is UV/IR finite. Leave all
    // finite masters symbolic; only known physical pole slots are assigned.
    let pole_replacements = massive_master_poles(&candidate, &masses);
    for slot in 0..3 {
        let pole = actual[slot].replace_multiple(pole_replacements.iter());
        assert_equal(
            &pole,
            &Atom::new(),
            &format!("massive native pole slot {slot}"),
        );
    }

    let before_repeat = cache_sizes(&candidate);
    let before_signatures = master_signatures(&candidate);
    clear_domains();
    let repeated = candidate.compile(request).unwrap();
    assert_eq!(repeated, candidate_slots);
    assert_eq!(cache_sizes(&candidate), before_repeat);
    assert_eq!(master_signatures(&candidate), before_signatures);
    assert!(
        domains().is_empty(),
        "repeat must not prepare another inverse"
    );
}

#[test]
fn scalar_geometry_upgrade_preserves_reversed_five_point_subset_order_and_cache() {
    let mass = Atom::var(symbolica::symbol!("cayley_cache_upgrade::mass_offset"));
    let (_, _, r) = physical_fixture(7, mass);
    let active = vec![6, 4, 2, 0, 5];
    let reversed: Vec<usize> = active.iter().rev().copied().collect();
    let mut compiler = ScalarCompiler::with_aliases(r.clone()).unwrap();

    clear_domains();
    let scalar = compiler.scalar_geometry(&active).unwrap();
    assert!(scalar.inverse.is_none());
    assert_eq!(compiler.geometry.len(), 1);
    assert!(
        domains().is_empty(),
        "the lazy scalar solve stays unchanged"
    );

    let full = compiler.geometry(&active).unwrap();
    assert_eq!(domains(), ["rational", "rational_inverse"]);
    assert_eq!(
        compiler.geometry.len(),
        1,
        "upgrade replaces the same cache entry"
    );
    assert_inverse_identity(&r, &active, &full);
    for (actual, expected) in full.b.iter().zip(&scalar.b) {
        assert_equal(actual, expected, "upgrade preserves b in supplied order");
    }
    assert_equal(&full.sum_b, &scalar.sum_b, "upgrade preserves B");

    let before_repeat = cache_sizes(&compiler);
    clear_domains();
    let repeated = compiler.geometry(&active).unwrap();
    let scalar_after_upgrade = compiler.scalar_geometry(&active).unwrap();
    assert_eq!(repeated.inverse, full.inverse);
    assert_eq!(scalar_after_upgrade.inverse, full.inverse);
    assert_eq!(cache_sizes(&compiler), before_repeat);
    assert!(
        domains().is_empty(),
        "both cache queries reuse the full inverse"
    );

    let reverse_geometry = compiler.geometry(&reversed).unwrap();
    assert_eq!(domains(), ["rational", "rational_inverse"]);
    assert_eq!(
        compiler.geometry.len(),
        2,
        "ordered subset keys stay distinct"
    );
    assert_inverse_identity(&r, &reversed, &reverse_geometry);
    let inverse = full.inverse.as_ref().unwrap();
    let reverse_inverse = reverse_geometry.inverse.as_ref().unwrap();
    for i in 0..active.len() {
        assert_equal(
            &reverse_geometry.b[i],
            &full.b[active.len() - 1 - i],
            "reversed subset preserves b permutation",
        );
        for j in 0..active.len() {
            assert_equal(
                &reverse_inverse[i][j],
                &inverse[active.len() - 1 - i][active.len() - 1 - j],
                "reversed subset preserves inverse permutation",
            );
        }
    }
    assert_equal(
        &reverse_geometry.sum_b,
        &full.sum_b,
        "reversed subset preserves B",
    );
    assert!(
        compiler.cache.is_empty(),
        "geometry queries do not compile integrals"
    );
    assert!(compiler.leaves.is_empty());
    assert!(compiler.aliases().is_empty());
}

#[test]
fn opaque_mass_geometry_keeps_original_atom_inverse_in_actual_caller() {
    let opaque_mass = symbolica::parse!("sin(cayley_opaque_caller::mass_offset)");
    let (_, _, r) = physical_fixture(5, opaque_mass.clone());
    let active = vec![0, 1, 2, 3, 4];
    let reference = original_atom_geometry(&r, &active);
    let mut compiler = ScalarCompiler::with_aliases(r.clone()).unwrap();

    clear_domains();
    let actual = compiler.geometry(&active).unwrap();
    assert_eq!(
        domains(),
        ["atom"],
        "opaque mass must decline exact inverse adapter"
    );
    assert_inverse_identity(&r, &active, &actual);
    let inverse = actual.inverse.as_ref().unwrap();
    let reference_inverse = reference.inverse.as_ref().unwrap();
    assert!(
        inverse
            .iter()
            .flatten()
            .any(|entry| entry.contains(opaque_mass.clone()))
    );
    for (actual_row, expected_row) in inverse.iter().zip(reference_inverse) {
        for (actual, expected) in actual_row.iter().zip(expected_row) {
            assert_equal(
                actual,
                expected,
                "opaque caller matches original AtomField inverse",
            );
        }
    }
    for (actual, expected) in actual.b.iter().zip(&reference.b) {
        assert_equal(actual, expected, "opaque caller b");
    }
    assert_equal(&actual.sum_b, &reference.sum_b, "opaque caller B");
    let before_repeat = cache_sizes(&compiler);
    clear_domains();
    assert_eq!(compiler.geometry(&active).unwrap().inverse, actual.inverse);
    assert_eq!(cache_sizes(&compiler), before_repeat);
    assert!(domains().is_empty());
    assert!(compiler.leaves.is_empty());
}

#[test]
fn inverse_preparation_failure_propagates_without_exceptional_recurrence() {
    let (_, _, r) = physical_fixture(
        5,
        Atom::var(symbolica::symbol!("cayley_fatal::mass_offset")),
    );
    let mut compiler = ScalarCompiler::with_aliases(r).unwrap();
    routing_rational::TEST_INVERSE_PREPARATION_ERROR.with(|slot| {
        *slot.borrow_mut() = Some("injected exact inverse certificate failure".into());
    });
    let result = compiler.compile(ScalarRequest {
        powers: vec![2, 1, 1, 1, 1],
        dimension_shift: 0,
    });
    let error =
        result.expect_err("an internal preparation failure cannot yield integral coefficients");
    assert!(error.contains("injected exact inverse certificate failure"));
    assert!(
        compiler.exceptional_geometry.is_empty(),
        "internal failure must not select physical singular recurrences"
    );
    assert!(compiler.geometry.is_empty());
    assert!(compiler.leaves.is_empty());
    assert!(routing_rational::TEST_INVERSE_PREPARATION_ERROR.with(|slot| slot.borrow().is_none()));
}
