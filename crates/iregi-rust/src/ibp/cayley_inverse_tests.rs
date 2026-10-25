//! Exact field-equivalence tests for the private five-point inverse adapter.
use super::*;
use symbolica::{parse, symbol};

fn atom_inverse(rows: &[Vec<Atom>]) -> Matrix<symbolica::domains::atom::AtomField> {
    Matrix::from_nested_vec(rows.to_vec(), super::super::exact_field())
        .unwrap()
        .inv()
        .unwrap()
}

fn assert_inverse_and_old_field(rows: &[Vec<Atom>], values: &[Vec<Atom>]) {
    let field = super::super::exact_field();
    let original = Matrix::from_nested_vec(rows.to_vec(), field.clone()).unwrap();
    let candidate = Matrix::from_nested_vec(values.to_vec(), field.clone()).unwrap();
    let identity = Matrix::identity(5, field);
    assert_eq!(&original * &candidate, identity);
    assert_eq!(&candidate * &original, identity);
    let old = atom_inverse(rows);
    for (new, old) in candidate.iter().zip(old.iter()) {
        assert!((new - old).together().is_zero());
    }
}

fn diagonal(value: Atom) -> Vec<Vec<Atom>> {
    (0..5)
        .map(|i| {
            (0..5)
                .map(|j| if i == j { value.clone() } else { Atom::Zero })
                .collect()
        })
        .collect()
}

#[test]
fn five_point_affine_inverse_matches_original_atom_field() {
    let rows = vec![
        vec![
            parse!("5+cayley_inverse_test::x"),
            parse!("1/2"),
            Atom::Zero,
            Atom::one(),
            Atom::Zero,
        ],
        vec![
            parse!("1/2"),
            parse!("6+cayley_inverse_test::y"),
            Atom::one(),
            Atom::Zero,
            Atom::Zero,
        ],
        vec![
            Atom::Zero,
            Atom::one(),
            Atom::num(7),
            Atom::one(),
            Atom::Zero,
        ],
        vec![
            Atom::one(),
            Atom::Zero,
            Atom::one(),
            Atom::num(8),
            parse!("1/3"),
        ],
        vec![
            Atom::Zero,
            Atom::Zero,
            Atom::Zero,
            parse!("1/3"),
            Atom::num(9),
        ],
    ];
    let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
    assert!(matches!(system.entries, RoutingEntries::Rational(_)));
    let inverse = system.inverse_five().unwrap().unwrap();
    assert_inverse_and_old_field(&rows, &inverse);
}

#[test]
fn five_point_generic_distance_mass_aliases_use_shared_exact_map() {
    let masses: Vec<_> = (0..5)
        .map(|i| parse!(&format!("cayley_generic::m{i}")))
        .collect();
    let rows: Vec<Vec<_>> = (0..5)
        .map(|i| {
            (0..5)
                .map(|j| {
                    let distance = if i == j {
                        Atom::Zero
                    } else {
                        parse!(&format!("cayley_generic::s{}_{}", i.min(j), i.max(j)))
                    };
                    distance - &masses[i] - &masses[j]
                })
                .collect()
        })
        .collect();
    let system = PreparedRoutingSystem::new(rows).unwrap();
    assert!(matches!(system.entries, RoutingEntries::Rational(_)));
    assert_eq!(system.variables.len(), 15);
    // inverse_five certifies both products in Symbolica's converted field.
    // Do not repeat the expensive generic AtomField constructor here.
    let inverse = system.inverse_five().unwrap().unwrap();
    assert_eq!(inverse.len(), 5);
    for i in 0..5 {
        for j in 0..5 {
            assert!((&inverse[i][j] - &inverse[j][i]).together().is_zero());
        }
    }
    assert!(inverse.iter().flatten().any(|x| !x.is_zero()));
}

#[test]
fn five_point_gaussian_inverse_matches_original_complex_atom_field() {
    let mut rows = diagonal(parse!("3+cayley_inverse_test::x"));
    rows[0][1] = parse!("𝑖");
    rows[1][0] = parse!("𝑖");
    rows[1][2] = parse!("1/3");
    rows[2][1] = parse!("1/3");
    let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
    assert!(matches!(system.entries, RoutingEntries::Gaussian(_)));
    assert_eq!(
        system.variables.len(),
        1,
        "literal i must remain a coefficient"
    );
    assert_inverse_and_old_field(&rows, &system.inverse_five().unwrap().unwrap());
}

#[test]
fn five_point_structural_singularity_remains_a_strict_failure() {
    let rows = vec![vec![parse!("1+cayley_inverse_test::x"); 5]; 5];
    let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
    assert!(
        matches!(system.inverse_five().unwrap_err(), super::super::GeometryError::Singular(ref error) if error=="Singular")
    );
    assert!(
        Matrix::from_nested_vec(rows, super::super::exact_field())
            .unwrap()
            .inv()
            .is_err()
    );
}

#[test]
fn five_point_formal_chart_boundary_is_not_specialized_at_preparation() {
    let mut rows = diagonal(Atom::one());
    rows[0][0] = parse!("cayley_inverse_test::x");
    let inverse = PreparedRoutingSystem::new(rows.clone())
        .unwrap()
        .inverse_five()
        .unwrap()
        .unwrap();
    assert!(
        (&inverse[0][0] - parse!("1/cayley_inverse_test::x"))
            .together()
            .is_zero()
    );
    assert_inverse_and_old_field(&rows, &inverse);
}

#[test]
fn five_point_nonaffine_opaque_and_float_entries_keep_original_path() {
    for entry in [
        parse!("sin(cayley_inverse_test::x)"),
        parse!("1/cayley_inverse_test::x"),
        parse!("cayley_inverse_test::x*cayley_inverse_test::y"),
        parse!("cayley_inverse_test::x^2"),
        parse!("cayley_inverse_test::x^100000"),
        Atom::num(0.5),
    ] {
        let mut rows = diagonal(Atom::one());
        rows[0][0] = entry;
        let system = PreparedRoutingSystem::new(rows).unwrap();
        assert!(matches!(system.entries, RoutingEntries::Atom(_)));
        assert!(system.inverse_five().unwrap().is_none());
    }
}

#[test]
fn inverse_adapter_declines_other_sizes_and_rectangular_routing_systems() {
    for (rows, columns) in [(4, 4), (6, 6), (5, 6)] {
        let entries = (0..rows)
            .map(|i| (0..columns).map(|j| Atom::num(i64::from(i == j))).collect())
            .collect();
        assert!(
            PreparedRoutingSystem::new(entries)
                .unwrap()
                .inverse_five()
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn five_point_inverse_preserves_first_and_mixed_symbolic_derivatives() {
    let mut rows = diagonal(Atom::num(3));
    rows[0][0] = parse!("1+cayley_inverse_test::x");
    rows[1][1] = Atom::num(2);
    rows[0][1] = parse!("cayley_inverse_test::y");
    rows[1][0] = parse!("cayley_inverse_test::y");
    let old = atom_inverse(&rows);
    let candidate = PreparedRoutingSystem::new(rows)
        .unwrap()
        .inverse_five()
        .unwrap()
        .unwrap();
    let x = symbol!("cayley_inverse_test::x");
    let y = symbol!("cayley_inverse_test::y");
    for i in 0..5 {
        for j in 0..5 {
            assert!(
                (candidate[i][j].derivative(x) - old[(i as u32, j as u32)].derivative(x))
                    .together()
                    .is_zero()
            );
            assert!(
                (candidate[i][j].derivative(x).derivative(y)
                    - old[(i as u32, j as u32)].derivative(x).derivative(y))
                .together()
                .is_zero()
            );
        }
    }
    assert!(
        !candidate[0][0]
            .derivative(x)
            .derivative(y)
            .together()
            .is_zero()
    );
}
