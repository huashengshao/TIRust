use super::*;
use crate::basis::tensor_basis;

fn fixture() -> (Vec<[Atom; 4]>, Vec<Atom>) {
    (
        exact_shifts(&[[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]]),
        [1, 2, 3].into_iter().map(Atom::num).collect(),
    )
}

fn exact_shifts(rows: &[[i64; 4]]) -> Vec<[Atom; 4]> {
    rows.iter().map(|row| row.map(Atom::num)).collect()
}

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("affine_triangle_tests::{name}")))
}

fn identity(shifts: &[[Atom; 4]], masses: &[Atom], coefficients: &[Atom]) {
    let q: [Atom; 4] = std::array::from_fn(|mu| variable(&format!("q{mu}")));
    let sum =
        shifts
            .iter()
            .zip(masses)
            .zip(coefficients)
            .fold(Atom::new(), |sum, ((p, mass), c)| {
                let momentum = std::array::from_fn(|mu| &q[mu] + &p[mu]);
                sum + c * (dot(&momentum, &momentum) - mass)
            });
    assert!((sum - Atom::num(1)).together().is_zero());
}

#[test]
fn fixture_coefficients_are_derived_and_match_existing_denominator_solver() {
    let (shifts, masses) = fixture();
    let chart = triangle_affine_chart(&shifts, &masses, 1).unwrap().unwrap();
    let expected = [
        -Atom::num(1) / Atom::num(2),
        Atom::num(1),
        -Atom::num(1) / Atom::num(2),
    ];
    for (actual, expected) in chart.coefficients.iter().zip(expected) {
        assert!((actual - expected).together().is_zero());
    }
    let mut existing = PvCompiler::new(shifts.clone(), masses.clone()).unwrap();
    let factors = existing.denominator_identity(&[0, 1, 2]).unwrap();
    for (actual, old) in chart.coefficients.iter().zip(factors) {
        assert!((actual - old).together().is_zero());
    }
    identity(&shifts, &masses, &chart.coefficients);
    assert!(
        triangle_affine_chart(&shifts, &masses, 0)
            .unwrap()
            .is_none()
    );
}

#[test]
fn dynamic_translated_routing_and_masses_keep_exact_same_q_identity() {
    let translation: [Atom; 4] = std::array::from_fn(|mu| variable(&format!("a{mu}")));
    let vector: [Atom; 4] = std::array::from_fn(|mu| variable(&format!("v{mu}")));
    let ratio = variable("ratio");
    let shifts = vec![
        translation.clone(),
        std::array::from_fn(|mu| &translation[mu] + &vector[mu]),
        std::array::from_fn(|mu| &translation[mu] + &ratio * &vector[mu]),
    ];
    let masses: Vec<_> = (0..3).map(|i| variable(&format!("m{i}"))).collect();
    let chart = triangle_affine_chart(&shifts, &masses, 2).unwrap().unwrap();
    identity(&shifts, &masses, &chart.coefficients);
    for coefficient in &chart.coefficients {
        assert!(
            translation
                .iter()
                .all(|a| !coefficient.contains(a.as_view()))
        );
    }
    let symbolica::atom::AtomView::Var(mass) = masses[2].as_view() else {
        unreachable!()
    };
    assert!(
        chart
            .coefficients
            .iter()
            .any(|c| !c.derivative(mass.get_symbol()).together().is_zero()),
        "mass dependence must remain symbolic, including its Dual derivatives"
    );
    let predicates = triangle_affine_polynomials(&shifts, &masses).unwrap();
    assert!(
        predicates
            .rank_constraints
            .iter()
            .all(|p| p.together().is_zero())
    );
}

#[test]
fn all_coordinate_axes_and_reordered_denominators_use_derived_charts() {
    for axis in 0..4 {
        let shifts: Vec<[Atom; 4]> = (0..3)
            .map(|i| std::array::from_fn(|mu| Atom::num(if mu == axis { i as i64 } else { 0 })))
            .collect();
        let masses: Vec<_> = [1, 2, 3].into_iter().map(Atom::num).collect();
        let chart = triangle_affine_chart(&shifts, &masses, axis)
            .unwrap()
            .unwrap();
        identity(&shifts, &masses, &chart.coefficients);
        let reversed_shifts: Vec<_> = shifts.iter().rev().cloned().collect();
        let reversed_masses: Vec<_> = masses.iter().rev().cloned().collect();
        let reversed = triangle_affine_chart(&reversed_shifts, &reversed_masses, axis)
            .unwrap()
            .unwrap();
        identity(&reversed_shifts, &reversed_masses, &reversed.coefficients);
        for i in 0..3 {
            assert!(
                (&chart.coefficients[i] - &reversed.coefficients[2 - i])
                    .together()
                    .is_zero()
            );
        }
    }
}

#[test]
fn rank_six_recipe_has_all_210_moments_and_only_existing_bubble_children() {
    let (shifts, masses) = fixture();
    let mut compiler = PvCompiler::new(shifts, masses).unwrap();
    let basis: Vec<_> = tensor_basis(6).into_iter().map(|m| m.0).collect();
    let moments = compiler
        .compile_affine_descent_chart_moments(&key(1), &[], &basis, AffineDescentBudget::default())
        .unwrap();
    assert_eq!(moments.len(), 210);
    assert!(compiler.leaves().iter().all(|leaf| leaf.active.len() <= 2));
    assert!(compiler.leaves().iter().any(|leaf| leaf.active == [0, 1]));
    assert!(compiler.leaves().iter().any(|leaf| leaf.active == [0, 2]));
    assert!(compiler.leaves().iter().any(|leaf| leaf.active == [1, 2]));
    assert!(
        !compiler.geometry.contains_key(&vec![0, 1, 2]),
        "no triangle Gram inverse may be prepared"
    );
    assert!(!compiler.moments.keys().any(|(active, _)| active.len() == 3));
}

#[test]
fn no_chart_is_an_error_and_does_not_generate_successful_zero_outputs() {
    let (shifts, _) = fixture();
    let masses: Vec<_> = [1, 2, 1].into_iter().map(Atom::num).collect();
    let predicates = triangle_affine_polynomials(&shifts, &masses).unwrap();
    assert!(
        predicates
            .charts
            .iter()
            .all(|(_, determinant)| determinant.is_zero())
    );
    let mut compiler = PvCompiler::new(shifts, masses).unwrap();
    assert!(
        compiler
            .compile_affine_descent_chart_moments(
                &key(1),
                &[],
                &[[0; 4]],
                AffineDescentBudget::default()
            )
            .is_err()
    );
    assert!(compiler.leaves().is_empty());
}

fn key(axis: usize) -> AffineChartKey {
    AffineChartKey {
        coordinate_rows: vec![axis],
        relative_columns: vec![0, 1],
    }
}
fn triangle_affine_chart(
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    axis: usize,
) -> Result<Option<AffineDescentChart>, String> {
    affine_descent_chart(shifts, masses, &key(axis), AffineDescentBudget::default())
}
fn triangle_affine_polynomials(
    shifts: &[[Atom; 4]],
    masses: &[Atom],
) -> Result<AffineDescentPolynomials, String> {
    affine_descent_polynomials(shifts, masses, 1, AffineDescentBudget::default())
}

#[test]
fn generic_profile_index_layouts_and_small_exact_resource_checks() {
    let budget = AffineDescentBudget::default();
    // These are index-only contracts: no symbolic N6/N7 determinant or
    // cofactor expressions are prepared or numerically qualified here.
    for (n, rank, expected_constraints, expected_charts, expected_cofactors) in [
        (2, 0, 4, 1, 2),
        (3, 1, 6, 4, 12),
        (4, 2, 4, 6, 24),
        (5, 3, 1, 4, 20),
        (6, 4, 0, 1, 6),
        (7, 4, 0, 6, 42),
    ] {
        let minor_rows = subsets(4, rank + 1, budget.max_constraints).unwrap();
        let minor_columns = if minor_rows.is_empty() { Vec::new() }
            else { subsets(n - 1, rank + 1, budget.max_constraints).unwrap() };
        let rows = subsets(4, rank, budget.max_charts).unwrap();
        let columns = subsets(n - 1, rank + 1, budget.max_charts).unwrap();
        assert_eq!(minor_rows.len() * minor_columns.len(), expected_constraints);
        assert_eq!(rows.len() * columns.len(), expected_charts);
        assert_eq!(rows.len() * columns.len() * n, expected_cofactors);
    }
    assert_eq!(subsets(6, 5, budget.max_charts).unwrap(), [
        vec![0, 1, 2, 3, 4], vec![0, 1, 2, 3, 5], vec![0, 1, 2, 4, 5],
        vec![0, 1, 3, 4, 5], vec![0, 2, 3, 4, 5], vec![1, 2, 3, 4, 5],
    ]);
    // Retain actual Matrix-based output-layout and resource rejection checks
    // on the immediate exact N3 fixture, rather than preparing large systems.
    let (shifts, masses) = fixture();
    let prepared = affine_descent_polynomials(&shifts, &masses, 1, budget).unwrap();
    assert_eq!((prepared.rank_constraints.len(), prepared.charts.len(), prepared.cofactors.len()), (6, 4, 12));
    let mut limited = budget; limited.max_charts = 2;
    assert!(affine_descent_polynomials(&shifts, &masses, 1, limited).is_err());
    let mut limited = budget; limited.max_cofactors = 1;
    assert!(affine_descent_polynomials(&shifts, &masses, 1, limited).is_err());
    assert!(affine_descent_polynomials(&shifts, &masses, 2, budget).is_err());
}

#[test]
#[ignore="expensive full symbolic N2..N7 preparation stress; prior fixed90 timeout retained; lightweight normal layout test is separate"]
fn generic_profile_layouts_and_explicit_resource_limits_remain_finite() {
    for (n, rank, constraints, charts) in [
        (2, 0, 4, 1),
        (3, 1, 6, 4),
        (4, 2, 4, 6),
        (5, 3, 1, 4),
        (6, 4, 0, 1),
        (7, 4, 0, 6),
    ] {
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| std::array::from_fn(|mu| variable(&format!("layout_p{i}_{mu}"))))
            .collect();
        let masses: Vec<_> = (0..n).map(|i| variable(&format!("layout_m{i}"))).collect();
        let p = affine_descent_polynomials(&shifts, &masses, rank, AffineDescentBudget::default())
            .unwrap();
        assert_eq!(p.rank_constraints.len(), constraints);
        assert_eq!(p.charts.len(), charts);
        assert_eq!(p.cofactors.len(), charts * n);
    }
    let (shifts, masses) = fixture();
    let mut budget = AffineDescentBudget::default();
    budget.max_charts = 2;
    assert!(affine_descent_polynomials(&shifts, &masses, 1, budget).is_err());
    let mut budget = AffineDescentBudget::default();
    budget.max_cofactors = 1;
    assert!(affine_descent_polynomials(&shifts, &masses, 1, budget).is_err());
    assert!(
        affine_descent_polynomials(&shifts, &masses, 2, AffineDescentBudget::default()).is_err()
    );
}

#[test]
fn every_raw_cramer_numerator_matches_solve_and_omitted_coordinate_residuals() {
    let translation: [Atom; 4] = std::array::from_fn(|mu| variable(&format!("residual_a{mu}")));
    let vector: [Atom; 4] = std::array::from_fn(|mu| variable(&format!("residual_v{mu}")));
    let ratio = variable("residual_ratio");
    let shifts = vec![
        translation.clone(),
        std::array::from_fn(|mu| &translation[mu] + &vector[mu]),
        std::array::from_fn(|mu| &translation[mu] + &ratio * &vector[mu]),
    ];
    let masses: Vec<_> = (0..3)
        .map(|i| variable(&format!("residual_m{i}")))
        .collect();
    let p =
        affine_descent_polynomials(&shifts, &masses, 1, AffineDescentBudget::default()).unwrap();
    for (index, (key, determinant)) in p.charts.iter().enumerate() {
        let chart = affine_descent_chart(&shifts, &masses, key, AffineDescentBudget::default())
            .unwrap()
            .unwrap();
        let numerators = &p.cofactors[index * 3..(index + 1) * 3];
        for child in 0..3 {
            assert!(
                (&chart.coefficients[child] * determinant - &numerators[child])
                    .together()
                    .is_zero()
            );
        }
        assert!(Atom::add_many(numerators).together().is_zero());
        for mu in 0..4 {
            let residual = (0..3).fold(Atom::new(), |sum, child| {
                sum + &numerators[child] * &shifts[child][mu]
            });
            assert!(
                residual.together().is_zero(),
                "including every omitted coordinate row"
            );
        }
        let constant = (0..3).fold(Atom::new(), |sum, child| {
            sum + &numerators[child] * (dot(&shifts[child], &shifts[child]) - &masses[child])
        });
        assert!((constant - determinant).together().is_zero());
    }
}

#[test]
fn certified_zero_mask_omits_singular_child_before_it_is_compiled() {
    let shifts: Vec<[Atom; 4]> = (0..3)
        .map(|i| std::array::from_fn(|mu| variable(&format!("mask_p{i}_{mu}"))))
        .collect();
    let masses: Vec<_> = (0..3).map(|i| variable(&format!("mask_m{i}"))).collect();
    let mut compiler = PvCompiler::with_aliases(shifts, masses).unwrap();
    compiler
        .compile_affine_descent_chart_moments(
            &key(1),
            &[2],
            &[[0; 4]],
            AffineDescentBudget::default(),
        )
        .unwrap();
    assert!(!compiler.moments.contains_key(&(vec![0, 1], [0; 4])));
    assert!(!compiler.leaves().iter().any(|leaf| leaf.active == [0, 1]));
    assert!(compiler.leaves().iter().any(|leaf| leaf.active == [0, 2]));
    assert!(compiler.leaves().iter().any(|leaf| leaf.active == [1, 2]));
}
