//! Private proposal: change only exact routing-system preparation arithmetic.
//!
//! The recurrence, support cache, relation ordering and numerical evaluator are
//! unchanged. Symbolica performs conversion, elimination and verification.
//! Explicit affine entries with exact rational/Gaussian coefficients use the
//! library rational-polynomial field; every other input retains AtomField.

use std::sync::Arc;

use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    coefficient::ConvertToRing,
    domains::{
        algebraic::AlgebraicExtension,
        integer::{IntegerRing, Z},
        rational::{Q, RationalField},
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    poly::PolyVariable,
    tensors::matrix::Matrix,
};

type RationalEntry = RationalPolynomial<IntegerRing, u16>;
type GaussianField = AlgebraicExtension<RationalField>;
type GaussianEntry = RationalPolynomial<GaussianField, u16>;

// Test-only observation proves which domain the real ScalarCompiler caller
// selects. Thread-local storage cannot mix traces from parallel test threads.
#[cfg(test)]
thread_local! {
    pub(super) static TEST_DOMAINS: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
    pub(super) static TEST_INVERSE_PREPARATION_ERROR: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

enum RoutingEntries {
    Rational(Vec<Vec<RationalEntry>>),
    Gaussian(Vec<Vec<GaussianEntry>>),
    Atom(Vec<Vec<Atom>>),
}

/// One converted five-row routing system, reusable across constrained solves.
pub(super) struct PreparedRoutingSystem {
    entries: RoutingEntries,
    variables: Arc<Vec<PolyVariable>>,
    columns: usize,
}

impl PreparedRoutingSystem {
    pub(super) fn new(rows: Vec<Vec<Atom>>) -> Result<Self, String> {
        let columns = rows.first().map_or(0, Vec::len);
        // Preserve the original matrix shape validation before domain selection.
        Matrix::from_nested_vec(rows.clone(), super::exact_field())
            .map_err(|error| format!("routing denominator matrix: {error:?}"))?;
        let gaussian = GaussianField::complex(Q);
        // The private fast subset is a <=5-by-6 matrix of explicit sums of
        // constants and scalar multiples of bare symbols. It includes current
        // generic N6/N7 routing matrices with squared-distance aliases. Keeping
        // other syntax on AtomField avoids imposing u16 exponent limits on
        // arbitrary direct caller expressions. This is a syntactic whitelist;
        // no expansion, degree arithmetic or local polynomial engine is used.
        let mut affine = rows.len() <= 5 && columns <= 6;
        let mut rational = true;
        let mut gaussian_coefficients = true;
        let mut variables = Vec::new();
        for entry in rows.iter().flatten() {
            entry.visitor(&mut |atom| {
                match atom {
                    AtomView::Var(_) => {
                        let variable = PolyVariable::try_from(atom.to_owned())
                            .expect("a bare Symbolica variable is a polynomial variable");
                        if !variables.contains(&variable) {
                            variables.push(variable);
                        }
                    }
                    AtomView::Num(number) => {
                        rational &= Q
                            .try_element_from_coefficient_view(number.get_coeff_view())
                            .is_ok();
                        gaussian_coefficients &= gaussian
                            .try_element_from_coefficient_view(number.get_coeff_view())
                            .is_ok();
                    }
                    AtomView::Mul(product) => {
                        let mut symbols = 0;
                        for factor in product {
                            match factor {
                                AtomView::Num(_) => {}
                                AtomView::Var(_) => symbols += 1,
                                _ => affine = false,
                            }
                        }
                        affine &= symbols <= 1;
                    }
                    AtomView::Pow(_) | AtomView::Fun(_) => affine = false,
                    AtomView::Add(_) => {}
                }
                true
            });
        }
        let variables = Arc::new(variables);
        let entries = if affine && rational {
            let entries = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| rational_entry(entry, &variables))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            RoutingEntries::Rational(entries)
        } else if affine && gaussian_coefficients {
            let entries = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|entry| gaussian_entry(entry, &variables))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            RoutingEntries::Gaussian(entries)
        } else {
            RoutingEntries::Atom(rows)
        };
        #[cfg(test)]
        TEST_DOMAINS.with(|domains| {
            domains.borrow_mut().push(match &entries {
                RoutingEntries::Rational(_) => "rational",
                RoutingEntries::Gaussian(_) => "gaussian",
                RoutingEntries::Atom(_) => "atom",
            });
        });
        Ok(Self {
            entries,
            variables,
            columns,
        })
    }

    /// Solve the original rows, optionally appending x_fixed=rhs_last.
    /// Only the routing caller's literal zero/one RHS entries are admitted.
    /// `None` means the same linear-solve failure that the old caller ignores
    /// while searching for a homogeneous relation. Conversion/certificate
    /// failures are explicit errors and never select a different algebra path.
    pub(super) fn solve_any(
        &self,
        rhs: &[Atom],
        fixed: Option<usize>,
    ) -> Result<Option<Vec<Atom>>, String> {
        if fixed.is_some_and(|index| index >= self.columns) {
            return Err("routing constraint is outside the matrix".into());
        }
        if rhs
            .iter()
            .any(|entry| !entry.is_zero() && entry != &Atom::one())
        {
            return Err("routing right-hand side must contain literal zeros and ones".into());
        }
        macro_rules! solve_exact {
            ($entries:expr, $field:expr, $convert:ident) => {{
                let mut entries = $entries.clone();
                if let Some(fixed) = fixed {
                    entries.push(
                        (0..self.columns)
                            .map(|column| {
                                $convert(&Atom::num(i64::from(column == fixed)), &self.variables)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                let field = $field;
                let matrix = Matrix::from_nested_vec(entries, field.clone())
                    .map_err(|error| format!("routing exact matrix: {error:?}"))?;
                let rhs = Matrix::new_vec(
                    rhs.iter()
                        .map(|entry| $convert(entry, &self.variables))
                        .collect::<Result<Vec<_>, _>>()?,
                    field,
                );
                match matrix.solve_any(&rhs) {
                    Ok(solution) => {
                        if &matrix * &solution != rhs {
                            return Err("routing exact solution identity failed".into());
                        }
                        Ok(Some(
                            solution
                                .into_vec()
                                .into_iter()
                                .map(|entry| entry.to_expression())
                                .collect(),
                        ))
                    }
                    Err(_) => Ok(None),
                }
            }};
        }
        match &self.entries {
            RoutingEntries::Rational(entries) => {
                solve_exact!(entries, RationalPolynomialField::new(Z), rational_entry)
            }
            RoutingEntries::Gaussian(entries) => solve_exact!(
                entries,
                RationalPolynomialField::new(GaussianField::complex(Q)),
                gaussian_entry
            ),
            RoutingEntries::Atom(entries) => {
                let mut entries = entries.clone();
                if let Some(fixed) = fixed {
                    entries.push(
                        (0..self.columns)
                            .map(|column| Atom::num(i64::from(column == fixed)))
                            .collect(),
                    );
                }
                let matrix = Matrix::from_nested_vec(entries, super::exact_field())
                    .map_err(|error| format!("routing fallback matrix: {error:?}"))?;
                let rhs = Matrix::new_vec(rhs.to_vec(), super::exact_field());
                Ok(matrix.solve_any(&rhs).ok().map(Matrix::into_vec))
            }
        }
    }

    /// Strict inverse for the existing eligible affine five-point geometry.
    /// Ineligible syntax and every other shape retain the caller's AtomField
    /// inverse. Symbolica performs inversion and both identity certificates;
    /// no sampled values or Taylor components enter this preparation step.
    pub(super) fn inverse_five(&self) -> Result<Option<Vec<Vec<Atom>>>, super::GeometryError> {
        let rows = match &self.entries {
            RoutingEntries::Rational(rows) => rows.len(),
            RoutingEntries::Gaussian(rows) => rows.len(),
            RoutingEntries::Atom(rows) => rows.len(),
        };
        if self.columns != 5 || rows != 5 {
            return Ok(None);
        }
        macro_rules! invert_exact {
            ($entries:expr, $field:expr, $domain:literal) => {{
                #[cfg(test)]
                TEST_DOMAINS.with(|domains| domains.borrow_mut().push($domain));
                #[cfg(test)]
                if let Some(error) =
                    TEST_INVERSE_PREPARATION_ERROR.with(|slot| slot.borrow_mut().take())
                {
                    return Err(super::GeometryError::Preparation(error));
                }
                let field = $field;
                let matrix =
                    Matrix::from_nested_vec($entries.clone(), field.clone()).map_err(|error| {
                        super::GeometryError::Preparation(format!(
                            "exact Cayley inverse matrix: {error:?}"
                        ))
                    })?;
                let inverse = matrix.inv().map_err(|error| match error {
                    symbolica::tensors::matrix::MatrixError::Singular => {
                        super::GeometryError::Singular("Singular".into())
                    }
                    other => super::GeometryError::Preparation(format!("{other:?}")),
                })?;
                let identity = Matrix::identity(5, field);
                if &matrix * &inverse != identity || &inverse * &matrix != identity {
                    return Err(super::GeometryError::Preparation(
                        "exact Cayley inverse identity failed".into(),
                    ));
                }
                let entries: Vec<_> = inverse
                    .into_vec()
                    .into_iter()
                    .map(|entry| entry.to_expression())
                    .collect();
                Ok(Some(
                    entries.chunks_exact(5).map(<[Atom]>::to_vec).collect(),
                ))
            }};
        }
        match &self.entries {
            RoutingEntries::Rational(entries) => {
                invert_exact!(entries, RationalPolynomialField::new(Z), "rational_inverse")
            }
            RoutingEntries::Gaussian(entries) => invert_exact!(
                entries,
                RationalPolynomialField::new(GaussianField::complex(Q)),
                "gaussian_inverse"
            ),
            RoutingEntries::Atom(_) => Ok(None),
        }
    }
}

fn rational_entry(
    entry: &Atom,
    variables: &Arc<Vec<PolyVariable>>,
) -> Result<RationalEntry, String> {
    let value: RationalEntry = entry
        .try_to_rational_polynomial(&Q, &Z, Some(variables.clone()))
        .map_err(|error| format!("routing rational conversion: {error}"))?;
    if value.get_variables().as_ref() != variables.as_ref() || !value.denominator.is_constant() {
        return Err("routing rational conversion violated its polynomial variable map".into());
    }
    Ok(value)
}

fn gaussian_entry(
    entry: &Atom,
    variables: &Arc<Vec<PolyVariable>>,
) -> Result<GaussianEntry, String> {
    let gaussian = GaussianField::complex(Q);
    let value: GaussianEntry = entry
        .try_to_rational_polynomial(&gaussian, &gaussian, Some(variables.clone()))
        .map_err(|error| format!("routing Gaussian conversion: {error}"))?;
    if value.get_variables().as_ref() != variables.as_ref() || !value.denominator.is_constant() {
        return Err("routing Gaussian conversion violated its polynomial variable map".into());
    }
    Ok(value)
}

#[cfg(test)]
#[path = "cayley_inverse_tests.rs"]
mod cayley_inverse_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::parse;

    fn verify(rows: &[Vec<Atom>], rhs: &[Atom], fixed: Option<usize>, values: &[Atom]) {
        assert_eq!(rhs.len(), rows.len() + usize::from(fixed.is_some()));
        assert!(rows.iter().all(|row| row.len() == values.len()));
        for (row, expected) in rows.iter().zip(rhs) {
            let actual = row
                .iter()
                .zip(values)
                .map(|(coefficient, value)| coefficient * value)
                .sum::<Atom>();
            assert!((actual - expected).together().is_zero());
        }
        if let Some(fixed) = fixed {
            assert!((&values[fixed] - rhs.last().unwrap()).together().is_zero());
        }
    }

    #[test]
    fn inhomogeneous_polynomial_and_distance_alias_rows_reconstruct_exactly() {
        let rows = vec![
            vec![parse!("routing_test::p"), Atom::one()],
            vec![parse!("routing_test::s-routing_test::m"), Atom::num(2)],
        ];
        let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
        assert!(matches!(system.entries, RoutingEntries::Rational(_)));
        let rhs = vec![Atom::Zero, Atom::one()];
        let values = system.solve_any(&rhs, None).unwrap().unwrap();
        verify(&rows, &rhs, None, &values);
        let old = Matrix::from_nested_vec(rows, super::super::exact_field())
            .unwrap()
            .solve_any(&Matrix::new_vec(rhs, super::super::exact_field()))
            .unwrap()
            .into_vec();
        for (new, old) in values.iter().zip(old) {
            assert!((new - old).together().is_zero());
        }
    }

    #[test]
    fn underdetermined_solution_keeps_original_library_free_variable_choice() {
        let rows = vec![
            vec![Atom::one(), Atom::Zero, Atom::one()],
            vec![Atom::Zero, Atom::one(), Atom::one()],
        ];
        let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
        let rhs = vec![Atom::Zero, Atom::one()];
        let values = system.solve_any(&rhs, None).unwrap().unwrap();
        assert_eq!(values, vec![Atom::Zero, Atom::one(), Atom::Zero]);
        verify(&rows, &rhs, None, &values);
    }

    #[test]
    fn inconsistent_constant_then_constrained_homogeneous_relation_is_preserved() {
        let rows = vec![
            vec![Atom::one(), Atom::num(2)],
            vec![Atom::num(2), Atom::num(4)],
        ];
        let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
        assert!(
            system
                .solve_any(&[Atom::Zero, Atom::one()], None)
                .unwrap()
                .is_none()
        );
        let rhs = vec![Atom::Zero, Atom::Zero, Atom::one()];
        let values = system.solve_any(&rhs, Some(0)).unwrap().unwrap();
        verify(&rows, &rhs, Some(0), &values);
        assert_eq!(values, vec![Atom::one(), -Atom::one() / Atom::num(2)]);
    }

    #[test]
    fn gaussian_coefficients_use_exact_existing_complex_field() {
        let rows = vec![
            vec![parse!("𝑖"), Atom::one()],
            vec![Atom::one(), parse!("𝑖")],
        ];
        let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
        assert!(matches!(system.entries, RoutingEntries::Gaussian(_)));
        let rhs = vec![Atom::one(), Atom::Zero];
        let values = system.solve_any(&rhs, None).unwrap().unwrap();
        verify(&rows, &rhs, None, &values);
        assert!(
            (&values[0] + parse!("𝑖") / Atom::num(2))
                .together()
                .is_zero()
        );
        assert!(
            (&values[1] - Atom::one() / Atom::num(2))
                .together()
                .is_zero()
        );
    }

    #[test]
    fn rational_and_gaussian_fractional_symbolic_entries_roundtrip_exactly() {
        for (coefficient, gaussian) in [
            (parse!("routing_test::a/3+1/7"), false),
            (parse!("(1+𝑖)*routing_test::a/3+(2-𝑖)/7"), true),
        ] {
            let rows = vec![vec![coefficient]];
            let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
            assert_eq!(
                matches!(system.entries, RoutingEntries::Gaussian(_)),
                gaussian
            );
            let rhs = vec![Atom::one()];
            let values = system.solve_any(&rhs, None).unwrap().unwrap();
            verify(&rows, &rhs, None, &values);
        }
    }

    #[test]
    fn real_n6_n7_five_row_systems_match_atom_field() {
        for n in [6, 7] {
            let mut rows = vec![
                vec![Atom::one(), Atom::Zero, Atom::Zero, Atom::Zero, Atom::one()],
                vec![Atom::Zero, Atom::one(), Atom::Zero, Atom::Zero, Atom::one()],
                vec![Atom::Zero, Atom::Zero, Atom::one(), Atom::Zero, Atom::one()],
                vec![Atom::Zero, Atom::Zero, Atom::Zero, Atom::one(), Atom::one()],
                vec![
                    Atom::one(),
                    Atom::num(-1),
                    Atom::num(-1),
                    Atom::num(-1),
                    parse!("-2-routing_test::delta"),
                ],
            ];
            if n == 7 {
                for (row, value) in rows.iter_mut().zip([
                    Atom::num(2),
                    Atom::one(),
                    Atom::Zero,
                    Atom::one(),
                    parse!("-routing_test::theta"),
                ]) {
                    row.push(value);
                }
            }
            let rhs = vec![Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero, Atom::one()];
            let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
            assert!(matches!(system.entries, RoutingEntries::Rational(_)));
            let values = system.solve_any(&rhs, None).unwrap().unwrap();
            verify(&rows, &rhs, None, &values);
            let old = Matrix::from_nested_vec(rows, super::super::exact_field())
                .unwrap()
                .solve_any(&Matrix::new_vec(rhs, super::super::exact_field()))
                .unwrap()
                .into_vec();
            assert_eq!(values.len(), n - 1);
            for (new, old) in values.iter().zip(old) {
                assert!((new - old).together().is_zero());
            }
        }
    }

    #[test]
    fn homogeneous_search_can_skip_an_inconsistent_first_fixed_column() {
        let rows = vec![vec![Atom::one(), Atom::Zero], vec![Atom::Zero, Atom::Zero]];
        let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
        let rhs = vec![Atom::Zero, Atom::Zero, Atom::one()];
        assert!(system.solve_any(&rhs, Some(0)).unwrap().is_none());
        let values = system.solve_any(&rhs, Some(1)).unwrap().unwrap();
        verify(&rows, &rhs, Some(1), &values);
    }

    #[test]
    fn high_degree_or_nonaffine_entries_keep_original_atom_field() {
        for coefficient in [
            parse!("(routing_test::x^40000+routing_test::y)^2"),
            parse!("routing_test::x*routing_test::y"),
            parse!("routing_test::x*(routing_test::y+1)"),
            parse!("routing_test::x^2"),
        ] {
            let system = PreparedRoutingSystem::new(vec![vec![coefficient]]).unwrap();
            assert!(matches!(system.entries, RoutingEntries::Atom(_)));
        }
        // Generic interfaces above seven propagators retain their old domain.
        let system = PreparedRoutingSystem::new(vec![vec![Atom::one(); 7]; 5]).unwrap();
        assert!(matches!(system.entries, RoutingEntries::Atom(_)));
    }

    #[test]
    fn opaque_and_nonpolynomial_coefficients_keep_original_atom_field() {
        for coefficient in [parse!("sin(routing_test::a)"), parse!("1/routing_test::a")] {
            let rows = vec![vec![coefficient]];
            let system = PreparedRoutingSystem::new(rows.clone()).unwrap();
            assert!(matches!(system.entries, RoutingEntries::Atom(_)));
            let rhs = vec![Atom::one()];
            let values = system.solve_any(&rhs, None).unwrap().unwrap();
            verify(&rows, &rhs, None, &values);
        }
    }

    #[test]
    fn floating_literals_are_not_silently_rationalized() {
        let system = PreparedRoutingSystem::new(vec![vec![Atom::num(0.5)]]).unwrap();
        assert!(matches!(system.entries, RoutingEntries::Atom(_)));
    }

    #[test]
    fn invalid_constraint_and_unknown_rhs_symbols_fail_explicitly() {
        let system = PreparedRoutingSystem::new(vec![vec![Atom::one()]]).unwrap();
        assert!(
            system
                .solve_any(&[Atom::Zero, Atom::one()], Some(1))
                .is_err()
        );
        assert!(
            system
                .solve_any(&[parse!("routing_test::undeclared")], None)
                .is_err()
        );
        let system = PreparedRoutingSystem::new(vec![vec![parse!("routing_test::x")]]).unwrap();
        assert!(
            system
                .solve_any(&[parse!("routing_test::x^70000")], None)
                .is_err()
        );
    }

    #[test]
    fn actual_scalar_compiler_generic_distance_aliases_select_rational_routing() {
        // This follows plan.rs::compile_scalar_tensors: generic Cartesian
        // coordinates, bare independent distance aliases, symbolic masses,
        // with_aliases/new+enable_aliases, then set_routing. with_aliases alone
        // does not introduce the s_ij aliases; the plan caller creates them.
        for n in [6, 7] {
            let shifts: Vec<[Atom; 4]> = (0..n)
                .map(|i| {
                    std::array::from_fn(|mu| {
                        Atom::var(symbolica::symbol!(format!("routing_actual::p{i}_{mu}")))
                    })
                })
                .collect();
            let masses: Vec<_> = (0..n)
                .map(|i| Atom::var(symbolica::symbol!(format!("routing_actual::m2_{i}"))))
                .collect();
            let distances: Vec<Vec<_>> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            if i == j {
                                Atom::Zero
                            } else {
                                Atom::var(symbolica::symbol!(format!(
                                    "routing_actual::s{}_{}",
                                    i.min(j),
                                    i.max(j)
                                )))
                            }
                        })
                        .collect()
                })
                .collect();
            let r = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| &distances[i][j] - &masses[i] - &masses[j])
                        .collect()
                })
                .collect();
            let mut compiler = super::super::ScalarCompiler::with_aliases(r).unwrap();
            compiler.set_routing(shifts).unwrap();
            TEST_DOMAINS.with(|domains| domains.borrow_mut().clear());
            let active: Vec<_> = (0..n).collect();
            let relation = compiler.routing_relation(&active).unwrap().unwrap();
            assert!(relation.constant_one);
            assert_eq!(relation.coefficients.len(), n);
            assert_eq!(compiler.routing_relations.len(), 1);
            TEST_DOMAINS.with(|domains| {
                assert_eq!(domains.borrow().as_slice(), ["rational"]);
                domains.borrow_mut().clear();
            });
            // The second query must reuse the structural relation cache.
            assert!(
                compiler
                    .routing_relation(&active)
                    .unwrap()
                    .unwrap()
                    .constant_one
            );
            TEST_DOMAINS.with(|domains| assert!(domains.borrow().is_empty()));
        }
    }

    #[test]
    fn actual_scalar_compiler_unaliased_quadratic_routing_keeps_atom_field() {
        let a = parse!("routing_unaliased::a");
        let mass = parse!("routing_unaliased::m2");
        let delta = parse!("routing_unaliased::delta");
        let shifts = vec![
            [Atom::Zero, Atom::Zero, Atom::Zero, Atom::Zero],
            [a.clone(), Atom::Zero, Atom::Zero, Atom::Zero],
            [Atom::Zero, Atom::one(), Atom::Zero, Atom::Zero],
            [Atom::Zero, Atom::Zero, Atom::one(), Atom::Zero],
            [Atom::Zero, Atom::Zero, Atom::Zero, Atom::one()],
            [a, Atom::one(), Atom::one(), Atom::one()],
        ];
        let masses: Vec<_> = (0..6)
            .map(|i| if i == 5 { &mass + &delta } else { mass.clone() })
            .collect();
        let r = (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| {
                        let d: [Atom; 4] =
                            std::array::from_fn(|mu| &shifts[i][mu] - &shifts[j][mu]);
                        d[0].pow(2)
                            - d[1].pow(2)
                            - d[2].pow(2)
                            - d[3].pow(2)
                            - &masses[i]
                            - &masses[j]
                    })
                    .collect()
            })
            .collect();
        let mut compiler = super::super::ScalarCompiler::with_aliases(r).unwrap();
        compiler.set_routing(shifts).unwrap();
        TEST_DOMAINS.with(|domains| domains.borrow_mut().clear());
        let relation = compiler
            .routing_relation(&[0, 1, 2, 3, 4, 5])
            .unwrap()
            .unwrap();
        assert!(relation.constant_one);
        TEST_DOMAINS.with(|domains| assert_eq!(domains.borrow().as_slice(), ["atom"]));
        let coefficients: Vec<_> = relation
            .coefficients
            .iter()
            .map(|coefficient| {
                compiler
                    .aliases()
                    .iter()
                    .rev()
                    .fold(coefficient.clone(), |value, (alias, body)| {
                        value.replace(alias.clone()).with(body.clone())
                    })
            })
            .collect();
        for (coefficient, expected) in coefficients.iter().zip([-3, 1, 1, 1, 1, -1]) {
            assert!(
                (coefficient * &delta - Atom::num(expected))
                    .together()
                    .is_zero()
            );
        }
    }
}
