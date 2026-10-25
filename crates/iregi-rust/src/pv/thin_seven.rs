//! Geometry for the private thin seven-point dispatcher.
//!
//! These are exact preserved-coordinate identities. No tensor child is prepared
//! or evaluated here. Symbolica owns determinants, inverses, matrix products and
//! coefficient arithmetic; the integer maps only select ordered propagators.

use super::{Atom, AtomCore, Matrix, PreparationTimer, PvCompiler, coordinate_rows, dot, field};
use std::sync::Arc;
use symbolica::{
    atom::AtomView,
    coefficient::ConvertToRing,
    domains::{
        algebraic::AlgebraicExtension,
        integer::{IntegerRing, Z},
        rational::{Q, RationalField},
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    poly::PolyVariable,
};

type GeometryCoefficient = RationalPolynomial<IntegerRing, u16>;
type GeometryField = RationalPolynomialField<IntegerRing, u16>;

/// Preserve one complete ordered symbol map for every entry of a chart.
/// Generic plans use the 28 shift symbols followed by the 7 mass-squared symbols;
/// exact source fixtures may use fewer symbols. Opaque functions are excluded.
fn geometry_variables(compiler: &PvCompiler) -> Result<Arc<Vec<PolyVariable>>, String> {
    let mut variables = Vec::new();
    for value in compiler
        .shifts
        .iter()
        .flatten()
        .chain(&compiler.mass_squared)
    {
        for variable in value.get_all_indeterminates(true) {
            if !matches!(variable, AtomView::Var(_)) {
                return Err("thin geometry requires polynomial coordinates and masses".into());
            }
            let variable = PolyVariable::try_from(variable.to_owned())
                .map_err(|error| format!("thin geometry variable: {error}"))?;
            if !variables.contains(&variable) {
                variables.push(variable);
            }
        }
    }
    Ok(Arc::new(variables))
}

/// Symbolica converts and normalizes every entry once, before elimination.
/// Conversion can append indeterminates, so enforce the complete declared map.
fn exact_geometry_matrix(
    matrix: &Matrix<super::AtomField>,
    variables: &Arc<Vec<PolyVariable>>,
) -> Result<Matrix<GeometryField>, String> {
    let rows = matrix
        .row_iter()
        .map(|row| {
            row.iter()
                .map(|entry| {
                    let value: GeometryCoefficient = entry
                        .try_to_rational_polynomial(&Q, &Z, Some(variables.clone()))
                        .map_err(|error| format!("thin geometry exact conversion: {error}"))?;
                    if value.get_variables().as_ref() != variables.as_ref() {
                        return Err("thin geometry conversion added undeclared variables".into());
                    }
                    if !value.denominator.is_constant() {
                        return Err("thin geometry matrix entries must be polynomials".into());
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Matrix::from_nested_vec(rows, GeometryField::new(Z))
        .map_err(|error| format!("thin geometry exact matrix: {error}"))
}

/// Validate the exact complex compatibility inputs using Symbolica's existing
/// Gaussian coefficient field. This performs no matrix solve and never puts i
/// in the geometry map. Unsupported coefficients or nonpolynomial entries fail
/// before the unchanged AtomField elimination is invoked.
fn validate_complex_geometry_matrix(
    matrix: &Matrix<super::AtomField>,
    variables: &Arc<Vec<PolyVariable>>,
) -> Result<(), String> {
    let field = AlgebraicExtension::complex(Q);
    for entry in matrix.row_iter().flatten() {
        let value: RationalPolynomial<AlgebraicExtension<RationalField>, u16> = entry
            .try_to_rational_polynomial(&field, &field, Some(variables.clone()))
            .map_err(|error| format!("thin geometry exact complex conversion: {error}"))?;
        if value.get_variables().as_ref() != variables.as_ref() {
            return Err("thin geometry complex conversion added undeclared variables".into());
        }
        if !value.denominator.is_constant() {
            return Err("thin geometry complex matrix entries must be polynomials".into());
        }
    }
    Ok(())
}

/// One candidate for the complete seven-point parent expression.
#[derive(Clone, Debug)]
pub(crate) struct ThinSevenChart {
    /// Omitted column among the six non-origin parent lines. The absent line is
    /// `active[omitted + 1]`, and its weight is structurally zero in every column.
    pub omitted: usize,
    /// Exact coordinate polynomial without internally generated PV aliases.
    /// This is a chart predicate, not a numerical nonzero certificate.
    pub determinant: Atom,
    /// Constant-one column followed by the four Cartesian numerator columns.
    /// Each vector uses the complete seven-line parent order.
    pub weights: [Vec<Atom>; 5],
}

/// A six-line coefficient system, including a structurally unavailable system.
#[derive(Clone, Debug)]
pub(crate) struct ThinSixGeometry {
    /// Ordered original propagator indices; the first is this child's origin.
    pub active: Vec<usize>,
    pub determinant: Atom,
    /// `None` means exact structural singularity, never a successful zero.
    /// A demanded unavailable child must reject its parent representation.
    pub weights: Option<[Vec<Atom>; 5]>,
}

#[derive(Clone, Debug)]
pub(crate) struct ThinSevenGeometry {
    pub active: Vec<usize>,
    /// Highest-priority chart first, matching the existing PV selection order:
    /// omitted columns 5,4,3,2,1,0, with exact singular charts absent.
    /// Select complete assembled chart expressions lazily; do not select the
    /// five columns independently or evaluate inactive child inverses eagerly.
    pub charts: Vec<ThinSevenChart>,
    /// Entry `i` omits `active[i]`, preserving every other line's order.
    pub six: Vec<ThinSixGeometry>,
}

impl PvCompiler {
    /// Construct geometry only. The caller enforces the private preserved-shift,
    /// unit-power native Cartesian domain before using these coefficients.
    pub(crate) fn thin_seven_geometry(
        &mut self,
        active: &[usize],
    ) -> Result<ThinSevenGeometry, String> {
        validate_active(self, active, 7)?;
        let mut six = Vec::with_capacity(7);
        for omitted in 0..7 {
            let subset: Vec<_> = active
                .iter()
                .enumerate()
                .filter_map(|(position, &line)| (position != omitted).then_some(line))
                .collect();
            six.push(self.thin_six_geometry(&subset)?);
        }
        let mut charts = Vec::with_capacity(6);
        for omitted in (0..6).rev() {
            // The chart omits one non-origin parent line, so its matrix and
            // every RHS equal this already prepared six-line system. Reuse
            // those aliases rather than repeat six exact inversions.
            let removed = omitted + 1;
            let child = &six[removed];
            let Some(child_weights) = &child.weights else {
                continue;
            };
            let weights = std::array::from_fn(|column| {
                let mut weights = vec![Atom::new(); 7];
                for (local, parent_position) in
                    (0..7).filter(|&position| position != removed).enumerate()
                {
                    weights[parent_position] = child_weights[column][local].clone();
                }
                weights
            });
            charts.push(ThinSevenChart {
                omitted,
                determinant: child.determinant.clone(),
                weights,
            });
        }
        if charts.is_empty() {
            return Err(
                "thin seven-point coordinate system has no structurally regular chart".into(),
            );
        }
        Ok(ThinSevenGeometry {
            active: active.to_vec(),
            charts,
            six,
        })
    }

    /// The five RHS columns for an ordered six-line subset. In particular,
    /// omitting the original origin changes the last Cartesian RHS entries.
    pub(crate) fn thin_six_geometry(
        &mut self,
        active: &[usize],
    ) -> Result<ThinSixGeometry, String> {
        validate_active(self, active, 6)?;
        let columns = [0, 1, 2, 3, 4];
        let (determinant, weights) = solve_chart(self, active, &columns)?;
        Ok(ThinSixGeometry {
            active: active.to_vec(),
            determinant,
            weights,
        })
    }
}

fn validate_active(compiler: &PvCompiler, active: &[usize], expected: usize) -> Result<(), String> {
    if active.len() != expected
        || active.iter().enumerate().any(|(position, &line)| {
            line >= compiler.shifts.len() || active[..position].contains(&line)
        })
    {
        return Err(format!(
            "thin geometry requires {expected} distinct in-range ordered propagators"
        ));
    }
    if !compiler.moments.is_empty() {
        return Err("thin geometry requires a compiler with no prepared tensor moments".into());
    }
    Ok(())
}

/// Select the Q→Z path structurally, before any conversion or elimination.
/// In particular, Symbolica stores i as a numeric coefficient, not a variable.
/// Direct exact fixtures using another coefficient domain retain the original
/// AtomField implementation; a failed rational-path invariant never falls back.
fn has_only_rational_coefficients(compiler: &PvCompiler) -> bool {
    compiler
        .shifts
        .iter()
        .flatten()
        .chain(&compiler.mass_squared)
        .all(|value| {
            let mut supported = true;
            value.visitor(&mut |atom| {
                if let AtomView::Num(number) = atom {
                    supported &= Q
                        .try_element_from_coefficient_view(number.get_coeff_view())
                        .is_ok();
                }
                supported
            });
            supported
        })
}

fn solve_chart(
    compiler: &mut PvCompiler,
    active: &[usize],
    columns: &[usize],
) -> Result<(Atom, Option<[Vec<Atom>; 5]>), String> {
    // Keep opaque input functions outside this experiment on both paths.
    let variables = geometry_variables(compiler)?;
    // Validate original declarations as well as assembled matrices: common
    // offsets can cancel from matrix entries, hiding an unsupported input.
    let inputs = Matrix::new_vec(
        compiler
            .shifts
            .iter()
            .flatten()
            .chain(&compiler.mass_squared)
            .cloned()
            .collect(),
        field(),
    );
    if has_only_rational_coefficients(compiler) {
        exact_geometry_matrix(&inputs, &variables)?;
        solve_chart_rational(compiler, active, columns, &variables)
    } else {
        validate_complex_geometry_matrix(&inputs, &variables)?;
        solve_chart_atom(compiler, active, columns, &variables)
    }
}

/// Solve a five-column relative-coordinate chart against the constant and four
/// Cartesian RHS columns. The raw matrix deliberately uses coordinate distances,
/// not supplied/snapped invariants or coefficient aliases.
fn solve_chart_rational(
    compiler: &mut PvCompiler,
    active: &[usize],
    columns: &[usize],
    variables: &Arc<Vec<PolyVariable>>,
) -> Result<(Atom, Option<[Vec<Atom>; 5]>), String> {
    let _chart = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 six-line chart {active:?}"),
    );
    let _matrix = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 matrix assembly {active:?}"),
    );
    let origin = active[0];
    let mut rows = coordinate_rows(&compiler.shifts, active);
    rows.push(
        active[1..]
            .iter()
            .map(|&line| {
                let delta = std::array::from_fn(|mu| {
                    &compiler.shifts[line][mu] - &compiler.shifts[origin][mu]
                });
                dot(&delta, &delta) - &compiler.mass_squared[line] + &compiler.mass_squared[origin]
            })
            .collect(),
    );
    let matrix = Matrix::from_nested_vec(
        rows.iter()
            .map(|row| columns.iter().map(|&column| row[column].clone()).collect())
            .collect(),
        field(),
    )
    .map_err(|error| format!("thin coordinate chart matrix: {error}"))?;
    let matrix = exact_geometry_matrix(&matrix, variables)?;
    drop(_matrix);
    let _determinant = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 determinant {active:?}"),
    );
    let exact_determinant = matrix
        .det()
        .map_err(|error| format!("thin coordinate chart determinant: {error:?}"))?;
    if !exact_determinant.denominator.is_constant() {
        return Err("thin coordinate determinant acquired a nonpolynomial denominator".into());
    }
    let determinant = exact_determinant.to_expression();
    drop(_determinant);
    if determinant.is_zero() {
        return Ok((determinant, None));
    }
    let _rhs = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 RHS assembly {active:?}"),
    );
    let rhs = Matrix::from_nested_vec(
        (0..5)
            .map(|row| {
                (0..5)
                    .map(|column| {
                        if column == 0 {
                            Atom::num(i64::from(row == 4))
                        } else if row == 4 {
                            -&compiler.shifts[origin][column - 1]
                        } else if row == column - 1 {
                            Atom::num(if row == 0 { 1 } else { -1 }) / Atom::num(2)
                        } else {
                            Atom::new()
                        }
                    })
                    .collect()
            })
            .collect(),
        field(),
    )
    .map_err(|error| format!("thin coordinate chart RHS: {error}"))?;
    let rhs = exact_geometry_matrix(&rhs, variables)?;
    drop(_rhs);
    let _inverse = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 inverse {active:?}"),
    );
    let inverse = matrix
        .inv()
        .map_err(|error| format!("thin coordinate chart inverse: {error:?}"))?;
    drop(_inverse);
    let _product = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 inverse times RHS {active:?}"),
    );
    let solutions = &inverse * &rhs;
    drop(_product);
    let _certificate = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 exact solution certificate {active:?}"),
    );
    if &matrix * &solutions != rhs {
        return Err("thin coordinate exact solution identity failed".into());
    }
    for solution in solutions.row_iter().flatten() {
        // Symbolica's canonical rational-function multiplication must cancel
        // every variable denominator against det(M). This excludes removable
        // Gaussian pivots from the returned expressions' numerical domain.
        if !(solution * &exact_determinant).denominator.is_constant() {
            return Err("thin coordinate solution retained a pole outside its determinant".into());
        }
    }
    drop(_certificate);
    let _aliases = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 weight aliases {active:?}"),
    );
    let weights = std::array::from_fn(|column| {
        let mut weights = vec![Atom::new(); active.len()];
        for (row, &selected) in columns.iter().enumerate() {
            weights[selected + 1] = solutions[(row as u32, column as u32)].to_expression();
        }
        weights[0] = -Atom::add_many(&weights[1..]);
        weights
            .into_iter()
            .map(|weight| compiler.bind_alias(weight))
            .collect()
    });
    drop(_aliases);
    Ok((determinant, Some(weights)))
}

/// Compatibility implementation copied from the marker-only baseline. This
/// branch has no polynomial-domain preparation or speedup qualification.
fn solve_chart_atom(
    compiler: &mut PvCompiler,
    active: &[usize],
    columns: &[usize],
    variables: &Arc<Vec<PolyVariable>>,
) -> Result<(Atom, Option<[Vec<Atom>; 5]>), String> {
    let _chart = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 six-line chart {active:?}"),
    );
    let _matrix = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 matrix assembly {active:?}"),
    );
    let origin = active[0];
    let mut rows = coordinate_rows(&compiler.shifts, active);
    rows.push(
        active[1..]
            .iter()
            .map(|&line| {
                let delta = std::array::from_fn(|mu| {
                    &compiler.shifts[line][mu] - &compiler.shifts[origin][mu]
                });
                dot(&delta, &delta) - &compiler.mass_squared[line] + &compiler.mass_squared[origin]
            })
            .collect(),
    );
    let matrix = Matrix::from_nested_vec(
        rows.iter()
            .map(|row| columns.iter().map(|&column| row[column].clone()).collect())
            .collect(),
        field(),
    )
    .map_err(|error| format!("thin coordinate chart matrix: {error}"))?;
    validate_complex_geometry_matrix(&matrix, variables)?;
    drop(_matrix);
    let _determinant = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 determinant {active:?}"),
    );
    let determinant = matrix
        .det()
        .map_err(|error| format!("thin coordinate chart determinant: {error:?}"))?;
    drop(_determinant);
    if determinant.is_zero() {
        return Ok((determinant, None));
    }
    let _rhs = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 RHS assembly {active:?}"),
    );
    let rhs = Matrix::from_nested_vec(
        (0..5)
            .map(|row| {
                (0..5)
                    .map(|column| {
                        if column == 0 {
                            Atom::num(i64::from(row == 4))
                        } else if row == 4 {
                            -&compiler.shifts[origin][column - 1]
                        } else if row == column - 1 {
                            Atom::num(if row == 0 { 1 } else { -1 }) / Atom::num(2)
                        } else {
                            Atom::new()
                        }
                    })
                    .collect()
            })
            .collect(),
        field(),
    )
    .map_err(|error| format!("thin coordinate chart RHS: {error}"))?;
    validate_complex_geometry_matrix(&rhs, variables)?;
    drop(_rhs);
    let _inverse = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 inverse {active:?}"),
    );
    let inverse = matrix
        .inv()
        .map_err(|error| format!("thin coordinate chart inverse: {error:?}"))?;
    drop(_inverse);
    let _product = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 inverse times RHS {active:?}"),
    );
    let solutions = &inverse * &rhs;
    drop(_product);
    let _aliases = PreparationTimer::new(
        compiler.profile_preparation,
        format_args!("thin N7 weight aliases {active:?}"),
    );
    let weights = std::array::from_fn(|column| {
        let mut weights = vec![Atom::new(); active.len()];
        for (row, &selected) in columns.iter().enumerate() {
            weights[selected + 1] = solutions[(row as u32, column as u32)].clone();
        }
        weights[0] = -Atom::add_many(&weights[1..]);
        weights
            .into_iter()
            .map(|weight| compiler.bind_alias(weight))
            .collect()
    });
    drop(_aliases);
    Ok((determinant, Some(weights)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Masses make f_i=t_i^5 at the unshifted origin. This is an algebraic
    // geometry fixture, not an asserted physical integration domain.
    fn moment_curve() -> (Vec<[Atom; 4]>, Vec<Atom>) {
        let shifts: Vec<_> = (0_i64..7)
            .map(|t| std::array::from_fn(|mu| Atom::num(t.pow(mu as u32 + 1))))
            .collect();
        let masses = shifts
            .iter()
            .enumerate()
            .map(|(i, p)| dot(p, p) - Atom::num((i as i64).pow(5)))
            .collect();
        (shifts, masses)
    }

    fn assert_integrand_columns(
        shifts: &[[Atom; 4]],
        masses: &[Atom],
        active: &[usize],
        weights: &[Vec<Atom>; 5],
    ) {
        let q: [Atom; 4] = std::array::from_fn(|mu| {
            Atom::var(symbolica::symbol!(format!(
                "thin_seven_geometry_test::q{mu}"
            )))
        });
        let perpendicular = Atom::var(symbolica::symbol!(
            "thin_seven_geometry_test::q_perp_squared"
        ));
        let denominators: Vec<_> = active
            .iter()
            .map(|&line| {
                let shifted = std::array::from_fn(|mu| &q[mu] + &shifts[line][mu]);
                dot(&shifted, &shifted) + &perpendicular - &masses[line]
            })
            .collect();
        for column in 0..5 {
            assert_eq!(weights[column].len(), active.len());
            let actual = Atom::add_many(
                &weights[column]
                    .iter()
                    .zip(&denominators)
                    .map(|(weight, denominator)| weight * denominator)
                    .collect::<Vec<_>>(),
            );
            let expected = if column == 0 {
                Atom::num(1)
            } else {
                q[column - 1].clone()
            };
            assert!(
                (actual - expected).expand().together().is_zero(),
                "column {column}"
            );
        }
    }

    #[test]
    fn thin_seven_geometry_six_chart_order_and_all_five_rhs_identities() {
        let (mut shifts, masses) = moment_curve();
        for point in &mut shifts {
            for (mu, coordinate) in point.iter_mut().enumerate() {
                *coordinate = &*coordinate + Atom::num((mu + 1) as i64) / Atom::num(3);
            }
        }
        let active: Vec<_> = (0..7).rev().collect();
        let mut compiler = PvCompiler::new(shifts.clone(), masses.clone()).unwrap();
        let geometry = compiler.thin_seven_geometry(&active).unwrap();
        assert_eq!(geometry.active, active);
        assert_eq!(
            geometry
                .charts
                .iter()
                .map(|chart| chart.omitted)
                .collect::<Vec<_>>(),
            vec![5, 4, 3, 2, 1, 0]
        );
        for chart in &geometry.charts {
            assert!(!chart.determinant.is_zero());
            assert!(
                chart
                    .weights
                    .iter()
                    .all(|column| column[chart.omitted + 1].is_zero())
            );
            let child = &geometry.six[chart.omitted + 1];
            assert_eq!(chart.determinant, child.determinant);
            for column in 0..5 {
                let embedded: Vec<_> = chart.weights[column]
                    .iter()
                    .enumerate()
                    .filter_map(|(position, weight)| {
                        (position != chart.omitted + 1).then_some(weight.clone())
                    })
                    .collect();
                assert_eq!(embedded, child.weights.as_ref().unwrap()[column]);
            }
            assert_integrand_columns(&shifts, &masses, &active, &chart.weights);
        }
        assert_eq!(geometry.six.len(), 7);
        for (omitted, child) in geometry.six.iter().enumerate() {
            let expected: Vec<_> = active
                .iter()
                .enumerate()
                .filter_map(|(position, &line)| (position != omitted).then_some(line))
                .collect();
            assert_eq!(child.active, expected);
            assert!(!child.determinant.is_zero());
            assert_integrand_columns(
                &shifts,
                &masses,
                &child.active,
                child.weights.as_ref().unwrap(),
            );
        }
    }

    #[test]
    fn thin_seven_geometry_preserves_exact_complex_atom_coefficients() {
        let (mut shifts, mut masses) = moment_curve();
        for point in &mut shifts {
            point[0] = &point[0] + Atom::i() / Atom::num(3);
        }
        for (line, mass) in masses.iter_mut().enumerate() {
            *mass = &*mass + Atom::i() * Atom::num((line + 1) as i64);
        }
        let mut compiler = PvCompiler::new(shifts.clone(), masses.clone()).unwrap();
        assert!(!has_only_rational_coefficients(&compiler));
        // No i, opaque placeholder, or hidden symbol enters the variable map.
        assert!(geometry_variables(&compiler).unwrap().is_empty());
        let active = [0, 1, 2, 3, 4, 5];
        let geometry = compiler.thin_six_geometry(&active).unwrap();
        assert!(!geometry.determinant.is_zero());
        assert_integrand_columns(
            &shifts,
            &masses,
            &active,
            geometry.weights.as_ref().unwrap(),
        );
        // A common unsupported mass offset cancels from M. It must still fail
        // declaration validation before either coefficient-domain solver.
        let x = Atom::var(symbolica::symbol!(
            "thin_seven_geometry_test::bad_mass_pole"
        ));
        for mass in &mut masses {
            *mass = &*mass + Atom::i() / &x;
        }
        let mut malformed = PvCompiler::new(shifts, masses).unwrap();
        assert!(malformed.thin_six_geometry(&active).is_err());
    }

    #[test]
    fn thin_seven_geometry_retains_unavailable_six_children() {
        let (mut shifts, mut masses) = moment_curve();
        shifts[6] = shifts[0].clone();
        masses[6] = masses[0].clone();
        let mut compiler = PvCompiler::new(shifts, masses).unwrap();
        let geometry = compiler
            .thin_seven_geometry(&(0..7).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(geometry.charts.len(), 1);
        assert_eq!(geometry.charts[0].omitted, 5);
        for (omitted, child) in geometry.six.iter().enumerate() {
            let regular = omitted == 0 || omitted == 6;
            assert_eq!(child.weights.is_some(), regular);
            assert_eq!(child.determinant.is_zero(), !regular);
        }
    }

    #[test]
    fn thin_seven_geometry_keeps_primal_zero_mixed_direction_weight() {
        let (shifts, masses) = moment_curve();
        let active = [0, 1, 2, 3, 4, 5];
        let mut original = PvCompiler::new(shifts.clone(), masses.clone()).unwrap();
        let original = original
            .thin_six_geometry(&active)
            .unwrap()
            .weights
            .unwrap();
        let c = original[0][0].clone();
        assert!(!c.is_zero());
        let x_symbol = symbolica::symbol!("thin_seven_geometry_test::x");
        let y_symbol = symbolica::symbol!("thin_seven_geometry_test::y");
        let x = Atom::var(x_symbol);
        let y = Atom::var(y_symbol);
        let translation = &original[1][0] / &c + &x * &y;
        let translated: Vec<_> = shifts
            .iter()
            .map(|point| {
                let mut point = point.clone();
                point[0] = &point[0] + &translation;
                point
            })
            .collect();
        let mut compiler = PvCompiler::new(translated, masses).unwrap();
        let weights = compiler
            .thin_six_geometry(&active)
            .unwrap()
            .weights
            .unwrap();
        let weight = &weights[1][0];
        assert!(!weight.is_zero());
        assert!((weight + &x * &y * &c).expand().together().is_zero());
        assert!(
            weight
                .replace(x.clone())
                .with(Atom::new())
                .replace(y.clone())
                .with(Atom::new())
                .is_zero()
        );
        assert!(
            (weight.derivative(x_symbol).derivative(y_symbol) + c)
                .expand()
                .together()
                .is_zero()
        );
    }

    #[test]
    fn thin_seven_geometry_rejects_invalid_or_all_singular_parent() {
        let (shifts, masses) = moment_curve();
        let mut compiler = PvCompiler::new(shifts, masses).unwrap();
        for active in [
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 1, 2, 3, 4, 5, 5],
            vec![0, 1, 2, 3, 4, 5, 7],
        ] {
            assert!(compiler.thin_seven_geometry(&active).is_err());
        }
        let shifts = vec![std::array::from_fn(|_| Atom::new()); 7];
        let mut compiler = PvCompiler::new(shifts, vec![Atom::num(1); 7]).unwrap();
        assert!(
            compiler
                .thin_seven_geometry(&(0..7).collect::<Vec<_>>())
                .is_err()
        );
    }
}

#[cfg(test)]
mod rational_domain_tests {
    use super::*;

    fn variable(name: &str) -> Atom {
        Atom::var(symbolica::symbol!(format!("thin_n7_exact_domain::{name}")))
    }

    fn map(values: &[Atom]) -> Arc<Vec<PolyVariable>> {
        Arc::new(
            values
                .iter()
                .cloned()
                .map(PolyVariable::try_from)
                .collect::<Result<_, _>>()
                .unwrap(),
        )
    }

    #[test]
    fn exact_domain_preserves_full_ordered_generic_variable_map() {
        let shifts: Vec<[Atom; 4]> = (0..7)
            .map(|i| std::array::from_fn(|mu| variable(&format!("p{i}_{mu}"))))
            .collect();
        let masses: Vec<_> = (0..7).map(|i| variable(&format!("m{i}"))).collect();
        let parameters: Vec<_> = shifts.iter().flatten().chain(&masses).cloned().collect();
        let compiler = PvCompiler::new(shifts, masses).unwrap();
        let variables = geometry_variables(&compiler).unwrap();
        assert_eq!(variables.as_ref(), map(&parameters).as_ref());
        assert_eq!(variables.len(), 35);
        let raw = Matrix::new_vec(
            vec![parameters[34].clone(), Atom::new(), Atom::num(1)],
            field(),
        );
        let exact = exact_geometry_matrix(&raw, &variables).unwrap();
        for entry in exact.row_iter().flatten() {
            assert_eq!(entry.get_variables().as_ref(), variables.as_ref());
        }
    }

    #[test]
    fn exact_domain_matches_visible_atom_determinant_and_all_rhs_columns() {
        let x = variable("x");
        let y = variable("y");
        let variables = map(&[x.clone(), y.clone()]);
        let raw = Matrix::from_nested_vec(
            vec![
                vec![x.clone(), Atom::num(1), &y / Atom::num(3)],
                vec![Atom::num(2), y.clone(), Atom::num(1)],
                vec![Atom::num(1), &x + &y, Atom::num(4)],
            ],
            field(),
        )
        .unwrap();
        let raw_rhs = Matrix::from_nested_vec(
            vec![
                vec![
                    Atom::num(1),
                    Atom::num(1) / Atom::num(2),
                    x.clone(),
                    y.clone(),
                    Atom::new(),
                ],
                vec![
                    Atom::new(),
                    Atom::num(-1) / Atom::num(2),
                    y.clone(),
                    Atom::num(1),
                    x.clone(),
                ],
                vec![Atom::num(1), -&x, Atom::new(), -&y, Atom::num(2)],
            ],
            field(),
        )
        .unwrap();
        let exact = exact_geometry_matrix(&raw, &variables).unwrap();
        let rhs = exact_geometry_matrix(&raw_rhs, &variables).unwrap();
        let determinant = exact.det().unwrap();
        assert!(determinant.denominator.is_constant());
        assert!(
            (&determinant.to_expression() - raw.det().unwrap())
                .together()
                .is_zero()
        );
        let solutions = &exact.inv().unwrap() * &rhs;
        assert_eq!(&exact * &solutions, rhs);
        let original = &raw.inv().unwrap() * &raw_rhs;
        for (actual, expected) in solutions
            .row_iter()
            .flatten()
            .zip(original.row_iter().flatten())
        {
            assert!((&actual.to_expression() - expected).together().is_zero());
            assert!((actual * &determinant).denominator.is_constant());
        }
    }

    #[test]
    fn exact_domain_cancels_a_removable_symbolic_pivot() {
        let x = variable("pivot");
        let variables = map(&[x.clone()]);
        let raw = Matrix::from_nested_vec(
            vec![vec![x, Atom::num(1)], vec![Atom::num(1), Atom::new()]],
            field(),
        )
        .unwrap();
        let exact = exact_geometry_matrix(&raw, &variables).unwrap();
        let determinant = exact.det().unwrap();
        assert_eq!(determinant.to_expression(), Atom::num(-1));
        let inverse = exact.inv().unwrap();
        for entry in inverse.row_iter().flatten() {
            assert!(
                entry.denominator.is_constant(),
                "removable pivot must not survive at x=0"
            );
            assert!((entry * &determinant).denominator.is_constant());
        }
        assert_eq!((&exact * &inverse)[(0, 0)].to_expression(), Atom::num(1));
        assert_eq!((&exact * &inverse)[(1, 1)].to_expression(), Atom::num(1));
    }

    #[test]
    fn exact_domain_rejects_added_variables_opaque_calls_and_nonpolynomial_entries() {
        let x = variable("allowed");
        let y = variable("undeclared");
        let variables = map(&[x.clone()]);
        for entry in [
            y,
            symbolica::atom::Symbol::SIN.call((x.clone(),)),
            Atom::num(1) / (&x + Atom::num(1)),
        ] {
            let raw = Matrix::new_vec(vec![entry.clone()], field());
            assert!(exact_geometry_matrix(&raw, &variables).is_err());
            assert!(validate_complex_geometry_matrix(&raw, &variables).is_err());
            let complex = Matrix::new_vec(vec![&entry + Atom::i()], field());
            assert!(validate_complex_geometry_matrix(&complex, &variables).is_err());
        }
    }
}
