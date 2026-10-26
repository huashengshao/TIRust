//! Generic, guarded constant-denominator descent proposal. Child module of pv.
//! Index subsets enumerate matrix charts only; Symbolica owns all algebra.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AffineDescentBudget {
    pub max_propagators: usize,
    pub max_charts: usize,
    pub max_constraints: usize,
    pub max_cofactors: usize,
}

impl Default for AffineDescentBudget {
    fn default() -> Self {
        // Preserve the existing inner-planner budget. It is not a theorem
        // that all these domains are supported or numerically qualified.
        Self {
            max_propagators: 16,
            max_charts: 4096,
            max_constraints: 4096,
            max_cofactors: 65536,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AffineChartKey {
    pub coordinate_rows: Vec<usize>,
    /// Columns of relative shifts, i.e. original denominator index minus one.
    pub relative_columns: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct AffineDescentChart {
    pub key: AffineChartKey,
    pub determinant: Atom,
    pub coefficients: Vec<Atom>,
}

#[derive(Clone, Debug)]
pub(crate) struct AffineDescentPolynomials {
    pub rank_constraints: Vec<Atom>,
    pub charts: Vec<(AffineChartKey, Atom)>,
    /// Chart-major denseN raw Cramer numerators, like the existing planar
    /// cofactor helper. Only full-Taylor exact zeros may suppress a child.
    pub cofactors: Vec<Atom>,
}

/// Finite combinatorial bookkeeping, not a polynomial or elimination engine.
fn subsets(length: usize, count: usize, cap: usize) -> Result<Vec<Vec<usize>>, String> {
    fn visit(
        length: usize,
        count: usize,
        start: usize,
        selected: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
        cap: usize,
    ) -> Result<(), String> {
        if selected.len() == count {
            if result.len() >= cap {
                return Err("affine chart enumeration budget exceeded".into());
            }
            result.push(selected.clone());
            return Ok(());
        }
        let remaining = count - selected.len();
        if remaining > length.saturating_sub(start) {
            return Ok(());
        }
        for next in start..=length - remaining {
            selected.push(next);
            visit(length, count, next + 1, selected, result, cap)?;
            selected.pop();
        }
        Ok(())
    }
    if count > length {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    visit(length, count, 0, &mut Vec::new(), &mut result, cap)?;
    Ok(result)
}

fn affine_rows(
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    budget: AffineDescentBudget,
) -> Result<Vec<Vec<Atom>>, String> {
    if shifts.len() < 2 || shifts.len() != masses.len() || shifts.len() > budget.max_propagators {
        return Err("affine descent needs matching N>=2 shifts/masses within its explicit propagator budget".into());
    }
    let active: Vec<_> = (0..shifts.len()).collect();
    let mut rows = coordinate_rows(shifts, &active);
    rows.push(
        (1..shifts.len())
            .map(|i| {
                let relative = std::array::from_fn(|mu| &shifts[i][mu] - &shifts[0][mu]);
                dot(&relative, &relative) - &masses[i] + &masses[0]
            })
            .collect(),
    );
    Ok(rows)
}

fn affine_chart_matrix(
    rows: &[Vec<Atom>],
    key: &AffineChartKey,
) -> Result<Matrix<AtomField>, String> {
    let rank = key.coordinate_rows.len();
    let increasing = |indices: &[usize], limit: usize| {
        indices.iter().all(|&i| i < limit) && indices.windows(2).all(|pair| pair[0] < pair[1])
    };
    if rank > 4
        || key.relative_columns.len() != rank + 1
        || !increasing(&key.coordinate_rows, 4)
        || !increasing(&key.relative_columns, rows[0].len())
    {
        return Err("invalid affine descent chart indices".into());
    }
    let selected_rows: Vec<_> = key
        .coordinate_rows
        .iter()
        .copied()
        .chain(std::iter::once(4))
        .collect();
    Matrix::from_nested_vec(
        selected_rows
            .into_iter()
            .map(|row| {
                key.relative_columns
                    .iter()
                    .map(|&column| rows[row][column].clone())
                    .collect()
            })
            .collect(),
        field(),
    )
    .map_err(|error| format!("affine descent chart: {error}"))
}

/// Exact coordinate rank<=r predicates and every constant-one chart for that
/// profile. N6/r4 gives one chart; N7/r4 gives the six existing five-column
/// charts. Lower profiles cover affine-degenerate families such as N3/r1.
/// Rank-four profiles require no coordinate-rank constraints in four space-time
/// dimensions. A profile without a regular chart never returns a zero result.
pub(crate) fn affine_descent_polynomials(
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    coordinate_rank: usize,
    budget: AffineDescentBudget,
) -> Result<AffineDescentPolynomials, String> {
    let rows = affine_rows(shifts, masses, budget)?;
    let columns = shifts.len() - 1;
    if coordinate_rank > 4 || coordinate_rank + 1 > columns {
        return Err("affine coordinate rank must be <=min(4,N-2)".into());
    }
    let minor_rows = subsets(4, coordinate_rank + 1, budget.max_constraints)?;
    let minor_columns = if minor_rows.is_empty() {
        Vec::new()
    } else {
        subsets(columns, coordinate_rank + 1, budget.max_constraints)?
    };
    if minor_rows
        .len()
        .checked_mul(minor_columns.len())
        .is_none_or(|count| count > budget.max_constraints)
    {
        return Err("affine rank-constraint budget exceeded".into());
    }
    let mut rank_constraints = Vec::new();
    for selected_rows in &minor_rows {
        for selected_columns in &minor_columns {
            let matrix = Matrix::from_nested_vec(
                selected_rows
                    .iter()
                    .map(|&row| {
                        selected_columns
                            .iter()
                            .map(|&column| rows[row][column].clone())
                            .collect()
                    })
                    .collect(),
                field(),
            )
            .map_err(|error| format!("affine rank minor: {error}"))?;
            rank_constraints.push(
                matrix
                    .det()
                    .map_err(|error| format!("affine rank determinant: {error:?}"))?
                    .together(),
            );
        }
    }
    let chart_rows = subsets(4, coordinate_rank, budget.max_charts)?;
    let chart_columns = subsets(columns, coordinate_rank + 1, budget.max_charts)?;
    if chart_rows
        .len()
        .checked_mul(chart_columns.len())
        .is_none_or(|count| count > budget.max_charts)
    {
        return Err("affine constant-chart budget exceeded".into());
    }
    let mut charts = Vec::new();
    let count = chart_rows
        .len()
        .checked_mul(chart_columns.len())
        .and_then(|count| count.checked_mul(shifts.len()))
        .ok_or_else(|| "affine cofactor count overflow".to_owned())?;
    if count > budget.max_cofactors {
        return Err("affine cofactor budget exceeded".into());
    }
    let mut cofactors = Vec::with_capacity(count);
    for selected_rows in chart_rows {
        for selected_columns in &chart_columns {
            let key = AffineChartKey {
                coordinate_rows: selected_rows.clone(),
                relative_columns: selected_columns.clone(),
            };
            let matrix = affine_chart_matrix(&rows, &key)?;
            let determinant = matrix
                .det()
                .map_err(|error| format!("affine constant determinant: {error:?}"))?
                .together();
            // Reuse planar_chart_cofactor_polynomials' exact construction:
            // replace each selected column by the unit-last RHS, then ask
            // Symbolica for the determinant. No hand-coded Cramer formula.
            let size = key.relative_columns.len();
            let mut numerators = vec![Atom::new(); shifts.len()];
            for (column, &relative) in key.relative_columns.iter().enumerate() {
                let replaced = (0..size)
                    .map(|row| {
                        (0..size)
                            .map(|j| {
                                if j == column {
                                    Atom::num(i64::from(row + 1 == size))
                                } else {
                                    matrix[(row as u32, j as u32)].clone()
                                }
                            })
                            .collect()
                    })
                    .collect();
                numerators[relative + 1] = Matrix::from_nested_vec(replaced, field())
                    .map_err(|error| format!("affine cofactor matrix: {error}"))?
                    .det()
                    .map_err(|error| format!("affine cofactor determinant: {error:?}"))?
                    .together();
            }
            numerators[0] = -Atom::add_many(&numerators[1..]);
            cofactors.extend(numerators);
            charts.push((key, determinant));
        }
    }
    Ok(AffineDescentPolynomials {
        rank_constraints,
        charts,
        cofactors,
    })
}

pub(crate) fn affine_descent_chart(
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    key: &AffineChartKey,
    budget: AffineDescentBudget,
) -> Result<Option<AffineDescentChart>, String> {
    let rows = affine_rows(shifts, masses, budget)?;
    let matrix = affine_chart_matrix(&rows, key)?;
    let determinant = matrix
        .det()
        .map_err(|error| format!("affine constant determinant: {error:?}"))?
        .together();
    if determinant.is_zero() {
        return Ok(None);
    }
    let size = key.relative_columns.len();
    let rhs = Matrix::new_vec(
        (0..size)
            .map(|row| Atom::num(i64::from(row + 1 == size)))
            .collect(),
        field(),
    );
    let solution = matrix
        .solve(&rhs)
        .map_err(|error| format!("affine constant solve: {error:?}"))?;
    let mut coefficients = vec![Atom::new(); shifts.len()];
    for (row, &column) in key.relative_columns.iter().enumerate() {
        coefficients[column + 1] = solution[(row as u32, 0)].clone().together();
    }
    coefficients[0] = -Atom::add_many(&coefficients[1..]);
    Ok(Some(AffineDescentChart {
        key: key.clone(),
        determinant,
        coefficients,
    }))
}

impl PvCompiler {
    /// This private recipe is authorized ONLY after the generic guard accepts
    /// this exact key/profile on the complete represented Taylor input row.
    /// It compiles existing same-q pinches without generating the parent Gram
    /// inverse or parent scalar recurrence. Inner unsupported/singular children
    /// still fail explicitly; this is not an arbitrary-N completion.
    pub(crate) fn compile_affine_descent_chart_moments(
        &mut self,
        key: &AffineChartKey,
        inactive: &[usize],
        basis: &[[u16; 4]],
        budget: AffineDescentBudget,
    ) -> Result<Vec<[Atom; 4]>, String> {
        if self.shifts.len() > 16 {
            return Err(
                "affine descent recipe preserves the existing 16-propagator inner-planner budget"
                    .into(),
            );
        }
        if !self.moments.is_empty() || !self.geometry.is_empty() {
            return Err("affine descent needs an unused compiler".into());
        }
        if inactive.len() >= self.shifts.len()
            || inactive.iter().any(|&child| child >= self.shifts.len())
            || inactive.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err("affine inactive children must be distinct ordered in-range indices and retain a child".into());
        }
        if basis
            .iter()
            .any(|alpha| alpha.iter().map(|&n| u32::from(n)).sum::<u32>() > 6)
        {
            return Err("this affine descent proposal has a rank-six checkpoint".into());
        }
        let chart = affine_descent_chart(&self.shifts, &self.mass_squared, key, budget)?
            .ok_or_else(|| "affine descent chart is structurally singular".to_owned())?;
        // The wrapper certifies each inactive raw cofactor in every Taylor
        // component BEFORE this method is called; primal zeros never suffice.
        // Omit before recursion so 0 * an undefined child is never generated.
        let factors: Vec<_> = chart
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(child, factor)| {
                if inactive.binary_search(&child).is_ok() {
                    Atom::new()
                } else {
                    self.bind_alias(factor)
                }
            })
            .collect();
        let active: Vec<_> = (0..self.shifts.len()).collect();
        let mut output = Vec::with_capacity(basis.len());
        for &exponents in basis {
            let mut value = zero();
            for (omitted, factor) in factors.iter().enumerate() {
                if factor.is_zero() {
                    continue;
                }
                let child = self.moment(&without(&active, omitted), exponents)?;
                accumulate(&mut value, factor, &child);
            }
            let value = self.cache_value(value);
            output.push(if self.alias_enabled {
                value
            } else {
                value.map(|expression| expression.replace_multiple(self.substitutions.iter()))
            });
        }
        Ok(output)
    }
}

#[cfg(test)]
#[path = "affine_descent_tests.rs"]
mod tests;
