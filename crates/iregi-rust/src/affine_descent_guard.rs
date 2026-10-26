//! Generic-N source proposal. Existing exact polynomial certificates own all
//! full-Taylor arithmetic; only lane/parameter/chart bookkeeping is local.
use crate::{
    InputLayout, KinematicsBatch, ReductionError,
    exact_zero_certificate::{
        ExactZeroCertificate, ExactZeroConstraintPlan, ExactZeroConstraintWorkspace,
    },
    execution::validate_input_row,
    pv::{AffineChartKey, AffineDescentBudget, affine_descent_polynomials},
};
use iregi_masters::NativeFloat;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        dual::{DualNumberStructure, HyperDual},
        float::Complex,
        rational::Rational,
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AffineDescentSelection {
    Chart { index: usize, inactive: Vec<usize> },
    OffAffineFamily { constraint: usize, component: usize },
    NoRegularChart,
    Unresolved { working_bits: u32 },
    InvalidInput(String),
}

#[derive(Clone)]
pub(crate) struct AffineDescentGuardPlan {
    parameters: Vec<Atom>,
    propagators: usize,
    components: usize,
    rank_constraints: usize,
    charts: Vec<AffineChartKey>,
    exact: ExactZeroConstraintPlan,
}

impl AffineDescentGuardPlan {
    pub(crate) fn new(
        n: usize,
        coordinate_rank: usize,
        budget: AffineDescentBudget,
    ) -> Result<Self, ReductionError> {
        Self::prepare(n, coordinate_rank, budget, None)
    }
    pub(crate) fn new_dual<D: DualNumberStructure>(
        n: usize,
        coordinate_rank: usize,
        budget: AffineDescentBudget,
        dual: &D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(dual)
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        Self::prepare(
            n,
            coordinate_rank,
            budget,
            Some(
                dual.get_shape()
                    .into_iter()
                    .map(<[usize]>::to_vec)
                    .collect(),
            ),
        )
    }
    fn prepare(
        n: usize,
        rank: usize,
        budget: AffineDescentBudget,
        shape: Option<Vec<Vec<usize>>>,
    ) -> Result<Self, ReductionError> {
        // Check before allocating shifts or enumerating any chart.
        if n < 2 || n > budget.max_propagators || rank > 4 || rank + 2 > n {
            return Err(ReductionError::Preparation(
                "invalid affine descent profile or resource budget".into(),
            ));
        }
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| {
                std::array::from_fn(|mu| {
                    Atom::var(symbolica::symbol!(format!(
                        "iregi_affine_descent::p{i}_{mu}"
                    )))
                })
            })
            .collect();
        let masses: Vec<_> = (0..n)
            .map(|i| Atom::var(symbolica::symbol!(format!("iregi_affine_descent::m2_{i}"))))
            .collect();
        let mut parameters: Vec<_> = shifts.iter().flatten().cloned().collect();
        parameters.extend(masses.iter().cloned());
        parameters.push(Atom::var(symbolica::symbol!("iregi_affine_descent::mu")));
        let p = affine_descent_polynomials(&shifts, &masses, rank, budget)
            .map_err(ReductionError::Preparation)?;
        let rank_constraints = p.rank_constraints.len();
        let charts = p.charts.iter().map(|(key, _)| key.clone()).collect();
        let mut constraints = p.rank_constraints;
        constraints.extend(p.charts.into_iter().map(|(_, determinant)| determinant));
        constraints.extend(p.cofactors);
        let exact = match &shape {
            None => ExactZeroConstraintPlan::new(&parameters, &constraints),
            Some(shape) => ExactZeroConstraintPlan::new_dual(
                &parameters,
                &constraints,
                &HyperDual::<u8>::new(shape.clone()),
            ),
        }?;
        Ok(Self {
            parameters,
            propagators: n,
            components: shape.as_ref().map_or(1, Vec::len),
            rank_constraints,
            charts,
            exact,
        })
    }
    pub(crate) fn parameters(&self) -> &[Atom] {
        &self.parameters
    }
    pub(crate) fn charts(&self) -> &[AffineChartKey] {
        &self.charts
    }
    pub(crate) fn instantiate(&self) -> AffineDescentGuardWorkspace {
        AffineDescentGuardWorkspace {
            propagators: self.propagators,
            parameters: self.parameters.len(),
            components: self.components,
            rank_constraints: self.rank_constraints,
            charts: self.charts.len(),
            evaluated_rows: 0,
            passes: 0,
            exact: self.exact.instantiate(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct AffineDescentGuardWorkspace {
    propagators: usize,
    parameters: usize,
    components: usize,
    rank_constraints: usize,
    charts: usize,
    evaluated_rows: usize,
    passes: usize,
    exact: ExactZeroConstraintWorkspace,
}

impl AffineDescentGuardWorkspace {
    pub(crate) fn set_precision_budget(
        &mut self,
        initial: u32,
        maximum: u32,
    ) -> Result<(), ReductionError> {
        self.exact.set_precision_budget(initial, maximum)
    }
    pub(crate) fn evaluated_rows(&self) -> usize {
        self.evaluated_rows
    }
    pub(crate) fn passes(&self) -> usize {
        self.passes
    }
    fn check_selected(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
        selected: &[Vec<(usize, usize)>],
    ) -> Result<Vec<ExactZeroCertificate>, ReductionError> {
        let result = self.exact.check_selected_batch(input, rows, selected);
        self.evaluated_rows = self
            .evaluated_rows
            .checked_add(self.exact.evaluated_rows())
            .ok_or_else(|| ReductionError::Evaluation("affine certificate rows overflow".into()))?;
        self.passes = self
            .passes
            .checked_add(self.exact.passes())
            .ok_or_else(|| {
                ReductionError::Evaluation("affine certificate passes overflow".into())
            })?;
        result
    }
    fn width(&self, rows: usize) -> Result<(usize, usize), ReductionError> {
        let width = self
            .parameters
            .checked_mul(self.components)
            .ok_or_else(|| {
                ReductionError::InvalidInput("affine descent row width overflow".into())
            })?;
        let total = rows.checked_mul(width).ok_or_else(|| {
            ReductionError::InvalidInput("affine descent batch length overflow".into())
        })?;
        Ok((width, total))
    }
    pub(crate) fn select_batch<T: NativeFloat>(
        &mut self,
        input: &KinematicsBatch<T>,
    ) -> Result<Vec<AffineDescentSelection>, ReductionError> {
        self.evaluated_rows = 0;
        self.passes = 0;
        let (width, total) = self.width(input.rows)?;
        if input.components != self.components || input.values.len() != total {
            return Err(ReductionError::InvalidInput(
                "affine descent input length/Taylor shape mismatch".into(),
            ));
        }
        let n = self.propagators;
        let layout = InputLayout {
            shifts: 0..4 * n,
            scalar_products: None,
            masses: 4 * n..5 * n,
            scale: 5 * n,
        };
        let mut result = vec![AffineDescentSelection::NoRegularChart; input.rows];
        let mut exact = Vec::with_capacity(input.values.len());
        let mut active = Vec::with_capacity(input.rows);
        for (lane, row) in input.values.chunks_exact(width).enumerate() {
            if let Some(status) = validate_input_row(row, &layout, self.components) {
                result[lane] = AffineDescentSelection::InvalidInput(format!("{status:?}"));
                continue;
            }
            let start = exact.len();
            for value in row {
                match (value.re.exact_rational(), value.im.exact_rational()) {
                    (Some(re), Some(im)) => exact.push(Complex::new(re, im)),
                    _ => break,
                }
            }
            if exact.len() == start + width {
                active.push(lane);
            } else {
                exact.truncate(start);
                result[lane] = AffineDescentSelection::InvalidInput(
                    "numeric type cannot certify exact represented affine inputs".into(),
                );
            }
        }
        let selected = self.select_batch_exact(&exact, active.len())?;
        for (lane, selection) in active.into_iter().zip(selected) {
            result[lane] = selection;
        }
        Ok(result)
    }
    pub(crate) fn select_batch_exact(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
    ) -> Result<Vec<AffineDescentSelection>, ReductionError> {
        self.evaluated_rows = 0;
        self.passes = 0;
        let (width, total) = self.width(rows)?;
        if input.len() != total {
            return Err(ReductionError::InvalidInput(
                "affine descent exact input length mismatch".into(),
            ));
        }
        let rank_selected = vec![
            (0..self.rank_constraints)
                .flat_map(|constraint| {
                    (0..self.components).map(move |component| (constraint, component))
                })
                .collect();
            rows
        ];
        let proofs = self.check_selected(input, rows, &rank_selected)?;
        let mut result = vec![AffineDescentSelection::NoRegularChart; rows];
        let mut active = Vec::with_capacity(rows);
        let mut packed = Vec::with_capacity(input.len());
        for (lane, proof) in proofs.into_iter().enumerate() {
            match proof {
                ExactZeroCertificate::Zero => {
                    active.push(lane);
                    packed.extend_from_slice(&input[lane * width..(lane + 1) * width]);
                }
                ExactZeroCertificate::Nonzero {
                    constraint,
                    component,
                } => {
                    result[lane] = AffineDescentSelection::OffAffineFamily {
                        constraint,
                        component,
                    };
                }
                ExactZeroCertificate::Unresolved { working_bits } => {
                    result[lane] = AffineDescentSelection::Unresolved { working_bits };
                }
            }
        }
        // Reject zero-containing enclosures and zero-primal nilpotents.
        // Only a proved nonzero primal authorizes a formal series inverse.
        let chart_selected = vec![
            (self.rank_constraints..self.rank_constraints + self.charts)
                .map(|constraint| (constraint, 0))
                .collect();
            active.len()
        ];
        let proofs = self.check_selected(&packed, active.len(), &chart_selected)?;
        let mut charted_lanes = Vec::new();
        let mut charted_indices = Vec::new();
        let mut charted_exact = Vec::new();
        for (compact, (lane, proof)) in active.into_iter().zip(proofs).enumerate() {
            result[lane] = match proof {
                ExactZeroCertificate::Nonzero {
                    constraint,
                    component: 0,
                } if (self.rank_constraints..self.rank_constraints + self.charts)
                    .contains(&constraint) =>
                {
                    let index = constraint - self.rank_constraints;
                    charted_lanes.push(lane);
                    charted_indices.push(index);
                    charted_exact
                        .extend_from_slice(&packed[compact * width..(compact + 1) * width]);
                    AffineDescentSelection::Chart {
                        index,
                        inactive: Vec::new(),
                    }
                }
                ExactZeroCertificate::Zero => AffineDescentSelection::NoRegularChart,
                ExactZeroCertificate::Unresolved { working_bits } => {
                    AffineDescentSelection::Unresolved { working_bits }
                }
                _ => {
                    return Err(ReductionError::Evaluation(
                        "invalid affine descent certificate result".into(),
                    ));
                }
            };
        }
        // Reuse the existing CertifiedPlanar mask protocol with dynamicN
        // indices. Unknown/nonzero cofactors retain their children. Only an
        // exact full-Taylor Zero is eligible for omission before recursion.
        let start = self.rank_constraints + self.charts;
        for child in 0..self.propagators {
            let selected = charted_indices
                .iter()
                .map(|&index| {
                    (0..self.components)
                        .map(|component| (start + index * self.propagators + child, component))
                        .collect()
                })
                .collect::<Vec<_>>();
            let proofs = self.check_selected(&charted_exact, charted_lanes.len(), &selected)?;
            for (&lane, proof) in charted_lanes.iter().zip(proofs) {
                if proof == ExactZeroCertificate::Zero {
                    if let AffineDescentSelection::Chart { inactive, .. } = &mut result[lane] {
                        inactive.push(child);
                    }
                }
            }
        }
        for lane in charted_lanes {
            if matches!(&result[lane], AffineDescentSelection::Chart { inactive, .. }
                if inactive.len() == self.propagators)
            {
                return Err(ReductionError::Evaluation(
                    "all affine cofactors zero despite regular chart".into(),
                ));
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "affine_descent_guard_tests.rs"]
mod tests;
