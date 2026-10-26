//! Generic-N source proposal. Existing exact polynomial certificates own all
//! full-Taylor arithmetic; only lane/parameter/chart bookkeeping is local.
use crate::{
    InputLayout, InvariantInput, KinematicsBatch, ReductionError,
    exact_zero_certificate::{
        ExactZeroCertificate, ExactZeroConstraintPlan, ExactZeroConstraintWorkspace,
    },
    execution::validate_input_row,
    pv::{AffineChartKey, AffineDescentBudget, affine_descent_polynomials,
        affine_descent_polynomials_with_distances},
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
    layout: InputLayout,
    propagators: usize,
    components: usize,
    rank_constraints: usize,
    charts: Vec<AffineChartKey>,
    exact: ExactZeroConstraintPlan,
    compatibility: Option<ExactZeroConstraintPlan>,
}

impl AffineDescentGuardPlan {
    pub(crate) fn new(
        n: usize,
        coordinate_rank: usize,
        budget: AffineDescentBudget,
    ) -> Result<Self, ReductionError> {
        Self::prepare(n, coordinate_rank, budget, None, InvariantInput::FromShifts)
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
            InvariantInput::FromShifts,
        )
    }

    /// Private full supplied-G geometry. Construction supplies polynomial
    /// instructions only: each workspace call must freshly certify every
    /// represented compatibility component before rank/chart/cofactor tests.
    pub(crate) fn new_checked_supplied(
        n: usize,
        coordinate_rank: usize,
        budget: AffineDescentBudget,
    ) -> Result<Self, ReductionError> {
        Self::prepare(n, coordinate_rank, budget, None, InvariantInput::SuppliedScalarProducts)
    }

    pub(crate) fn new_checked_supplied_dual<D: DualNumberStructure>(
        n: usize,
        coordinate_rank: usize,
        budget: AffineDescentBudget,
        dual: &D,
    ) -> Result<Self, ReductionError> {
        iregi_masters::validate_dual_structure(dual)
            .map_err(|error| ReductionError::Preparation(error.to_string()))?;
        Self::prepare(
            n, coordinate_rank, budget,
            Some(dual.get_shape().into_iter().map(<[usize]>::to_vec).collect()),
            InvariantInput::SuppliedScalarProducts,
        )
    }

    fn prepare(
        n: usize,
        rank: usize,
        budget: AffineDescentBudget,
        shape: Option<Vec<Vec<usize>>>,
        invariants: InvariantInput,
    ) -> Result<Self, ReductionError> {
        // Check before allocating shifts or enumerating any chart.
        if n < 2 || n > budget.max_propagators || rank > 4 || rank + 2 > n {
            return Err(ReductionError::Preparation(
                "invalid affine descent profile or resource budget".into(),
            ));
        }
        let layout = geometry_layout(n, invariants)?;
        let (mut parameters, compatibility_constraints) = if invariants == InvariantInput::SuppliedScalarProducts {
            crate::certified_tensor::compatibility_constraints(n)
        } else {
            let parameters = (0..n).flat_map(|i| (0..4).map(move |mu| {
                Atom::var(symbolica::symbol!(format!("iregi_affine_descent::p{i}_{mu}")))
            })).collect();
            (parameters, Vec::new())
        };
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| std::array::from_fn(|mu| parameters[4 * i + mu].clone()))
            .collect();
        let masses: Vec<_> = (0..n)
            .map(|i| Atom::var(symbolica::symbol!(format!("iregi_affine_descent::m2_{i}"))))
            .collect();
        let distances = if let Some(products) = &layout.scalar_products {
            let mut gram = vec![vec![Atom::new(); n]; n];
            let mut argument = products.start;
            for i in 0..n { for j in i..n {
                gram[i][j] = parameters[argument].clone();
                gram[j][i] = parameters[argument].clone();
                argument += 1;
            }}
            Some((0..n).map(|i| (0..n).map(|j|
                &gram[i][i] + &gram[j][j] - Atom::num(2) * &gram[i][j]
            ).collect::<Vec<Atom>>()).collect::<Vec<Vec<Atom>>>())
        } else { None };
        parameters.extend(masses.iter().cloned());
        parameters.push(Atom::var(symbolica::symbol!("iregi_affine_descent::mu")));
        let p = match &distances {
            None => affine_descent_polynomials(&shifts, &masses, rank, budget),
            Some(distances) => affine_descent_polynomials_with_distances(&shifts, &masses, distances, rank, budget),
        }.map_err(ReductionError::Preparation)?;
        let rank_constraints = p.rank_constraints.len();
        let charts = p.charts.iter().map(|(key, _)| key.clone()).collect();
        let mut constraints = p.rank_constraints;
        constraints.extend(p.charts.into_iter().map(|(_, determinant)| determinant));
        constraints.extend(p.cofactors);
        let exact = polynomial_plan(&parameters, &constraints, &shape)?;
        let compatibility = if compatibility_constraints.is_empty() { None }
            else { Some(polynomial_plan(&parameters, &compatibility_constraints, &shape)?) };
        Ok(Self {
            parameters,
            layout,
            propagators: n,
            components: shape.as_ref().map_or(1, Vec::len),
            rank_constraints,
            charts,
            exact,
            compatibility,
        })
    }
    pub(crate) fn parameters(&self) -> &[Atom] {
        &self.parameters
    }
    pub(crate) fn input_layout(&self) -> &InputLayout { &self.layout }
    pub(crate) fn charts(&self) -> &[AffineChartKey] {
        &self.charts
    }
    pub(crate) fn instantiate(&self) -> AffineDescentGuardWorkspace {
        AffineDescentGuardWorkspace {
            propagators: self.propagators,
            layout: self.layout.clone(),
            parameters: self.parameters.len(),
            components: self.components,
            rank_constraints: self.rank_constraints,
            charts: self.charts.len(),
            evaluated_rows: 0,
            passes: 0,
            exact: self.exact.instantiate(),
            compatibility: self.compatibility.as_ref().map(ExactZeroConstraintPlan::instantiate),
        }
    }
}

#[derive(Clone)]
pub(crate) struct AffineDescentGuardWorkspace {
    propagators: usize,
    layout: InputLayout,
    parameters: usize,
    components: usize,
    rank_constraints: usize,
    charts: usize,
    evaluated_rows: usize,
    passes: usize,
    exact: ExactZeroConstraintWorkspace,
    compatibility: Option<ExactZeroConstraintWorkspace>,
}

impl AffineDescentGuardWorkspace {
    pub(crate) fn set_precision_budget(
        &mut self,
        initial: u32,
        maximum: u32,
    ) -> Result<(), ReductionError> {
        self.exact.set_precision_budget(initial, maximum)?;
        if let Some(compatibility) = &mut self.compatibility {
            compatibility.set_precision_budget(initial, maximum)?;
        }
        Ok(())
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
        let mut result = vec![AffineDescentSelection::NoRegularChart; input.rows];
        let mut exact = Vec::with_capacity(input.values.len());
        let mut active = Vec::with_capacity(input.rows);
        for (lane, row) in input.values.chunks_exact(width).enumerate() {
            if let Some(status) = validate_input_row(row, &self.layout, self.components) {
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
        let Some(compatibility) = &mut self.compatibility else {
            return self.select_affine_exact(input, rows);
        };
        // This is deliberately a complete check_batch, not a selected-primal
        // or selected-component test. No rank fallback may accept a bad G jet.
        let proofs = compatibility.check_batch(input, rows);
        self.evaluated_rows = self.evaluated_rows.checked_add(compatibility.evaluated_rows())
            .ok_or_else(|| ReductionError::Evaluation("affine compatibility rows overflow".into()))?;
        self.passes = self.passes.checked_add(compatibility.passes())
            .ok_or_else(|| ReductionError::Evaluation("affine compatibility passes overflow".into()))?;
        let proofs = proofs?;
        if proofs.len() != rows {
            return Err(ReductionError::Evaluation("affine compatibility certificate row count mismatch".into()));
        }
        let mut result = vec![AffineDescentSelection::NoRegularChart; rows];
        let mut active = Vec::with_capacity(rows);
        let mut packed = Vec::with_capacity(input.len());
        for (lane, proof) in proofs.into_iter().enumerate() {
            match proof {
                ExactZeroCertificate::Zero => {
                    active.push(lane);
                    packed.extend_from_slice(&input[lane * width..(lane + 1) * width]);
                }
                ExactZeroCertificate::Nonzero { constraint, component } => {
                    result[lane] = AffineDescentSelection::InvalidInput(format!(
                        "supplied-G/Cartesian compatibility constraint {constraint} is certified nonzero in Taylor component {component}; input was preserved"
                    ));
                }
                ExactZeroCertificate::Unresolved { working_bits } => {
                    result[lane] = AffineDescentSelection::Unresolved { working_bits };
                }
            }
        }
        // Keep cumulative proof counters. The former selection body must not
        // reset or omit the compatibility work when scattering its sub-batch.
        let selections = self.select_affine_exact(&packed, active.len())?;
        if selections.len() != active.len() {
            return Err(ReductionError::Evaluation("affine checked selection row count mismatch".into()));
        }
        for (lane, selection) in active.into_iter().zip(selections) { result[lane] = selection; }
        Ok(result)
    }

    fn select_affine_exact(
        &mut self,
        input: &[Complex<Rational>],
        rows: usize,
    ) -> Result<Vec<AffineDescentSelection>, ReductionError> {
        let (width, total) = self.width(rows)?;
        if input.len() != total {
            return Err(ReductionError::InvalidInput("affine private exact input length mismatch".into()));
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

fn polynomial_plan(
    parameters: &[Atom],
    constraints: &[Atom],
    shape: &Option<Vec<Vec<usize>>>,
) -> Result<ExactZeroConstraintPlan, ReductionError> {
    match shape {
        None => ExactZeroConstraintPlan::new(parameters, constraints),
        Some(shape) => ExactZeroConstraintPlan::new_dual(
            parameters, constraints, &HyperDual::<u8>::new(shape.clone()),
        ),
    }
}

fn geometry_layout(n: usize, invariants: InvariantInput) -> Result<InputLayout, ReductionError> {
    let overflow = || ReductionError::Preparation("affine guard geometry dimensions overflow".into());
    let shift_end = n.checked_mul(4).ok_or_else(overflow)?;
    let scalar_products = if invariants == InvariantInput::SuppliedScalarProducts {
        let next = n.checked_add(1).ok_or_else(overflow)?;
        let count = n.checked_mul(next).ok_or_else(overflow)? / 2;
        let end = shift_end.checked_add(count).ok_or_else(overflow)?;
        Some(shift_end..end)
    } else { None };
    let mass_start = scalar_products.as_ref().map_or(shift_end, |range| range.end);
    let scale = mass_start.checked_add(n).ok_or_else(overflow)?;
    scale.checked_add(1).ok_or_else(overflow)?;
    Ok(InputLayout { shifts: 0..shift_end, scalar_products, masses: mass_start..scale, scale })
}

#[cfg(test)]
#[path = "affine_descent_guard_tests.rs"]
mod tests;

#[cfg(test)]
mod checked_supplied_tests {
    use super::*;
    use symbolica::domains::float::SingleFloat;

    fn c(value: i64) -> Complex<Rational> {
        Complex::new(Rational::from(value), Rational::from(0))
    }
    fn dual() -> HyperDual<u8> {
        HyperDual::new(vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]])
    }
    fn triangle() -> Vec<Complex<Rational>> {
        // Same collinear Cartesian fixture as the existing guard tests; the
        // six upper-G entries are explicit independent exact input data.
        [0,0,0,0, 0,1,0,0, 0,2,0,0, 0,0,0,-1,-2,-4, 1,2,3,3]
            .into_iter().map(c).collect()
    }
    fn planar_five() -> Vec<Complex<Rational>> {
        [0,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1, 0,1,1,1,
         0,0,0,0,0, -1,0,0,-1, -1,0,-1, -1,-1,-3, 4,9,16,25,36,3]
            .into_iter().map(c).collect()
    }
    fn jets(row: Vec<Complex<Rational>>) -> Vec<Complex<Rational>> {
        row.into_iter().flat_map(|value| [value,c(0),c(0),c(0)]).collect()
    }

    #[test]
    fn checked_guard_uses_original_full_layout_and_recertifies_bad_g_before_rank_selection() {
        let plan = AffineDescentGuardPlan::new_checked_supplied(3,1,AffineDescentBudget::default()).unwrap();
        assert_eq!(plan.parameters().len(),22);
        assert_eq!(plan.input_layout().shifts,0..12);
        assert_eq!(plan.input_layout().scalar_products,Some(12..18));
        assert_eq!(plan.input_layout().masses,18..21);
        assert_eq!(plan.input_layout().scale,21);
        let (geometry, _) = crate::certified_tensor::compatibility_constraints(3);
        assert_eq!(&plan.parameters()[..18],geometry.as_slice());
        let mut workspace = plan.instantiate();
        let good = triangle();
        let mut bad = good.clone(); bad[12]=c(1);
        let input = [good.clone(),bad.clone(),good.clone()].concat();
        let original = input.clone();
        let selected = workspace.select_batch_exact(&input,3).unwrap();
        assert_eq!(input,original);
        assert!(matches!(selected[0],AffineDescentSelection::Chart{..}));
        assert!(matches!(&selected[1],AffineDescentSelection::InvalidInput(reason)
            if reason.contains("supplied-G/Cartesian") && reason.contains("component 0")));
        assert_eq!(selected[2],selected[0]);
        assert!(workspace.evaluated_rows()>3);
        assert!(workspace.passes()>1);
        assert!(matches!(workspace.select_batch_exact(&bad,1).unwrap()[0],AffineDescentSelection::InvalidInput(_)));
        assert_eq!(workspace.evaluated_rows(),1);
        assert_eq!(workspace.passes(),1);
        assert_eq!(workspace.exact.evaluated_rows(),0,"bad G cannot reach rank/chart/cofactor selection");
        assert_eq!(workspace.select_batch_exact(&good,1).unwrap(),[selected[0].clone()]);
        let original_plan = AffineDescentGuardPlan::new(3,1,AffineDescentBudget::default()).unwrap();
        assert_eq!(original_plan.parameters().len(),16);
        assert!(original_plan.input_layout().scalar_products.is_none());
        assert!(original_plan.compatibility.is_none());
    }

    #[test]
    fn checked_guard_keeps_ordered_mixed_compatibility_and_validates_full_numeric_rows() {
        let plan = AffineDescentGuardPlan::new_checked_supplied_dual(3,1,AffineDescentBudget::default(),&dual()).unwrap();
        let mut workspace = plan.instantiate();
        let mut good = jets(triangle());
        for line in 0..3 {
            // Common routing translation need not alter authoritative relative G.
            good[(4*line)*4+1]=c(2);
            good[(4*line+1)*4+2]=c(3);
            good[(4*line+2)*4+3]=c(5);
        }
        let mut mixed = good.clone(); mixed[12*4+3]=c(1);
        let exact = [good.clone(),mixed.clone(),good.clone()].concat();
        let selected = workspace.select_batch_exact(&exact,3).unwrap();
        assert!(matches!(selected[0],AffineDescentSelection::Chart{..}));
        assert!(matches!(&selected[1],AffineDescentSelection::InvalidInput(reason)
            if reason.contains("Taylor component 3")));
        assert_eq!(selected[2],selected[0]);
        let numeric = |row:&[Complex<Rational>]| row.iter().map(|value|
            Complex::new(0_f64.from_rational(&value.re),0_f64.from_rational(&value.im))).collect::<Vec<_>>();
        let numeric_good = numeric(&good);
        let mut bad_mu = numeric_good.clone(); bad_mu[21*4].re = -1.;
        let mut bad_mass = numeric_good.clone(); bad_mass[18*4].im = 1.;
        let input = KinematicsBatch { rows:4,components:4,values:[numeric_good.clone(),numeric(&mixed),bad_mu,bad_mass].concat() };
        let before = input.values.clone();
        let selected = workspace.select_batch(&input).unwrap();
        assert_eq!(input.values,before);
        assert!(matches!(selected[0],AffineDescentSelection::Chart{..}));
        assert!(matches!(&selected[1],AffineDescentSelection::InvalidInput(reason) if reason.contains("Taylor component 3")));
        assert!(matches!(&selected[2],AffineDescentSelection::InvalidInput(reason) if reason.contains("mu must be positive")));
        assert!(matches!(&selected[3],AffineDescentSelection::InvalidInput(reason) if reason.contains("squared masses")));
        assert_eq!(workspace.select_batch_exact(&good,1).unwrap(),[selected[0].clone()]);
        assert!(workspace.select_batch_exact(&[],0).unwrap().is_empty());
        assert_eq!(workspace.evaluated_rows(),0);
        assert_eq!(workspace.passes(),0);
    }

    #[test]
    fn checked_five_point_rank_selection_keeps_regular_primal_distinct_from_transverse_mixed_jet() {
        let plan = AffineDescentGuardPlan::new_checked_supplied_dual(5,3,AffineDescentBudget::default(),&dual()).unwrap();
        assert_eq!(plan.parameters().len(),41);
        assert_eq!(plan.rank_constraints,1,"N5/r3 has one full coordinate determinant");
        let mut workspace = plan.instantiate();
        let good = jets(planar_five());
        let mut transverse = good.clone();
        // p4.time=st preserves G in this algebra because (st)^2=0, but leaves
        // the planar variety at a mixed order despite the identical primal.
        transverse[16*4+3]=c(1);
        let mut regular = good.clone(); regular[16*4]=c(1); regular[34*4]=c(-2);
        let input = [good.clone(),transverse,regular,good].concat();
        let original = input.clone();
        let selected = workspace.select_batch_exact(&input,4).unwrap();
        assert_eq!(input,original);
        assert!(matches!(selected[0],AffineDescentSelection::Chart{..}));
        assert_eq!(selected[1],AffineDescentSelection::OffAffineFamily {constraint:0,component:3});
        assert_eq!(selected[2],AffineDescentSelection::OffAffineFamily {constraint:0,component:0});
        assert_eq!(selected[3],selected[0]);
        let order = [3,1,0,2];
        let reordered = order.iter().flat_map(|&lane| input[lane*164..(lane+1)*164].iter().cloned()).collect::<Vec<_>>();
        let actual = workspace.select_batch_exact(&reordered,4).unwrap();
        for (lane,&original) in order.iter().enumerate() { assert_eq!(actual[lane],selected[original]); }
        for (lane,expected) in selected.into_iter().enumerate() {
            assert_eq!(workspace.select_batch_exact(&input[lane*164..(lane+1)*164],1).unwrap(),[expected]);
        }
    }

    #[test]
    fn checked_compatibility_precision_budget_never_projects_tiny_bad_products_to_zero() {
        let plan = AffineDescentGuardPlan::new_checked_supplied(3,1,AffineDescentBudget::default()).unwrap();
        let mut workspace = plan.instantiate();
        let mut tiny = triangle(); tiny[15].re += Rational::from((1,2)).pow(300);
        workspace.set_precision_budget(128,128).unwrap();
        assert_eq!(workspace.select_batch_exact(&tiny,1).unwrap(),[AffineDescentSelection::Unresolved {working_bits:128}]);
        assert_eq!(workspace.exact.evaluated_rows(),0);
        workspace.set_precision_budget(128,1024).unwrap();
        assert!(matches!(workspace.select_batch_exact(&tiny,1).unwrap()[0],AffineDescentSelection::InvalidInput(_)));
        assert!(workspace.select_batch_exact(&tiny[..21],1).is_err());
        assert_eq!(workspace.evaluated_rows(),0);
        assert_eq!(workspace.passes(),0);
    }
}
