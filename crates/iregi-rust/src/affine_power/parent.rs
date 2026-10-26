//! Reuse the guarded executor for a small imported-child recurrence graph.
//! Child integrals remain numerical inputs; only the existing constant-one
//! routing relation is emitted by the metadata DAG.
use super::{Shape, dag::AffinePowerDag};
use crate::{
    InputLayout, ReductionError, ReductionPlan, ReductionSettings,
    basis::tensor_basis,
    pv::{AffineChartKey, AffineDescentBudget, affine_descent_chart},
    stability::quadrature::{
        QuadratureStabilityPlan, imported_affine::PreparedImportedAffineQuadrature,
    },
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::dual::HyperDual,
};

pub(super) struct ParentRecipe {
    pub plan: ReductionPlan,
    pub quadrature: PreparedImportedAffineQuadrature,
    pub checks: QuadratureStabilityPlan,
    pub exact_zero_outputs: Vec<bool>,
    pub geometry_parameters: usize,
}

fn prep(message: impl Into<String>) -> ReductionError {
    ReductionError::Preparation(message.into())
}
fn mul(a: usize, b: usize) -> Result<usize, ReductionError> {
    a.checked_mul(b)
        .ok_or_else(|| prep("affine parent dimensions overflow"))
}
fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("iregi_affine_power::{name}")))
}

impl ParentRecipe {
    pub fn prepare(
        settings: &ReductionSettings,
        key: &AffineChartKey,
        inactive: &[usize],
        dag: &AffinePowerDag,
        geometry: &[Atom],
        shape: &Shape,
        chart_budget: AffineDescentBudget,
        max_parameters: usize,
    ) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        let geometry_count = mul(n, 5)?
            .checked_add(1)
            .ok_or_else(|| prep("affine parent parameter count overflows"))?;
        if geometry.len() != geometry_count {
            return Err(prep("affine guard geometry layout differs"));
        }
        let basis = tensor_basis(settings.rank);
        let outputs = mul(basis.len(), 4)?;
        let imported = mul(dag.terminals.len(), outputs)?;
        let parameters_count = geometry_count
            .checked_add(mul(imported, 2)?)
            .ok_or_else(|| prep("affine imported parameter count overflows"))?;
        if parameters_count > max_parameters {
            return Err(prep("affine imported parent parameter budget exceeded"));
        }
        let order = shape.as_ref().map_or(Ok(0u16), |shape| {
            shape.iter().try_fold(0u16, |maximum, row| {
                row.iter()
                    .try_fold(0usize, |sum, &d| sum.checked_add(d))
                    .and_then(|d| u16::try_from(d).ok())
                    .map(|d| maximum.max(d))
                    .ok_or_else(|| prep("affine analytic order exceeds u16"))
            })
        })?;
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| std::array::from_fn(|mu| geometry[4 * i + mu].clone()))
            .collect();
        let masses = geometry[4 * n..5 * n].to_vec();
        let chart = affine_descent_chart(&shifts, &masses, key, chart_budget)
            .map_err(prep)?
            .ok_or_else(|| prep("affine selected chart is structurally singular"))?;
        if inactive.iter().any(|&line| line >= n) || inactive.windows(2).any(|w| w[0] >= w[1]) {
            return Err(prep(
                "affine inactive coefficient indices differ from the certified profile",
            ));
        }
        let mut aliases = Vec::new();
        let coefficients: Vec<_> = chart
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(line, expression)| {
                if inactive.binary_search(&line).is_ok() {
                    Atom::new()
                } else {
                    let handle = variable(&format!("coefficient_{line}"));
                    aliases.push((handle.clone(), expression));
                    handle
                }
            })
            .collect();
        let coefficient_aliases = aliases.clone();
        let values: Vec<_> = (0..imported)
            .map(|i| variable(&format!("child_value_{i}")))
            .collect();
        let amplitudes: Vec<_> = (0..imported)
            .map(|i| variable(&format!("child_amplitude_{i}")))
            .collect();
        let mut external_primal_seeds = Vec::with_capacity(imported);
        for (terminal, source) in dag.terminals.iter().enumerate() {
            let power_sum = source
                .remaining_powers
                .iter()
                .try_fold(0i32, |sum, &p| sum.checked_add(i32::from(p)))
                .ok_or_else(|| prep("affine child dimension sum overflows"))?;
            let base = 4i32
                .checked_add(2 * i32::from(settings.dimension_shift))
                .and_then(|v| power_sum.checked_mul(2).and_then(|p| v.checked_sub(p)))
                .ok_or_else(|| prep("affine child mass dimension overflows"))?;
            for (moment, monomial) in basis.iter().enumerate() {
                let dimension = base
                    .checked_add(
                        i32::try_from(monomial.rank())
                            .map_err(|_| prep("affine rank dimension overflows"))?,
                    )
                    .ok_or_else(|| prep("affine child tensor dimension overflows"))?;
                for pole in 0..4 {
                    let index = terminal * outputs + 4 * moment + pole;
                    external_primal_seeds.push((
                        values[index].clone(),
                        amplitudes[index].clone(),
                        dimension,
                    ));
                }
            }
        }
        let mut moments = Vec::with_capacity(basis.len());
        for moment in 0..basis.len() {
            let mut slots: [Atom; 4] = std::array::from_fn(|_| Atom::new());
            for (pole, slot) in slots.iter_mut().enumerate() {
                let output = moment * 4 + pole;
                let leaves: Vec<_> = (0..dag.terminals.len())
                    .map(|t| values[t * outputs + output].clone())
                    .collect();
                let emitted = dag
                    .emit(&coefficients, &leaves, &format!("output_{output}"))
                    .map_err(prep)?;
                aliases.extend(emitted.aliases);
                *slot = emitted.root;
            }
            moments.push(slots);
        }
        // Error propagation uses the SAME immutable coefficient vector and DAG
        // as the central outputs. Symbolica projects effective signed weights;
        // no handwritten coefficient or Taylor multiplication is introduced.
        let markers: Vec<_> = (0..dag.terminals.len())
            .map(|i| variable(&format!("error_source_{i}")))
            .collect();
        let representative = dag
            .emit(&coefficients, &markers, "error_representative")
            .map_err(prep)?;
        let mut projection_aliases = coefficient_aliases;
        projection_aliases.extend(representative.aliases);
        let carrier = shape.as_ref().map(|s| HyperDual::<u8>::new(s.clone()));
        let quadrature = PreparedImportedAffineQuadrature::prepare(
            geometry,
            &projection_aliases,
            &representative.root,
            &markers,
            outputs,
            carrier.as_ref(),
        )?;
        let mut parameters = geometry.to_vec();
        parameters.extend(values);
        parameters.extend(amplitudes);
        let layout = InputLayout {
            shifts: 0..4 * n,
            scalar_products: None,
            masses: 4 * n..5 * n,
            scale: 5 * n,
        };
        let distances: Vec<Vec<_>> = shifts
            .iter()
            .map(|p| {
                shifts
                    .iter()
                    .map(|q| crate::plan::squared_difference(p, q))
                    .collect()
            })
            .collect();
        let plan = ReductionPlan::new_affine_power_parent(
            settings.clone(),
            order,
            parameters,
            external_primal_seeds,
            layout,
            distances,
            moments,
            aliases,
            vec![format!(
                "Certified constant-denominator positive-power DAG: {} states, {} edges, {} pinched outputs; unchanged q, full rank {}, h={}; child methods reported separately",
                dag.nodes.len(),
                dag.edge_count,
                dag.terminals.len(),
                settings.rank,
                settings.dimension_shift
            )],
        )?;
        let exact_zero_outputs = plan
            .moments
            .iter()
            .flatten()
            .map(AtomCore::is_zero)
            .collect();
        let checks = QuadratureStabilityPlan::imported_dense_checks(&plan)?;
        Ok(Self {
            plan,
            quadrature,
            checks,
            exact_zero_outputs,
            geometry_parameters: geometry_count,
        })
    }
}
