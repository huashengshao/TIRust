//! Reuse the guarded executor for a small imported-child recurrence graph.
//! Child integrals remain numerical inputs; only the existing constant-one
//! routing relation is emitted by the metadata DAG.
use super::{Shape, dag::AffinePowerDag};
use crate::{
    InputLayout, InvariantInput, ReductionError, ReductionPlan, ReductionSettings,
    basis::tensor_basis,
    pv::{
        AffineChartKey, AffineDescentBudget, affine_descent_chart,
        affine_descent_chart_with_distances,
    },
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
        let shift_end = mul(n, 4)?;
        let supplied = settings.invariants == InvariantInput::SuppliedScalarProducts;
        let products = if supplied {
            n.checked_add(1)
                .and_then(|m| n.checked_mul(m))
                .map(|v| v / 2)
                .ok_or_else(|| prep("affine supplied-product count overflows"))?
        } else {
            0
        };
        let mass_start = shift_end
            .checked_add(products)
            .ok_or_else(|| prep("affine mass layout overflows"))?;
        let scale = mass_start
            .checked_add(n)
            .ok_or_else(|| prep("affine scale layout overflows"))?;
        let geometry_count = scale
            .checked_add(1)
            .ok_or_else(|| prep("affine parent parameter count overflows"))?;
        let layout = InputLayout {
            shifts: 0..shift_end,
            scalar_products: supplied.then_some(shift_end..mass_start),
            masses: mass_start..scale,
            scale,
        };
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
        let masses = geometry[layout.masses.clone()].to_vec();
        let distances: Vec<Vec<Atom>> = if let Some(range) = &layout.scalar_products {
            let mut original_products = vec![vec![Atom::new(); n]; n];
            let mut parameter = range.start;
            for i in 0..n {
                for j in i..n {
                    let original = geometry[parameter].clone();
                    parameter += 1;
                    original_products[i][j] = original.clone();
                    original_products[j][i] = original;
                }
            }
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            &original_products[i][i] + &original_products[j][j]
                                - Atom::num(2) * &original_products[i][j]
                        })
                        .collect()
                })
                .collect()
        } else {
            shifts
                .iter()
                .map(|p| {
                    shifts
                        .iter()
                        .map(|q| crate::plan::squared_difference(p, q))
                        .collect()
                })
                .collect()
        };
        let chart = if supplied {
            affine_descent_chart_with_distances(&shifts, &masses, &distances, key, chart_budget)
        } else {
            affine_descent_chart(&shifts, &masses, key, chart_budget)
        }
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
        let constructor = if supplied {
            ReductionPlan::new_checked_affine_power_parent
        } else {
            ReductionPlan::new_affine_power_parent
        };
        let plan = constructor(
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

#[cfg(test)]
mod supplied_parent_tests {
    use super::*;
    use crate::affine_power::dag::DagBudget;
    use crate::{Algorithm, Convention, InvariantInput, TensorWeights};
    #[test]
    fn checked_parent_retains_original_gram_dimensions_and_requested_conventions() {
        let (mut geometry, _) = crate::certified_tensor::compatibility_constraints(3);
        geometry.extend(
            (0..3).map(|i| Atom::var(symbolica::symbol!(format!("iregi_affine_descent::m2_{i}")))),
        );
        geometry.push(Atom::var(symbolica::symbol!("iregi_affine_descent::mu")));
        assert_eq!(geometry.len(), 22);
        let key = AffineChartKey {
            coordinate_rows: vec![1],
            relative_columns: vec![0, 1],
        };
        let dag = AffinePowerDag::new(
            vec![2, 1, 1],
            &[],
            DagBudget {
                max_nodes: 128,
                max_edges: 256,
                max_terminals: 64,
            },
        )
        .unwrap();
        for dimension_shift in [-1, 0, 1] {
            let mut settings = ReductionSettings::new(3, 1);
            settings.invariants = InvariantInput::SuppliedScalarProducts;
            settings.powers = vec![2, 1, 1];
            settings.dimension_shift = dimension_shift;
            settings.algorithm = Algorithm::Ibp;
            settings.convention = Convention::MadLoop;
            settings.tensor_weights = TensorWeights::Multinomial;
            let recipe = ParentRecipe::prepare(
                &settings,
                &key,
                &[],
                &dag,
                &geometry,
                &None,
                AffineDescentBudget::default(),
                100_000,
            )
            .unwrap();
            assert_eq!(recipe.geometry_parameters, 22);
            assert_eq!(recipe.plan.layout.scalar_products, Some(12..18));
            assert_eq!(recipe.plan.layout.masses, 18..21);
            assert_eq!(recipe.plan.layout.scale, 21);
            assert_eq!(&recipe.plan.parameters[..22], geometry.as_slice());
            assert_eq!(
                serde_json::to_value(recipe.plan.settings()).unwrap(),
                serde_json::to_value(&settings).unwrap()
            );
            let original_r01 = &geometry[12] + &geometry[15] - Atom::num(2) * &geometry[13];
            assert!(
                (&recipe.plan.raw_distances[0][1] - original_r01)
                    .together()
                    .is_zero()
            );
            assert!(recipe.plan.leaves.is_empty());
            assert!(recipe.plan.master_arguments.is_empty());
            assert!(!recipe.plan.external_primal_seeds.is_empty());
            assert!(
                recipe.plan.with_analytic_jet_order(2).is_err(),
                "imported checked parent cannot rebuild as a generic unrelated plan"
            );
            assert!(
                ReductionPlan::new_affine_power_parent(
                    settings.clone(),
                    0,
                    recipe.plan.parameters.clone(),
                    recipe.plan.external_primal_seeds.clone(),
                    recipe.plan.layout.clone(),
                    recipe.plan.raw_distances.clone(),
                    recipe.plan.moments.clone(),
                    recipe.plan.aliases.clone(),
                    vec![]
                )
                .is_err(),
                "existing Cartesian private constructor still rejects supplied G"
            );
        }
    }
}
