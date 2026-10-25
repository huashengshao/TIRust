//! Private flattened N7 plan. Symbolica owns both coefficient levels and jets.
use super::*;
use crate::pv::{PreparationTimer, ThinSevenGeometry};
use symbolica::atom::{AtomView, Symbol};

impl ReductionPlan {
    pub(crate) fn new_thin_seven(
        settings: ReductionSettings,
        analytic_jet_order: u16,
    ) -> Result<Self, ReductionError> {
        use crate::types::{Convention, TensorWeights};
        if settings.propagators != 7
            || settings.rank > 7
            || settings.powers != [1; 7]
            || settings.dimension_shift != 0
            || settings.invariants != InvariantInput::FromShifts
            || settings.input_policy != InputPolicy::Preserve
            || settings.convention != Convention::IregiNative
            || settings.tensor_weights != TensorWeights::Components
            || settings.algorithm != Algorithm::PassarinoVeltman
        {
            return Err(ReductionError::Unsupported(
                "private thin N7 requires rank<=7, unit powers, h=0, preserved shifts, PV, native raw components".into(),
            ));
        }
        let shifts: Vec<[Atom; 4]> = (0..7)
            .map(|i| std::array::from_fn(|mu| variable(&format!("p{i}_{mu}"))))
            .collect();
        let masses: Vec<_> = (0..7).map(|i| variable(&format!("m2_{i}"))).collect();
        let mut parameters: Vec<_> = shifts.iter().flatten().cloned().collect();
        parameters.extend(masses.iter().cloned());
        parameters.push(variable("mu"));
        let layout = InputLayout {
            shifts: 0..28,
            scalar_products: None,
            masses: 28..35,
            scale: 35,
        };
        let distances: Vec<Vec<_>> = shifts
            .iter()
            .map(|p| shifts.iter().map(|q| squared_difference(p, q)).collect())
            .collect();
        let mut compiler = PvCompiler::with_distances(shifts, masses, distances.clone(), true)
            .map_err(ReductionError::Preparation)?;
        let profile = std::env::var_os("IREGI_PV_PREPARATION_PROFILE").is_some();
        let _geometry = PreparationTimer::new(profile, format_args!("thin N7 all geometry"));
        let geometry = compiler
            .thin_seven_geometry(&(0..7).collect::<Vec<_>>())
            .map_err(ReductionError::Preparation)?;
        drop(_geometry);
        let _layout = PreparationTimer::new(profile, format_args!("thin N7 parent basis and imported layout"));
        let child_rank = settings.rank.saturating_sub(2);
        let child_basis = tensor_basis(child_rank);
        let child_slots = 21 * 4 * child_basis.len();
        let values: Vec<_> = (0..child_slots)
            .map(|i| variable(&format!("thin7_child_{i}")))
            .collect();
        let amplitudes: Vec<_> = (0..child_slots)
            .map(|i| variable(&format!("thin7_amplitude_{i}")))
            .collect();
        let external_primal_seeds = values
            .iter()
            .zip(&amplitudes)
            .enumerate()
            .map(|(i, (value, amplitude))| {
                (
                    value.clone(),
                    amplitude.clone(),
                    child_basis[(i % (4 * child_basis.len())) / 4].rank() as i32 - 6,
                )
            })
            .collect();
        parameters.extend(values.iter().cloned());
        parameters.extend(amplitudes);
        let basis = tensor_basis(settings.rank);
        let mut aliases = compiler.aliases().to_vec();
        let mut bind = |body: Atom| {
            if matches!(body.as_view(), AtomView::Num(_) | AtomView::Var(_)) {
                return body;
            }
            let name = variable(&format!("thin7_intermediate_{}", aliases.len()));
            aliases.push((name.clone(), body));
            name
        };
        drop(_layout);
        let _assembly = PreparationTimer::new(profile, format_args!("thin N7 parent assembly"));
        let mut moments =
            assemble_thin_seven_moments(&basis, &child_basis, &geometry, &values, &mut bind)
                .map_err(ReductionError::Preparation)?;
        drop(_assembly);
        let _uv = PreparationTimer::new(profile, format_args!("thin N7 final UV rule"));
        // Imported children already have native normalization. This shared
        // final rule is separate from the pre-rule exact pole-cancellation tests.
        apply_final_uv_power_counting(&settings, &basis, &mut moments)?;
        drop(_uv);
        Ok(Self {
            execution_cache: Default::default(),
            variant_cache: Default::default(),
            settings,
            basis,
            parameters,
            external_primal_seeds,
            layout,
            raw_distances: distances.clone(),
            distances,
            moments,
            leaves: Vec::new(),
            master_arguments: Vec::new(),
            aliases,
            snapping_aliases: Vec::new(),
            snapping_diagnostics: Vec::new(),
            snapping_threshold: None,
            snapping_zero_image: None,
            snapping_zero_image_required: false,
            fallback: None,
            algorithm_used: Algorithm::PassarinoVeltman,
            preparation_notes: vec![format!(
                "Private thin N7: one packed batch of all 21 N5 rank-{child_rank} subsets, two guarded coefficient levels, no numerical cofactor pruning or quadrature route support"
            )],
            special_kinematics: true,
            analytic_jet_order,
            bubble_limit: None,
            zero_invariants: None,
            certified_scalar_basis: None,
            certified_planar_chart: None,
            certified_planar_mask: 0,
            thin_six_child_rank: None,
            thin_seven_child_rank: Some(child_rank),
        })
    }
}

/// Two numerator lowerings, with the constant column at rank zero. This is
/// structural index dispatch; all multiplication, addition and IFs are Atom APIs.
pub(crate) fn assemble_thin_seven_moments(
    basis: &[Monomial],
    child_basis: &[Monomial],
    geometry: &ThinSevenGeometry,
    values: &[Atom],
    bind: &mut impl FnMut(Atom) -> Atom,
) -> Result<Vec<[Atom; 4]>, String> {
    let child_slots = child_basis
        .len()
        .checked_mul(4)
        .ok_or("N7 child width overflow")?;
    if geometry.active != (0..7).collect::<Vec<_>>()
        || geometry.six.len() != 7
        || geometry.charts.is_empty()
        || values.len()
            != 21usize
                .checked_mul(child_slots)
                .ok_or("N7 child count overflow")?
        || child_basis.is_empty()
    {
        return Err("invalid thin N7 geometry or imported child layout".into());
    }
    let child_indices: HashMap<_, _> = child_basis
        .iter()
        .enumerate()
        .map(|(i, monomial)| (monomial.0, i))
        .collect();
    if child_indices.len() != child_basis.len() {
        return Err("duplicate thin N7 child monomial".into());
    }
    for (omitted, six) in geometry.six.iter().enumerate() {
        let expected: Vec<_> = (0..7).filter(|&i| i != omitted).collect();
        if six.active != expected
            || six
                .weights
                .as_ref()
                .is_some_and(|w| w.iter().any(|c| c.len() != 6))
        {
            return Err("invalid ordered thin N7 six-line geometry".into());
        }
    }
    let mut seen_charts = std::collections::HashSet::new();
    for chart in &geometry.charts {
        if chart.omitted >= 6
            || !seen_charts.insert(chart.omitted)
            || chart.weights.iter().any(|c| c.len() != 7)
            || chart
                .weights
                .iter()
                .any(|c| !c[chart.omitted + 1].is_zero())
        {
            return Err("invalid thin N7 chart ordering or structural zero".into());
        }
    }
    // A single imported N5 value can occur along two ordered pinch paths.
    // Coalesce that leaf, while retaining an alias for every distinct N6
    // moment so its own local denominator/cancellation checks remain visible.
    let mut six_moments: HashMap<(usize, [u16; 4], usize), Atom> = HashMap::new();
    let mut result = Vec::with_capacity(basis.len());
    for monomial in basis {
        let (outer_column, remainder) = lowered(monomial.0);
        let (inner_column, child_exponent) = lowered(remainder);
        let child_index = *child_indices
            .get(&child_exponent)
            .ok_or("thin N7 imported child basis omits a demanded monomial")?;
        let mut candidates = Vec::with_capacity(geometry.charts.len());
        for chart in &geometry.charts {
            let mut output: [Atom; 4] = std::array::from_fn(|_| Atom::new());
            for (pole, slot) in output.iter_mut().enumerate() {
                let mut terms = Vec::new();
                for omitted in 0..7 {
                    let outer = &chart.weights[outer_column][omitted];
                    // Only a structural Atom zero permits omission. A zero
                    // primal or cofactor center never removes a Taylor path.
                    if outer.is_zero() {
                        continue;
                    }
                    let key = (omitted, remainder, pole);
                    let inner = if let Some(value) = six_moments.get(&key) {
                        value.clone()
                    } else {
                        let six = &geometry.six[omitted];
                        let weights = six.weights.as_ref().ok_or_else(|| {
                            format!("thin N7 demanded six-line geometry omitting {omitted} is structurally singular; this restricted dispatcher has no homogeneous fallback")
                        })?;
                        let mut inner_terms = Vec::new();
                        for (local, &second_omitted) in six.active.iter().enumerate() {
                            let weight = &weights[inner_column][local];
                            if weight.is_zero() {
                                continue;
                            }
                            let subset = crate::thin_n7::subset_index(omitted, second_omitted)
                                .ok_or("invalid thin N7 double-pinch index")?;
                            inner_terms.push(
                                weight * &values[subset * child_slots + 4 * child_index + pole],
                            );
                        }
                        let value = bind(Atom::add_many(&inner_terms));
                        six_moments.insert(key, value.clone());
                        value
                    };
                    terms.push(outer * &inner);
                }
                *slot = Atom::add_many(&terms);
            }
            candidates.push((&chart.determinant, output));
        }
        let (_, mut output) = candidates.pop().ok_or("no thin N7 chart")?;
        // Preserve the existing chart priority. The fallback still contains
        // its active inverse if every numerical chart is singular; never zero.
        for (determinant, candidate) in candidates.into_iter().rev() {
            for (slot, when_true) in output.iter_mut().zip(candidate) {
                if *slot != when_true {
                    *slot = Symbol::IF.call((determinant, &when_true, &*slot));
                }
            }
        }
        result.push(output);
    }
    Ok(result)
}

fn lowered(mut exponent: [u16; 4]) -> (usize, [u16; 4]) {
    if let Some(mu) = exponent.iter().position(|&n| n != 0) {
        exponent[mu] -= 1;
        (mu + 1, exponent)
    } else {
        (0, exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thin_seven_constructor_rejects_unsupported_settings_before_geometry() {
        for fault in 0..8 {
            let mut settings = ReductionSettings::new(7, 0);
            settings.algorithm = Algorithm::PassarinoVeltman;
            match fault {
                0 => settings.propagators = 6,
                1 => settings.rank = 8,
                2 => settings.powers[0] = 2,
                3 => settings.dimension_shift = 1,
                4 => settings.convention = crate::Convention::OneLoop,
                5 => settings.tensor_weights = crate::TensorWeights::Multinomial,
                6 => settings.algorithm = Algorithm::Ibp,
                7 => settings.powers.clear(),
                _ => unreachable!(),
            }
            assert!(matches!(
                ReductionPlan::new_thin_seven(settings, 0),
                Err(ReductionError::Unsupported(_))
            ));
        }
    }

    #[test]
    #[ignore = "generic N7 chart preparation; requires a separate explicit source-qualified runtime budget"]
    fn thin_seven_variant_rebuild_retains_child_topology_and_native_context() {
        let mut settings = ReductionSettings::new(7, 0);
        settings.algorithm = Algorithm::PassarinoVeltman;
        let plan = ReductionPlan::new_thin_seven(settings, 0).unwrap();
        assert_eq!(plan.thin_six_child_rank, None);
        assert_eq!(plan.thin_seven_child_rank, Some(0));
        assert_eq!(plan.parameters.len(), 36 + 2 * 21 * 4);
        assert_eq!(plan.external_primal_seeds.len(), 21 * 4);
        let rebuilt = plan.with_analytic_jet_order(2).unwrap();
        let cached = plan.with_analytic_jet_order(2).unwrap();
        assert!(Arc::ptr_eq(&rebuilt, &cached));
        assert_eq!(rebuilt.thin_six_child_rank, None);
        assert_eq!(rebuilt.thin_seven_child_rank, Some(0));
        assert_eq!(rebuilt.analytic_jet_order, 2);
        assert_eq!(rebuilt.parameters, plan.parameters);
        assert_eq!(rebuilt.external_primal_seeds, plan.external_primal_seeds);
        assert_eq!(rebuilt.basis(), plan.basis());
        assert_eq!(rebuilt.expressions(), plan.expressions());
        assert!(
            plan.with_zero_invariants(ZeroInvariantProfile {
                masses: vec![false; 7],
                distances: Vec::new()
            })
            .is_err()
        );
        for (six, seven) in [(Some(0), Some(0)), (None, Some(1)), (None, None)] {
            assert!(
                plan.variant(PlanVariant {
                    analytic_jet_order: 2,
                    bubble_equal_masses: None,
                    zero_invariants: None,
                    certified_planar_mask: 0,
                    thin_six_child_rank: six,
                    thin_seven_child_rank: seven,
                })
                .is_err()
            );
        }
    }
}
