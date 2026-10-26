//! Metadata constructor for a proof-selected constant-identity power DAG.
//! The owning frontstage supplies Symbolica expressions; no child integral
//! graph is inlined into this parent and no sampled proof enters its cache.
use super::*;

impl ReductionPlan {
    pub(crate) fn new_affine_power_parent(
        settings: ReductionSettings,
        analytic_jet_order: u16,
        parameters: Vec<Atom>,
        external_primal_seeds: Vec<(Atom, Atom, i32)>,
        layout: InputLayout,
        distances: Vec<Vec<Atom>>,
        moments: Vec<[Atom; 4]>,
        aliases: Vec<(Atom, Atom)>,
        notes: Vec<String>,
    ) -> Result<Self, ReductionError> {
        if settings.invariants != InvariantInput::FromShifts {
            return Err(ReductionError::InvalidInput(
                "affine-power Cartesian parent requires FromShifts".into(),
            ));
        }
        Self::build_affine_power_parent(
            settings,
            analytic_jet_order,
            parameters,
            external_primal_seeds,
            layout,
            distances,
            moments,
            aliases,
            notes,
        )
    }

    /// Structural imported recipe only. The checked owning workspace repeats
    /// full-Taylor p/G/rank/chart/cofactor certification before using this plan.
    pub(crate) fn new_checked_affine_power_parent(
        settings: ReductionSettings,
        analytic_jet_order: u16,
        parameters: Vec<Atom>,
        external_primal_seeds: Vec<(Atom, Atom, i32)>,
        layout: InputLayout,
        distances: Vec<Vec<Atom>>,
        moments: Vec<[Atom; 4]>,
        aliases: Vec<(Atom, Atom)>,
        notes: Vec<String>,
    ) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        if !(2..=6).contains(&n)
            || settings.rank > 6
            || settings.invariants != InvariantInput::SuppliedScalarProducts
            || layout.shifts != (0..4 * n)
            || layout.scalar_products != Some(4 * n..4 * n + n * (n + 1) / 2)
            || layout.masses != (4 * n + n * (n + 1) / 2..5 * n + n * (n + 1) / 2)
            || layout.scale != 5 * n + n * (n + 1) / 2
        {
            return Err(ReductionError::Unsupported("private checked affine-power parent requires complete preserved supplied-G N2..6/rank<=6 geometry".into()));
        }
        Self::build_affine_power_parent(
            settings,
            analytic_jet_order,
            parameters,
            external_primal_seeds,
            layout,
            distances,
            moments,
            aliases,
            notes,
        )
    }

    /// Imported low-point observables, with their original formal geometry.
    /// This identity performs no affine/pinching reduction and requires no
    /// equality between independent supplied products and Cartesian shifts.
    /// The selected source values/amplitudes are re-guarded at target dimensions.
    pub(crate) fn new_lowpoint_imported_tensor_parent(
        settings: ReductionSettings,
        analytic_jet_order: u16,
        parameters: Vec<Atom>,
        external_primal_seeds: Vec<(Atom, Atom, i32)>,
        layout: InputLayout,
        distances: Vec<Vec<Atom>>,
        moments: Vec<[Atom; 4]>,
        aliases: Vec<(Atom, Atom)>,
        notes: Vec<String>,
    ) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        if !(2..=5).contains(&n)
            || layout.shifts != (0..4 * n)
            || layout.masses.len() != n
            || layout.scale != layout.masses.end
            || layout.scale >= parameters.len()
            || distances.len() != n
            || distances.iter().any(|row| row.len() != n)
        {
            return Err(ReductionError::InvalidInput(
                "low-point imported identity requires complete original N2..5 geometry".into(),
            ));
        }
        let expected = match settings.invariants {
            InvariantInput::FromShifts => {
                if layout.scalar_products.is_some() || layout.masses != (4 * n..5 * n) {
                    return Err(ReductionError::InvalidInput(
                        "imported Cartesian identity layout differs".into(),
                    ));
                }
                let shifts: Vec<[Atom; 4]> = (0..n)
                    .map(|i| std::array::from_fn(|mu| parameters[4 * i + mu].clone()))
                    .collect();
                shifts
                    .iter()
                    .map(|p| {
                        shifts
                            .iter()
                            .map(|q| squared_difference(p, q))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            }
            InvariantInput::SuppliedScalarProducts => {
                let end = 4 * n + n * (n + 1) / 2;
                if layout.scalar_products != Some(4 * n..end) || layout.masses != (end..end + n) {
                    return Err(ReductionError::InvalidInput(
                        "imported supplied-product identity layout differs".into(),
                    ));
                }
                let mut products = vec![vec![Atom::new(); n]; n];
                let mut argument = 4 * n;
                for i in 0..n {
                    for j in i..n {
                        products[i][j] = parameters[argument].clone();
                        products[j][i] = products[i][j].clone();
                        argument += 1;
                    }
                }
                (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| {
                                &products[i][i] + &products[j][j] - Atom::num(2) * &products[i][j]
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            }
        };
        for i in 0..n {
            for j in 0..n {
                if !(&distances[i][j] - &expected[i][j])
                    .expand()
                    .together()
                    .is_zero()
                {
                    return Err(ReductionError::Preparation(
                        "imported identity distances do not retain their original formal geometry"
                            .into(),
                    ));
                }
            }
        }
        Self::build_affine_power_parent(
            settings,
            analytic_jet_order,
            parameters,
            external_primal_seeds,
            layout,
            distances,
            moments,
            aliases,
            notes,
        )
    }

    fn build_affine_power_parent(
        settings: ReductionSettings,
        analytic_jet_order: u16,
        parameters: Vec<Atom>,
        external_primal_seeds: Vec<(Atom, Atom, i32)>,
        layout: InputLayout,
        distances: Vec<Vec<Atom>>,
        mut moments: Vec<[Atom; 4]>,
        aliases: Vec<(Atom, Atom)>,
        notes: Vec<String>,
    ) -> Result<Self, ReductionError> {
        if settings.propagators < 2
            || settings.powers.len() != settings.propagators
            || settings.powers.contains(&0)
            || settings.input_policy != InputPolicy::Preserve
        {
            return Err(ReductionError::InvalidInput(
                "affine-power parent requires preserved positive powers".into(),
            ));
        }
        let basis = tensor_basis(settings.rank);
        if moments.len() != basis.len()
            || layout.scale >= parameters.len()
            || distances.len() != settings.propagators
            || distances
                .iter()
                .any(|row| row.len() != settings.propagators)
            || external_primal_seeds.is_empty()
        {
            return Err(ReductionError::Preparation(
                "affine-power parent has inconsistent imported expressions or layout".into(),
            ));
        }
        // Child outputs already have the requested normalization and tensor
        // weights at the SAME h. Only the existing final-output UV rule applies.
        apply_final_uv_power_counting(&settings, &basis, &mut moments)?;
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
            snapping_compatibility: Vec::new(),
            fallback: None,
            // This composite frontstage reports its actual child algorithms
            // separately. Do not advertise a pure IBP or pure PV backend here.
            algorithm_used: Algorithm::Hybrid,
            preparation_notes: notes,
            special_kinematics: true,
            analytic_jet_order,
            bubble_limit: None,
            zero_invariants: None,
            certified_invariant_chart: None,
            certified_planar_chart: None,
            certified_planar_mask: 0,
            certified_affine_chart: None,
            thin_six_child_rank: None,
            thin_seven_child_rank: None,
        })
    }
}
