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
        mut moments: Vec<[Atom; 4]>,
        aliases: Vec<(Atom, Atom)>,
        notes: Vec<String>,
    ) -> Result<Self, ReductionError> {
        if settings.propagators < 2
            || settings.powers.len() != settings.propagators
            || settings.powers.contains(&0)
            || settings.invariants != InvariantInput::FromShifts
            || settings.input_policy != InputPolicy::Preserve
        {
            return Err(ReductionError::InvalidInput(
                "affine-power parent requires preserved Cartesian positive powers".into(),
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
