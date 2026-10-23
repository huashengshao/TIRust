//! Exact topology preparation. No sampled values are cached in a plan.

use crate::{
    basis::{Monomial, tensor_basis},
    convention::convert_convention,
    ibp::{MasterLeaf, MasterLeafKind, ScalarCompiler, ScalarRequest},
    pv::PvCompiler,
    snapping::{LegacySnapping, SnapDiagnostic, exact_threshold},
    types::{
        Algorithm, InputLayout, InputPolicy, InvariantInput, ReductionError, ReductionSettings,
    },
};
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{Arc, Mutex, OnceLock},
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{atom::AtomField, integer::Integer},
};

#[derive(Clone, Debug)]
pub struct ReductionPlan {
    pub(crate) execution_cache: crate::execution::ExactExecutionCache,
    variant_cache: StructuralPlanCache,
    pub(crate) settings: ReductionSettings,
    pub(crate) basis: Vec<Monomial>,
    pub(crate) parameters: Vec<Atom>,
    pub(crate) layout: InputLayout,
    pub(crate) distances: Vec<Vec<Atom>>,
    /// Polynomial distances in original inputs, before the optional on-shell
    /// policy. These certify numerical zeros without undoing intentional snaps.
    pub(crate) raw_distances: Vec<Vec<Atom>>,
    pub(crate) moments: Vec<[Atom; 4]>,
    pub(crate) leaves: Vec<MasterLeaf>,
    pub(crate) master_arguments: Vec<Vec<Atom>>,
    pub(crate) aliases: Vec<(Atom, Atom)>,
    /// Invariant-only definitions required by argument and diagnostic lowering.
    pub(crate) snapping_aliases: Vec<(Atom, Atom)>,
    pub(crate) snapping_diagnostics: Vec<SnapDiagnostic>,
    /// The effective threshold, absent for preserved and supplied-G inputs.
    pub(crate) snapping_threshold: Option<Atom>,
    /// A zero-threshold negative reference can violate the zero-distance
    /// diagonal assumed by the recurrence. Reject only demanded matrix paths.
    pub(crate) snapping_zero_image: Option<Atom>,
    pub(crate) snapping_zero_image_required: bool,
    pub(crate) fallback: Option<Box<ReductionPlan>>,
    algorithm_used: Algorithm,
    preparation_notes: Vec<String>,
    pub(crate) special_kinematics: bool,
    pub(crate) analytic_jet_order: u16,
    bubble_limit: Option<(u16, bool)>,
    zero_invariants: Option<ZeroInvariantProfile>,
    /// A structural chart whose full-Taylor invariant constraints are checked
    /// by the private certified scalar entry point before numerical execution.
    certified_scalar_basis: Option<[usize; 4]>,
}

/// Exact constraints are structural: numerical values are never stored here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ZeroInvariantProfile {
    pub masses: Vec<bool>,
    pub distances: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PlanVariant {
    analytic_jet_order: u16,
    bubble_equal_masses: Option<bool>,
    zero_invariants: Option<ZeroInvariantProfile>,
}

type StructuralPlanCache =
    Arc<Mutex<HashMap<PlanVariant, Arc<OnceLock<Result<Arc<ReductionPlan>, ReductionError>>>>>>;

impl ReductionPlan {
    pub fn new(settings: ReductionSettings) -> Result<Self, ReductionError> {
        Self::build(settings, None, None, 0, None)
    }

    /// Only the certified scalar wrapper may prepare this chart. A plan holds
    /// no numerical certificate; that wrapper rechecks every input batch.
    pub(crate) fn new_certified_scalar(
        settings: ReductionSettings,
        basis: [usize; 4],
    ) -> Result<Self, ReductionError> {
        Self::build(settings, None, None, 0, Some(basis))
    }

    /// An analytic Taylor neighborhood selected only for exact lightlike bubbles.
    /// `jet_order` bounds the total derivative degree of the caller's Dual layout.
    pub(crate) fn lightlike_bubble(
        settings: ReductionSettings,
        jet_order: u16,
        equal_masses: bool,
    ) -> Result<Self, ReductionError> {
        if settings.propagators != 2 {
            return Err(ReductionError::InvalidInput(
                "lightlike bubble requires two propagators".into(),
            ));
        }
        Self::build(
            settings,
            Some((jet_order, equal_masses)),
            None,
            jet_order,
            None,
        )
    }

    pub(crate) fn with_analytic_jet_order(&self, order: u16) -> Result<Arc<Self>, ReductionError> {
        self.variant(PlanVariant {
            analytic_jet_order: order,
            bubble_equal_masses: self.bubble_limit.map(|(_, equal)| equal),
            zero_invariants: self.zero_invariants.clone(),
        })
    }

    /// Prepare an exact invariant profile. The runtime must verify each selected
    /// constraint is zero in every Taylor component before dispatching a lane.
    pub(crate) fn with_zero_invariants(
        &self,
        mut profile: ZeroInvariantProfile,
    ) -> Result<Arc<Self>, ReductionError> {
        let n = self.settings.propagators;
        if profile.masses.len() != n
            || profile
                .distances
                .iter()
                .any(|&(i, j)| i >= n || j >= n || i == j)
        {
            return Err(ReductionError::InvalidInput(
                "invalid zero-invariant profile".into(),
            ));
        }
        for pair in &mut profile.distances {
            if pair.0 > pair.1 {
                *pair = (pair.1, pair.0);
            }
        }
        profile.distances.sort_unstable();
        profile.distances.dedup();
        self.variant(PlanVariant {
            analytic_jet_order: self.analytic_jet_order,
            bubble_equal_masses: None,
            zero_invariants: Some(profile),
        })
    }

    fn variant(&self, variant: PlanVariant) -> Result<Arc<Self>, ReductionError> {
        if variant.analytic_jet_order == self.analytic_jet_order
            && variant.bubble_equal_masses == self.bubble_limit.map(|(_, equal)| equal)
            && variant.zero_invariants == self.zero_invariants
        {
            // Do not insert self into its own cache: even an immutable clone
            // would retain this cache and create an ownership cycle.
            return Ok(Arc::new(self.clone()));
        }
        let entry = self
            .variant_cache
            .lock()
            .map_err(|_| ReductionError::Preparation("structural plan cache lock poisoned".into()))?
            .entry(variant.clone())
            .or_default()
            .clone();
        // No cache lock is held while preparing fallback/profile recurrences.
        // Every newly built child receives fresh caches, with no parent link.
        entry
            .get_or_init(|| {
                Self::build(
                    self.settings.clone(),
                    variant
                        .bubble_equal_masses
                        .map(|equal| (variant.analytic_jet_order, equal)),
                    variant.zero_invariants,
                    variant.analytic_jet_order,
                    self.certified_scalar_basis,
                )
                .map(Arc::new)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| match error {
                ReductionError::InvalidInput(reason) => {
                    ReductionError::InvalidInput(reason.clone())
                }
                ReductionError::Preparation(reason) => ReductionError::Preparation(reason.clone()),
                ReductionError::Unsupported(reason) => ReductionError::Unsupported(reason.clone()),
                ReductionError::Evaluation(reason) => ReductionError::Evaluation(reason.clone()),
            })
    }

    fn build(
        settings: ReductionSettings,
        bubble_limit: Option<(u16, bool)>,
        zero_invariants: Option<ZeroInvariantProfile>,
        analytic_jet_order: u16,
        certified_scalar_basis: Option<[usize; 4]>,
    ) -> Result<Self, ReductionError> {
        let n = settings.propagators;
        if n == 0 || settings.powers.len() != n || settings.powers.iter().any(|&p| p == 0) {
            return Err(ReductionError::InvalidInput(
                "provide a nonempty topology and one positive power per propagator".into(),
            ));
        }
        if settings.rank > 32 || n > 16 {
            return Err(ReductionError::Unsupported(
                "preparation budget exceeds 32 tensor indices or 16 propagators".into(),
            ));
        }
        // High-point pinching identities require a four-dimensional external
        // Gram matrix. Arbitrary supplied invariants, including Dual directions
        // transverse to that variety, do not satisfy those identities. Until
        // every Taylor component is checked, do not accept this input domain.
        if let Some(chart) = certified_scalar_basis {
            if !(6..=7).contains(&n)
                || settings.rank != 0
                || settings.algorithm != Algorithm::Ibp
                || settings.invariants != InvariantInput::SuppliedScalarProducts
                || chart.iter().any(|&index| index == 0 || index >= n)
                || chart
                    .iter()
                    .enumerate()
                    .any(|(i, index)| chart[..i].contains(index))
            {
                return Err(ReductionError::InvalidInput(
                    "certified scalar charts require six or seven propagators, rank-zero IBP, supplied scalar products, and four distinct nonzero propagator indices".into(),
                ));
            }
        }
        if settings.invariants == InvariantInput::SuppliedScalarProducts
            && n >= 6
            && certified_scalar_basis.is_none()
        {
            return Err(ReductionError::Unsupported(
                "supplied scalar products with six or more propagators require validated four-dimensional Gram constraints in every Taylor component; coordinate inputs remain supported".into(),
            ));
        }
        let requested_threshold = match &settings.input_policy {
            InputPolicy::Preserve => None,
            InputPolicy::LegacyOnShell { threshold } => Some(exact_threshold(threshold)?),
        };
        let snapping_threshold = if settings.invariants == InvariantInput::FromShifts {
            requested_threshold
        } else {
            None
        };
        if snapping_threshold.is_some() && n >= 5 {
            return Err(ReductionError::Unsupported(
                "legacy on-shell snapping for five or more propagators awaits independent validation of coordinate identities with non-bilinear snapped invariants".into(),
            ));
        }
        let shifts: Vec<[Atom; 4]> = (0..n)
            .map(|i| std::array::from_fn(|mu| variable(&format!("p{i}_{mu}"))))
            .collect();
        let masses: Vec<_> = (0..n).map(|i| variable(&format!("m2_{i}"))).collect();
        let snapping = match (&settings.input_policy, settings.invariants) {
            (InputPolicy::LegacyOnShell { threshold }, InvariantInput::FromShifts) => Some(
                Rc::new(RefCell::new(LegacySnapping::new(&masses, threshold)?)),
            ),
            _ => None,
        };
        let mu = variable("mu");
        let mut parameters: Vec<Atom> = shifts.iter().flatten().cloned().collect();
        let mut distances = vec![vec![Atom::new(); n]; n];
        let mut raw_distances = distances.clone();
        let scalar_products = match settings.invariants {
            InvariantInput::FromShifts => {
                for i in 0..n {
                    for j in i + 1..n {
                        let raw = squared_difference(&shifts[i], &shifts[j]);
                        raw_distances[i][j] = raw.clone();
                        raw_distances[j][i] = raw.clone();
                        distances[i][j] = if let Some(snapping) = &snapping {
                            snapping
                                .borrow_mut()
                                .snap_scalar_product(raw, format!("(p{i}-p{j})²"))
                        } else {
                            raw
                        };
                        distances[j][i] = distances[i][j].clone();
                    }
                }
                None
            }
            InvariantInput::SuppliedScalarProducts => {
                let start = parameters.len();
                let mut products = vec![vec![Atom::new(); n]; n];
                for i in 0..n {
                    for j in i..n {
                        let value = variable(&format!("g{i}_{j}"));
                        products[i][j] = value.clone();
                        products[j][i] = value.clone();
                        parameters.push(value);
                    }
                }
                for i in 0..n {
                    for j in i + 1..n {
                        distances[i][j] =
                            &products[i][i] + &products[j][j] - Atom::num(2) * &products[i][j];
                        distances[j][i] = distances[i][j].clone();
                        raw_distances[i][j] = distances[i][j].clone();
                        raw_distances[j][i] = distances[i][j].clone();
                    }
                }
                Some(start..parameters.len())
            }
        };
        let mass_start = parameters.len();
        parameters.extend(masses.iter().cloned());
        let scale = parameters.len();
        parameters.push(mu.clone());
        let layout = InputLayout {
            shifts: 0..4 * n,
            scalar_products,
            masses: mass_start..scale,
            scale,
        };
        let basis = tensor_basis(settings.rank);
        let ordinary = settings.dimension_shift == 0 && settings.powers.iter().all(|&p| p == 1);
        let special_kinematics = bubble_limit.is_some() || zero_invariants.is_some();
        let algorithm_used = if !ordinary || special_kinematics {
            Algorithm::Ibp
        } else {
            settings.algorithm
        };
        if !ordinary && settings.algorithm == Algorithm::PassarinoVeltman {
            return Err(ReductionError::Unsupported("PV currently requires unit propagator powers and zero dimension shift; use IBP or hybrid".into()));
        }
        let mut preparation_notes = Vec::new();
        if matches!(settings.input_policy, InputPolicy::LegacyOnShell { .. }) {
            preparation_notes.push(if snapping.is_some() {
                "Legacy on-shell policy applies to individual scalar products; input momenta and masses are preserved".into()
            } else {
                "Supplied scalar products bypass the legacy on-shell policy, matching the original supplied-invariant interface".into()
            });
        }
        let (mut moments, leaves, mut aliases, algorithm_used) = match algorithm_used {
            Algorithm::Ibp => {
                match compile_ibp(
                    &settings,
                    &basis,
                    &shifts,
                    &masses,
                    &distances,
                    bubble_limit,
                    zero_invariants.as_ref(),
                    certified_scalar_basis,
                ) {
                    Ok((moments, leaves, aliases)) => (moments, leaves, aliases, Algorithm::Ibp),
                    Err(error)
                        if ordinary
                            && n >= 5
                            && !special_kinematics
                            && certified_scalar_basis.is_none() =>
                    {
                        // Original IBP also completes its four-dimensional
                        // high-point tensor system with PV at zero dimension
                        // coefficients. This does not replace an independent
                        // higher-dimensional scalar integral by box masters.
                        preparation_notes.push(format!("IBP tensor completion uses PV: {error}"));
                        let (moments, leaves, aliases) = compile_pv(
                            &basis,
                            &shifts,
                            &masses,
                            &distances,
                            analytic_jet_order,
                            snapping.clone(),
                        )?;
                        (moments, leaves, aliases, Algorithm::PassarinoVeltman)
                    }
                    Err(error) => return Err(error),
                }
            }
            Algorithm::PassarinoVeltman | Algorithm::Hybrid => {
                let pv = compile_pv(
                    &basis,
                    &shifts,
                    &masses,
                    &distances,
                    analytic_jet_order,
                    snapping.clone(),
                );
                match pv {
                    Ok((moments, leaves, aliases)) => {
                        (moments, leaves, aliases, Algorithm::PassarinoVeltman)
                    }
                    Err(error) if algorithm_used == Algorithm::Hybrid => {
                        preparation_notes
                            .push(format!("PV preparation failed; using IBP: {error}"));
                        let (moments, leaves, aliases) = compile_ibp(
                            &settings,
                            &basis,
                            &shifts,
                            &masses,
                            &distances,
                            bubble_limit,
                            zero_invariants.as_ref(),
                            certified_scalar_basis,
                        )?;
                        (moments, leaves, aliases, Algorithm::Ibp)
                    }
                    Err(error) => return Err(error),
                }
            }
        };
        let (snapping_aliases, snapping_diagnostics, snapping_zero_image) =
            snapping.as_ref().map_or_else(
                || (Vec::new(), Vec::new(), None),
                |snapping| {
                    let mut snapping = snapping.borrow_mut();
                    // The compatibility-only query must not create independent
                    // boundary diagnostics or make an unused policy active.
                    let diagnostics = snapping.diagnostics().to_vec();
                    let zero_image = snapping_threshold
                        .as_ref()
                        .filter(|value| value.is_zero())
                        .map(|_| {
                            snapping.snap_scalar_product(Atom::Zero, "zero-distance compatibility")
                        });
                    (snapping.aliases().to_vec(), diagnostics, zero_image)
                },
            );
        let snapping_zero_image_required = snapping_zero_image.is_some()
            && n == 1
            && algorithm_used == Algorithm::Ibp
            && (settings.rank > 0 || !ordinary);
        if !snapping_aliases.is_empty() {
            let mut ordered = snapping_aliases.clone();
            ordered.append(&mut aliases);
            aliases = ordered;
        }
        let base_uv_degree = 4 + 2 * i32::from(settings.dimension_shift)
            - 2 * settings
                .powers
                .iter()
                .map(|&power| i32::from(power))
                .sum::<i32>();
        for (moment, monomial) in moments.iter_mut().zip(&basis) {
            // Only the requested output is UV convergent. Intermediate UV
            // poles remain available to the recurrence's finite rational terms.
            let rank = i32::try_from(monomial.rank()).map_err(|_| {
                ReductionError::Unsupported("tensor rank exceeds UV power-counting range".into())
            })?;
            if base_uv_degree + rank < 0 {
                moment[0] = Atom::Zero;
            }
            *moment = convert_convention(moment, *monomial, &settings);
        }
        let master_arguments = leaves
            .iter()
            .map(|leaf| {
                let active = &leaf.active;
                let mut args = Vec::new();
                let mut distance =
                    |i: usize, j: usize| args.push(distances[active[i]][active[j]].clone());
                match (leaf.kind, active.len()) {
                    (MasterLeafKind::FourDimensional, 1) => {}
                    (MasterLeafKind::FourDimensional, 2) => distance(0, 1),
                    (MasterLeafKind::FourDimensional, 3) => {
                        distance(0, 1);
                        distance(1, 2);
                        distance(0, 2);
                    }
                    (MasterLeafKind::FourDimensional, 4) => {
                        distance(0, 1);
                        distance(1, 2);
                        distance(2, 3);
                        distance(0, 3);
                        distance(0, 2);
                        distance(1, 3);
                    }
                    (MasterLeafKind::SixDimensionalPentagon, 5) => {
                        for i in 0..5 {
                            for j in i + 1..5 {
                                distance(i, j);
                            }
                        }
                    }
                    _ => unreachable!("scalar compiler emitted an inconsistent master kind"),
                }
                args.extend(active.iter().map(|&i| masses[i].clone()));
                args.push(mu.pow(2));
                args
            })
            .collect();
        let fallback = if !special_kinematics
            && settings.algorithm == Algorithm::Hybrid
            && algorithm_used == Algorithm::PassarinoVeltman
        {
            let mut fallback_settings = settings.clone();
            fallback_settings.algorithm = Algorithm::Ibp;
            match Self::build(
                fallback_settings,
                None,
                None,
                analytic_jet_order,
                certified_scalar_basis,
            ) {
                Ok(plan) if plan.algorithm_used == Algorithm::Ibp => Some(Box::new(plan)),
                Ok(_) => {
                    preparation_notes.push("Independent IBP fallback is unavailable for this topology; its tensor completion also uses PV".into());
                    None
                }
                Err(error) => {
                    preparation_notes.push(format!("IBP fallback preparation failed: {error}"));
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            execution_cache: Default::default(),
            variant_cache: Default::default(),
            settings,
            basis,
            parameters,
            layout,
            distances,
            raw_distances,
            moments,
            leaves,
            master_arguments,
            aliases,
            snapping_aliases,
            snapping_diagnostics,
            snapping_threshold,
            snapping_zero_image,
            snapping_zero_image_required,
            fallback,
            algorithm_used,
            preparation_notes,
            special_kinematics,
            analytic_jet_order,
            bubble_limit,
            zero_invariants,
            certified_scalar_basis,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    pub fn input_layout(&self) -> &InputLayout {
        &self.layout
    }
    pub fn settings(&self) -> &ReductionSettings {
        &self.settings
    }
    pub fn basis(&self) -> &[Monomial] {
        &self.basis
    }
    pub fn master_count(&self) -> usize {
        self.leaves.len()
    }
    pub fn algorithm_used(&self) -> Algorithm {
        self.algorithm_used
    }
    /// An IBP invariant profile cannot replace the distinct direct-dot Gram
    /// matrix selected by legacy PV snapping.
    pub(crate) fn zero_profile_dispatch_allowed(&self) -> bool {
        !self.special_kinematics
            // Rounded invariant evaluations cannot establish new exact-zero
            // constraints beyond the certified full-Taylor Gram relations.
            && self.certified_scalar_basis.is_none()
            && !(self.snapping_threshold.is_some()
                && self.algorithm_used == Algorithm::PassarinoVeltman)
    }
    pub fn preparation_notes(&self) -> &[String] {
        &self.preparation_notes
    }
    /// Shared Symbolica recurrence nodes referenced by `expressions()`.
    pub fn aliases(&self) -> &[(Atom, Atom)] {
        &self.aliases
    }
    pub fn expressions(&self) -> &[[Atom; 4]] {
        &self.moments
    }

    /// Locally guarded comparison probes for the effective legacy policy.
    /// Runtime evaluation must also respect each probe owner's branch demand.
    pub fn snapping_diagnostics(&self) -> &[SnapDiagnostic] {
        &self.snapping_diagnostics
    }

    /// Exact threshold expression, absent when the policy has no effect.
    pub fn snapping_threshold(&self) -> Option<&Atom> {
        self.snapping_threshold.as_ref()
    }
}

fn compile_pv(
    basis: &[Monomial],
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    distances: &[Vec<Atom>],
    analytic_jet_order: u16,
    snapping: Option<Rc<RefCell<LegacySnapping>>>,
) -> Result<(Vec<[Atom; 4]>, Vec<MasterLeaf>, Vec<(Atom, Atom)>), ReductionError> {
    let mut compiler =
        PvCompiler::with_distances(shifts.to_vec(), masses.to_vec(), distances.to_vec(), true)
            .map_err(ReductionError::Preparation)?;
    compiler
        .set_analytic_jet_order(analytic_jet_order)
        .map_err(ReductionError::Preparation)?;
    if let Some(snapping) = snapping {
        compiler
            .set_gram_scalar_product_transform(move |raw| {
                snapping
                    .borrow_mut()
                    .snap_scalar_product(raw, "PV Gram scalar product")
            })
            .map_err(ReductionError::Preparation)?;
    }
    let moments = basis
        .iter()
        .map(|m| compiler.compile_moment(m.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ReductionError::Preparation)?;
    Ok((
        moments,
        compiler.leaves().to_vec(),
        compiler.aliases().to_vec(),
    ))
}

fn compile_ibp(
    settings: &ReductionSettings,
    basis: &[Monomial],
    shifts: &[[Atom; 4]],
    masses: &[Atom],
    supplied_distances: &[Vec<Atom>],
    bubble_limit: Option<(u16, bool)>,
    zero_invariants: Option<&ZeroInvariantProfile>,
    certified_basis: Option<[usize; 4]>,
) -> Result<(Vec<[Atom; 4]>, Vec<MasterLeaf>, Vec<(Atom, Atom)>), ReductionError> {
    // Independent squared-distance symbols keep exact matrix elimination small.
    // Symbolica shares their definitions across the recurrence graph.
    let n = shifts.len();
    let mut aliases = Vec::new();
    let mut distances = vec![vec![Atom::new(); n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let s = variable(&format!("s{i}_{j}"));
            distances[i][j] = s.clone();
            distances[j][i] = s.clone();
            aliases.push((s, supplied_distances[i][j].clone()));
        }
    }
    let r = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| &distances[i][j] - &masses[i] - &masses[j])
                .collect()
        })
        .collect();
    let mut compiler = match bubble_limit {
        Some((order, equal_masses)) => ScalarCompiler::lightlike_bubble(r, order, equal_masses),
        None => match zero_invariants {
            Some(profile) => {
                ScalarCompiler::with_zero_invariants(r, &profile.masses, &profile.distances)
            }
            None => ScalarCompiler::new(r),
        },
    }
    .map_err(ReductionError::Preparation)?;
    if let Some(basis) = certified_basis {
        // The opt-in scalar wrapper certifies that these four Gram covectors
        // preserve every external linear dependency in every Taylor component.
        // They are coordinates on the span, not reconstructed Cartesian shifts;
        // original squared distances and masses remain authoritative in R.
        let coordinates =
            certified_covector_aliases(basis, &distances, supplied_distances, &mut aliases);
        compiler
            .set_certified_linear_coordinates(coordinates)
            .map_err(ReductionError::Preparation)?;
    } else if settings.invariants == InvariantInput::FromShifts
        && settings.input_policy == InputPolicy::Preserve
    {
        compiler
            .set_routing(shifts.to_vec())
            .map_err(ReductionError::Preparation)?;
    }
    compiler
        .enable_aliases()
        .map_err(ReductionError::Preparation)?;
    compiler
        .enable_physical_poles()
        .map_err(ReductionError::Preparation)?;
    compiler
        .enable_gram_regular_bubbles()
        .map_err(ReductionError::Preparation)?;
    if settings.dimension_shift != 0 || settings.powers.iter().any(|&power| power != 1) {
        // Ordinary four-dimensional high-point tensors retain the existing
        // PV completion. Genuine generalized requests require the independent
        // shifted master; its backend enforces a finite parameter domain.
        compiler
            .enable_six_dimensional_pentagon()
            .map_err(ReductionError::Preparation)?;
    }
    let t: [Atom; 4] = std::array::from_fn(|i| variable(&format!("aux{i}")));
    let metric = t[0].pow(2) - t[1].pow(2) - t[2].pow(2) - t[3].pow(2);
    let projections: Vec<Atom> = shifts
        .iter()
        .map(|p| t.iter().zip(p).map(|(a, b)| a * b).sum())
        .collect();
    let mut by_exponent: HashMap<[u16; 4], [Atom; 4]> = HashMap::new();
    for rank in 0..=settings.rank {
        let mut generating: [Atom; 4] = std::array::from_fn(|_| Atom::new());
        for pairs in 0..=rank / 2 {
            let scalar_h = settings
                .dimension_shift
                .checked_add((rank - pairs) as i16)
                .ok_or_else(|| ReductionError::InvalidInput("dimension shift overflow".into()))?;
            let front = Atom::num(Integer::factorial(u32::from(rank)))
                / Atom::num(Integer::factorial(u32::from(pairs)))
                * Atom::num(-4).pow(-i64::from(pairs))
                * metric.pow(i64::from(pairs));
            for counts in compositions(rank - 2 * pairs, n) {
                let mut powers = settings.powers.clone();
                let mut coefficient = front.clone();
                for i in 0..n {
                    powers[i] = powers[i].checked_add(counts[i]).ok_or_else(|| {
                        ReductionError::InvalidInput("propagator power overflow".into())
                    })?;
                    if counts[i] != 0 {
                        coefficient = coefficient * projections[i].pow(i64::from(counts[i]));
                    }
                    coefficient = coefficient
                        * Atom::num(Integer::binom(
                            i64::from(settings.powers[i]) + i64::from(counts[i]) - 1,
                            i64::from(counts[i]),
                        ));
                }
                let value = compiler
                    .compile(ScalarRequest {
                        powers,
                        dimension_shift: scalar_h,
                    })
                    .map_err(ReductionError::Preparation)?;
                for pole in 0..4 {
                    generating[pole] = &generating[pole] + &coefficient * &value[pole];
                }
            }
        }
        // Symbolica performs the polynomial expansion and coefficient collection.
        for (pole, expression) in generating.into_iter().enumerate() {
            let field = AtomField {
                statistical_zero_test: false,
                cancel_check_on_division: true,
                custom_normalization: None,
            };
            let polynomial = expression
                .expand()
                .to_polynomial_in_vars_with_field::<u16>(&t, &field);
            for term in &polynomial {
                let exponents: [u16; 4] = term.exponents.try_into().map_err(|_| {
                    ReductionError::Preparation("unexpected generating polynomial basis".into())
                })?;
                let entry = by_exponent
                    .entry(exponents)
                    .or_insert_with(|| std::array::from_fn(|_| Atom::new()));
                let weight = multinomial(exponents);
                entry[pole] = term.coefficient / weight;
            }
        }
    }
    let moments = basis
        .iter()
        .map(|m| {
            by_exponent
                .remove(&m.0)
                .unwrap_or_else(|| std::array::from_fn(|_| Atom::new()))
        })
        .collect();
    aliases.extend_from_slice(compiler.aliases());
    Ok((moments, compiler.leaves, aliases))
}

/// Share the certified Gram covectors without expanding their input polynomials
/// inside every exact matrix operation. Symmetric entries share one handle.
/// These are coordinates on the certified span; physical R remains unchanged.
fn certified_covector_aliases(
    basis: [usize; 4],
    distances: &[Vec<Atom>],
    supplied_distances: &[Vec<Atom>],
    aliases: &mut Vec<(Atom, Atom)>,
) -> Vec<[Atom; 4]> {
    let mut entries = HashMap::new();
    (0..distances.len())
        .map(|i| {
            std::array::from_fn(|axis| {
                let anchor = basis[axis];
                if i == 0 {
                    return Atom::new();
                }
                if i == anchor {
                    return distances[0][i].clone();
                }
                let (first, second) = (anchor.min(i), anchor.max(i));
                entries
                    .entry((first, second))
                    .or_insert_with(|| {
                        let handle = variable(&format!("cert_k{first}_{second}"));
                        // Expand once in the original G inputs: this cancels
                        // repeated diagonal terms before numerical evaluation.
                        // Combining evaluated squared-distance aliases instead
                        // could introduce avoidable floating-point cancellation.
                        let body = ((&supplied_distances[0][anchor] + &supplied_distances[0][i]
                            - &supplied_distances[anchor][i])
                            / Atom::num(2))
                        .expand();
                        aliases.push((handle.clone(), body));
                        handle
                    })
                    .clone()
            })
        })
        .collect()
}

fn multinomial(exponents: [u16; 4]) -> Atom {
    let rank: u32 = exponents.iter().map(|&e| u32::from(e)).sum();
    exponents
        .into_iter()
        .fold(Atom::num(Integer::factorial(rank)), |value, e| {
            value / Atom::num(Integer::factorial(u32::from(e)))
        })
}

fn variable(name: &str) -> Atom {
    Atom::var(symbolica::symbol!(format!("iregi::{name}")))
}
fn squared_difference(a: &[Atom; 4], b: &[Atom; 4]) -> Atom {
    (&a[0] - &b[0]).pow(2)
        - (&a[1] - &b[1]).pow(2)
        - (&a[2] - &b[2]).pow(2)
        - (&a[3] - &b[3]).pow(2)
}

// Integer index tuples describing tensor structures, not polynomial arithmetic.
fn compositions(total: u16, count: usize) -> Vec<Vec<u16>> {
    if count == 1 {
        return vec![vec![total]];
    }
    let mut result = Vec::new();
    for first in 0..=total {
        for mut tail in compositions(total - first, count - 1) {
            tail.insert(0, first);
            result.push(tail);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Convention, Workspace};
    use iregi_masters::Float;
    use symbolica::domains::{dual::HyperDual, float::Complex, rational::Rational};

    fn settings(n: usize, rank: u16) -> ReductionSettings {
        let mut settings = ReductionSettings::new(n, rank);
        settings.algorithm = Algorithm::Ibp;
        settings.convention = Convention::OneLoop;
        settings
    }

    #[test]
    fn certified_covector_aliases_preserve_exact_inputs_and_mixed_dual_components() {
        use symbolica::{
            domains::float::SingleFloat,
            evaluate::{Dualizer, ExpressionEvaluator},
        };

        for (n, basis) in [(6, [1, 2, 3, 4]), (6, [2, 3, 4, 5]), (7, [2, 4, 5, 6])] {
            let mut gram = vec![vec![Atom::new(); n]; n];
            let mut parameters = Vec::new();
            for i in 0..n {
                for j in i..n {
                    let value = variable(&format!("covector_test_g{i}_{j}"));
                    gram[i][j] = value.clone();
                    gram[j][i] = value.clone();
                    parameters.push(value);
                }
            }
            let mut distances = vec![vec![Atom::new(); n]; n];
            let mut raw = distances.clone();
            let mut aliases = Vec::new();
            for i in 0..n {
                for j in i + 1..n {
                    let value = variable(&format!("covector_test_s{i}_{j}"));
                    let body = &gram[i][i] + &gram[j][j] - Atom::num(2) * &gram[i][j];
                    distances[i][j] = value.clone();
                    distances[j][i] = value.clone();
                    raw[i][j] = body.clone();
                    raw[j][i] = body.clone();
                    aliases.push((value, body));
                }
            }
            let coordinates = certified_covector_aliases(basis, &distances, &raw, &mut aliases);
            assert!(coordinates[0].iter().all(AtomCore::is_zero));
            for (axis, &anchor) in basis.iter().enumerate() {
                assert_eq!(coordinates[anchor][axis], distances[0][anchor]);
                for (other_axis, &other) in basis.iter().enumerate() {
                    assert_eq!(coordinates[other][axis], coordinates[anchor][other_axis]);
                }
            }
            let mut expressions = Vec::new();
            for (i, coordinate) in coordinates.iter().enumerate() {
                for (axis, &anchor) in basis.iter().enumerate() {
                    let expected = &gram[anchor][i] - &gram[0][anchor] - &gram[0][i] + &gram[0][0];
                    let expanded = aliases
                        .iter()
                        .rev()
                        .fold(coordinate[axis].clone(), |value, (handle, body)| {
                            value.replace(handle.clone()).with(body.clone())
                        });
                    assert!((expanded - &expected).together().is_zero());
                    expressions.extend([coordinate[axis].clone(), expected]);
                }
            }
            // This exercises alias propagation through Symbolica's Dualizer,
            // including reordered, mixed and second-order Taylor components.
            let shape = vec![
                vec![0, 0],
                vec![0, 1],
                vec![1, 0],
                vec![1, 1],
                vec![0, 2],
                vec![2, 0],
            ];
            let exact: ExpressionEvaluator<Complex<Rational>> =
                Atom::evaluator_multiple(&expressions, &parameters)
                    .add_aliases(aliases)
                    .unwrap()
                    .direct_translation(true)
                    .horner_iterations(0)
                    .cores(1)
                    .build()
                    .unwrap();
            let prototype = Float::new(256);
            let convert = |value: &Complex<Rational>| {
                Complex::new(
                    prototype.from_rational(&value.re),
                    prototype.from_rational(&value.im),
                )
            };
            let mut evaluator = exact
                .vectorize(&Dualizer::new(
                    HyperDual::<Complex<Rational>>::new(shape.clone()),
                    Vec::new(),
                ))
                .unwrap()
                .map_coeff_with_prec(&convert, 256);
            let inputs = (0..parameters.len() * shape.len())
                .map(|i| {
                    let argument = (i / shape.len()) as i64;
                    let component = (i % shape.len()) as i64;
                    Complex::new(
                        Rational::from(
                            (argument + 3).pow(2) * (component + 1).pow(3)
                                + (argument + 1) * (component + 5),
                        ),
                        Rational::from(
                            (argument + 2).pow(3) * (component + 2).pow(2)
                                + (argument + 7) * (component + 3),
                        ),
                    )
                })
                .map(|value| convert(&value))
                .collect::<Vec<_>>();
            // These integer jets and the linear half-integer coefficients are
            // represented exactly at 256 bits; equality needs no tolerance.
            let mut outputs = vec![
                convert(&Complex::new(Rational::from(0), Rational::from(0)));
                expressions.len() * shape.len()
            ];
            evaluator.evaluate(&inputs, &mut outputs);
            let mut distinct_mixed_components = false;
            for pair in outputs.chunks_exact(2 * shape.len()) {
                assert_eq!(&pair[..shape.len()], &pair[shape.len()..]);
                distinct_mixed_components |= pair[3] != pair[0] && pair[4] != pair[5];
            }
            assert!(distinct_mixed_components);
        }
    }

    #[test]
    fn certified_scalar_chart_rejects_unsupported_domains_and_invalid_bases() {
        let mut configuration = settings(6, 0);
        configuration.invariants = InvariantInput::SuppliedScalarProducts;
        assert!(matches!(
            ReductionPlan::new(configuration.clone()),
            Err(ReductionError::Unsupported(_))
        ));
        for chart in [[0, 1, 2, 3], [1, 1, 3, 4], [1, 2, 3, 6]] {
            assert!(matches!(
                ReductionPlan::new_certified_scalar(configuration.clone(), chart),
                Err(ReductionError::InvalidInput(_))
            ));
        }
        let mut non_scalar = configuration.clone();
        non_scalar.rank = 1;
        let mut coordinate_input = configuration.clone();
        coordinate_input.invariants = InvariantInput::FromShifts;
        let mut pv = configuration.clone();
        pv.algorithm = Algorithm::PassarinoVeltman;
        let mut hybrid = configuration;
        hybrid.algorithm = Algorithm::Hybrid;
        let mut five = settings(5, 0);
        five.invariants = InvariantInput::SuppliedScalarProducts;
        let mut eight = settings(8, 0);
        eight.invariants = InvariantInput::SuppliedScalarProducts;
        for invalid in [non_scalar, coordinate_input, pv, hybrid, five, eight] {
            assert!(matches!(
                ReductionPlan::new_certified_scalar(invalid, [1, 2, 3, 4]),
                Err(ReductionError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn nontrivial_dual_specializations_share_exact_work_across_precisions() {
        let plan = ReductionPlan::new(settings(1, 0)).unwrap();
        let shape = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
        let binary = Workspace::new_dual(&plan, 0., shape.clone()).unwrap();
        let specialized = plan.with_analytic_jet_order(1).unwrap();
        let first = specialized
            .execution_cache
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .get()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        let precise = Workspace::new_dual(&plan, Float::new(256), shape).unwrap();
        let again = plan.with_analytic_jet_order(1).unwrap();
        let second = again
            .execution_cache
            .lock()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .get()
            .unwrap()
            .as_ref()
            .unwrap()
            .clone();
        assert!(Arc::ptr_eq(&specialized, &again));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(binary.binary_precision(), 53);
        assert_eq!(precise.binary_precision(), 256);

        // Same total degree, different Taylor-component order: the tensor
        // plan is shared, while each exact flattened program remains distinct.
        let first_shape = vec![vec![0, 0], vec![1, 0], vec![0, 1]];
        let second_shape = vec![vec![0, 0], vec![0, 1], vec![1, 0]];
        let first = Workspace::new_dual(
            &plan,
            0.,
            HyperDual::<Complex<Rational>>::new(first_shape.clone()),
        )
        .unwrap();
        let second = Workspace::new_dual(
            &plan,
            0.,
            HyperDual::<Complex<Rational>>::new(second_shape.clone()),
        )
        .unwrap();
        assert_eq!(first.dual_shape().unwrap(), first_shape);
        assert_eq!(second.dual_shape().unwrap(), second_shape);
        let cache = specialized.execution_cache.lock().unwrap();
        assert_eq!(cache.len(), 3);
        let first = cache[&Some(first_shape)].get().unwrap().as_ref().unwrap();
        let second = cache[&Some(second_shape)].get().unwrap().as_ref().unwrap();
        assert!(!Arc::ptr_eq(first, second));
    }

    #[test]
    fn canonical_profiles_are_shared_and_variant_caches_do_not_form_cycles() {
        let weak = {
            let plan = ReductionPlan::new(settings(2, 0)).unwrap();
            let first = plan
                .with_zero_invariants(ZeroInvariantProfile {
                    masses: vec![true, true],
                    distances: vec![(1, 0), (0, 1)],
                })
                .unwrap();
            let second = plan
                .with_zero_invariants(ZeroInvariantProfile {
                    masses: vec![true, true],
                    distances: vec![(0, 1)],
                })
                .unwrap();
            assert!(Arc::ptr_eq(&first, &second));
            let _binary = Workspace::new(&first, 0.).unwrap();
            let _precise = Workspace::new(&second, Float::new(256)).unwrap();
            assert_eq!(first.execution_cache.lock().unwrap().len(), 1);
            Arc::downgrade(&first)
        };
        assert!(weak.upgrade().is_none());
    }

    fn assert_concurrent_variant_lookup(plan: Arc<ReductionPlan>) {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let plan = plan.clone();
                std::thread::spawn(move || plan.with_analytic_jet_order(2).unwrap())
            })
            .collect();
        let variants: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert!(
            variants
                .iter()
                .all(|variant| Arc::ptr_eq(variant, &variants[0]))
        );
        let weak = Arc::downgrade(&variants[0]);
        drop(variants);
        drop(plan);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn warm_specializations_support_concurrent_cache_lookup() {
        let plan = Arc::new(ReductionPlan::new(settings(1, 0)).unwrap());
        // Initialize Symbolica's exact graph on the current licensed thread.
        // Worker threads receive only the completed Rust cache entry, so they
        // cannot enter any Symbolica construction or Atom-cloning code.
        let variant = plan.with_analytic_jet_order(2).unwrap();
        let entry = plan
            .variant_cache
            .lock()
            .unwrap()
            .get(&PlanVariant {
                analytic_jet_order: 2,
                bubble_equal_masses: None,
                zero_invariants: None,
            })
            .unwrap()
            .clone();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let entry = entry.clone();
                std::thread::spawn(move || entry.get().unwrap().as_ref().unwrap().clone())
            })
            .collect();
        for handle in handles {
            assert!(Arc::ptr_eq(&handle.join().unwrap(), &variant));
        }
        let weak = Arc::downgrade(&variant);
        drop(variant);
        drop(entry);
        drop(plan);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    #[ignore = "cold preparation on worker threads requires a multithread-enabled Symbolica license"]
    fn cold_specializations_are_initialized_once_across_threads() {
        assert_concurrent_variant_lookup(Arc::new(ReductionPlan::new(settings(1, 0)).unwrap()));
    }

    #[test]
    fn final_uv_power_counting_preserves_ir_and_dimension_shifted_residues() {
        eprintln!("UV fixture: dotted bubble in four dimensions");
        let mut dotted = settings(2, 0);
        dotted.powers = vec![2, 1];
        let convergent = ReductionPlan::new(dotted.clone()).unwrap();
        assert!(convergent.moments[0][0].is_zero());
        assert!(!convergent.moments[0][3].is_zero());
        eprintln!("UV fixture: dotted bubble in six dimensions");
        dotted.dimension_shift = 1;
        let shifted = ReductionPlan::new(dotted).unwrap();
        assert!(!shifted.moments[0][0].is_zero());

        eprintln!("UV fixture: scaleless bubble UV/IR cancellation");
        let bubble = ReductionPlan::new(settings(2, 0))
            .unwrap()
            .with_zero_invariants(ZeroInvariantProfile {
                masses: vec![true; 2],
                distances: vec![(0, 1)],
            })
            .unwrap();
        assert_eq!(bubble.moments[0][0], Atom::num(1));
        assert_eq!(bubble.moments[0][1], Atom::num(-1));

        eprintln!("UV fixture: massless on-shell triangle IR double pole");
        let triangle = ReductionPlan::new(settings(3, 0))
            .unwrap()
            .with_zero_invariants(ZeroInvariantProfile {
                masses: vec![true; 3],
                distances: vec![(0, 1), (0, 2)],
            })
            .unwrap();
        assert!(triangle.moments[0][0].is_zero());
        assert!(!triangle.moments[0][2].is_zero());
    }

    #[test]
    #[ignore = "expensive generic five-to-seven-point symbolic preparation; run with an explicit time budget"]
    fn final_uv_power_counting_high_point_symbolic_preparation() {
        for (n, rank) in [(5, 2), (6, 0), (7, 0)] {
            let started = std::time::Instant::now();
            eprintln!("UV high-point fixture: preparing {n} propagators, rank {rank}");
            let mut configuration = settings(n, rank);
            configuration.algorithm = Algorithm::PassarinoVeltman;
            let plan = ReductionPlan::new(configuration).unwrap();
            assert!(plan.moments.iter().all(|moment| moment[0].is_zero()));
            assert!(plan.moments.iter().any(|moment| !moment[3].is_zero()));
            eprintln!(
                "UV high-point fixture: {n} propagators, rank {rank} passed in {:?}",
                started.elapsed()
            );
        }
    }
}
